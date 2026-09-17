"""Secure output-space ensembling for completed point-cloud predictions.

This module intentionally depends only on NumPy and the Python standard
library.  Source prediction trees are treated as untrusted immutable evidence:
all paths, manifests, identities, hashes, shapes, and values are checked before
an owned staging directory is atomically published without replacement.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping

import numpy as np

from pcdenoise.data.archive import (
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
)
from pcdenoise.submission import (
    _NPY_HEADER_ALLOWANCE,
    _assert_tree_unchanged,
    _decode_npy,
    _expected_directories,
    _identity,
    _open_owned_file,
    _read_owned,
    _scan_tree,
)


_FORMULA = "(1-beta)*A+beta*B"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAMPLE_ID_RE = re.compile(r"^[0-9]{8}/[0-9a-f]{28,32}$")
_MANIFEST_NAME = "inference_manifest.json"
_MAX_MANIFEST_BYTES = 16 << 20


class EnsembleValidationError(ValueError):
    """Raised when source predictions cannot form a valid ensemble."""


class _OutputRecoveryError(RuntimeError):
    """Signals that a published raw directory could not be rolled back."""


@dataclass(frozen=True)
class _SourceSample:
    sample_id: str
    relative_path: str
    point_count: int
    input_sha256: str
    output_sha256: str


@dataclass(frozen=True)
class _PredictionSource:
    tree: object
    manifest_sha256: str
    model_reference: Mapping[str, object]
    samples: Mapping[str, _SourceSample]


@dataclass
class _RecipeStage:
    path: Path
    parent: Path
    destination: Path
    descriptor: int
    device: int
    inode: int
    expected_size: int
    expected_sha256: str
    published: bool = False


def _validated_beta(beta: object) -> float:
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError("beta must be a finite real number in [0, 1]")
    value = float(beta)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("beta must be a finite real number in [0, 1]")
    # Normalize negative zero so the recipe has one canonical endpoint.
    return 0.0 if value == 0.0 else value


def _canonical_json_bytes(
    value: Mapping[str, object] | list[object],
    *,
    pretty: bool,
) -> bytes:
    options: dict[str, object] = {
        "allow_nan": False,
        "ensure_ascii": True,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
        encoded = json.dumps(value, **options) + "\n"
    else:
        options["separators"] = (",", ":")
        encoded = json.dumps(value, **options)
    return encoded.encode("utf-8")


def _duplicate_rejecting_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EnsembleValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str):
    raise EnsembleValidationError(f"non-finite JSON constant: {value}")


def _decode_manifest(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_invalid_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EnsembleValidationError(
            f"{label} inference manifest is invalid JSON"
        ) from error
    if not isinstance(value, dict):
        raise EnsembleValidationError(
            f"{label} inference manifest root must be a mapping"
        )
    return value


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _validated_model_reference(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EnsembleValidationError(
            f"{label} model_reference must be a mapping"
        )
    checkpoint_sha256 = value.get("checkpoint_sha256")
    config_sha256 = value.get("config_sha256")
    checkpoint_step = value.get("checkpoint_step")
    if not _valid_sha256(checkpoint_sha256):
        raise EnsembleValidationError(
            f"{label} checkpoint hash is invalid"
        )
    if not _valid_sha256(config_sha256):
        raise EnsembleValidationError(f"{label} config hash is invalid")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise EnsembleValidationError(
            f"{label} checkpoint step is invalid"
        )
    # Re-encode to detach the returned provenance from the decoded manifest.
    return json.loads(
        _canonical_json_bytes(value, pretty=False).decode("utf-8")
    )


def _validated_sample_ids(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise EnsembleValidationError(
            f"{label} sample IDs must be a non-empty list"
        )
    if not all(
        isinstance(sample_id, str)
        and bool(_SAMPLE_ID_RE.fullmatch(sample_id))
        for sample_id in value
    ):
        raise EnsembleValidationError(
            f"{label} sample IDs contain an invalid ID"
        )
    if len(set(value)) != len(value):
        raise EnsembleValidationError(
            f"{label} sample IDs contain duplicates"
        )
    if value != sorted(value):
        raise EnsembleValidationError(
            f"{label} sample IDs must be sorted"
        )
    return list(value)


def _scan_prediction_source(
    prediction_dir: Path | str,
    *,
    label: str,
) -> _PredictionSource:
    tree = _scan_tree(prediction_dir, label=label)
    if _MANIFEST_NAME not in tree.files:
        raise EnsembleValidationError(
            f"{label} inference manifest is required"
        )
    manifest_path = tree.root / _MANIFEST_NAME
    with _open_owned_file(
        manifest_path,
        label=f"{label} inference manifest",
        expected_identity=tree.files[_MANIFEST_NAME],
    ) as owned:
        payload = _read_owned(owned, max_bytes=_MAX_MANIFEST_BYTES)
    manifest = _decode_manifest(payload, label=label)

    if manifest.get("format") != "pcdenoise_prediction_v1":
        raise EnsembleValidationError(
            f"{label} inference manifest format is invalid"
        )
    if manifest.get("format_version") != 1:
        raise EnsembleValidationError(
            f"{label} inference manifest format_version is invalid"
        )
    if manifest.get("status") != "completed":
        raise EnsembleValidationError(
            f"{label} inference manifest status must be completed"
        )
    sample_ids = _validated_sample_ids(
        manifest.get("sample_ids"),
        label=label,
    )
    sample_count = manifest.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count != len(sample_ids)
    ):
        raise EnsembleValidationError(
            f"{label} inference manifest sample count mismatch"
        )
    model_reference = _validated_model_reference(
        manifest.get("model_reference"),
        label=label,
    )

    expected_files = {
        f"shapenet/{sample_id}/denoised.npy" for sample_id in sample_ids
    }
    allowed_files = expected_files | {_MANIFEST_NAME}
    actual_files = set(tree.files)
    if actual_files != allowed_files:
        unexpected = sorted(actual_files - allowed_files)
        missing = sorted(allowed_files - actual_files)
        raise EnsembleValidationError(
            f"{label} files are not exact; "
            f"unexpected={unexpected[:3]} missing={missing[:3]}"
        )
    expected_directories = _expected_directories(iter(sample_ids))
    actual_directories = set(tree.directories)
    if actual_directories != expected_directories:
        unexpected = sorted(actual_directories - expected_directories)
        missing = sorted(expected_directories - actual_directories)
        raise EnsembleValidationError(
            f"{label} directories are not exact; "
            f"unexpected={unexpected[:3]} missing={missing[:3]}"
        )

    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) != len(sample_ids):
        raise EnsembleValidationError(
            f"{label} sample records are incomplete"
        )
    samples: dict[str, _SourceSample] = {}
    for record in records:
        if not isinstance(record, dict):
            raise EnsembleValidationError(
                f"{label} sample record must be a mapping"
            )
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in sample_ids:
            raise EnsembleValidationError(
                f"{label} sample record has an unknown sample ID"
            )
        if sample_id in samples:
            raise EnsembleValidationError(
                f"{label} duplicate sample record: {sample_id}"
            )
        relative_path = f"shapenet/{sample_id}/denoised.npy"
        if record.get("relative_path") != relative_path:
            raise EnsembleValidationError(
                f"{label} relative path mismatch: {sample_id}"
            )
        point_count = record.get("point_count")
        if (
            isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count <= 0
        ):
            raise EnsembleValidationError(
                f"{label} point count is invalid: {sample_id}"
            )
        input_sha256 = record.get("input_sha256")
        output_sha256 = record.get("output_sha256")
        if not _valid_sha256(input_sha256):
            raise EnsembleValidationError(
                f"{label} input hash is invalid: {sample_id}"
            )
        if not _valid_sha256(output_sha256):
            raise EnsembleValidationError(
                f"{label} output hash is invalid: {sample_id}"
            )
        samples[sample_id] = _SourceSample(
            sample_id=sample_id,
            relative_path=relative_path,
            point_count=point_count,
            input_sha256=input_sha256,
            output_sha256=output_sha256,
        )
    if sorted(samples) != sample_ids:
        raise EnsembleValidationError(
            f"{label} sample records do not match sample IDs"
        )
    return _PredictionSource(
        tree=tree,
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        model_reference=model_reference,
        samples=dict(sorted(samples.items())),
    )


def _check_source_compatibility(
    source_a: _PredictionSource,
    source_b: _PredictionSource,
) -> list[str]:
    sample_ids_a = sorted(source_a.samples)
    sample_ids_b = sorted(source_b.samples)
    if sample_ids_a != sample_ids_b:
        raise EnsembleValidationError(
            "prediction source sample IDs do not match"
        )
    for sample_id in sample_ids_a:
        sample_a = source_a.samples[sample_id]
        sample_b = source_b.samples[sample_id]
        if sample_a.input_sha256 != sample_b.input_sha256:
            raise EnsembleValidationError(
                f"authoritative input hash mismatch: {sample_id}"
            )
        if sample_a.point_count != sample_b.point_count:
            raise EnsembleValidationError(
                f"prediction source point count mismatch: {sample_id}"
            )
    return sample_ids_a


def _load_prediction(
    source: _PredictionSource,
    sample: _SourceSample,
    *,
    label: str,
) -> np.ndarray:
    identity = source.tree.files[sample.relative_path]
    path = source.tree.root.joinpath(
        *PurePosixPath(sample.relative_path).parts
    )
    max_bytes = (
        sample.point_count * 3 * np.dtype(np.float32).itemsize
        + _NPY_HEADER_ALLOWANCE
    )
    with _open_owned_file(
        path,
        label=f"{label} prediction {sample.sample_id}",
        expected_identity=identity,
    ) as owned:
        payload = _read_owned(owned, max_bytes=max_bytes)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != sample.output_sha256:
        raise EnsembleValidationError(
            f"{label} output hash differs from inference manifest: "
            f"{sample.sample_id}"
        )
    values = _decode_npy(
        payload,
        label=f"{label} prediction {sample.sample_id}",
        expected_shape=(sample.point_count, 3),
    )
    return np.ascontiguousarray(values, dtype=np.float32)


def _member_recipe(source: _PredictionSource) -> dict[str, object]:
    reference = source.model_reference
    return {
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "checkpoint_step": reference["checkpoint_step"],
        "config_sha256": reference["config_sha256"],
        "inference_manifest_sha256": source.manifest_sha256,
    }


def _build_recipe(
    *,
    source_a: _PredictionSource,
    source_b: _PredictionSource,
    sample_ids: list[str],
    beta: float,
) -> tuple[dict[str, object], bytes, str, str, str]:
    members = {
        "a": _member_recipe(source_a),
        "b": _member_recipe(source_b),
    }
    primary_member = "a" if (1.0 - beta) >= beta else "b"
    config = {
        "format": "pcdenoise_ensemble_config_v1",
        "format_version": 1,
        "formula": _FORMULA,
        "beta": beta,
        "members": members,
    }
    config_sha256 = hashlib.sha256(
        _canonical_json_bytes(config, pretty=False)
    ).hexdigest()
    recipe: dict[str, object] = {
        "format": "pcdenoise_ensemble_recipe_v1",
        "format_version": 1,
        "formula": _FORMULA,
        "beta": beta,
        "primary_member": primary_member,
        "config_sha256": config_sha256,
        "members": members,
        "samples": [
            {
                "sample_id": sample_id,
                "point_count": source_a.samples[sample_id].point_count,
                "input_sha256": (
                    source_a.samples[sample_id].input_sha256
                ),
                "source_output_sha256": {
                    "a": source_a.samples[sample_id].output_sha256,
                    "b": source_b.samples[sample_id].output_sha256,
                },
            }
            for sample_id in sample_ids
        ],
    }
    payload = _canonical_json_bytes(recipe, pretty=True)
    recipe_sha256 = hashlib.sha256(payload).hexdigest()
    return (
        recipe,
        payload,
        recipe_sha256,
        config_sha256,
        primary_member,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while writing ensemble artifact")
        offset += written


def _write_output_npy(path: Path, values: np.ndarray) -> str:
    array = np.asarray(values)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] != 3
        or not np.isfinite(array).all()
    ):
        raise EnsembleValidationError(
            "ensemble output must have finite float32 shape (N,3)"
        )
    buffer = io.BytesIO()
    np.save(
        buffer,
        np.ascontiguousarray(array),
        allow_pickle=False,
    )
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _write_stage_manifest(
    path: Path,
    manifest: Mapping[str, object],
) -> str:
    payload = _canonical_json_bytes(manifest, pretty=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _hash_owned_descriptor(
    descriptor: int,
    *,
    label: str,
) -> tuple[str, object]:
    before = _identity(os.fstat(descriptor))
    if not stat.S_ISREG(before.mode):
        raise RuntimeError(f"{label} is not a regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    remaining = before.size
    while remaining:
        chunk = os.read(descriptor, min(8 << 20, remaining))
        if not chunk:
            raise RuntimeError(f"{label} ended while being hashed")
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise RuntimeError(f"{label} grew while being hashed")
    after = _identity(os.fstat(descriptor))
    if after != before:
        raise RuntimeError(f"{label} changed while being hashed")
    return digest.hexdigest(), before


def _validate_output_stage_tree(
    root: Path,
    *,
    sample_ids: list[str],
    expected_hashes: Mapping[str, str],
    label: str,
):
    tree = _scan_tree(root, label=label)
    expected_files = {
        f"shapenet/{sample_id}/denoised.npy" for sample_id in sample_ids
    } | {_MANIFEST_NAME}
    if set(expected_hashes) != expected_files:
        raise RuntimeError("ensemble stage hash contract is incomplete")
    if set(tree.files) != expected_files:
        unexpected = sorted(set(tree.files) - expected_files)
        missing = sorted(expected_files - set(tree.files))
        raise RuntimeError(
            "ensemble stage files are not exact; "
            f"unexpected={unexpected[:3]} missing={missing[:3]}"
        )
    expected_directories = _expected_directories(iter(sample_ids))
    if set(tree.directories) != expected_directories:
        unexpected = sorted(
            set(tree.directories) - expected_directories
        )
        missing = sorted(expected_directories - set(tree.directories))
        raise RuntimeError(
            "ensemble stage directories are not exact; "
            f"unexpected={unexpected[:3]} missing={missing[:3]}"
        )
    for relative_path in sorted(expected_files):
        path = tree.root.joinpath(*PurePosixPath(relative_path).parts)
        with _open_owned_file(
            path,
            label=f"{label} {relative_path}",
            expected_identity=tree.files[relative_path],
        ) as owned:
            actual_sha256, _ = _hash_owned_descriptor(
                owned.descriptor,
                label=f"{label} {relative_path}",
            )
        if actual_sha256 != expected_hashes[relative_path]:
            raise RuntimeError(
                f"ensemble stage hash changed: {relative_path}"
            )
    _assert_tree_unchanged(tree, label=label)
    return tree


def _same_tree_content(before, after) -> bool:
    return (
        before.root_identity.device == after.root_identity.device
        and before.root_identity.inode == after.root_identity.inode
        and before.directories == after.directories
        and before.files == after.files
    )


def _staging_path_is_owned(stage) -> bool:
    try:
        metadata = os.lstat(stage.path)
    except FileNotFoundError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_dev == stage.device
        and metadata.st_ino == stage.inode
    )


def _publish_verified_output_stage(
    stage,
    *,
    sample_ids: list[str],
    expected_hashes: Mapping[str, str],
    boundary_check: Callable[[], None],
) -> Path:
    before = _validate_output_stage_tree(
        stage.path,
        sample_ids=sample_ids,
        expected_hashes=expected_hashes,
        label="ensemble output stage",
    )
    if (
        before.root_identity.device != stage.device
        or before.root_identity.inode != stage.inode
    ):
        raise RuntimeError("ensemble output stage identity changed")
    boundary_check()
    publication_completed = False
    try:
        published = _publish_owned_stage(stage)
        publication_completed = True
        if published != stage.destination:
            raise RuntimeError(
                "ensemble output published to an unexpected path"
            )
        after = _validate_output_stage_tree(
            published,
            sample_ids=sample_ids,
            expected_hashes=expected_hashes,
            label="published ensemble output",
        )
        if not _same_tree_content(before, after):
            raise RuntimeError(
                "ensemble output stage changed during publication"
            )
        boundary_check()
        return published
    except BaseException as error:
        if publication_completed or not _staging_path_is_owned(stage):
            # There is no race-free POSIX operation that renames a directory
            # only if its pathname still resolves to an already-open inode.
            # Once publication may have occurred, never mutate that pathname
            # again.  Preserve the matching recipe and report an explicit
            # recovery state instead of risking a foreign namespace entry.
            raise _OutputRecoveryError(
                "ensemble output publication completed or became "
                "uncertain before validation finished"
            ) from error
        raise


def _recipe_destination(path: Path | str) -> tuple[Path, Path]:
    requested = Path(path)
    if not requested.name:
        raise ValueError(f"recipe_output must name a new file: {requested}")
    if os.path.lexists(os.fspath(requested)):
        raise FileExistsError(f"refusing to overwrite recipe: {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve()
    destination = parent / requested.name
    if os.path.lexists(os.fspath(destination)):
        raise FileExistsError(f"refusing to overwrite recipe: {destination}")
    return parent, destination


def _create_recipe_stage(
    recipe_output: Path | str,
    payload: bytes,
) -> _RecipeStage:
    parent, destination = _recipe_destination(recipe_output)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.ensemble-recipe-",
        dir=os.fspath(parent),
    )
    path = Path(raw_path)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"recipe stage is not a regular file: {path}"
            )
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        if metadata.st_size != len(payload):
            raise RuntimeError(
                "ensemble recipe stage has an unexpected size"
            )
        return _RecipeStage(
            path=path,
            parent=parent,
            destination=destination,
            descriptor=descriptor,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            expected_size=len(payload),
            expected_sha256=expected_sha256,
        )
    except BaseException:
        os.close(descriptor)
        if os.path.lexists(os.fspath(path)):
            os.unlink(path)
        raise


def _recipe_identity_matches(stage: _RecipeStage, path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_dev == stage.device
        and metadata.st_ino == stage.inode
    )


def _publish_recipe_stage(stage: _RecipeStage) -> None:
    descriptor_metadata = os.fstat(stage.descriptor)
    if (
        not stat.S_ISREG(descriptor_metadata.st_mode)
        or descriptor_metadata.st_dev != stage.device
        or descriptor_metadata.st_ino != stage.inode
        or not _recipe_identity_matches(stage, stage.path)
    ):
        raise RuntimeError("ensemble recipe stage identity changed")
    if os.path.lexists(os.fspath(stage.destination)):
        raise FileExistsError(
            f"refusing to overwrite recipe: {stage.destination}"
        )
    parent_descriptor = os.open(
        stage.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.link(
            stage.path.name,
            stage.destination.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        stage.published = True
        if not _recipe_identity_matches(stage, stage.destination):
            raise RuntimeError(
                "published ensemble recipe identity changed"
            )
        os.unlink(stage.path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _verify_published_recipe(stage: _RecipeStage) -> None:
    if not stage.published or not _recipe_identity_matches(
        stage,
        stage.destination,
    ):
        raise RuntimeError("published ensemble recipe identity changed")
    actual_sha256, descriptor_identity = _hash_owned_descriptor(
        stage.descriptor,
        label="published ensemble recipe",
    )
    if (
        descriptor_identity.device != stage.device
        or descriptor_identity.inode != stage.inode
        or descriptor_identity.size != stage.expected_size
        or actual_sha256 != stage.expected_sha256
    ):
        raise RuntimeError(
            "published ensemble recipe hash or size changed"
        )
    destination_identity = _identity(os.lstat(stage.destination))
    if destination_identity != descriptor_identity:
        raise RuntimeError(
            "published ensemble recipe path changed while being verified"
        )
    if os.fstat(stage.descriptor).st_nlink < 1:
        raise RuntimeError("published ensemble recipe was unlinked")


def _cleanup_recipe_stage(stage: _RecipeStage) -> None:
    cleanup_error: BaseException | None = None
    try:
        if os.path.lexists(os.fspath(stage.path)):
            if not _recipe_identity_matches(stage, stage.path):
                raise RuntimeError(
                    "refusing to remove changed ensemble recipe stage"
                )
            os.unlink(stage.path)
        if stage.published and os.path.lexists(
            os.fspath(stage.destination)
        ):
            if not _recipe_identity_matches(stage, stage.destination):
                raise RuntimeError(
                    "refusing to remove changed published ensemble recipe"
                )
            os.unlink(stage.destination)
    except BaseException as error:
        cleanup_error = error
    finally:
        os.close(stage.descriptor)
    if cleanup_error is not None:
        raise cleanup_error


def _close_recipe_stage(stage: _RecipeStage) -> None:
    """Release the owned FD without touching a reused temporary pathname."""

    descriptor = stage.descriptor
    stage.descriptor = -1
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except OSError:
        # At a successful publication boundary the descriptor has no further
        # durability role.  Retrying close is unsafe because the fd number may
        # already have been reused, and a close error must not turn a complete
        # raw+recipe pair into a reported publication failure.
        pass


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_destination_separation(
    *,
    pred_a: Path | str,
    pred_b: Path | str,
    output_dir: Path | str,
    recipe_output: Path | str,
) -> None:
    source_roots = [
        Path(pred_a).resolve(),
        Path(pred_b).resolve(),
    ]
    output = Path(output_dir).resolve(strict=False)
    recipe = Path(recipe_output).resolve(strict=False)
    if _path_within(recipe, output):
        raise ValueError("recipe_output must be outside output_dir")
    for source in source_roots:
        if _path_within(output, source) or _path_within(recipe, source):
            raise ValueError(
                "ensemble destinations must be outside prediction sources"
            )


def build_prediction_ensemble(
    *,
    pred_a: Path | str,
    pred_b: Path | str,
    output_dir: Path | str,
    beta: float = 0.75,
    recipe_output: Path | str,
) -> dict[str, object]:
    """Validate, blend, and atomically publish two prediction directories.

    The exact float32 operation is ``(1-beta) * A + beta * B``.  Member A is
    the primary member for equal weights; otherwise the larger-weight member
    supplies ``model_reference.checkpoint_step``.
    """

    weight_b = _validated_beta(beta)
    output = Path(output_dir)
    recipe_path = Path(recipe_output)
    if not output.name:
        raise ValueError(f"output_dir must name a new directory: {output}")
    if os.path.lexists(os.fspath(output)):
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if os.path.lexists(os.fspath(recipe_path)):
        raise FileExistsError(
            f"refusing to overwrite recipe: {recipe_path}"
        )
    _validate_destination_separation(
        pred_a=pred_a,
        pred_b=pred_b,
        output_dir=output,
        recipe_output=recipe_path,
    )

    source_a = _scan_prediction_source(pred_a, label="prediction A")
    source_b = _scan_prediction_source(pred_b, label="prediction B")
    sample_ids = _check_source_compatibility(source_a, source_b)
    (
        recipe,
        recipe_payload,
        recipe_sha256,
        config_sha256,
        primary_member,
    ) = _build_recipe(
        source_a=source_a,
        source_b=source_b,
        sample_ids=sample_ids,
        beta=weight_b,
    )

    stage = _create_owned_stage(output)
    recipe_stage: _RecipeStage | None = None
    published_output: Path | None = None
    sample_records: list[dict[str, object]] = []
    stage_hashes: dict[str, str] = {}
    weight_a32 = np.float32(1.0 - weight_b)
    weight_b32 = np.float32(weight_b)
    try:
        for sample_id in sample_ids:
            sample_a = source_a.samples[sample_id]
            sample_b = source_b.samples[sample_id]
            values_a = _load_prediction(
                source_a,
                sample_a,
                label="prediction A",
            )
            values_b = _load_prediction(
                source_b,
                sample_b,
                label="prediction B",
            )
            if values_a.shape != values_b.shape:
                raise EnsembleValidationError(
                    f"prediction source shape mismatch: {sample_id}"
                )
            blended = np.add(
                np.multiply(
                    values_a,
                    weight_a32,
                    dtype=np.float32,
                ),
                np.multiply(
                    values_b,
                    weight_b32,
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
            if (
                blended.shape != values_a.shape
                or blended.dtype != np.float32
                or not np.isfinite(blended).all()
            ):
                raise EnsembleValidationError(
                    f"ensemble output is invalid: {sample_id}"
                )
            relative_path = sample_a.relative_path
            output_sha256 = _write_output_npy(
                stage.path.joinpath(
                    *PurePosixPath(relative_path).parts
                ),
                blended,
            )
            stage_hashes[relative_path] = output_sha256
            sample_records.append(
                {
                    "sample_id": sample_id,
                    "point_count": sample_a.point_count,
                    "input_sha256": sample_a.input_sha256,
                    "output_sha256": output_sha256,
                    "relative_path": relative_path,
                    "details": {
                        "ensemble_formula": _FORMULA,
                        "ensemble_beta": weight_b,
                        "source_output_sha256": {
                            "a": sample_a.output_sha256,
                            "b": sample_b.output_sha256,
                        },
                    },
                }
            )

        members = recipe["members"]
        primary_step = members[primary_member]["checkpoint_step"]
        model_reference = {
            "checkpoint_sha256": recipe_sha256,
            "checkpoint_step": primary_step,
            "config_sha256": config_sha256,
            "ensemble": {
                "formula": _FORMULA,
                "beta": weight_b,
                "primary_member": primary_member,
                "recipe_sha256": recipe_sha256,
                "members": members,
            },
        }
        manifest = {
            "format": "pcdenoise_prediction_v1",
            "format_version": 1,
            "status": "completed",
            "sample_count": len(sample_ids),
            "sample_ids": sample_ids,
            "model_reference": model_reference,
            "samples": sample_records,
        }
        manifest_sha256 = _write_stage_manifest(
            stage.path / _MANIFEST_NAME,
            manifest,
        )
        stage_hashes[_MANIFEST_NAME] = manifest_sha256

        _assert_tree_unchanged(source_a.tree, label="prediction A")
        _assert_tree_unchanged(source_b.tree, label="prediction B")
        recipe_stage = _create_recipe_stage(
            recipe_path,
            recipe_payload,
        )
        _publish_recipe_stage(recipe_stage)
        published_output = _publish_verified_output_stage(
            stage,
            sample_ids=sample_ids,
            expected_hashes=stage_hashes,
            boundary_check=lambda: _verify_published_recipe(
                recipe_stage
            ),
        )
    except BaseException as error:
        cleanup_errors: list[BaseException] = []
        preserve_recipe = isinstance(error, _OutputRecoveryError)
        try:
            _cleanup_owned_stage(stage)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if recipe_stage is not None:
            try:
                if preserve_recipe:
                    # A filesystem-level recovery failure left the owned raw
                    # publication state uncertain.  Never perform another
                    # pathname rename: keep the exact recipe visible as
                    # auditable evidence rather than risk touching a foreign
                    # namespace entry.
                    _close_recipe_stage(recipe_stage)
                else:
                    _cleanup_recipe_stage(recipe_stage)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            details = "; ".join(
                str(cleanup_error)
                for cleanup_error in cleanup_errors
            )
            raise RuntimeError(
                f"ensemble operation failed; cleanup failed: {details}"
            ) from error
        raise

    assert recipe_stage is not None
    _close_recipe_stage(recipe_stage)
    return {
        "status": "completed",
        "output_dir": str(published_output),
        "recipe_path": str(recipe_stage.destination),
        "recipe_sha256": recipe_sha256,
        "inference_manifest_sha256": manifest_sha256,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "formula": _FORMULA,
        "beta": weight_b,
        "primary_member": primary_member,
        "model_reference": model_reference,
    }


__all__ = [
    "EnsembleValidationError",
    "build_prediction_ensemble",
]
