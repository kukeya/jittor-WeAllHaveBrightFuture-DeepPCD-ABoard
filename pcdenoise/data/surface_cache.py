"""Deterministic, immutable clean-surface cache for PGD training.

The cache is generated with NumPy only.  Every selected mesh is sampled with
an independent RNG stream and normalized with the official bbox-center /
maximum-radius transform before publication.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import math
import os
import stat
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .archive import (
    MODEL_RE,
    SYNSET_RE,
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
)
from .mesh_dataset import fit_unit_sphere, load_obj_bytes, sample_surface
from .split import _validated_split_manifest


TRAIN_SPLIT_FORMAT = "pcdenoise_shape_ids_v1"
SURFACE_CACHE_FORMAT = "pcdenoise_train_surface_cache_v1"
SURFACE_CACHE_VERSION = 1
_SELECTION_ALGORITHM = (
    "coverage_first_capacity_largest_remainder_then_sha256_rank_v1"
)
_SELECTION_RANK_DERIVATION = (
    "sha256(le_u64(seed)||NUL||ascii(surface-cache-selection-v1)||"
    "NUL||ascii(shape_id)); ascending(digest,shape_id)"
)
_QUOTA_FORMULA = (
    "one_per_synset_then_floor((target-S)*(count-1)/"
    "sum(count-1)); leftover by descending integer remainder, "
    "ascending synset tie-break"
)
_SEED_DERIVATION = (
    "le_u64(first8(sha256(le_u64(global_seed)||NUL||"
    "ascii(surface-cache-sampling-v1)||NUL||ascii(shape_id))))"
)
_RNG_STREAM_POLICY = "one_independent_surface_sampling_stream_per_shape"
_TRAIN_SPLIT_KEYS = frozenset(
    (
        "format",
        "format_version",
        "split",
        "count",
        "shape_ids",
        "split_sha256",
    )
)
_RNG_KEYS = frozenset(
    (
        "library",
        "numpy_version",
        "generator",
        "bit_generator",
        "stream_policy",
    )
)
_SAMPLE_KEYS = frozenset(
    (
        "shape_id",
        "derived_seed",
        "mesh_sha256",
        "clean_sha256",
        "clean_file_bytes",
        "relative_path",
        "normalization_center",
        "normalization_scale",
    )
)
_MANIFEST_KEYS = frozenset(
    (
        "format",
        "format_version",
        "shape_count",
        "shape_ids",
        "num_points",
        "array_contract",
        "coordinate_frame",
        "seed",
        "seed_derivation",
        "rng",
        "selection",
        "split_sha256",
        "train_split_sha256",
        "split_manifest_sha256",
        "source_sha256",
        "samples_sha256",
        "samples",
        "content_sha256",
    )
)
_SELECTION_KEYS = frozenset(
    (
        "algorithm",
        "seed",
        "source_shape_count",
        "target_shape_count",
        "synset_count",
        "coverage_policy",
        "quota_formula",
        "rank_derivation",
        "per_synset",
        "selected_shape_ids",
    )
)
_PER_SYNSET_KEYS = frozenset(
    (
        "source_count",
        "capacity_after_coverage",
        "base_coverage_count",
        "proportional_floor",
        "remainder_numerator",
        "selected_count",
        "selected_shape_ids",
    )
)
_RENAME_NOREPLACE = 1


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) < 2**64
    ):
        raise ValueError("seed must be an integer in [0, 2**64)")
    return int(value)


def _shape_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("every shape ID must be a string")
    parts = value.split("/")
    if (
        len(parts) != 2
        or SYNSET_RE.fullmatch(parts[0]) is None
        or MODEL_RE.fullmatch(parts[1]) is None
    ):
        raise ValueError(
            "shape ID must have form <8-digit synset>/<28-32 hex model>"
        )
    return value


def _sha256_value(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _snapshot_descriptor(
    descriptor: int,
    *,
    source: str,
) -> tuple[bytes, str]:
    """Read and hash exactly one pinned regular-file descriptor."""

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"source must be a regular file: {source}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 8 << 20)
        if not chunk:
            break
        digest.update(chunk)
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise RuntimeError(f"source changed while reading: {source}")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise RuntimeError(f"source size changed while reading: {source}")
    return payload, digest.hexdigest()


def _require_nofollow() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError(
            getattr(os, "ENOTSUP", 95),
            "O_NOFOLLOW is required; unsafe fallback is forbidden",
        )
    return int(nofollow)


def _open_directory(path: os.PathLike[str] | str) -> int:
    """Open a directory without following any component of its path."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | _require_nofollow()
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                flags,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"source must be a directory: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _snapshot_at(
    directory_descriptor: int,
    name: str,
    *,
    source: str,
) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _require_nofollow()
    descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    try:
        return _snapshot_descriptor(descriptor, source=source)
    finally:
        os.close(descriptor)


def _snapshot_regular_path(path: Path) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _require_nofollow()
    descriptor = os.open(path, flags)
    try:
        return _snapshot_descriptor(descriptor, source=os.fspath(path))
    finally:
        os.close(descriptor)


def _open_child_directory(
    parent_descriptor: int,
    name: str,
) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | _require_nofollow(),
        dir_fd=parent_descriptor,
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"path component is not a directory: {name}")
    return descriptor


def _snapshot_relative_file(
    root_descriptor: int,
    relative_path: str,
) -> tuple[bytes, str]:
    """Snapshot a relative file without following any intermediate symlink."""

    parts = Path(relative_path).parts
    if (
        not parts
        or Path(relative_path).is_absolute()
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"unsafe relative cache path: {relative_path!r}")
    current = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            next_descriptor = _open_child_directory(current, component)
            os.close(current)
            current = next_descriptor
        return _snapshot_at(
            current,
            parts[-1],
            source=relative_path,
        )
    finally:
        os.close(current)


def _read_json(payload: bytes, *, label: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error


def _read_train_bundle(
    train_split: Path,
    *,
    expected_train_split_sha256: str | None = None,
    expected_split_manifest_sha256: str | None = None,
) -> tuple[list[str], str, str, str]:
    """Read train.json and its sibling manifest through one pinned directory."""

    if expected_train_split_sha256 is None:
        raise ValueError("expected train split SHA256 is required")
    if expected_split_manifest_sha256 is None:
        raise ValueError("expected split manifest SHA256 is required")
    expected_train = _sha256_value(
        expected_train_split_sha256,
        name="expected train split SHA256",
    )
    expected_manifest = _sha256_value(
        expected_split_manifest_sha256,
        name="expected split manifest SHA256",
    )
    if train_split.name != "train.json":
        raise ValueError("train_split must name train.json")
    parent_descriptor = _open_directory(train_split.parent)
    try:
        train_payload, train_actual = _snapshot_at(
            parent_descriptor,
            "train.json",
            source=os.fspath(train_split),
        )
        manifest_payload, manifest_actual = _snapshot_at(
            parent_descriptor,
            "manifest.json",
            source=os.fspath(train_split.with_name("manifest.json")),
        )
    finally:
        os.close(parent_descriptor)
    if train_actual != expected_train:
        raise ValueError("train split actual SHA does not match expected SHA")
    if manifest_actual != expected_manifest:
        raise ValueError(
            "split manifest actual SHA does not match expected SHA"
        )

    train_document = _read_json(train_payload, label="train split")
    manifest_document = _read_json(
        manifest_payload,
        label="sibling split manifest",
    )
    if not isinstance(train_document, dict):
        raise ValueError("train split must be a JSON object")
    if set(train_document) != _TRAIN_SPLIT_KEYS:
        raise ValueError("train split fields are invalid")
    if (
        train_document.get("format") != TRAIN_SPLIT_FORMAT
        or isinstance(train_document.get("format_version"), bool)
        or train_document.get("format_version") != 1
        or train_document.get("split") != "train"
    ):
        raise ValueError("train split metadata is invalid")
    raw_ids = train_document.get("shape_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("train split shape_ids must be a nonempty list")
    shape_ids = [_shape_id(value) for value in raw_ids]
    if (
        shape_ids != sorted(shape_ids)
        or len(set(shape_ids)) != len(shape_ids)
        or isinstance(train_document.get("count"), bool)
        or train_document.get("count") != len(shape_ids)
    ):
        raise ValueError("train split shape ordering/count is invalid")
    split_sha256 = _sha256_value(
        train_document.get("split_sha256"),
        name="train split_sha256",
    )
    try:
        validated_manifest = _validated_split_manifest(manifest_document)
    except ValueError as error:
        raise ValueError("sibling split manifest is invalid") from error
    if (
        validated_manifest.get("train") != shape_ids
        or validated_manifest.get("counts", {}).get("train") != len(shape_ids)
        or validated_manifest.get("split_sha256") != split_sha256
    ):
        raise ValueError("train split and sibling manifest do not match")
    return shape_ids, split_sha256, train_actual, manifest_actual


def _selection_rank(seed: int, shape_id: str) -> bytes:
    return hashlib.sha256(
        seed.to_bytes(8, "little", signed=False)
        + b"\0surface-cache-selection-v1\0"
        + shape_id.encode("ascii")
    ).digest()


def select_shape_ids(
    shape_ids: Iterable[str],
    *,
    select_count: int,
    seed: int,
) -> tuple[list[str], dict[str, object]]:
    """Select a category-covering subset with exact integer apportionment."""

    target = _positive_integer(select_count, name="select_count")
    global_seed = _seed(seed)
    ids = [_shape_id(value) for value in shape_ids]
    if not ids:
        raise ValueError("shape_ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError("shape_ids must not contain duplicates")
    ids.sort()
    if target > len(ids):
        raise ValueError("select_count exceeds the train split size")

    groups: dict[str, list[str]] = defaultdict(list)
    for shape_id in ids:
        groups[shape_id.split("/", 1)[0]].append(shape_id)
    synsets = sorted(groups)
    if target < len(synsets):
        raise ValueError(
            "select_count is too small to cover every train synset"
        )

    remaining = target - len(synsets)
    capacities = {
        synset: len(groups[synset]) - 1 for synset in synsets
    }
    total_capacity = sum(capacities.values())
    quotas: dict[str, int] = {}
    floors: dict[str, int] = {}
    remainders: dict[str, int] = {}
    if total_capacity == 0:
        if remaining != 0:
            raise ValueError("selection capacity is inconsistent")
        for synset in synsets:
            floors[synset] = 0
            remainders[synset] = 0
            quotas[synset] = 1
    else:
        for synset in synsets:
            numerator = remaining * capacities[synset]
            floors[synset], remainders[synset] = divmod(
                numerator,
                total_capacity,
            )
            quotas[synset] = 1 + floors[synset]
        unassigned = target - sum(quotas.values())
        remainder_order = sorted(
            synsets,
            key=lambda synset: (-remainders[synset], synset),
        )
        for synset in remainder_order[:unassigned]:
            quotas[synset] += 1

    selected: list[str] = []
    per_synset: dict[str, dict[str, object]] = {}
    for synset in synsets:
        ranked = sorted(
            groups[synset],
            key=lambda shape_id: (
                _selection_rank(global_seed, shape_id),
                shape_id,
            ),
        )
        chosen = sorted(ranked[: quotas[synset]])
        selected.extend(chosen)
        per_synset[synset] = {
            "source_count": len(groups[synset]),
            "capacity_after_coverage": capacities[synset],
            "base_coverage_count": 1,
            "proportional_floor": floors[synset],
            "remainder_numerator": remainders[synset],
            "selected_count": len(chosen),
            "selected_shape_ids": chosen,
        }
    selected.sort()
    if len(selected) != target or len(set(selected)) != target:
        raise RuntimeError("selection algorithm violated its exact-size contract")
    contract: dict[str, object] = {
        "algorithm": _SELECTION_ALGORITHM,
        "seed": global_seed,
        "source_shape_count": len(ids),
        "target_shape_count": target,
        "synset_count": len(synsets),
        "coverage_policy": "select_at_least_one_shape_from_every_train_synset",
        "quota_formula": _QUOTA_FORMULA,
        "rank_derivation": _SELECTION_RANK_DERIVATION,
        "per_synset": per_synset,
        "selected_shape_ids": selected,
    }
    return selected, contract


def _derived_seed(global_seed: int, shape_id: str) -> int:
    digest = hashlib.sha256(
        global_seed.to_bytes(8, "little", signed=False)
        + b"\0surface-cache-sampling-v1\0"
        + shape_id.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _snapshot_mesh_payload_from_root(
    root_descriptor: int,
    shape_id: str,
) -> tuple[bytes, str]:
    parts = _shape_id(shape_id).split("/")
    opened: list[int] = []
    current = root_descriptor
    try:
        for component in (*parts, "models"):
            descriptor = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | _require_nofollow(),
                dir_fd=current,
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(descriptor)
                raise ValueError(
                    f"mesh path component is not a directory: {component}"
                )
            opened.append(descriptor)
            current = descriptor
        return _snapshot_at(
            current,
            "model_normalized.obj",
            source=f"{shape_id}/models/model_normalized.obj",
        )
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _snapshot_mesh_from_root(
    root_descriptor: int,
    shape_id: str,
) -> tuple[object, str]:
    payload, mesh_sha256 = _snapshot_mesh_payload_from_root(
        root_descriptor,
        shape_id,
    )
    mesh = load_obj_bytes(payload)
    return mesh, mesh_sha256


def _generate_surface_sample_from_root(
    root_descriptor: int,
    *,
    shape_id: str,
    num_points: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    per_shape_seed = _derived_seed(seed, shape_id)
    mesh, mesh_sha256 = _snapshot_mesh_from_root(
        root_descriptor,
        shape_id,
    )
    generator = np.random.default_rng(per_shape_seed)
    raw_clean = sample_surface(mesh, num_points, generator)
    transform = fit_unit_sphere(raw_clean)
    clean = np.ascontiguousarray(transform.apply(raw_clean), dtype=np.float32)
    if clean.shape != (num_points, 3) or not np.isfinite(clean).all():
        raise RuntimeError("surface sample violated the array contract")
    metadata: dict[str, object] = {
        "shape_id": shape_id,
        "derived_seed": per_shape_seed,
        "mesh_sha256": mesh_sha256,
        "normalization_center": [
            float(value) for value in transform.center
        ],
        "normalization_scale": float(transform.scale),
    }
    return clean, metadata


def reproduce_surface_sample(
    *,
    mesh_root: os.PathLike[str] | str,
    shape_id: str,
    num_points: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Regenerate one selected cloud from its pinned mesh snapshot."""

    count = _positive_integer(num_points, name="num_points")
    global_seed = _seed(seed)
    validated_shape_id = _shape_id(shape_id)
    root_descriptor = _open_directory(mesh_root)
    try:
        return _generate_surface_sample_from_root(
            root_descriptor,
            shape_id=validated_shape_id,
            num_points=count,
            seed=global_seed,
        )
    finally:
        os.close(root_descriptor)


def _npy_payload(array: np.ndarray, *, num_points: int) -> bytes:
    values = np.asarray(array)
    if (
        values.dtype != np.float32
        or values.shape != (num_points, 3)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("surface cache array contract was violated")
    stream = io.BytesIO()
    np.save(
        stream,
        np.ascontiguousarray(values),
        allow_pickle=False,
    )
    return stream.getvalue()


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | _require_nofollow()
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"output is not a regular file: {path}")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing cache file")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    _write_exclusive(path, payload)


def _fsync_directory_tree(directory_descriptor: int) -> None:
    """Persist every already-fsynced file entry before directory publication."""

    for name in sorted(os.listdir(directory_descriptor)):
        metadata = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_child_directory(directory_descriptor, name)
            try:
                pinned = os.fstat(child)
                if (
                    pinned.st_dev != metadata.st_dev
                    or pinned.st_ino != metadata.st_ino
                ):
                    raise RuntimeError(
                        f"output directory changed before fsync: {name}"
                    )
                _fsync_directory_tree(child)
            finally:
                os.close(child)
        elif not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"unsafe output entry before publication: {name}"
            )
    os.fsync(directory_descriptor)


def _rename_noreplace_in_parent(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace rollback is unavailable",
            destination_name,
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _durably_publish_owned_stage(stage: object) -> Path:
    stage_descriptor = _open_directory(stage.path)
    try:
        metadata = os.fstat(stage_descriptor)
        if (
            metadata.st_dev != stage.device
            or metadata.st_ino != stage.inode
        ):
            raise RuntimeError("staging identity changed before fsync")
        _fsync_directory_tree(stage_descriptor)
    finally:
        os.close(stage_descriptor)
    parent_descriptor = _open_directory(stage.parent)
    try:
        os.fsync(parent_descriptor)
        published = _publish_owned_stage(stage)
        try:
            os.fsync(parent_descriptor)
        except BaseException as publish_sync_error:
            try:
                published_metadata = os.stat(
                    stage.destination.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(published_metadata.st_mode)
                    or published_metadata.st_dev != stage.device
                    or published_metadata.st_ino != stage.inode
                ):
                    raise RuntimeError(
                        "published directory identity changed before rollback"
                    )
                _rename_noreplace_in_parent(
                    parent_descriptor,
                    stage.destination.name,
                    stage.path.name,
                )
                restored = os.stat(
                    stage.path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(restored.st_mode)
                    or restored.st_dev != stage.device
                    or restored.st_ino != stage.inode
                ):
                    raise RuntimeError(
                        "rolled-back staging directory identity changed"
                    )
            except BaseException as rollback_error:
                raise RuntimeError(
                    "post-publication fsync failed and the exact output "
                    "could not be rolled back"
                ) from rollback_error
            try:
                os.fsync(parent_descriptor)
            except OSError as rollback_sync_error:
                publish_sync_error.add_note(
                    "rollback succeeded, but parent fsync also failed: "
                    f"{rollback_sync_error}"
                )
            raise publish_sync_error
        return published
    finally:
        os.close(parent_descriptor)


def _source_hashes() -> dict[str, str]:
    module_dir = Path(__file__).resolve().parent
    paths = {
        "pcdenoise/data/archive.py": module_dir / "archive.py",
        "pcdenoise/data/mesh_dataset.py": module_dir / "mesh_dataset.py",
        "pcdenoise/data/split.py": module_dir / "split.py",
        "pcdenoise/data/surface_cache.py": Path(__file__).resolve(),
    }
    result = {}
    for name, path in paths.items():
        _, result[name] = _snapshot_regular_path(path)
    return dict(sorted(result.items()))


def _manifest_content_digest(manifest: Mapping[str, object]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    return _canonical_digest(unsigned)


def _sample_records_digest(records: Sequence[Mapping[str, object]]) -> str:
    return _canonical_digest(list(records))


def build_surface_cache(
    *,
    mesh_root: os.PathLike[str] | str,
    train_split: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    select_count: int = 4096,
    num_points: int = 50_000,
    seed: int = 20260726,
    workers: int = 1,
    expected_train_split_sha256: str | None = None,
    expected_split_manifest_sha256: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Build and atomically publish a deterministic clean training cache."""

    target = _positive_integer(select_count, name="select_count")
    count = _positive_integer(num_points, name="num_points")
    global_seed = _seed(seed)
    worker_count = _positive_integer(workers, name="workers")
    if worker_count > 256:
        raise ValueError("workers must not exceed 256")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    source_sha256 = _source_hashes()
    (
        train_ids,
        split_sha256,
        train_split_sha256,
        split_manifest_sha256,
    ) = _read_train_bundle(
        Path(train_split),
        expected_train_split_sha256=expected_train_split_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
    )
    selected_ids, selection = select_shape_ids(
        train_ids,
        select_count=target,
        seed=global_seed,
    )
    root_descriptor = _open_directory(mesh_root)
    stage = None
    try:
        stage = _create_owned_stage(output_dir)
        shapenet_dir = stage.path / "shapenet"
        shapenet_dir.mkdir()
        for synset in sorted(
            {shape_id.split("/", 1)[0] for shape_id in selected_ids}
        ):
            (shapenet_dir / synset).mkdir()

        def generate_and_write(shape_id: str) -> dict[str, object]:
            clean, metadata = _generate_surface_sample_from_root(
                root_descriptor,
                shape_id=shape_id,
                num_points=count,
                seed=global_seed,
            )
            synset, model = shape_id.split("/")
            sample_dir = shapenet_dir / synset / model
            sample_dir.mkdir()
            relative_path = (
                Path("shapenet") / synset / model / "clean.npy"
            ).as_posix()
            payload = _npy_payload(clean, num_points=count)
            _write_exclusive(sample_dir / "clean.npy", payload)
            return {
                **metadata,
                "clean_sha256": hashlib.sha256(payload).hexdigest(),
                "clean_file_bytes": len(payload),
                "relative_path": relative_path,
            }

        records: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            iterator = executor.map(generate_and_write, selected_ids)
            for completed, (shape_id, record) in enumerate(
                zip(selected_ids, iterator),
                start=1,
            ):
                if record.get("shape_id") != shape_id:
                    raise RuntimeError("worker result order changed")
                records.append(record)
                if progress is not None:
                    progress(completed, len(selected_ids), shape_id)

        if _source_hashes() != source_sha256:
            raise RuntimeError(
                "surface-cache source files changed during build"
            )
        manifest: dict[str, object] = {
            "format": SURFACE_CACHE_FORMAT,
            "format_version": SURFACE_CACHE_VERSION,
            "shape_count": len(selected_ids),
            "shape_ids": selected_ids,
            "num_points": count,
            "array_contract": {
                "container": "npy",
                "dtype": "float32",
                "shape": [count, 3],
            },
            "coordinate_frame": (
                "per_shape_clean_sample_bbox_center_max_radius_unit_sphere"
            ),
            "seed": global_seed,
            "seed_derivation": _SEED_DERIVATION,
            "rng": {
                "library": "numpy",
                "numpy_version": np.__version__,
                "generator": "numpy.random.Generator",
                "bit_generator": type(
                    np.random.default_rng(0).bit_generator
                ).__name__,
                "stream_policy": _RNG_STREAM_POLICY,
            },
            "selection": selection,
            "split_sha256": split_sha256,
            "train_split_sha256": train_split_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "source_sha256": source_sha256,
            "samples_sha256": _sample_records_digest(records),
            "samples": records,
        }
        manifest["content_sha256"] = _manifest_content_digest(manifest)
        _write_json(stage.path / "manifest.json", manifest)
        _durably_publish_owned_stage(stage)
        return manifest
    except BaseException:
        if stage is not None:
            _cleanup_owned_stage(stage)
        raise
    finally:
        os.close(root_descriptor)


def _validate_selection_contract(
    selection: object,
    *,
    shape_ids: list[str],
    global_seed: int,
) -> None:
    if not isinstance(selection, dict) or set(selection) != _SELECTION_KEYS:
        raise ValueError("surface cache selection contract is invalid")
    if (
        selection.get("algorithm") != _SELECTION_ALGORITHM
        or isinstance(selection.get("seed"), bool)
        or selection.get("seed") != global_seed
        or isinstance(selection.get("target_shape_count"), bool)
        or selection.get("target_shape_count") != len(shape_ids)
        or selection.get("selected_shape_ids") != shape_ids
        or selection.get("coverage_policy")
        != "select_at_least_one_shape_from_every_train_synset"
        or selection.get("rank_derivation") != _SELECTION_RANK_DERIVATION
        or selection.get("quota_formula") != _QUOTA_FORMULA
    ):
        raise ValueError("surface cache selection metadata is invalid")
    source_count = _positive_integer(
        selection.get("source_shape_count"),
        name="selection source_shape_count",
    )
    if source_count < len(shape_ids):
        raise ValueError("selection source count is too small")
    per_synset = selection.get("per_synset")
    if not isinstance(per_synset, dict) or not per_synset:
        raise ValueError("selection per_synset is invalid")
    if (
        isinstance(selection.get("synset_count"), bool)
        or selection.get("synset_count") != len(per_synset)
    ):
        raise ValueError("selection synset count is invalid")
    selected_union: list[str] = []
    source_sum = 0
    for synset in sorted(per_synset):
        if SYNSET_RE.fullmatch(synset) is None:
            raise ValueError("selection contains an invalid synset")
        entry = per_synset[synset]
        if not isinstance(entry, dict) or set(entry) != _PER_SYNSET_KEYS:
            raise ValueError("selection per-synset fields are invalid")
        group_ids_raw = entry.get("selected_shape_ids")
        if not isinstance(group_ids_raw, list) or not group_ids_raw:
            raise ValueError("selection per-synset IDs are invalid")
        group_ids = [_shape_id(value) for value in group_ids_raw]
        if (
            group_ids != sorted(group_ids)
            or any(not item.startswith(f"{synset}/") for item in group_ids)
            or isinstance(entry.get("selected_count"), bool)
            or entry.get("selected_count") != len(group_ids)
            or isinstance(entry.get("base_coverage_count"), bool)
            or entry.get("base_coverage_count") != 1
        ):
            raise ValueError("selection per-synset contract is invalid")
        group_source_count = _positive_integer(
            entry.get("source_count"),
            name=f"source_count for {synset}",
        )
        capacity = entry.get("capacity_after_coverage")
        floor = entry.get("proportional_floor")
        remainder = entry.get("remainder_numerator")
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, Integral)
            or int(capacity) != group_source_count - 1
            or isinstance(floor, bool)
            or not isinstance(floor, Integral)
            or int(floor) < 0
            or isinstance(remainder, bool)
            or not isinstance(remainder, Integral)
            or int(remainder) < 0
            or len(group_ids) > group_source_count
        ):
            raise ValueError("selection apportionment fields are invalid")
        source_sum += group_source_count
        selected_union.extend(group_ids)
    if source_sum != source_count or sorted(selected_union) != shape_ids:
        raise ValueError("selection IDs/counts are internally inconsistent")


def _load_npy_snapshot(
    root_descriptor: int,
    relative_path: str,
    *,
    expected_sha256: str,
    expected_file_bytes: int,
    num_points: int,
) -> np.ndarray:
    payload, actual_sha256 = _snapshot_relative_file(
        root_descriptor,
        relative_path,
    )
    if len(payload) != expected_file_bytes:
        raise ValueError(f"clean file byte count differs: {relative_path}")
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"clean SHA does not match manifest: {relative_path}"
        )
    try:
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid clean NPY: {relative_path}") from error
    if values.shape != (num_points, 3) or values.dtype != np.float32:
        raise ValueError("clean array contract does not match manifest")
    if not np.isfinite(values).all():
        raise ValueError("clean array contains non-finite values")
    center = (
        values.min(axis=0).astype(np.float64)
        + values.max(axis=0).astype(np.float64)
    ) * 0.5
    radius = float(
        np.linalg.norm(values.astype(np.float64) - center, axis=1).max()
    )
    if not np.allclose(center, 0.0, rtol=0.0, atol=3e-6):
        raise ValueError("clean array is not bbox-centered at the origin")
    if not math.isclose(radius, 1.0, rel_tol=3e-6, abs_tol=3e-6):
        raise ValueError("clean array is not normalized to the unit sphere")
    return np.ascontiguousarray(values)


def _regular_file_inventory_from_root(
    root_descriptor: int,
) -> set[str]:
    """Traverse an FD-anchored tree and reject every non-directory/link."""

    files: set[str] = set()

    def visit(directory_descriptor: int, prefix: tuple[str, ...]) -> None:
        for name in sorted(os.listdir(directory_descriptor)):
            if name in ("", ".", "..") or "/" in name or "\0" in name:
                raise ValueError("cache inventory contains an unsafe name")
            metadata = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            relative = "/".join((*prefix, name))
            if stat.S_ISDIR(metadata.st_mode):
                child = _open_child_directory(directory_descriptor, name)
                try:
                    pinned = os.fstat(child)
                    if (
                        pinned.st_dev != metadata.st_dev
                        or pinned.st_ino != metadata.st_ino
                    ):
                        raise RuntimeError(
                            f"cache directory changed during inventory: "
                            f"{relative}"
                        )
                    visit(child, (*prefix, name))
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | _require_nofollow()
                )
                descriptor = os.open(
                    name,
                    flags,
                    dir_fd=directory_descriptor,
                )
                try:
                    pinned = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(pinned.st_mode)
                        or pinned.st_dev != metadata.st_dev
                        or pinned.st_ino != metadata.st_ino
                    ):
                        raise RuntimeError(
                            f"cache file changed during inventory: {relative}"
                        )
                finally:
                    os.close(descriptor)
                files.add(relative)
            else:
                raise ValueError(
                    f"cache inventory contains unsafe entry: {relative}"
                )

    visit(root_descriptor, ())
    return files


def _verify_surface_cache_pinned(
    cache_dir: os.PathLike[str] | str,
    *,
    root_descriptor: int,
    mesh_root: os.PathLike[str] | str | None = None,
    train_split: os.PathLike[str] | str | None = None,
    expected_train_split_sha256: str | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_content_sha256: str | None = None,
    verify_files: bool = True,
) -> dict[str, object]:
    """Fail closed unless a published cache matches its complete contract."""

    if not isinstance(verify_files, bool):
        raise ValueError("verify_files must be a bool")
    root = Path(cache_dir)
    manifest_payload, _ = _snapshot_at(
        root_descriptor,
        "manifest.json",
        source=os.fspath(root / "manifest.json"),
    )
    manifest = _read_json(manifest_payload, label="surface cache manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != SURFACE_CACHE_FORMAT
        or isinstance(manifest.get("format_version"), bool)
        or manifest.get("format_version") != SURFACE_CACHE_VERSION
        or set(manifest) != _MANIFEST_KEYS
    ):
        raise ValueError("surface cache manifest metadata/fields are invalid")
    claimed_content = _sha256_value(
        manifest.get("content_sha256"),
        name="surface cache content SHA",
    )
    if expected_content_sha256 is not None:
        expected_content = _sha256_value(
            expected_content_sha256,
            name="expected surface cache content SHA",
        )
        if claimed_content != expected_content:
            raise ValueError("surface cache content SHA differs from expected")
    if _manifest_content_digest(manifest) != claimed_content:
        raise ValueError("surface cache content SHA/contract is invalid")

    shape_count = _positive_integer(
        manifest.get("shape_count"),
        name="manifest shape_count",
    )
    count = _positive_integer(
        manifest.get("num_points"),
        name="manifest num_points",
    )
    global_seed = _seed(manifest.get("seed"))
    if (
        manifest.get("seed_derivation") != _SEED_DERIVATION
        or manifest.get("coordinate_frame")
        != "per_shape_clean_sample_bbox_center_max_radius_unit_sphere"
        or manifest.get("array_contract")
        != {
            "container": "npy",
            "dtype": "float32",
            "shape": [count, 3],
        }
    ):
        raise ValueError("surface cache generation contract is invalid")
    shape_ids_raw = manifest.get("shape_ids")
    records = manifest.get("samples")
    if not isinstance(shape_ids_raw, list) or not isinstance(records, list):
        raise ValueError("surface cache shape/sample lists are invalid")
    shape_ids = [_shape_id(value) for value in shape_ids_raw]
    if (
        shape_ids != sorted(shape_ids)
        or len(set(shape_ids)) != len(shape_ids)
        or shape_count != len(shape_ids)
        or len(records) != len(shape_ids)
    ):
        raise ValueError("surface cache shape ordering/count is invalid")
    _validate_selection_contract(
        manifest.get("selection"),
        shape_ids=shape_ids,
        global_seed=global_seed,
    )
    if manifest.get("samples_sha256") != _sample_records_digest(records):
        raise ValueError("surface cache samples SHA is invalid")

    rng = manifest.get("rng")
    if (
        not isinstance(rng, dict)
        or set(rng) != _RNG_KEYS
        or rng.get("library") != "numpy"
        or rng.get("generator") != "numpy.random.Generator"
        or not isinstance(rng.get("numpy_version"), str)
        or not rng.get("numpy_version")
        or not isinstance(rng.get("bit_generator"), str)
        or not rng.get("bit_generator")
        or rng.get("stream_policy") != _RNG_STREAM_POLICY
    ):
        raise ValueError("surface cache RNG contract is invalid")
    _sha256_value(manifest.get("split_sha256"), name="split SHA")
    _sha256_value(
        manifest.get("train_split_sha256"),
        name="train split SHA",
    )
    _sha256_value(
        manifest.get("split_manifest_sha256"),
        name="split manifest SHA",
    )
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != {
        "pcdenoise/data/archive.py",
        "pcdenoise/data/mesh_dataset.py",
        "pcdenoise/data/split.py",
        "pcdenoise/data/surface_cache.py",
    }:
        raise ValueError("surface cache source hash inventory is invalid")
    for name, digest in sources.items():
        _sha256_value(digest, name=f"source SHA {name}")

    if train_split is not None:
        (
            train_ids,
            split_sha256,
            train_actual,
            split_manifest_actual,
        ) = _read_train_bundle(
            Path(train_split),
            expected_train_split_sha256=expected_train_split_sha256,
            expected_split_manifest_sha256=(
                expected_split_manifest_sha256
            ),
        )
        selected, selection = select_shape_ids(
            train_ids,
            select_count=shape_count,
            seed=global_seed,
        )
        if (
            split_sha256 != manifest.get("split_sha256")
            or train_actual != manifest.get("train_split_sha256")
            or split_manifest_actual
            != manifest.get("split_manifest_sha256")
            or selected != shape_ids
            or selection != manifest.get("selection")
        ):
            raise ValueError("surface cache split/selection binding is invalid")

    mesh_descriptor = (
        _open_directory(mesh_root) if mesh_root is not None else None
    )
    expected_files = {"manifest.json"}
    try:
        for index, (shape_id, record) in enumerate(zip(shape_ids, records)):
            if not isinstance(record, dict) or set(record) != _SAMPLE_KEYS:
                raise ValueError("surface cache sample record is invalid")
            if record.get("shape_id") != shape_id:
                raise ValueError(
                    f"surface cache sample order differs at index {index}"
                )
            derived_seed = _seed(record.get("derived_seed"))
            if derived_seed != _derived_seed(global_seed, shape_id):
                raise ValueError(f"derived seed is invalid for {shape_id}")
            mesh_sha = _sha256_value(
                record.get("mesh_sha256"),
                name=f"mesh SHA for {shape_id}",
            )
            clean_sha = _sha256_value(
                record.get("clean_sha256"),
                name=f"clean SHA for {shape_id}",
            )
            file_bytes = _positive_integer(
                record.get("clean_file_bytes"),
                name=f"clean_file_bytes for {shape_id}",
            )
            relative = (
                Path("shapenet") / shape_id / "clean.npy"
            ).as_posix()
            if record.get("relative_path") != relative:
                raise ValueError(f"relative path is invalid for {shape_id}")
            expected_files.add(relative)
            center_raw = record.get("normalization_center")
            if (
                not isinstance(center_raw, list)
                or len(center_raw) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, Real)
                    for value in center_raw
                )
                or not np.isfinite(
                    np.asarray(center_raw, dtype=np.float64)
                ).all()
            ):
                raise ValueError(
                    f"normalization center is invalid for {shape_id}"
                )
            _finite_positive(
                record.get("normalization_scale"),
                name=f"normalization scale for {shape_id}",
            )
            if verify_files:
                _load_npy_snapshot(
                    root_descriptor,
                    relative,
                    expected_sha256=clean_sha,
                    expected_file_bytes=file_bytes,
                    num_points=count,
                )
            if mesh_descriptor is not None:
                _, actual_mesh_sha = _snapshot_mesh_payload_from_root(
                    mesh_descriptor,
                    shape_id,
                )
                if actual_mesh_sha != mesh_sha:
                    raise ValueError(
                        f"mesh SHA does not match manifest for {shape_id}"
                    )
    finally:
        if mesh_descriptor is not None:
            os.close(mesh_descriptor)
    if (
        verify_files
        and _regular_file_inventory_from_root(root_descriptor)
        != expected_files
    ):
        raise ValueError("surface cache file inventory is invalid")
    return manifest


def verify_surface_cache(
    cache_dir: os.PathLike[str] | str,
    *,
    mesh_root: os.PathLike[str] | str | None = None,
    train_split: os.PathLike[str] | str | None = None,
    expected_train_split_sha256: str | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_content_sha256: str | None = None,
    verify_files: bool = True,
) -> dict[str, object]:
    """Verify all cache data through one pinned, no-follow root descriptor."""

    root_descriptor = _open_directory(cache_dir)
    try:
        return _verify_surface_cache_pinned(
            cache_dir,
            root_descriptor=root_descriptor,
            mesh_root=mesh_root,
            train_split=train_split,
            expected_train_split_sha256=expected_train_split_sha256,
            expected_split_manifest_sha256=(
                expected_split_manifest_sha256
            ),
            expected_content_sha256=expected_content_sha256,
            verify_files=verify_files,
        )
    finally:
        os.close(root_descriptor)


__all__ = [
    "SURFACE_CACHE_FORMAT",
    "SURFACE_CACHE_VERSION",
    "build_surface_cache",
    "reproduce_surface_sample",
    "select_shape_ids",
    "verify_surface_cache",
]
