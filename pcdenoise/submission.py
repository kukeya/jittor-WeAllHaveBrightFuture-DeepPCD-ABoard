"""Strict, reproducible packaging for official point-cloud submissions.

The extracted official noisy directory is the authority for sample IDs and
array shapes.  Prediction files are snapshotted without following symlinks,
written to a deterministic ZIP, and then read back in full before the ZIP and
its manifest are atomically published without replacing existing files.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator, Mapping

import numpy as np

from pcdenoise.data.archive import inspect_test_zip


_SYNSET_RE = re.compile(r"^[0-9]{8}$")
_MODEL_RE = re.compile(r"^[0-9a-f]{28,32}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_REGULAR_0644 = stat.S_IFREG | 0o644
_NPY_HEADER_ALLOWANCE = 1 << 20
_RENAME_NOREPLACE = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SubmissionValidationError(ValueError):
    """Raised when inputs, predictions, or the result archive are invalid."""


@dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _TreeSnapshot:
    root: Path
    root_identity: _Identity
    directories: Mapping[str, _Identity]
    files: Mapping[str, _Identity]


@dataclass(frozen=True)
class _InputSample:
    sample_id: str
    relative_path: str
    path: Path
    shape: tuple[int, int]
    sha256: str


@dataclass(frozen=True)
class _OwnedFile:
    path: Path
    descriptor: int
    identity: _Identity


@dataclass(frozen=True)
class _Stage:
    path: Path
    parent: Path
    identity: _Identity
    destination_name: str
    output_name: str
    manifest_name: str


@dataclass(frozen=True)
class _InferenceManifest:
    sha256: str
    model_reference: Mapping[str, object]
    output_sha256: Mapping[str, str]


def _identity(metadata: os.stat_result) -> _Identity:
    return _Identity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_directory_root(path: Path | str, *, label: str) -> Path:
    requested = Path(path)
    try:
        metadata = os.lstat(requested)
    except FileNotFoundError:
        raise FileNotFoundError(requested) from None
    if stat.S_ISLNK(metadata.st_mode):
        raise SubmissionValidationError(f"{label} root is a symlink: {requested}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise SubmissionValidationError(
            f"{label} root is not a directory: {requested}"
        )
    return requested.resolve()


def _scan_tree(path: Path | str, *, label: str) -> _TreeSnapshot:
    root = _require_directory_root(path, label=label)
    root_before = _identity(os.lstat(root))
    directories: dict[str, _Identity] = {}
    files: dict[str, _Identity] = {}

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        try:
            with os.scandir(directory) as entries:
                ordered = sorted(entries, key=lambda item: item.name)
        except OSError as error:
            raise SubmissionValidationError(
                f"could not scan {label} directory: {directory}"
            ) from error
        for entry in ordered:
            relative = prefix / entry.name
            relative_name = relative.as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise SubmissionValidationError(
                    f"could not inspect {label} entry: {relative_name}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise SubmissionValidationError(
                    f"{label} entry is a symlink: {relative_name}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                directories[relative_name] = _identity(metadata)
                visit(Path(entry.path), relative)
            elif stat.S_ISREG(metadata.st_mode):
                files[relative_name] = _identity(metadata)
            else:
                raise SubmissionValidationError(
                    f"{label} entry is not a regular file or directory: "
                    f"{relative_name}"
                )

    visit(root, PurePosixPath())
    root_after = _identity(os.lstat(root))
    if root_after != root_before:
        raise SubmissionValidationError(f"{label} root changed during scan")
    return _TreeSnapshot(
        root=root,
        root_identity=root_before,
        directories=dict(sorted(directories.items())),
        files=dict(sorted(files.items())),
    )


def _assert_tree_unchanged(
    previous: _TreeSnapshot,
    *,
    label: str,
) -> None:
    current = _scan_tree(previous.root, label=label)
    if current != previous:
        raise SubmissionValidationError(f"{label} tree changed during packaging")


@contextmanager
def _open_owned_file(
    path: Path | str,
    *,
    label: str,
    expected_identity: _Identity | None = None,
) -> Iterator[_OwnedFile]:
    requested = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as error:
        raise SubmissionValidationError(
            f"could not safely open {label}: {requested}"
        ) from error
    try:
        identity = _identity(os.fstat(descriptor))
        if not stat.S_ISREG(identity.mode):
            raise SubmissionValidationError(
                f"{label} is not a regular file: {requested}"
            )
        if expected_identity is not None and identity != expected_identity:
            raise SubmissionValidationError(
                f"{label} changed between scan and open: {requested}"
            )
        yield _OwnedFile(
            path=requested,
            descriptor=descriptor,
            identity=identity,
        )
        if _identity(os.fstat(descriptor)) != identity:
            raise SubmissionValidationError(
                f"{label} changed while being read: {requested}"
            )
    finally:
        os.close(descriptor)


def _read_owned(
    owned: _OwnedFile,
    *,
    max_bytes: int | None = None,
) -> bytes:
    if max_bytes is not None and owned.identity.size > max_bytes:
        raise SubmissionValidationError(
            f"file is too large for its expected array: {owned.path}"
        )
    os.lseek(owned.descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = owned.identity.size
    while remaining:
        chunk = os.read(owned.descriptor, min(1 << 20, remaining))
        if not chunk:
            raise SubmissionValidationError(
                f"file ended before its declared size: {owned.path}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(owned.descriptor, 1):
        raise SubmissionValidationError(
            f"file grew while being read: {owned.path}"
        )
    return b"".join(chunks)


def _hash_owned(owned: _OwnedFile) -> str:
    os.lseek(owned.descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    remaining = owned.identity.size
    while remaining:
        chunk = os.read(owned.descriptor, min(8 << 20, remaining))
        if not chunk:
            raise SubmissionValidationError(
                f"file ended before its declared size: {owned.path}"
            )
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(owned.descriptor, 1):
        raise SubmissionValidationError(
            f"file grew while being hashed: {owned.path}"
        )
    return digest.hexdigest()


def _hash_regular_file(
    path: Path | str,
    *,
    label: str,
) -> tuple[str, int]:
    with _open_owned_file(path, label=label) as owned:
        digest = _hash_owned(owned)
        return digest, owned.identity.size


def _decode_npy(
    payload: bytes,
    *,
    label: str,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    stream = io.BytesIO(payload)
    try:
        values = np.load(stream, allow_pickle=False)
    except (EOFError, OSError, ValueError) as error:
        raise SubmissionValidationError(f"invalid NPY for {label}") from error
    if stream.tell() != len(payload):
        raise SubmissionValidationError(f"NPY has trailing bytes for {label}")
    if values.dtype != np.float32:
        raise SubmissionValidationError(f"{label} must use float32")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 3:
        raise SubmissionValidationError(
            f"{label} must have non-empty shape (N,3)"
        )
    actual_shape = (int(values.shape[0]), int(values.shape[1]))
    if expected_shape is not None and actual_shape != expected_shape:
        raise SubmissionValidationError(
            f"{label} shape {actual_shape} does not match input "
            f"shape {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise SubmissionValidationError(f"{label} must contain only finite values")
    return values


def _sample_id_from_path(
    relative_path: str,
    *,
    filename: str,
    label: str,
) -> str:
    parts = PurePosixPath(relative_path).parts
    if (
        len(parts) != 4
        or parts[0] != "shapenet"
        or not _SYNSET_RE.fullmatch(parts[1])
        or not _MODEL_RE.fullmatch(parts[2])
        or parts[3] != filename
    ):
        raise SubmissionValidationError(
            f"unexpected {label} path: {relative_path}"
        )
    return f"{parts[1]}/{parts[2]}"


def _expected_directories(sample_ids: Iterator[str]) -> set[str]:
    expected = {"shapenet"}
    for sample_id in sample_ids:
        synset, model = sample_id.split("/", 1)
        expected.add(f"shapenet/{synset}")
        expected.add(f"shapenet/{synset}/{model}")
    return expected


def _scan_inputs(
    input_dir: Path | str,
    *,
    expected_sample_count: int,
    expected_point_count: int | None,
) -> tuple[_TreeSnapshot, dict[str, _InputSample]]:
    if (
        isinstance(expected_sample_count, bool)
        or not isinstance(expected_sample_count, int)
        or expected_sample_count <= 0
    ):
        raise ValueError("expected_sample_count must be a positive integer")
    if expected_point_count is not None and (
        isinstance(expected_point_count, bool)
        or not isinstance(expected_point_count, int)
        or expected_point_count <= 0
    ):
        raise ValueError("expected_point_count must be a positive integer or None")
    tree = _scan_tree(input_dir, label="authoritative input")
    indexed_paths: dict[str, tuple[str, _Identity]] = {}
    for relative_path, identity in tree.files.items():
        sample_id = _sample_id_from_path(
            relative_path,
            filename="noisy.npy",
            label="authoritative input",
        )
        if sample_id in indexed_paths:
            raise SubmissionValidationError(
                f"duplicate authoritative input sample: {sample_id}"
            )
        indexed_paths[sample_id] = (relative_path, identity)
    if len(indexed_paths) != expected_sample_count:
        raise SubmissionValidationError(
            "authoritative input sample count mismatch: "
            f"expected {expected_sample_count}, got {len(indexed_paths)}"
        )
    expected_directories = _expected_directories(iter(indexed_paths))
    actual_directories = set(tree.directories)
    if actual_directories != expected_directories:
        unexpected = sorted(actual_directories - expected_directories)
        missing = sorted(expected_directories - actual_directories)
        raise SubmissionValidationError(
            "authoritative input directories are not exact; "
            f"unexpected={unexpected[:3]} missing={missing[:3]}"
        )

    samples: dict[str, _InputSample] = {}
    for sample_id in sorted(indexed_paths):
        relative_path, identity = indexed_paths[sample_id]
        path = tree.root.joinpath(*PurePosixPath(relative_path).parts)
        with _open_owned_file(
            path,
            label=f"authoritative input {sample_id}",
            expected_identity=identity,
        ) as owned:
            payload = _read_owned(owned)
        values = _decode_npy(
            payload,
            label=f"authoritative input {sample_id}",
        )
        if (
            expected_point_count is not None
            and values.shape[0] != expected_point_count
        ):
            raise SubmissionValidationError(
                f"authoritative input {sample_id} point count "
                f"{values.shape[0]} does not match required "
                f"{expected_point_count}"
            )
        samples[sample_id] = _InputSample(
            sample_id=sample_id,
            relative_path=relative_path,
            path=path,
            shape=(int(values.shape[0]), 3),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    _assert_tree_unchanged(tree, label="authoritative input")
    return tree, samples


def _scan_predictions(
    prediction_dir: Path | str,
    *,
    samples: Mapping[str, _InputSample],
    model_sha256: str,
) -> tuple[_TreeSnapshot, _InferenceManifest]:
    tree = _scan_tree(prediction_dir, label="prediction")
    metadata_name = "inference_manifest.json"
    expected_files = {
        f"shapenet/{sample_id}/denoised.npy" for sample_id in samples
    }
    allowed_files = set(expected_files) | {metadata_name}
    actual_files = set(tree.files)
    if actual_files != allowed_files:
        unexpected = sorted(actual_files - allowed_files)
        missing = sorted(expected_files - actual_files)
        raise SubmissionValidationError(
            "prediction files are not exact; "
            f"unexpected={unexpected[:3]} missing={missing[:3]}"
        )
    expected_directories = _expected_directories(iter(samples))
    actual_directories = set(tree.directories)
    if actual_directories != expected_directories:
        unexpected = sorted(actual_directories - expected_directories)
        missing = sorted(expected_directories - actual_directories)
        raise SubmissionValidationError(
            "prediction directories are not exact; "
            f"unexpected={unexpected[:3]} missing={missing[:3]}"
        )

    metadata_path = tree.root / metadata_name
    with _open_owned_file(
        metadata_path,
        label="raw inference manifest",
        expected_identity=tree.files[metadata_name],
    ) as owned:
        payload = _read_owned(owned, max_bytes=16 << 20)
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SubmissionValidationError(
            "invalid raw inference manifest"
        ) from error
    if not isinstance(metadata, dict) or metadata.get("status") != "completed":
        raise SubmissionValidationError(
            "raw inference manifest must report completed status"
        )
    if metadata.get("sample_count") != len(samples):
        raise SubmissionValidationError(
            "raw inference manifest sample_count mismatch"
        )
    if metadata.get("sample_ids") != sorted(samples):
        raise SubmissionValidationError(
            "raw inference manifest sample IDs mismatch"
        )
    reference = metadata.get("model_reference")
    if not isinstance(reference, dict):
        raise SubmissionValidationError(
            "raw inference manifest model_reference is required"
        )
    if reference.get("checkpoint_sha256") != model_sha256:
        raise SubmissionValidationError(
            "raw inference manifest model hash mismatch"
        )
    config_sha256 = reference.get("config_sha256")
    if not isinstance(config_sha256, str) or not _SHA256_RE.fullmatch(
        config_sha256
    ):
        raise SubmissionValidationError(
            "raw inference manifest config hash is required"
        )
    checkpoint_step = reference.get("checkpoint_step")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise SubmissionValidationError(
            "raw inference manifest checkpoint_step is required"
        )

    records = metadata.get("samples")
    if not isinstance(records, list) or len(records) != len(samples):
        raise SubmissionValidationError(
            "raw inference manifest sample records are incomplete"
        )
    claims: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise SubmissionValidationError(
                "raw inference manifest sample record must be a mapping"
            )
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in samples:
            raise SubmissionValidationError(
                "raw inference manifest contains an unknown sample ID"
            )
        if sample_id in claims:
            raise SubmissionValidationError(
                f"raw inference manifest duplicate sample: {sample_id}"
            )
        sample = samples[sample_id]
        expected_path = f"shapenet/{sample_id}/denoised.npy"
        if record.get("relative_path") != expected_path:
            raise SubmissionValidationError(
                f"raw inference manifest relative path mismatch: {sample_id}"
            )
        if record.get("point_count") != sample.shape[0]:
            raise SubmissionValidationError(
                f"raw inference manifest point count mismatch: {sample_id}"
            )
        if record.get("input_sha256") != sample.sha256:
            raise SubmissionValidationError(
                f"raw inference manifest input hash mismatch: {sample_id}"
            )
        output_sha256 = record.get("output_sha256")
        if not isinstance(output_sha256, str) or not _SHA256_RE.fullmatch(
            output_sha256
        ):
            raise SubmissionValidationError(
                f"raw inference manifest output hash is invalid: {sample_id}"
            )
        claims[sample_id] = output_sha256
    if sorted(claims) != sorted(samples):
        raise SubmissionValidationError(
            "raw inference manifest sample records do not match inputs"
        )
    reference_copy = json.loads(
        json.dumps(
            reference,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return tree, _InferenceManifest(
        sha256=hashlib.sha256(payload).hexdigest(),
        model_reference=reference_copy,
        output_sha256=claims,
    )


def _prediction_payload(
    tree: _TreeSnapshot,
    sample: _InputSample,
    *,
    expected_output_sha256: str,
) -> tuple[bytes, str]:
    relative_path = f"shapenet/{sample.sample_id}/denoised.npy"
    identity = tree.files[relative_path]
    path = tree.root.joinpath(*PurePosixPath(relative_path).parts)
    max_bytes = sample.shape[0] * 3 * np.dtype(np.float32).itemsize
    max_bytes += _NPY_HEADER_ALLOWANCE
    with _open_owned_file(
        path,
        label=f"prediction {sample.sample_id}",
        expected_identity=identity,
    ) as owned:
        payload = _read_owned(owned, max_bytes=max_bytes)
    _decode_npy(
        payload,
        label=f"prediction {sample.sample_id}",
        expected_shape=sample.shape,
    )
    output_sha256 = hashlib.sha256(payload).hexdigest()
    if output_sha256 != expected_output_sha256:
        raise SubmissionValidationError(
            f"prediction output hash differs from inference manifest: "
            f"{sample.sample_id}"
        )
    return payload, output_sha256


def _completion_manifest_check(
    input_root: Path,
    *,
    archive_sha256: str,
    sample_count: int,
) -> str | None:
    path = input_root.parent / "completion_manifest.json"
    if not _lexists(path):
        return None
    with _open_owned_file(path, label="prepared completion manifest") as owned:
        payload = _read_owned(owned, max_bytes=1 << 20)
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SubmissionValidationError(
            "invalid prepared completion manifest"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise SubmissionValidationError(
            "prepared completion manifest is not complete"
        )
    if manifest.get("test_npy_count") != sample_count:
        raise SubmissionValidationError(
            "prepared completion manifest sample count mismatch"
        )
    if manifest.get("test_archive_sha256") != archive_sha256:
        raise SubmissionValidationError(
            "input archive hash does not match prepared completion manifest"
        )
    return hashlib.sha256(payload).hexdigest()


def _inspect_archive_binding(
    input_archive: Path | str,
    *,
    samples: Mapping[str, _InputSample],
) -> dict[str, object]:
    inventory = inspect_test_zip(input_archive)
    archive_ids = inventory.get("shape_ids")
    archive_paths = inventory.get("normalized_files")
    expected_ids = sorted(samples)
    expected_paths = sorted(
        sample.relative_path for sample in samples.values()
    )
    if archive_ids != expected_ids:
        raise SubmissionValidationError(
            "official archive inventory IDs do not match prepared inputs"
        )
    if archive_paths != expected_paths:
        raise SubmissionValidationError(
            "official archive inventory paths do not match prepared inputs"
        )
    if inventory.get("npy_count") != len(samples):
        raise SubmissionValidationError(
            "official archive inventory sample count mismatch"
        )
    return inventory


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = _REGULAR_0644 << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    if info.filename.endswith("/"):
        raise SubmissionValidationError(
            f"unexpected ZIP directory member: {info.filename}"
        )
    if info.date_time != _FIXED_ZIP_TIME:
        raise SubmissionValidationError(
            f"non-reproducible ZIP timestamp: {info.filename}"
        )
    if info.compress_type != zipfile.ZIP_DEFLATED:
        raise SubmissionValidationError(
            f"unexpected ZIP compression: {info.filename}"
        )
    if info.create_system != 3:
        raise SubmissionValidationError(
            f"unexpected ZIP creator metadata: {info.filename}"
        )
    mode = (info.external_attr >> 16) & 0xFFFF
    if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o644:
        raise SubmissionValidationError(
            f"ZIP member is not regular mode 0644: {info.filename}"
        )
    if info.flag_bits & 0x1:
        raise SubmissionValidationError(
            f"encrypted ZIP member is forbidden: {info.filename}"
        )


def _validate_zip_owned(
    owned: _OwnedFile,
    *,
    samples: Mapping[str, _InputSample],
    expected_prediction_sha256: Mapping[str, str] | None,
) -> dict[str, object]:
    try:
        with os.fdopen(os.dup(owned.descriptor), "rb") as stream:
            with zipfile.ZipFile(stream, mode="r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise SubmissionValidationError(
                        "duplicate ZIP member names are forbidden"
                    )
                expected_names = [
                    f"shapenet/{sample_id}/denoised.npy"
                    for sample_id in sorted(samples)
                ]
                if names != expected_names:
                    unexpected = sorted(set(names) - set(expected_names))
                    missing = sorted(set(expected_names) - set(names))
                    raise SubmissionValidationError(
                        "unexpected ZIP root or member set; "
                        f"unexpected={unexpected[:3]} missing={missing[:3]}"
                    )
                for info, sample_id in zip(infos, sorted(samples)):
                    _validate_zip_info(info)
                    sample = samples[sample_id]
                    maximum = sample.shape[0] * 3 * np.dtype(np.float32).itemsize
                    maximum += _NPY_HEADER_ALLOWANCE
                    if info.file_size > maximum:
                        raise SubmissionValidationError(
                            f"ZIP member is too large: {info.filename}"
                        )
                bad_crc = archive.testzip()
                if bad_crc is not None:
                    raise SubmissionValidationError(
                        f"ZIP CRC failed for {bad_crc}"
                    )

                sample_records: list[dict[str, object]] = []
                for info, sample_id in zip(infos, sorted(samples)):
                    sample = samples[sample_id]
                    maximum = sample.shape[0] * 3 * np.dtype(np.float32).itemsize
                    maximum += _NPY_HEADER_ALLOWANCE
                    if info.file_size > maximum:
                        raise SubmissionValidationError(
                            f"ZIP member is too large: {info.filename}"
                        )
                    with archive.open(info, mode="r") as member:
                        payload = member.read(maximum + 1)
                    if len(payload) > maximum:
                        raise SubmissionValidationError(
                            f"ZIP member is too large: {info.filename}"
                        )
                    _decode_npy(
                        payload,
                        label=f"ZIP prediction {sample_id}",
                        expected_shape=sample.shape,
                    )
                    prediction_sha256 = hashlib.sha256(payload).hexdigest()
                    if (
                        expected_prediction_sha256 is not None
                        and prediction_sha256
                        != expected_prediction_sha256[sample_id]
                    ):
                        raise SubmissionValidationError(
                            f"ZIP bytes differ from raw prediction: {sample_id}"
                        )
                    sample_records.append(
                        {
                            "sample_id": sample_id,
                            "point_count": sample.shape[0],
                            "shape": list(sample.shape),
                            "input_sha256": sample.sha256,
                            "prediction_sha256": prediction_sha256,
                            "zip_path": info.filename,
                            "zip_crc32": f"{info.CRC:08x}",
                            "npy_bytes": info.file_size,
                        }
                    )
    except (
        EOFError,
        OSError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        zlib.error,
    ) as error:
        raise SubmissionValidationError(f"invalid submission ZIP: {error}") from error

    zip_sha256 = _hash_owned(owned)
    return {
        "sample_count": len(samples),
        "result_zip_sha256": zip_sha256,
        "result_zip_bytes": owned.identity.size,
        "crc_check": "passed",
        "samples": sample_records,
    }


def validate_submission_zip(
    *,
    input_dir: Path | str,
    zip_path: Path | str,
    expected_sample_count: int = 200,
    expected_point_count: int | None = 50000,
) -> dict[str, object]:
    """Reopen and strictly validate an existing competition submission ZIP."""

    _, samples = _scan_inputs(
        input_dir,
        expected_sample_count=expected_sample_count,
        expected_point_count=expected_point_count,
    )
    with _open_owned_file(zip_path, label="submission ZIP") as owned:
        return _validate_zip_owned(
            owned,
            samples=samples,
            expected_prediction_sha256=None,
        )


def _create_stage(output_zip: Path, manifest_path: Path) -> _Stage:
    if output_zip.name != "result.zip":
        raise ValueError("output_zip must be named result.zip")
    if manifest_path.name != "manifest.json":
        raise ValueError("manifest_path must be named manifest.json")
    output_directory = output_zip.parent.resolve(strict=False)
    manifest_directory = manifest_path.parent.resolve(strict=False)
    if output_directory != manifest_directory:
        raise ValueError("result.zip and manifest.json must share one directory")
    if not output_directory.name:
        raise ValueError("output_zip must be inside a new package directory")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    parent = output_directory.parent.resolve()
    destination = parent / output_directory.name
    if _lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite package directory: {destination}"
        )
    prefix = f".{destination.name}.submission-stage-"
    stage_path = Path(
        tempfile.mkdtemp(
            prefix=prefix,
            dir=os.fspath(parent),
        )
    ).resolve()
    identity = _identity(os.lstat(stage_path))
    if not stat.S_ISDIR(identity.mode):
        raise RuntimeError(f"submission stage is not a directory: {stage_path}")
    return _Stage(
        path=stage_path,
        parent=parent,
        identity=identity,
        destination_name=destination.name,
        output_name=output_zip.name,
        manifest_name=manifest_path.name,
    )


def _verify_stage(stage: _Stage) -> None:
    if (
        stage.path.parent != stage.parent
        or not stage.path.name.startswith(
            f".{stage.destination_name}.submission-stage-"
        )
    ):
        raise RuntimeError(f"refusing unsafe staging operation: {stage.path}")
    current = _identity(os.lstat(stage.path))
    if (
        not stat.S_ISDIR(current.mode)
        or current.device != stage.identity.device
        or current.inode != stage.identity.inode
    ):
        raise RuntimeError(f"submission stage identity changed: {stage.path}")


def _cleanup_stage(stage: _Stage) -> None:
    if not _lexists(stage.path):
        return
    _verify_stage(stage)
    shutil.rmtree(stage.path)


def _rename_noreplace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
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
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            "refusing to overwrite package directory",
            destination_name,
        )
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
            destination_name,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination_name,
    )


def _publish_stage_directory(
    stage: _Stage,
    *,
    expected_members: Mapping[str, tuple[_Identity, str]],
) -> Path:
    _verify_stage(stage)
    stage_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    stage_flags |= getattr(os, "O_NOFOLLOW", 0)
    stage_descriptor = os.open(
        stage.path,
        stage_flags,
    )
    parent_descriptor = os.open(
        stage.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    member_descriptors: list[int] = []
    renamed = False
    try:
        opened_stage = _identity(os.fstat(stage_descriptor))
        if (
            not stat.S_ISDIR(opened_stage.mode)
            or opened_stage.device != stage.identity.device
            or opened_stage.inode != stage.identity.inode
        ):
            raise RuntimeError("submission stage identity changed before publish")
        if set(expected_members) != {
            stage.output_name,
            stage.manifest_name,
        }:
            raise RuntimeError("submission stage member contract is incomplete")
        for name in (stage.output_name, stage.manifest_name):
            expected_identity, expected_sha256 = expected_members[name]
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(name, flags, dir_fd=stage_descriptor)
            member_descriptors.append(descriptor)
            actual_identity = _identity(os.fstat(descriptor))
            if actual_identity != expected_identity:
                raise RuntimeError(
                    f"staged {name} identity changed after validation"
                )
            owned = _OwnedFile(
                path=stage.path / name,
                descriptor=descriptor,
                identity=actual_identity,
            )
            if _hash_owned(owned) != expected_sha256:
                raise RuntimeError(
                    f"staged {name} hash changed after validation"
                )
            if _identity(os.fstat(descriptor)) != actual_identity:
                raise RuntimeError(f"staged {name} changed while hashing")
        os.fsync(stage_descriptor)
        _rename_noreplace(
            parent_descriptor,
            stage.path.name,
            stage.destination_name,
        )
        renamed = True
        published_directory = _identity(
            os.stat(
                stage.destination_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        )
        if (
            published_directory.device != stage.identity.device
            or published_directory.inode != stage.identity.inode
        ):
            raise RuntimeError(
                "published package directory identity does not match stage"
            )
        for descriptor, name in zip(
            member_descriptors,
            (stage.output_name, stage.manifest_name),
        ):
            if _identity(os.fstat(descriptor)) != expected_members[name][0]:
                raise RuntimeError(
                    f"staged {name} changed during directory publication"
                )
        os.fsync(parent_descriptor)
    except BaseException as original:
        if renamed:
            try:
                _rename_noreplace(
                    parent_descriptor,
                    stage.destination_name,
                    stage.path.name,
                )
                os.fsync(parent_descriptor)
            except OSError as rollback_error:
                raise RuntimeError(
                    "package publication failed and atomic rollback failed: "
                    f"{rollback_error}"
                ) from original
        raise
    finally:
        for descriptor in member_descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.close(parent_descriptor)
        except OSError:
            pass
        try:
            os.close(stage_descriptor)
        except OSError:
            pass
    return stage.parent / stage.destination_name


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> str:
    payload = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o644)
    return hashlib.sha256(payload).hexdigest()


def build_submission(
    *,
    input_dir: Path | str,
    prediction_dir: Path | str,
    input_archive: Path | str,
    model_file: Path | str,
    run_id: str,
    output_zip: Path | str,
    manifest_path: Path | str | None = None,
    expected_sample_count: int = 200,
    expected_point_count: int | None = 50000,
) -> dict[str, object]:
    """Validate raw predictions and atomically publish one package directory."""

    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must use 1-128 ASCII letters, digits, dot, dash, or underscore"
        )
    output = Path(output_zip)
    manifest_output = (
        Path(manifest_path)
        if manifest_path is not None
        else output.with_name("manifest.json")
    )
    stage = _create_stage(output, manifest_output)
    try:
        input_tree, samples = _scan_inputs(
            input_dir,
            expected_sample_count=expected_sample_count,
            expected_point_count=expected_point_count,
        )
        archive_inventory = _inspect_archive_binding(
            input_archive,
            samples=samples,
        )
        archive_sha256 = str(archive_inventory["archive_sha256"])
        archive_bytes = int(archive_inventory["compressed_bytes"])
        model_sha256, model_bytes = _hash_regular_file(
            model_file,
            label="model file",
        )
        completion_manifest_sha256 = _completion_manifest_check(
            input_tree.root,
            archive_sha256=archive_sha256,
            sample_count=len(samples),
        )
        prediction_tree, inference_manifest = _scan_predictions(
            prediction_dir,
            samples=samples,
            model_sha256=model_sha256,
        )

        staged_zip = stage.path / stage.output_name
        prediction_hashes: dict[str, str] = {}
        with zipfile.ZipFile(
            staged_zip,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            archive.comment = b""
            for sample_id in sorted(samples):
                payload, prediction_sha256 = _prediction_payload(
                    prediction_tree,
                    samples[sample_id],
                    expected_output_sha256=(
                        inference_manifest.output_sha256[sample_id]
                    ),
                )
                prediction_hashes[sample_id] = prediction_sha256
                archive.writestr(
                    _zip_info(f"shapenet/{sample_id}/denoised.npy"),
                    payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.chmod(staged_zip, 0o644)
        with staged_zip.open("rb") as stream:
            os.fsync(stream.fileno())

        _assert_tree_unchanged(input_tree, label="authoritative input")
        _assert_tree_unchanged(prediction_tree, label="prediction")
        with _open_owned_file(staged_zip, label="staged submission ZIP") as owned:
            zip_report = _validate_zip_owned(
                owned,
                samples=samples,
                expected_prediction_sha256=prediction_hashes,
            )
            validated_zip_identity = owned.identity

        input_inventory_payload = json.dumps(
            [
                {
                    "sample_id": sample.sample_id,
                    "shape": list(sample.shape),
                    "input_sha256": sample.sha256,
                }
                for sample in samples.values()
            ],
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        archive_inventory_payload = json.dumps(
            {
                "archive_sha256": archive_sha256,
                "normalized_files": archive_inventory["normalized_files"],
                "shape_ids": archive_inventory["shape_ids"],
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        manifest: dict[str, object] = {
            "format": "pcdenoise_submission_v1",
            "format_version": 1,
            "status": "validated",
            "run_id": run_id,
            "sample_count": len(samples),
            "input_archive": {
                "name": Path(input_archive).name,
                "sha256": archive_sha256,
                "bytes": archive_bytes,
            },
            "input_inventory_sha256": hashlib.sha256(
                input_inventory_payload
            ).hexdigest(),
            "archive_inventory_sha256": hashlib.sha256(
                archive_inventory_payload
            ).hexdigest(),
            "prepared_completion_manifest_sha256": (
                completion_manifest_sha256
            ),
            "raw_inference_manifest_sha256": inference_manifest.sha256,
            "inference_model_reference": (
                inference_manifest.model_reference
            ),
            "model": {
                "name": Path(model_file).name,
                "sha256": model_sha256,
                "bytes": model_bytes,
            },
            "result_zip": {
                "name": stage.output_name,
                "sha256": zip_report["result_zip_sha256"],
                "bytes": zip_report["result_zip_bytes"],
                "crc_check": zip_report["crc_check"],
            },
            "samples": zip_report["samples"],
        }
        staged_manifest = stage.path / stage.manifest_name
        manifest_sha256 = _write_manifest(staged_manifest, manifest)
        manifest_identity = _identity(os.lstat(staged_manifest))
        published_directory = _publish_stage_directory(
            stage,
            expected_members={
                stage.output_name: (
                    validated_zip_identity,
                    str(zip_report["result_zip_sha256"]),
                ),
                stage.manifest_name: (
                    manifest_identity,
                    manifest_sha256,
                ),
            },
        )
    except BaseException:
        _cleanup_stage(stage)
        raise
    _cleanup_stage(stage)
    return {
        "status": "validated",
        "run_id": run_id,
        "sample_count": len(samples),
        "package_dir": str(published_directory),
        "output_zip": str((published_directory / stage.output_name)),
        "manifest_path": str(
            published_directory / stage.manifest_name
        ),
        "result_zip_sha256": zip_report["result_zip_sha256"],
        "manifest_sha256": manifest_sha256,
        "crc_check": "passed",
    }


__all__ = [
    "SubmissionValidationError",
    "build_submission",
    "validate_submission_zip",
]
