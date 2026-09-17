"""Atomic batch prediction over competition-style noisy point clouds."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pcdenoise.data.archive import (
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
)
from pcdenoise.inference import canonical_inference_options, denoise_cloud


@dataclass(frozen=True)
class NoisySample:
    sample_id: str
    path: Path


def _sample_id(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    if "shapenet" not in parts:
        raise ValueError(f"noisy sample is outside shapenet layout: {path}")
    offset = parts.index("shapenet") + 1
    if len(parts) != offset + 3 or parts[-1] != "noisy.npy":
        raise ValueError(f"invalid noisy sample layout: {path}")
    synset, model = parts[offset], parts[offset + 1]
    if (
        len(synset) != 8
        or not synset.isdigit()
        or not 28 <= len(model) <= 32
        or any(character not in "0123456789abcdef" for character in model)
    ):
        raise ValueError(f"invalid noisy sample ID: {synset}/{model}")
    return f"{synset}/{model}"


def scan_noisy_inputs(root: Path | str) -> list[NoisySample]:
    input_root = Path(root)
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    indexed: dict[str, Path] = {}
    for path in sorted(input_root.rglob("noisy.npy")):
        sample_id = _sample_id(path, input_root)
        if sample_id in indexed:
            raise ValueError(f"duplicate noisy sample key: {sample_id}")
        indexed[sample_id] = path.resolve()
    if not indexed:
        raise ValueError("input root contains no noisy.npy samples")
    return [
        NoisySample(sample_id=sample_id, path=indexed[sample_id])
        for sample_id in sorted(indexed)
    ]


def _snapshot_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"input is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_noisy(path: Path) -> tuple[np.ndarray, str]:
    payload = _snapshot_regular(path)
    try:
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except (ValueError, OSError) as error:
        raise ValueError(f"invalid noisy NPY: {path}") from error
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 3
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"noisy NPY must have finite shape (N,3) float32: {path}"
        )
    return (
        np.ascontiguousarray(values),
        hashlib.sha256(payload).hexdigest(),
    )


def _json_snapshot(
    value: Mapping[str, object],
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must contain finite JSON-compatible values"
        ) from error
    result = json.loads(encoded)
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a mapping")
    return result


def _select_samples(
    samples: list[NoisySample],
    requested: Sequence[str] | None,
) -> list[NoisySample]:
    if requested is None:
        return samples
    if isinstance(requested, (str, bytes)) or not isinstance(
        requested, Sequence
    ):
        raise ValueError("sample_ids must be a sequence of IDs")
    selected_ids = list(requested)
    if not selected_ids:
        raise ValueError("sample_ids must not be empty")
    if not all(isinstance(sample_id, str) for sample_id in selected_ids):
        raise ValueError("sample_ids must contain strings")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("sample_ids contains duplicates")
    indexed = {sample.sample_id: sample for sample in samples}
    unknown = sorted(set(selected_ids) - set(indexed))
    if unknown:
        raise ValueError(f"sample_ids contains unknown IDs: {unknown[:3]}")
    return [indexed[sample_id] for sample_id in sorted(selected_ids)]


def _save_npy(path: Path, points: np.ndarray) -> str:
    values = np.asarray(points)
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 3
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("prediction violated the (N,3) float32 contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(
            stream,
            np.ascontiguousarray(values),
            allow_pickle=False,
        )
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(_snapshot_regular(path)).hexdigest()


def run_prediction(
    model: object,
    *,
    input_root: Path | str,
    output_dir: Path | str,
    patch_size: int = 1000,
    seed_k: float = 6,
    patch_batch_size: int = 5,
    niters: int = 1,
    normalization_mode: str = "noisy_max",
    robust_quantile: float | None = None,
    fusion_mode: str = "hard_best",
    iteration_damping: float = 1.0,
    sample_ids: Sequence[str] | None = None,
    model_reference: Mapping[str, object],
) -> dict[str, object]:
    """Denoise selected inputs and atomically publish one complete directory."""

    options = canonical_inference_options(
        normalization_mode=normalization_mode,
        robust_quantile=robust_quantile,
        fusion_mode=fusion_mode,
        iteration_damping=iteration_damping,
    )
    all_samples = scan_noisy_inputs(input_root)
    samples = _select_samples(all_samples, sample_ids)
    reference = _json_snapshot(model_reference, name="model_reference")
    stage = _create_owned_stage(output_dir)
    started = time.perf_counter()
    sample_records: list[dict[str, object]] = []
    try:
        for sample in samples:
            noisy, input_sha256 = _load_noisy(sample.path)
            sample_started = time.perf_counter()
            denoised, details = denoise_cloud(
                model,
                noisy,
                patch_size=patch_size,
                seed_k=seed_k,
                patch_batch_size=patch_batch_size,
                niters=niters,
                normalization_mode=str(options["normalization_mode"]),
                robust_quantile=options["robust_quantile"],
                fusion_mode=str(options["fusion_mode"]),
                iteration_damping=float(options["iteration_damping"]),
            )
            if denoised.shape != noisy.shape:
                raise RuntimeError(
                    f"{sample.sample_id}: output point count/shape changed"
                )
            synset, model_id = sample.sample_id.split("/")
            relative_path = (
                Path("shapenet")
                / synset
                / model_id
                / "denoised.npy"
            )
            output_sha256 = _save_npy(
                stage.path / relative_path,
                denoised.astype(np.float32, copy=False),
            )
            sample_records.append(
                {
                    "sample_id": sample.sample_id,
                    "point_count": len(noisy),
                    "input_sha256": input_sha256,
                    "output_sha256": output_sha256,
                    "relative_path": relative_path.as_posix(),
                    "elapsed_seconds": time.perf_counter() - sample_started,
                    "details": details,
                }
            )

        elapsed = time.perf_counter() - started
        manifest = {
            "format": "pcdenoise_prediction_v1",
            "format_version": 1,
            "status": "completed",
            "sample_count": len(samples),
            "sample_ids": [sample.sample_id for sample in samples],
            "patch_size": int(patch_size),
            "seed_k": float(seed_k),
            "patch_batch_size": int(patch_batch_size),
            "niters": int(niters),
            "normalization_mode": options["normalization_mode"],
            "robust_quantile": options["robust_quantile"],
            "fusion_mode": options["fusion_mode"],
            "iteration_damping": options["iteration_damping"],
            "model_reference": reference,
            "elapsed_seconds": elapsed,
            "samples": sample_records,
        }
        manifest_path = stage.path / "inference_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(
                manifest,
                stream,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        published = _publish_owned_stage(stage)
    except BaseException:
        _cleanup_owned_stage(stage)
        raise
    return {
        "output_dir": str(published),
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "elapsed_seconds": elapsed,
        "model_reference": reference,
    }


__all__ = [
    "NoisySample",
    "run_prediction",
    "scan_noisy_inputs",
]
