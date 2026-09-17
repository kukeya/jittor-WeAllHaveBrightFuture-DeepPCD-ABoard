"""Deterministic NumPy batch preparation for pure-Jittor PGD training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pcdenoise.data.noise import NOISE_PROFILES, ROTATION_MODES
from pcdenoise.data.pgd_training import (
    TRAINING_NORMALIZATION_MODES,
    sample_code_faithful_training_patch,
)
from pcdenoise.training.sampling import (
    EpochCoverage,
    coverage_statistics,
    epoch_batches,
)


_EPOCH_PATCH_SEED_DOMAIN = b"pcdenoise:epoch_patch_rng_v1"


def canonical_config_sha256(config: Mapping[str, object]) -> str:
    """Hash the semantic JSON configuration independently of key order."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    try:
        encoded = json.dumps(
            config,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise TypeError(
            "config must contain finite JSON-compatible values"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _nonnegative_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_integer(value: object, *, name: str) -> int:
    converted = _nonnegative_integer(value, name=name)
    if converted == 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _noise_range(minimum: object, maximum: object) -> tuple[float, float]:
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, Real)
        or isinstance(maximum, bool)
        or not isinstance(maximum, Real)
    ):
        raise ValueError("noise_min/noise_max must be finite numbers")
    low, high = float(minimum), float(maximum)
    if (
        not math.isfinite(low)
        or not math.isfinite(high)
        or low < 0.0
        or high <= low
    ):
        raise ValueError(
            "noise_min/noise_max must be finite, nonnegative, and increasing"
        )
    return low, high


def _sample_id(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    if "shapenet" not in parts:
        raise ValueError(f"clean sample is outside shapenet layout: {path}")
    offset = parts.index("shapenet") + 1
    if len(parts) != offset + 3 or parts[-1] != "clean.npy":
        raise ValueError(f"invalid clean sample layout: {path}")
    synset, model = parts[offset], parts[offset + 1]
    if (
        len(synset) != 8
        or not synset.isdigit()
        or not 28 <= len(model) <= 32
        or any(character not in "0123456789abcdef" for character in model)
    ):
        raise ValueError(f"invalid clean sample ID: {synset}/{model}")
    return f"{synset}/{model}"


@dataclass(frozen=True)
class CleanSample:
    sample_id: str
    path: Path


@dataclass(frozen=True)
class TrainingBatch:
    noisy: np.ndarray
    clean: np.ndarray
    sigmas: np.ndarray
    sample_ids: tuple[str, ...]
    center_indices: tuple[int, ...]
    noise_profiles: tuple[str, ...] = ()
    noise_scales: np.ndarray | None = None
    epoch: int | None = None
    batch_index: int | None = None
    epoch_batch_count: int | None = None
    epoch_size: int | None = None
    prefix_count: int | None = None
    prefix_coverage: EpochCoverage | None = None
    source_noise_scales: np.ndarray | None = None
    normalization_scales: np.ndarray | None = None


def scan_clean_cache(
    root: Path | str,
    *,
    filename: str = "clean.npy",
) -> list[CleanSample]:
    """Return the canonical sample list from one surface-cache directory."""

    cache_root = Path(root)
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    if filename != "clean.npy":
        raise ValueError("only the clean.npy surface-cache contract is supported")
    indexed: dict[str, Path] = {}
    for path in sorted(cache_root.rglob(filename)):
        sample_id = _sample_id(path, cache_root)
        if sample_id in indexed:
            raise ValueError(f"duplicate clean sample key: {sample_id}")
        indexed[sample_id] = path.resolve()
    if not indexed:
        raise ValueError("surface cache contains no clean.npy samples")
    return [
        CleanSample(sample_id=sample_id, path=indexed[sample_id])
        for sample_id in sorted(indexed)
    ]


@lru_cache(maxsize=128)
def _load_clean(path_text: str) -> np.ndarray:
    path = Path(path_text)
    values = np.load(path, allow_pickle=False)
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 3
    ):
        raise ValueError(
            f"clean cache array must have shape (N, 3) float32: {path}"
        )
    result = np.ascontiguousarray(values)
    if not np.isfinite(result).all():
        raise ValueError(f"clean cache array must be finite: {path}")
    result.setflags(write=False)
    return result


def _derived_seed(
    base_seed: int,
    *,
    step: int,
    batch_index: int,
) -> int:
    digest = hashlib.sha256(
        base_seed.to_bytes(8, "little", signed=False)
        + step.to_bytes(8, "little", signed=False)
        + batch_index.to_bytes(8, "little", signed=False)
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _epoch_patch_seed(
    base_seed: int,
    *,
    epoch: int,
    sample_id: str,
) -> int:
    """Derive one patch stream independently of epoch batch partitioning."""

    encoded_id = sample_id.encode("utf-8")
    digest = hashlib.sha256(
        _EPOCH_PATCH_SEED_DOMAIN
        + base_seed.to_bytes(8, "little", signed=False)
        + epoch.to_bytes(8, "little", signed=False)
        + len(encoded_id).to_bytes(8, "little", signed=False)
        + encoded_id
    ).digest()
    return int.from_bytes(digest[:16], "little", signed=False)


def build_training_batch(
    samples: Sequence[CleanSample],
    *,
    base_seed: int,
    step: int,
    batch_size: int,
    patch_size: int,
    noise_min: float = 0.005,
    noise_max: float = 0.020,
    rotate: bool = True,
) -> TrainingBatch:
    """Build a resume-stable batch from ``(seed, step, batch_index)``."""

    if not isinstance(samples, Sequence) or not samples:
        raise ValueError("samples must be a nonempty sequence")
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    if len({sample.sample_id for sample in ordered}) != len(ordered):
        raise ValueError("samples contains duplicate sample IDs")
    seed = _nonnegative_integer(base_seed, name="base_seed")
    if seed >= 2**64:
        raise ValueError("base_seed must be smaller than 2**64")
    actual_step = _nonnegative_integer(step, name="step")
    count = _positive_integer(batch_size, name="batch_size")
    points_per_patch = _positive_integer(patch_size, name="patch_size")
    low, high = _noise_range(noise_min, noise_max)
    if not isinstance(rotate, bool):
        raise ValueError("rotate must be a bool")

    noisy_patches = []
    clean_patches = []
    sigmas = []
    sample_ids = []
    center_indices = []
    for batch_index in range(count):
        rng = np.random.default_rng(
            _derived_seed(
                seed,
                step=actual_step,
                batch_index=batch_index,
            )
        )
        sample = ordered[int(rng.integers(0, len(ordered)))]
        clean_cloud = _load_clean(str(sample.path))
        if len(clean_cloud) < points_per_patch:
            raise ValueError(
                f"clean cache array shape has fewer than patch_size points: "
                f"{sample.path}"
            )
        patch = sample_code_faithful_training_patch(
            clean_cloud,
            rng,
            patch_size=points_per_patch,
            sigma_range=(low, high),
            rotate=rotate,
        )
        noisy_patches.append(patch.noisy)
        clean_patches.append(patch.clean)
        sigmas.append(patch.sigma)
        sample_ids.append(sample.sample_id)
        center_indices.append(patch.center_index)
    sigma_values = np.asarray(sigmas, dtype=np.float32)
    return TrainingBatch(
        noisy=np.ascontiguousarray(
            np.stack(noisy_patches),
            dtype=np.float32,
        ),
        clean=np.ascontiguousarray(
            np.stack(clean_patches),
            dtype=np.float32,
        ),
        sigmas=sigma_values,
        sample_ids=tuple(sample_ids),
        center_indices=tuple(center_indices),
        noise_profiles=("gaussian",) * count,
        noise_scales=sigma_values,
        source_noise_scales=sigma_values.copy(),
        normalization_scales=np.ones(count, dtype=np.float32),
    )


def build_epoch_training_batch(
    samples: Sequence[CleanSample],
    *,
    base_seed: int,
    epoch: int,
    batch_index: int,
    batch_size: int,
    patch_size: int,
    noise_profile: str,
    noise_min: float,
    noise_max: float,
    rotate: bool,
    rotation_mode: str = "euler_xyz",
    normalization_mode: str = "clean_unit",
) -> TrainingBatch:
    """Build one deterministic, without-replacement batch for an epoch.

    ``batch_index`` is zero based.  The final batch is not padded.  Each
    sample's patch RNG depends only on ``(base_seed, epoch, sample_id)`` so
    changing the batch size or resuming within an epoch cannot change its
    noise, rotation, center, or patch.
    """

    if not isinstance(samples, Sequence) or not samples:
        raise ValueError("samples must be a nonempty sequence")
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    if len({sample.sample_id for sample in ordered}) != len(ordered):
        raise ValueError("samples contains duplicate sample IDs")
    seed = _nonnegative_integer(base_seed, name="base_seed")
    if seed >= 2**64:
        raise ValueError("base_seed must be smaller than 2**64")
    epoch_index = _nonnegative_integer(epoch, name="epoch")
    if epoch_index >= 2**64:
        raise ValueError("epoch must be smaller than 2**64")
    selected_batch_index = _nonnegative_integer(
        batch_index,
        name="batch_index",
    )
    count = _positive_integer(batch_size, name="batch_size")
    points_per_patch = _positive_integer(patch_size, name="patch_size")
    low, high = _noise_range(noise_min, noise_max)
    if noise_profile not in NOISE_PROFILES:
        raise ValueError(
            f"noise_profile must be one of {NOISE_PROFILES}, "
            f"got {noise_profile!r}"
        )
    if not isinstance(rotate, bool):
        raise ValueError("rotate must be a bool")
    if rotation_mode not in ROTATION_MODES:
        raise ValueError(
            f"rotation_mode must be one of {ROTATION_MODES}, "
            f"got {rotation_mode!r}"
        )
    if normalization_mode not in TRAINING_NORMALIZATION_MODES:
        raise ValueError(
            "normalization_mode must be one of "
            f"{TRAINING_NORMALIZATION_MODES}, got {normalization_mode!r}"
        )

    batches = epoch_batches(
        len(ordered),
        batch_size=count,
        seed=seed,
        epoch=epoch_index,
    )
    if selected_batch_index >= len(batches):
        raise ValueError(
            "batch_index must be smaller than the number of batches "
            f"in the epoch ({len(batches)})"
        )
    selected_indices = batches[selected_batch_index]
    prefix_indices = np.concatenate(
        batches[: selected_batch_index + 1],
    )

    noisy_patches = []
    clean_patches = []
    noise_profiles = []
    noise_scales = []
    source_noise_scales = []
    normalization_scales = []
    sample_ids = []
    center_indices = []
    for sample_index in selected_indices:
        sample = ordered[int(sample_index)]
        clean_cloud = _load_clean(str(sample.path))
        if len(clean_cloud) < points_per_patch:
            raise ValueError(
                f"clean cache array shape has fewer than patch_size points: "
                f"{sample.path}"
            )
        rng = np.random.default_rng(
            _epoch_patch_seed(
                seed,
                epoch=epoch_index,
                sample_id=sample.sample_id,
            )
        )
        patch = sample_code_faithful_training_patch(
            clean_cloud,
            rng,
            patch_size=points_per_patch,
            noise_profile=noise_profile,
            noise_scale_range=(low, high),
            rotate=rotate,
            rotation_mode=rotation_mode,
            normalization_mode=normalization_mode,
        )
        noisy_patches.append(patch.noisy)
        clean_patches.append(patch.clean)
        noise_profiles.append(patch.noise_profile)
        noise_scales.append(patch.noise_scale)
        source_noise_scales.append(patch.source_noise_scale)
        normalization_scales.append(patch.normalization_scale)
        sample_ids.append(sample.sample_id)
        center_indices.append(patch.center_index)

    scale_values = np.asarray(noise_scales, dtype=np.float32)
    source_scale_values = np.asarray(
        source_noise_scales,
        dtype=np.float32,
    )
    normalization_scale_values = np.asarray(
        normalization_scales,
        dtype=np.float32,
    )
    return TrainingBatch(
        noisy=np.ascontiguousarray(
            np.stack(noisy_patches),
            dtype=np.float32,
        ),
        clean=np.ascontiguousarray(
            np.stack(clean_patches),
            dtype=np.float32,
        ),
        sigmas=scale_values,
        sample_ids=tuple(sample_ids),
        center_indices=tuple(center_indices),
        noise_profiles=tuple(noise_profiles),
        noise_scales=scale_values,
        epoch=epoch_index,
        batch_index=selected_batch_index,
        epoch_batch_count=len(batches),
        epoch_size=len(ordered),
        prefix_count=len(prefix_indices),
        prefix_coverage=coverage_statistics(
            prefix_indices,
            num_items=len(ordered),
        ),
        source_noise_scales=source_scale_values,
        normalization_scales=normalization_scale_values,
    )


__all__ = [
    "CleanSample",
    "TrainingBatch",
    "build_epoch_training_batch",
    "build_training_batch",
    "canonical_config_sha256",
    "scan_clean_cache",
]
