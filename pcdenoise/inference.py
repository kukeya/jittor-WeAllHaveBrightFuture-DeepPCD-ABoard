"""Code-faithful whole-cloud patch inference for pure-Jittor denoisers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from pcdenoise.data.mesh_dataset import UnitSphereTransform, fit_unit_sphere


NORMALIZATION_MODES = ("noisy_max",)
FUSION_MODES = ("hard_best",)


def _points(values: object, *, name: str) -> np.ndarray:
    points = np.asarray(values)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError(f"{name} must have finite shape (N, 3), N > 0")
    if not np.issubdtype(points.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    result = np.ascontiguousarray(points, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _mode(
    value: object,
    *,
    name: str,
    choices: tuple[str, ...],
) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"{name} must be one of {', '.join(repr(item) for item in choices)}"
        )
    return value


def canonical_inference_options(
    *,
    normalization_mode: object = "noisy_max",
    robust_quantile: object = None,
    fusion_mode: object = "hard_best",
    iteration_damping: object = 1.0,
) -> dict[str, object]:
    """Validate the fixed inference recipe used for the submitted result."""

    normalization = _mode(
        normalization_mode,
        name="normalization_mode",
        choices=NORMALIZATION_MODES,
    )
    fusion = _mode(
        fusion_mode,
        name="fusion_mode",
        choices=FUSION_MODES,
    )
    if robust_quantile is not None:
        raise ValueError("robust_quantile is not used by the submitted recipe")
    if (
        isinstance(iteration_damping, bool)
        or not isinstance(iteration_damping, Real)
        or float(iteration_damping) != 1.0
    ):
        raise ValueError("iteration_damping must be 1.0")
    return {
        "normalization_mode": normalization,
        "robust_quantile": None,
        "fusion_mode": fusion,
        "iteration_damping": 1.0,
    }


def _normalization_transform(
    noisy_points: np.ndarray,
    *,
    normalization_mode: str,
    robust_quantile: float | None,
) -> UnitSphereTransform:
    if normalization_mode != "noisy_max" or robust_quantile is not None:
        raise RuntimeError("canonical normalization options became invalid")
    return fit_unit_sphere(noisy_points)


def numpy_farthest_point_indices(
    points: np.ndarray,
    sample_count: int,
    *,
    start_index: int = 0,
) -> np.ndarray:
    """Return deterministic FPS indices for one non-differentiable cloud."""

    cloud = _points(points, name="points")
    count = _positive_integer(sample_count, name="sample_count")
    point_count = len(cloud)
    if count > point_count:
        raise ValueError("sample_count must not exceed the point count")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, Integral)
        or not 0 <= int(start_index) < point_count
    ):
        raise ValueError("start_index is out of range")

    coordinates = cloud.astype(np.float64)
    output = np.empty(count, dtype=np.int64)
    minimum_squared = np.full(point_count, np.inf, dtype=np.float64)
    selected = np.zeros(point_count, dtype=bool)
    current = int(start_index)
    for step in range(count):
        output[step] = current
        selected[current] = True
        delta = coordinates - coordinates[current]
        squared = np.einsum("ij,ij->i", delta, delta)
        minimum_squared = np.minimum(minimum_squared, squared)
        if step + 1 < count:
            scores = np.where(selected, -1.0, minimum_squared)
            # np.argmax returns the lower index for an exact distance tie.
            current = int(np.argmax(scores))
    return output


@dataclass(frozen=True)
class PatchPlan:
    """Immutable neighborhood and hard-best fallback plan."""

    seed_indices: np.ndarray
    indices: np.ndarray
    normalized_squared_distances: np.ndarray
    owner_patch: np.ndarray
    owner_local_index: np.ndarray
    uncovered_count: int

    @property
    def num_patches(self) -> int:
        return int(self.indices.shape[0])


def build_patch_plan(
    points: np.ndarray,
    *,
    patch_size: int = 1000,
    seed_k: float = 6,
) -> PatchPlan:
    """Build PGD FPS/KNN patches and its code-faithful hard-best fusion map."""

    cloud = _points(points, name="points")
    count = _positive_integer(patch_size, name="patch_size")
    density = _positive_real(seed_k, name="seed_k")
    point_count = len(cloud)
    if count > point_count:
        raise ValueError("patch_size must not exceed the point count")
    num_patches = int(density * point_count / count)
    if num_patches <= 0:
        raise ValueError(
            "seed_k * point_count / patch_size must produce at least one patch"
        )
    if num_patches > point_count:
        raise ValueError("seed_k produces more seed patches than points")

    seed_indices = numpy_farthest_point_indices(
        cloud,
        num_patches,
        start_index=0,
    )
    tree = cKDTree(cloud.astype(np.float64), compact_nodes=True)
    distances, indices = tree.query(
        cloud[seed_indices].astype(np.float64),
        k=count,
        workers=1,
    )
    distances = np.asarray(distances, dtype=np.float64).reshape(
        num_patches, count
    )
    indices = np.asarray(indices, dtype=np.int64).reshape(
        num_patches, count
    )

    # cKDTree orders by distance but does not promise an index tie break.
    # Canonicalizing every returned row makes ordinary ties deterministic.
    for patch_index in range(num_patches):
        order = np.lexsort(
            (indices[patch_index], distances[patch_index])
        )
        indices[patch_index] = indices[patch_index, order]
        distances[patch_index] = distances[patch_index, order]

    squared = distances * distances
    denominator = squared[:, -1:]
    denominator = np.maximum(
        denominator,
        np.finfo(np.float64).tiny,
    )
    normalized_squared = squared / denominator
    if not np.isfinite(normalized_squared).all():
        raise RuntimeError("patch distance normalization produced non-finite values")

    best_distance = np.full(point_count, np.inf, dtype=np.float64)
    owner_patch = np.full(point_count, -1, dtype=np.int64)
    owner_local = np.full(point_count, -1, dtype=np.int64)
    local_indices = np.arange(count, dtype=np.int64)
    for patch_index in range(num_patches):
        point_indices = indices[patch_index]
        candidates = normalized_squared[patch_index]
        better = candidates < best_distance[point_indices]
        accepted_points = point_indices[better]
        best_distance[accepted_points] = candidates[better]
        owner_patch[accepted_points] = patch_index
        owner_local[accepted_points] = local_indices[better]

    uncovered = int((owner_patch < 0).sum())
    return PatchPlan(
        seed_indices=np.ascontiguousarray(seed_indices),
        indices=np.ascontiguousarray(indices),
        normalized_squared_distances=np.ascontiguousarray(
            normalized_squared.astype(np.float32)
        ),
        owner_patch=np.ascontiguousarray(owner_patch),
        owner_local_index=np.ascontiguousarray(owner_local),
        uncovered_count=uncovered,
    )


def denoise_normalized_cloud(
    model: object,
    points: np.ndarray,
    *,
    patch_size: int = 1000,
    seed_k: float = 6,
    patch_batch_size: int = 5,
    fusion_mode: str = "hard_best",
) -> tuple[np.ndarray, dict[str, object]]:
    """Denoise one normalized cloud and fuse patch displacements in point order."""

    cloud = _points(points, name="points")
    batch_size = _positive_integer(
        patch_batch_size,
        name="patch_batch_size",
    )
    fusion = _mode(
        fusion_mode,
        name="fusion_mode",
        choices=FUSION_MODES,
    )
    plan = build_patch_plan(
        cloud,
        patch_size=patch_size,
        seed_k=seed_k,
    )
    centers = cloud[plan.seed_indices]
    centered_patches = (
        cloud[plan.indices] - centers[:, None, :]
    ).astype(np.float32)
    displacements = np.empty_like(centered_patches)

    eval_method = getattr(model, "eval", None)
    if not callable(eval_method):
        raise TypeError("model must provide eval() and be callable")
    is_training_method = getattr(model, "is_training", None)
    was_training = (
        bool(is_training_method())
        if callable(is_training_method)
        else False
    )
    eval_method()
    try:
        with jt.no_grad():
            for start in range(0, plan.num_patches, batch_size):
                stop = min(start + batch_size, plan.num_patches)
                inputs = jt.array(centered_patches[start:stop])
                predictions = model(inputs)
                if not isinstance(predictions, jt.Var):
                    raise TypeError("model must return one Jittor Var")
                actual = np.asarray(predictions.numpy(), dtype=np.float32)
                expected_shape = (stop - start, int(patch_size), 3)
                if actual.shape != expected_shape:
                    raise ValueError(
                        f"model output shape {actual.shape} does not match "
                        f"{expected_shape}"
                    )
                if not np.isfinite(actual).all():
                    raise RuntimeError("model produced non-finite predictions")
                displacements[start:stop] = (
                    actual - centered_patches[start:stop]
                )
    finally:
        if was_training:
            train_method = getattr(model, "train", None)
            if callable(train_method):
                train_method()

    covered = plan.owner_patch >= 0
    covered_ids = np.flatnonzero(covered)
    output = cloud.copy()
    if len(covered_ids):
        chosen_displacement = displacements[
            plan.owner_patch[covered_ids],
            plan.owner_local_index[covered_ids],
        ]
        output[covered_ids] = (
            cloud[covered_ids] + chosen_displacement
        ).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("patch displacement fusion produced non-finite output")
    return np.ascontiguousarray(output), {
        "point_count": len(cloud),
        "patch_size": int(patch_size),
        "seed_k": float(seed_k),
        "num_patches": plan.num_patches,
        "patch_batch_size": batch_size,
        "fusion_mode": fusion,
        "zero_weight_fallback_count": 0,
        "uncovered_count": plan.uncovered_count,
        "coverage_fraction": float(covered.mean()),
    }


def denoise_cloud(
    model: object,
    noisy_points: np.ndarray,
    *,
    patch_size: int = 1000,
    seed_k: float = 6,
    patch_batch_size: int = 5,
    niters: int = 1,
    normalization_mode: str = "noisy_max",
    robust_quantile: float | None = None,
    fusion_mode: str = "hard_best",
    iteration_damping: float = 1.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one selected noisy transform, run damped passes, and restore units."""

    noisy = _points(noisy_points, name="noisy_points")
    iteration_count = _positive_integer(niters, name="niters")
    if iteration_count != 1:
        raise ValueError("niters must be 1")
    options = canonical_inference_options(
        normalization_mode=normalization_mode,
        robust_quantile=robust_quantile,
        fusion_mode=fusion_mode,
        iteration_damping=iteration_damping,
    )
    transform = _normalization_transform(
        noisy,
        normalization_mode=str(options["normalization_mode"]),
        robust_quantile=options["robust_quantile"],
    )
    current = transform.apply(noisy)
    current, details = denoise_normalized_cloud(
        model,
        current,
        patch_size=patch_size,
        seed_k=seed_k,
        patch_batch_size=patch_batch_size,
        fusion_mode=str(options["fusion_mode"]),
    )
    details["iteration"] = 1
    details["residual_damping"] = 1.0
    restored = transform.restore(current)
    return restored, {
        "normalization_mode": options["normalization_mode"],
        "robust_quantile": options["robust_quantile"],
        "normalization_center": transform.center.tolist(),
        "normalization_scale": float(transform.scale),
        "fusion_mode": options["fusion_mode"],
        "iteration_damping": options["iteration_damping"],
        "niters": iteration_count,
        "iterations": [details],
    }


__all__ = [
    "PatchPlan",
    "FUSION_MODES",
    "NORMALIZATION_MODES",
    "build_patch_plan",
    "canonical_inference_options",
    "denoise_cloud",
    "denoise_normalized_cloud",
    "numpy_farthest_point_indices",
]
