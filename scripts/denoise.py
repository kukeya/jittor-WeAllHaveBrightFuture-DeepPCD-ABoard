#!/usr/bin/env python3
"""Denoise competition-style inputs with one verified Jittor checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import jittor as jt
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.models.factory import (
    build_pgd_model,
    pgd_model_architecture,
)
from pcdenoise.prediction import run_prediction
from pcdenoise.training.checkpoint import load_checkpoint
from pcdenoise.training.engine import canonical_config_sha256
from pcdenoise.training.runner import validate_training_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-config-sha256",
        help=(
            "expected canonical training-config digest stored in the "
            "checkpoint; when supplied, training-only paths are not required"
        ),
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--patch-batch-size", type=int, default=5)
    parser.add_argument("--niters", type=int, choices=(1,), default=1)
    parser.add_argument(
        "--normalization-mode",
        choices=("noisy_max",),
        default="noisy_max",
        help="per-sample noisy-cloud normalization (default: noisy_max)",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("hard_best",),
        default="hard_best",
        help="overlap displacement fusion (default: hard_best)",
    )
    parser.add_argument(
        "--iteration-damping",
        type=float,
        choices=(1.0,),
        default=1.0,
        help="fixed residual multiplier used by the submitted recipe",
    )
    parser.add_argument(
        "--sample-ids",
        type=Path,
        help="optional UTF-8 file with one <synset>/<model> ID per line",
    )
    return parser


def _sample_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError("--sample-ids file is empty")
    return lines


def main() -> int:
    args = _parser().parse_args()
    try:
        raw_config = yaml.safe_load(
            args.config.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid training config: {args.config}") from error
    if not isinstance(raw_config, dict):
        raise ValueError("training config root must be a mapping")
    if args.expected_config_sha256 is None:
        config = validate_training_config(raw_config)
        config_sha256 = canonical_config_sha256(config)
    else:
        # A frozen-checkpoint replay needs the architecture, conditioning
        # contract and CUDA flag, but must not require the multi-gigabyte
        # training cache on the review machine.  Keep the normal validator for
        # all semantic fields while substituting only training-only paths.
        # The caller separately pins both the config-file digest and the
        # canonical digest embedded in the checkpoint.
        portable_config = copy.deepcopy(raw_config)
        portable_data = portable_config.get("data")
        if not isinstance(portable_data, dict):
            raise ValueError("training config data must be a mapping")
        portable_data["train_cache"] = str(args.input_dir.resolve())
        portable_data["verify_cache"] = False
        for key in (
            "mesh_root",
            "train_split",
            "expected_train_split_sha256",
            "expected_split_manifest_sha256",
            "expected_content_sha256",
        ):
            portable_data.pop(key, None)
        config = validate_training_config(portable_config)
        config_sha256 = args.expected_config_sha256
    model_config = config["model"]
    patch_size = (
        int(args.patch_size)
        if args.patch_size is not None
        else int(model_config["patch_size"])
    )
    if patch_size != int(model_config["patch_size"]):
        raise ValueError(
            "--patch-size must match the checkpoint model patch_size"
        )

    use_cuda = bool(config["training"]["use_cuda"])
    if use_cuda and not jt.has_cuda:
        raise RuntimeError("checkpoint config requires CUDA")
    jt.flags.use_cuda = int(use_cuda)
    model = build_pgd_model(model_config, config["data"])
    checkpoint = load_checkpoint(
        args.checkpoint,
        model=model,
        expected_config_sha256=config_sha256,
    )
    model.eval()
    result = run_prediction(
        model,
        input_root=args.input_dir,
        output_dir=args.output_dir,
        patch_size=patch_size,
        seed_k=args.seed_k,
        patch_batch_size=args.patch_batch_size,
        niters=args.niters,
        normalization_mode=args.normalization_mode,
        robust_quantile=None,
        fusion_mode=args.fusion_mode,
        iteration_damping=args.iteration_damping,
        sample_ids=_sample_ids(args.sample_ids),
        model_reference={
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_step": checkpoint["step"],
            "config_sha256": config_sha256,
            "architecture": pgd_model_architecture(model_config),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
