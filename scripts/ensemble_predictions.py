#!/usr/bin/env python3
"""Blend two completed point-cloud prediction directories in output space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.ensemble import build_prediction_ensemble


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-a", type=Path, required=True)
    parser.add_argument("--pred-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--recipe-output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = build_prediction_ensemble(
        pred_a=args.pred_a,
        pred_b=args.pred_b,
        output_dir=args.output_dir,
        beta=args.beta,
        recipe_output=args.recipe_output,
    )
    print(
        json.dumps(
            result,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
