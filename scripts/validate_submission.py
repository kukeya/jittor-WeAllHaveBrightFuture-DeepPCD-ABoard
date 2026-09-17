#!/usr/bin/env python3
"""Build and independently validate an official ``result.zip`` submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.submission import build_submission


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="prepared official test root containing shapenet/.../noisy.npy",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="raw prediction root containing shapenet/.../denoised.npy",
    )
    parser.add_argument(
        "--input-archive",
        type=Path,
        required=True,
        help="official noisy test archive, used for the manifest hash",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        required=True,
        help="independently selected model checkpoint, hashed into the manifest",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-zip",
        type=Path,
        required=True,
        help=(
            "new atomic package path, normally "
            "submissions/<run_id>/package/result.zip"
        ),
    )
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--expected-point-count",
        type=int,
        default=50000,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_submission(
        input_dir=args.input_dir,
        prediction_dir=args.prediction_dir,
        input_archive=args.input_archive,
        model_file=args.model_file,
        run_id=args.run_id,
        output_zip=args.output_zip,
        manifest_path=args.manifest_path,
        expected_sample_count=args.expected_sample_count,
        expected_point_count=args.expected_point_count,
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
