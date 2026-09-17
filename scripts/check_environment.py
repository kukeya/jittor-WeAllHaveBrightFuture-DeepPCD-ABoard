#!/usr/bin/env python3
"""Verify the reproducible, pure-Jittor challenge environment.

The static mode validates ``environment.yml`` without importing third-party
packages.  The default runtime mode additionally verifies the installed
versions, compiler selection, CUDA devices, a Jittor optimizer step, AMP, and a
minimal CUDA ``jt.code`` operator.  The submitted point network uses Jittor's
native CUDA convolution path (``conv_opt=1``), so cuDNN is not required.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


EXPECTED_PYTHON = (3, 10)
EXPECTED_JITTOR = "1.3.10.0"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_CC_PATH = "/usr/bin/g++"
EXPECTED_NVCC_PATH = "/usr/local/cuda-12.4/bin/nvcc"
EXPECTED_CUDA_HOME = "/usr/local/cuda-12.4"
EXPECTED_CONV_OPT = "1"
SUPPORTED_CUDA_RELEASES = ("12.4",)
FORBIDDEN_FRAMEWORKS = ("torch", "tensorflow", "paddle", "mindspore")
FORBIDDEN_DEPENDENCY_NAMES = (
    "torch",
    "pytorch",
    "tensorflow",
    "paddle",
    "mindspore",
)


def contract_summary(
    *,
    cc_path: str = EXPECTED_CC_PATH,
    nvcc_path: str = EXPECTED_NVCC_PATH,
    cuda_home: str = EXPECTED_CUDA_HOME,
) -> dict[str, Any]:
    """Return the immutable version contract and selected toolchain paths."""

    return {
        "python": ".".join(map(str, EXPECTED_PYTHON)),
        "jittor": EXPECTED_JITTOR,
        "numpy": EXPECTED_NUMPY,
        "cc_path": cc_path,
        "nvcc_path": nvcc_path,
        "cuda_home": cuda_home,
        "conv_opt": EXPECTED_CONV_OPT,
    }


def detect_forbidden_frameworks(
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
) -> list[str]:
    """Return prohibited deep-learning frameworks visible to this interpreter."""

    detected: list[str] = []
    for package in FORBIDDEN_FRAMEWORKS:
        try:
            available = find_spec(package) is not None
        except (ImportError, AttributeError, ValueError):
            available = False
        if available:
            detected.append(package)
    return detected


def validate_environment_file(path: Path) -> list[str]:
    """Validate the dependency file using only the standard library."""

    if not path.is_file():
        return [f"dependency specification does not exist: {path}"]

    text = path.read_text(encoding="utf-8")
    normalized = text.lower()
    errors: list[str] = []
    required_patterns = {
        "environment name": r"(?m)^\s*name:\s*jittor\s*$",
        "Python 3.10": r"(?m)^\s*-\s*python=3\.10\s*$",
        "Jittor 1.3.10.0": r"(?m)^\s*-\s*jittor==1\.3\.10\.0\s*$",
        "NumPy 1.26.4": r"(?m)^\s*-\s*numpy==1\.26\.4\s*$",
        "GCC 10": r"(?m)^\s*-\s*gcc=10\s*$",
        "G++ 10": r"(?m)^\s*-\s*gxx=10\s*$",
        "libgomp": r"(?m)^\s*-\s*libgomp\s*$",
        "scipy": r"(?m)^\s*-\s*scipy(?:[<>=!~].*)?\s*$",
        "pyyaml": r"(?m)^\s*-\s*pyyaml(?:[<>=!~].*)?\s*$",
        "tensorboard": r"(?m)^\s*-\s*tensorboard(?:[<>=!~].*)?\s*$",
        "tensorboardX": r"(?m)^\s*-\s*tensorboardx(?:[<>=!~].*)?\s*$",
    }
    for description, pattern in required_patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE) is None:
            errors.append(f"missing required dependency contract: {description}")

    for package in FORBIDDEN_DEPENDENCY_NAMES:
        if package in normalized:
            errors.append(f"forbidden framework named in dependency file: {package}")
    return errors


def validate_selected_compiler_paths(
    actual_cc_path: Any,
    actual_nvcc_path: Any,
    *,
    expected_cc_path: str = EXPECTED_CC_PATH,
    expected_nvcc_path: str = EXPECTED_NVCC_PATH,
) -> list[str]:
    """Reject a Jittor compiler selection that differs from this invocation."""

    actual_cc = str(actual_cc_path)
    actual_nvcc = str(actual_nvcc_path)
    errors: list[str] = []
    if actual_cc != expected_cc_path:
        errors.append(
            "Jittor compiler.cc_path mismatch: "
            f"expected {expected_cc_path}, got {actual_cc!r}"
        )
    if actual_nvcc != expected_nvcc_path:
        errors.append(
            "Jittor compiler.nvcc_path mismatch: "
            f"expected {expected_nvcc_path}, got {actual_nvcc!r}"
        )
    return errors


def command_output(command: list[str]) -> tuple[int, str]:
    """Run a read-only metadata command and return status and combined output."""

    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout.strip()


def gpu_inventory() -> tuple[list[dict[str, str]], list[str]]:
    """Read NVIDIA device metadata without assuming an exact visible-GPU count."""

    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    status, output = command_output(query)
    if status != 0:
        fallback = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        fallback_status, fallback_output = command_output(fallback)
        if fallback_status != 0:
            return [], [f"nvidia-smi inventory failed: {output or fallback_output}"]
        output = fallback_output

    devices: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        device = {
            "index": fields[0],
            "name": fields[1],
            "memory_mib": fields[2],
        }
        if len(fields) >= 4:
            device["compute_capability"] = fields[3]
        devices.append(device)

    warnings: list[str] = []
    if len(devices) != 2:
        warnings.append(
            f"expected host inventory is two GPUs, but nvidia-smi reports {len(devices)}; "
            "single-GPU execution remains supported"
        )
    if devices and not all("4090" in device["name"] for device in devices):
        warnings.append("one or more reported GPUs are not RTX 4090 devices")
    return devices, warnings


def _finite_scalar(value: Any) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise RuntimeError(f"non-finite scalar: {scalar}")
    return scalar


def run_optimizer_smoke(jt: Any, np: Any) -> dict[str, Any]:
    """Run a CUDA forward/backward step and prove parameters were updated."""

    from jittor import nn

    jt.sync_all(True)
    model = nn.Linear(4, 2)
    parameters = list(model.parameters())
    if not parameters:
        raise RuntimeError("optimizer smoke model exposes no parameters")

    parameters_before = [
        np.array(parameter.numpy(), copy=True) for parameter in parameters
    ]
    if not all(np.isfinite(value).all() for value in parameters_before):
        raise RuntimeError("optimizer smoke parameters are non-finite before update")

    optimizer = nn.Adam(parameters, lr=1.0e-3)
    inputs = jt.array(
        np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
    )
    targets = jt.array(
        np.asarray(
            [[0.5, -0.5], [1.0, -1.0], [1.5, -1.5], [2.0, -2.0]],
            dtype=np.float32,
        )
    )
    prediction = model(inputs)
    loss = ((prediction - targets) ** 2).mean()
    optimizer.step(loss)
    jt.sync_all(True)

    loss_value = _finite_scalar(loss.item())
    parameters_after = [
        np.array(parameter.numpy(), copy=True) for parameter in parameters
    ]
    if not all(np.isfinite(value).all() for value in parameters_after):
        raise RuntimeError("optimizer smoke parameters are non-finite after update")

    per_parameter_updates = [
        float(np.max(np.abs(after - before)))
        for before, after in zip(parameters_before, parameters_after)
    ]
    max_abs_update = max(per_parameter_updates)
    if not math.isfinite(max_abs_update) or max_abs_update <= 0.0:
        raise RuntimeError(
            f"optimizer smoke did not update a parameter: {max_abs_update}"
        )

    return {
        "ok": True,
        "device": "cuda",
        "finite_loss": loss_value,
        "parameter_count": len(parameters),
        "parameter_shapes": [list(value.shape) for value in parameters_before],
        "parameters_before": [value.tolist() for value in parameters_before],
        "parameters_after": [value.tolist() for value in parameters_after],
        "per_parameter_max_abs_update": per_parameter_updates,
        "max_abs_update": max_abs_update,
    }


def run_amp_level5_smoke(jt: Any, np: Any) -> dict[str, Any]:
    """Exercise level-5 AMP with every floating operand created in one scope.

    At AMP level 5, Jittor marks array-like creation ops as float16-preferred.
    Creating float32 model parameters before entering the scope and then mixing
    them with float16-preferred forward ops triggers a Jittor 1.3.10.0 fusion
    dtype-relay error.  A real level-5 run therefore establishes the precision
    policy before model and data creation.
    """

    from jittor import nn

    jt.sync_all(True)
    with jt.flag_scope(amp_level=5):
        model = nn.Linear(4, 2)
        inputs = jt.array(
            np.asarray(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [3.0, 4.0, 5.0, 6.0],
                ],
                dtype=np.float32,
            )
        )
        targets = jt.array(
            np.asarray(
                [[0.5, -0.5], [1.0, -1.0], [1.5, -1.5], [2.0, -2.0]],
                dtype=np.float32,
            )
        )
        prediction = model(inputs)
        loss = ((prediction - targets) ** 2).mean()
    jt.sync_all(True)
    loss_value = _finite_scalar(loss.item())
    return {
        "ok": True,
        "amp_level": 5,
        "finite_loss": loss_value,
        "parameter_dtype": str(model.weight.dtype),
        "input_dtype": str(inputs.dtype),
        "target_dtype": str(targets.dtype),
        "loss_dtype": str(loss.dtype),
    }


def run_jittor_smokes(jt: Any, np: Any) -> dict[str, Any]:
    """Run optimizer, AMP, and custom-CUDA smoke checks."""

    jt.flags.use_cuda = 1
    jt.sync_all(True)

    optimizer_result = run_optimizer_smoke(jt, np)
    amp_result = run_amp_level5_smoke(jt, np)

    code_input = jt.array(np.asarray([1.0, -2.0, 3.5, 0.25], dtype=np.float32))
    jt.sync_all(True)
    code_output = jt.code(
        code_input.shape,
        code_input.dtype,
        [code_input],
        cuda_src=r"""
            __global__ static void multiply_by_two(@ARGS_DEF) {
                @PRECALC
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < in0_shape0)
                    @out(index) = @in0(index) * 2.0f;
            }
            multiply_by_two<<<1, 32>>>(@ARGS);
        """,
    )
    jt.sync_all(True)
    actual = code_output.numpy()
    expected = code_input.numpy() * np.float32(2.0)
    if not np.array_equal(actual, expected):
        raise RuntimeError(
            f"jt.code CUDA output mismatch: actual={actual.tolist()}, "
            f"expected={expected.tolist()}"
        )

    return {
        "optimizer": optimizer_result,
        "amp_level_5": amp_result,
        "jt_code_cuda": {
            "ok": True,
            "input": code_input.numpy().tolist(),
            "output": actual.tolist(),
        },
    }


def runtime_checks(
    *,
    expected_cc_path: str = EXPECTED_CC_PATH,
    expected_nvcc_path: str = EXPECTED_NVCC_PATH,
    expected_cuda_home: str = EXPECTED_CUDA_HOME,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Perform runtime checks and return details, errors, and warnings."""

    details: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    details["python_version"] = python_version
    if sys.version_info[:2] != EXPECTED_PYTHON:
        errors.append(
            f"Python version mismatch: expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, "
            f"got {python_version}"
        )

    forbidden = detect_forbidden_frameworks()
    details["forbidden_frameworks"] = forbidden
    if forbidden:
        errors.append(f"forbidden frameworks are importable: {', '.join(forbidden)}")

    environment_variables = {
        "cc_path": os.environ.get("cc_path"),
        "nvcc_path": os.environ.get("nvcc_path"),
        "CUDA_HOME": os.environ.get("CUDA_HOME"),
        "conv_opt": os.environ.get("conv_opt"),
    }
    details["environment_variables"] = environment_variables
    expected_variables = {
        "cc_path": expected_cc_path,
        "nvcc_path": expected_nvcc_path,
        "CUDA_HOME": expected_cuda_home,
        "conv_opt": EXPECTED_CONV_OPT,
    }
    for name, expected in expected_variables.items():
        actual = environment_variables[name]
        if actual != expected:
            errors.append(f"{name} mismatch: expected {expected}, got {actual!r}")

    for path in (expected_cc_path, expected_nvcc_path):
        if not Path(path).is_file():
            errors.append(f"required compiler executable does not exist: {path}")
    if not Path(expected_cuda_home).is_dir():
        errors.append(f"CUDA_HOME does not exist: {expected_cuda_home}")

    cc_status, cc_version = command_output(
        [expected_cc_path, "-dumpfullversion"]
    )
    details["cc_version"] = cc_version
    if cc_status != 0:
        errors.append(f"g++ -dumpfullversion failed with status {cc_status}")
    elif not cc_version.startswith("10."):
        errors.append(f"selected g++ is not major version 10: {cc_version!r}")

    nvcc_status, nvcc_output = command_output([expected_nvcc_path, "--version"])
    details["nvcc_version_output"] = nvcc_output
    if nvcc_status != 0:
        errors.append(f"nvcc --version failed with status {nvcc_status}")
    elif not any(
        f"release {release}" in nvcc_output
        for release in SUPPORTED_CUDA_RELEASES
    ):
        errors.append(
            "selected nvcc is not a supported CUDA release: "
            + ", ".join(SUPPORTED_CUDA_RELEASES)
        )

    devices, inventory_warnings = gpu_inventory()
    details["gpus"] = devices
    warnings.extend(inventory_warnings)
    if not devices:
        errors.append("no NVIDIA GPU could be inventoried")

    try:
        import numpy as np

        details["numpy_version"] = np.__version__
        if np.__version__ != EXPECTED_NUMPY:
            errors.append(
                f"NumPy version mismatch: expected {EXPECTED_NUMPY}, got {np.__version__}"
            )
    except Exception as exception:
        errors.append(f"NumPy import failed: {exception}")
        details["numpy_import_traceback"] = traceback.format_exc()
        return details, errors, warnings

    try:
        import jittor as jt

        details["jittor_version"] = jt.__version__
        if jt.__version__ != EXPECTED_JITTOR:
            errors.append(
                f"Jittor version mismatch: expected {EXPECTED_JITTOR}, got {jt.__version__}"
            )
        has_cuda = bool(jt.compiler.has_cuda)
        details["jittor_has_cuda"] = has_cuda
        selected_cc_path = str(getattr(jt.compiler, "cc_path", ""))
        selected_nvcc_path = str(getattr(jt.compiler, "nvcc_path", ""))
        details["jittor_cc_path"] = selected_cc_path
        details["jittor_nvcc_path"] = selected_nvcc_path
        errors.extend(
            validate_selected_compiler_paths(
                selected_cc_path,
                selected_nvcc_path,
                expected_cc_path=expected_cc_path,
                expected_nvcc_path=expected_nvcc_path,
            )
        )
        if not has_cuda:
            errors.append("Jittor reports compiler.has_cuda=False")
        if has_cuda:
            details["smokes"] = run_jittor_smokes(jt, np)
    except Exception as exception:
        errors.append(f"Jittor runtime smoke failed: {type(exception).__name__}: {exception}")
        details["jittor_traceback"] = traceback.format_exc()

    return details, errors, warnings


def build_summary(
    environment_file: Path,
    *,
    static_only: bool,
    expected_cc_path: str = EXPECTED_CC_PATH,
    expected_nvcc_path: str = EXPECTED_NVCC_PATH,
    expected_cuda_home: str = EXPECTED_CUDA_HOME,
) -> dict[str, Any]:
    """Build the complete check summary."""

    static_errors = validate_environment_file(environment_file)
    details: dict[str, Any] = {
        "environment_file": str(environment_file.resolve()),
    }
    errors = list(static_errors)
    warnings: list[str] = []
    mode = "static" if static_only else "runtime"
    if not static_only:
        runtime_details, runtime_errors, runtime_warnings = runtime_checks(
            expected_cc_path=expected_cc_path,
            expected_nvcc_path=expected_nvcc_path,
            expected_cuda_home=expected_cuda_home,
        )
        details["runtime"] = runtime_details
        errors.extend(runtime_errors)
        warnings.extend(runtime_warnings)
    return {
        "ok": not errors,
        "mode": mode,
        "contract": contract_summary(
            cc_path=expected_cc_path,
            nvcc_path=expected_nvcc_path,
            cuda_home=expected_cuda_home,
        ),
        "details": details,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-file",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "environment.yaml",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="validate only dependency contents; do not import Jittor",
    )
    parser.add_argument(
        "--cc-path",
        default=os.environ.get("cc_path", EXPECTED_CC_PATH),
        help="selected g++ executable; defaults to cc_path or the historical path",
    )
    parser.add_argument(
        "--nvcc-path",
        default=os.environ.get("nvcc_path", EXPECTED_NVCC_PATH),
        help="CUDA 12.4 nvcc path; defaults to nvcc_path or /usr/local/cuda-12.4/bin/nvcc",
    )
    parser.add_argument(
        "--cuda-home",
        default=os.environ.get("CUDA_HOME", EXPECTED_CUDA_HOME),
        help="CUDA 12.4 root; defaults to CUDA_HOME or /usr/local/cuda-12.4",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="also write the full JSON summary to this path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(
        args.environment_file,
        static_only=args.static_only,
        expected_cc_path=args.cc_path,
        expected_nvcc_path=args.nvcc_path,
        expected_cuda_home=args.cuda_home,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print("ENVIRONMENT_OK" if summary["ok"] else "ENVIRONMENT_FAILED")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
