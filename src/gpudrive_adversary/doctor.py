"""Structured Milestone-A host and GPUDrive runtime diagnostics."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .pins import (
    load_pins,
    required_checks_pass,
    repository_root,
    sha256_file,
    tree_sha256,
    verify_source_tree,
)
from .provenance import port_identity


def _check(
    name: str,
    ok: bool,
    *,
    expected: Any = None,
    observed: Any = None,
    required: bool = True,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "required": required,
        "expected": expected,
        "observed": observed,
        "message": message,
    }


def _command_probe(name: str, arguments: list[str]) -> dict[str, Any]:
    executable = shutil.which(name)
    if executable is None:
        return {"path": None, "version": None, "returncode": None}
    try:
        result = subprocess.run(
            [executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"path": executable, "version": str(exc), "returncode": None}
    output = (result.stdout or result.stderr).strip().splitlines()
    return {
        "path": executable,
        "version": output[0] if output else None,
        "returncode": result.returncode,
    }


def _runtime_probe(
    *, source: Path, pins: dict[str, Any], reference: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    modules: dict[str, Any] = {}
    for module_name in ("numpy", "torch", "madrona_gpudrive", "gpudrive"):
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            module_file = getattr(module, "__file__", None)
            modules[module_name] = {"version": version, "file": module_file}
            checks.append(_check(f"runtime.import.{module_name}", True, observed=module_file))
        except Exception as exc:  # Import failures may originate in native loaders.
            modules[module_name] = {"error": f"{type(exc).__name__}: {exc}"}
            checks.append(
                _check(
                    f"runtime.import.{module_name}",
                    False,
                    observed=modules[module_name]["error"],
                    message="Run inside the reference container after the pinned build.",
                )
            )

    runtime["modules"] = modules
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        torch_cuda_version = getattr(
            getattr(torch_module, "version", None), "cuda", None
        )
        cuda_available = bool(cuda and cuda.is_available())
        device_count = int(cuda.device_count()) if cuda else 0
        runtime["torch"] = {
            "cuda_version": torch_cuda_version,
            "cuda_available": cuda_available,
            "device_count": device_count,
            "device_names": [
                cuda.get_device_name(index) for index in range(cuda.device_count())
            ]
            if cuda
            else [],
        }
        checks.extend(
            [
                _check(
                    "runtime.torch_cuda_version",
                    torch_cuda_version == pins["reference_runtime"]["cuda"],
                    expected=pins["reference_runtime"]["cuda"],
                    observed=torch_cuda_version,
                    required=reference,
                ),
                _check(
                    "runtime.cuda_available",
                    cuda_available,
                    expected=True,
                    observed=cuda_available,
                    required=reference,
                ),
                _check(
                    "runtime.cuda_device_count",
                    device_count > 0,
                    expected="> 0",
                    observed=device_count,
                    required=reference,
                ),
            ]
        )

    gpudrive_path = modules.get("gpudrive", {}).get("file")
    gpudrive_from_source = False
    if gpudrive_path:
        try:
            gpudrive_from_source = Path(gpudrive_path).resolve().is_relative_to(
                source.resolve()
            )
        except OSError:
            pass
    checks.append(
        _check(
            "runtime.gpudrive_from_pinned_source",
            gpudrive_from_source,
            expected=str(source.resolve()),
            observed=gpudrive_path,
        )
    )

    native_path = modules.get("madrona_gpudrive", {}).get("file")
    runtime["native_extension_sha256"] = (
        sha256_file(native_path) if native_path and Path(native_path).is_file() else None
    )
    checks.append(
        _check(
            "runtime.native_extension_hashed",
            runtime["native_extension_sha256"] is not None,
            expected="SHA-256 of madrona_gpudrive native module",
            observed=runtime["native_extension_sha256"],
        )
    )
    cache_path = os.environ.get("MADRONA_MWGPU_KERNEL_CACHE")
    runtime["madrona_kernel_cache"] = {
        "path": cache_path,
        "tree_sha256": tree_sha256(cache_path) if cache_path else None,
    }
    checks.append(
        _check(
            "runtime.kernel_cache_configured",
            bool(cache_path),
            expected="MADRONA_MWGPU_KERNEL_CACHE set to a persistent path",
            observed=cache_path,
            required=reference,
            message="Required for the reference CUDA runtime; clear it after native code changes.",
        )
    )
    return checks, runtime


def build_doctor_report(
    *,
    source: Path,
    pin_path: Path | None = None,
    probe_runtime: bool = True,
    reference: bool = False,
    require_initialized_submodules: bool = True,
) -> dict[str, Any]:
    pins = load_pins(pin_path)
    reference_runtime = pins["reference_runtime"]
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "reference.runtime_probe_enabled",
            not reference or probe_runtime,
            expected=True,
            observed=probe_runtime,
            required=reference,
            message="A reference certificate cannot skip native runtime and CUDA probes.",
        )
    )

    current_python = platform.python_version()
    checks.append(
        _check(
            "host.python",
            current_python == reference_runtime["python"],
            expected=reference_runtime["python"],
            observed=current_python,
        )
    )

    system = platform.system().lower()
    machine = platform.machine().lower()
    machine_is_amd64 = machine in {"amd64", "x86_64"}
    checks.extend(
        [
            _check(
                "host.reference_platform",
                system == reference_runtime["platform"],
                expected=reference_runtime["platform"],
                observed=system,
                required=reference,
                message="Windows native is diagnostic only; use the Linux container for certified artifacts.",
            ),
            _check(
                "host.reference_architecture",
                machine_is_amd64,
                expected=reference_runtime["architecture"],
                observed=machine,
                required=reference,
            ),
        ]
    )

    command_specs = {
        "git": ["--version"],
        "cmake": ["--version"],
        "ninja": ["--version"],
        "nvcc": ["--version"],
        "nvidia-smi": ["--query-gpu=name,driver_version", "--format=csv,noheader"],
        "docker": ["--version"],
        "gcc": ["--version"],
        "g++": ["--version"],
    }
    commands: dict[str, Any] = {}
    for command, args in command_specs.items():
        probe = _command_probe(command, args)
        commands[command] = probe
        required = command == "git" or (
            reference
            and command
            in {"cmake", "ninja", "nvcc", "nvidia-smi", "gcc", "g++"}
        )
        checks.append(
            _check(
                f"host.command.{command}",
                probe["path"] is not None and probe["returncode"] == 0,
                expected="executable present and probe exits 0",
                observed=probe,
                required=required,
            )
        )

    checks.extend(
        verify_source_tree(
            source,
            pins,
            require_initialized_submodules=require_initialized_submodules,
        )
    )

    runtime: dict[str, Any] = {}
    if probe_runtime:
        runtime_checks, runtime = _runtime_probe(
            source=source, pins=pins, reference=reference
        )
        checks.extend(runtime_checks)

    expected_image_digest = reference_runtime["base_image_amd64_digest"]
    observed_image_digest = os.environ.get("GPUDRIVE_REFERENCE_IMAGE_DIGEST")
    checks.append(
        _check(
            "reference.base_image_digest",
            observed_image_digest == expected_image_digest,
            expected=expected_image_digest,
            observed=observed_image_digest,
            required=reference,
        )
    )

    port = port_identity(repository_root())
    checks.extend(
        [
            _check(
                "port.commit_fingerprinted",
                isinstance(port["commit"], str) and len(port["commit"]) == 40,
                expected="40-character Git commit",
                observed=port["commit"],
            ),
            _check(
                "port.dirty_state_recorded",
                isinstance(port["dirty"], bool),
                expected="boolean",
                observed=port["dirty"],
            ),
            _check(
                "port.diff_fingerprinted",
                isinstance(port["diff_sha256"], str)
                and len(port["diff_sha256"]) == 64,
                expected="SHA-256",
                observed=port["diff_sha256"],
            ),
            _check(
                "port.source_tree_fingerprinted",
                isinstance(port["source_tree_sha256"], str)
                and len(port["source_tree_sha256"]) == 64,
                expected="SHA-256",
                observed=port["source_tree_sha256"],
            ),
            _check(
                "port.declared_source_tree_matches",
                port["source_tree_matches_declared"],
                expected=port["declared_source_tree_sha256"],
                observed=port["source_tree_sha256"],
                required=reference,
            ),
        ]
    )

    return {
        "schema": "gpudrive_doctor_report",
        "schema_version": 1,
        "ok": required_checks_pass(checks),
        "checks": checks,
        "pins": pins,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
            "commands": commands,
        },
        "runtime": runtime,
        "source": str(source.resolve()),
        "port": port,
    }
