"""Safe acquisition and verification of the immutable victim checkpoint."""

from __future__ import annotations

import json
import hashlib
import math
import os
import struct
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ..pins import (
    canonical_json_sha256,
    repository_root,
    sha256_file,
)


class VictimCheckpointError(RuntimeError):
    """Raised when victim checkpoint provenance or content is invalid."""


def default_victim_pin_path() -> Path:
    return repository_root() / "configs/victim/pretrained_ppo.json"


def _is_sha(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def load_victim_pin(path: Path | str | None = None) -> dict[str, Any]:
    pin_path = Path(path) if path is not None else default_victim_pin_path()
    try:
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VictimCheckpointError(f"cannot load victim pin {pin_path}: {exc}") from exc
    if pin.get("schema") != "gpudrive_victim_policy_pin" or pin.get(
        "schema_version"
    ) != 1:
        raise VictimCheckpointError("unsupported victim pin schema")
    source = pin.get("source", {})
    if not _is_sha(source.get("revision"), 40):
        raise VictimCheckpointError("victim source revision must be an immutable Git SHA")
    files = source.get("files")
    if not isinstance(files, dict) or not files:
        raise VictimCheckpointError("victim source files must be a non-empty object")
    for name, specification in files.items():
        if Path(name).name != name or name in {".", ".."}:
            raise VictimCheckpointError(f"unsafe victim filename: {name!r}")
        if not _is_sha(specification.get("sha256"), 64):
            raise VictimCheckpointError(f"invalid SHA-256 for {name}")
        if not isinstance(specification.get("size"), int) or specification["size"] <= 0:
            raise VictimCheckpointError(f"invalid byte size for {name}")
    config = pin.get("model_config", {})
    if config.get("action_dim") != 91 or config.get("obs_dim") != 2984:
        raise VictimCheckpointError("victim model must use the pinned 91-action/2984-observation contract")
    if not _is_sha(pin.get("safetensors_contract", {}).get("state_dict_sha256"), 64):
        raise VictimCheckpointError("victim state-dictionary SHA-256 is missing")
    environment = pin.get("environment", {})
    if environment.get("max_cont_agents") != 1 or environment.get("victim_slot") != 0:
        raise VictimCheckpointError("victim environment must bind exactly slot 0")
    if environment.get("model_agent_layout") != 64:
        raise VictimCheckpointError("victim model layout must remain 64 agents")
    if environment.get("neutral_action_index") != 45:
        raise VictimCheckpointError("victim neutral action index must remain 45")
    if pin.get("failure_definition") is not None:
        raise VictimCheckpointError("Milestone B must not define research failure")
    return pin


def default_checkpoint_directory(pin: dict[str, Any] | None = None) -> Path:
    resolved = pin or load_victim_pin()
    repository_name = resolved["source"]["repository"].split("/")[-1]
    return (
        repository_root()
        / ".deps"
        / "checkpoints"
        / repository_name
        / resolved["source"]["revision"]
    )


def _json_without_duplicate_keys(payload: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VictimCheckpointError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VictimCheckpointError(f"invalid JSON: {exc}") from exc


def read_safetensors_contract(path: Path | str) -> dict[str, Any]:
    """Inspect safetensors metadata without importing Torch or deserializing pickle."""

    tensor_path = Path(path)
    file_size = tensor_path.stat().st_size
    with tensor_path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise VictimCheckpointError("safetensors file is shorter than its header prefix")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length == 0 or header_length > min(file_size - 8, 16 * 1024 * 1024):
            raise VictimCheckpointError("invalid safetensors header length")
        header = _json_without_duplicate_keys(handle.read(header_length))
    if not isinstance(header, dict):
        raise VictimCheckpointError("safetensors header must be a JSON object")

    tensors: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int, str]] = []
    parameter_count = 0
    data_size = file_size - 8 - header_length
    for name, specification in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(specification, dict):
            raise VictimCheckpointError(f"invalid tensor entry: {name}")
        dtype = specification.get("dtype")
        shape = specification.get("shape")
        offsets = specification.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(size, int) and size >= 0 for size in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
        ):
            raise VictimCheckpointError(f"malformed tensor specification: {name}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise VictimCheckpointError(f"tensor offsets outside file: {name}")
        intervals.append((start, end, name))
        parameter_count += math.prod(shape)
        tensors[name] = {"dtype": dtype, "shape": shape}
    for left, right in zip(sorted(intervals), sorted(intervals)[1:]):
        if left[1] > right[0]:
            raise VictimCheckpointError(
                f"overlapping safetensors data: {left[2]} and {right[2]}"
            )
    return {
        "tensor_count": len(tensors),
        "parameter_count": parameter_count,
        "tensors": tensors,
        "metadata": header.get("__metadata__"),
    }


def safetensors_state_sha256(path: Path | str) -> str:
    """Hash a pinned F32 safetensors file using the loaded-module state format."""

    tensor_path = Path(path)
    with tensor_path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise VictimCheckpointError("safetensors file is shorter than its header prefix")
        header_length = struct.unpack("<Q", prefix)[0]
        header = _json_without_duplicate_keys(handle.read(header_length))
        data_start = 8 + header_length
        digest = hashlib.sha256()
        for name in sorted(key for key in header if key != "__metadata__"):
            specification = header[name]
            if specification.get("dtype") != "F32":
                raise VictimCheckpointError(
                    f"unsupported state-hash dtype for {name}: {specification.get('dtype')}"
                )
            shape = specification.get("shape")
            offsets = specification.get("data_offsets")
            if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise VictimCheckpointError(f"malformed tensor specification: {name}")
            start, end = offsets
            handle.seek(data_start + start)
            payload = handle.read(end - start)
            if len(payload) != end - start:
                raise VictimCheckpointError(f"truncated tensor payload: {name}")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0torch.float32\0")
            digest.update(str(tuple(shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(payload)
            digest.update(b"\0")
    return digest.hexdigest()


def _check(
    name: str,
    ok: bool,
    expected: Any,
    observed: Any,
    *,
    required: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "required": required,
        "expected": expected,
        "observed": observed,
    }


def verify_checkpoint(
    directory: Path | str,
    pin: dict[str, Any] | None = None,
    *,
    gpudrive_source: Path | str | None = None,
) -> dict[str, Any]:
    resolved_pin = pin or load_victim_pin()
    root = Path(directory).resolve()
    checks: list[dict[str, Any]] = [
        _check("checkpoint.directory", root.is_dir(), "existing directory", str(root))
    ]
    if not root.is_dir():
        return {"ok": False, "directory": str(root), "checks": checks}

    for name, specification in sorted(resolved_pin["source"]["files"].items()):
        path = root / name
        if not specification["required_at_runtime"] and not path.exists():
            checks.append(
                _check(
                    f"checkpoint.file.{name}.optional",
                    False,
                    "absent or pinned bytes",
                    "absent",
                    required=False,
                )
            )
            continue
        size = path.stat().st_size if path.is_file() else None
        digest = sha256_file(path) if path.is_file() else None
        checks.extend(
            [
                _check(f"checkpoint.file.{name}.size", size == specification["size"], specification["size"], size),
                _check(f"checkpoint.file.{name}.sha256", digest == specification["sha256"], specification["sha256"], digest),
            ]
        )

    config_path = root / "config.json"
    config_payload = None
    try:
        config_payload = _json_without_duplicate_keys(config_path.read_bytes())
    except (OSError, VictimCheckpointError) as exc:
        checks.append(_check("checkpoint.config.parse", False, "valid JSON", str(exc)))
    if config_payload is not None:
        checks.append(
            _check(
                "checkpoint.config.exact",
                config_payload == resolved_pin["model_config"],
                resolved_pin["model_config"],
                config_payload,
            )
        )

    try:
        observed_contract = read_safetensors_contract(root / "model.safetensors")
        contract = resolved_pin["safetensors_contract"]
        expected_tensors = {
            name: {"dtype": contract["dtype"], "shape": shape}
            for name, shape in contract["required_shapes"].items()
        }
        observed_state_hash = safetensors_state_sha256(
            root / "model.safetensors"
        )
        checks.extend(
            [
                _check("checkpoint.safetensors.tensor_count", observed_contract["tensor_count"] == contract["tensor_count"], contract["tensor_count"], observed_contract["tensor_count"]),
                _check("checkpoint.safetensors.parameter_count", observed_contract["parameter_count"] == contract["parameter_count"], contract["parameter_count"], observed_contract["parameter_count"]),
                _check("checkpoint.safetensors.tensors", observed_contract["tensors"] == expected_tensors, expected_tensors, observed_contract["tensors"]),
                _check(
                    "checkpoint.safetensors.state_sha256",
                    observed_state_hash == contract["state_dict_sha256"],
                    contract["state_dict_sha256"],
                    observed_state_hash,
                ),
            ]
        )
    except (OSError, VictimCheckpointError) as exc:
        checks.append(
            _check("checkpoint.safetensors.parse", False, "valid pinned safetensors", str(exc))
        )

    if gpudrive_source is not None:
        source_root = Path(gpudrive_source).resolve()
        for relative, expected_hash in sorted(
            resolved_pin["upstream_source_hashes"].items()
        ):
            source_path = source_root / relative
            observed_hash = sha256_file(source_path) if source_path.is_file() else None
            checks.append(
                _check(
                    f"checkpoint.upstream_source.{relative}",
                    observed_hash == expected_hash,
                    expected_hash,
                    observed_hash,
                )
            )
    return {
        "schema": "gpudrive_victim_checkpoint_verification",
        "schema_version": 1,
        "ok": all(check["ok"] for check in checks if check["required"]),
        "directory": str(root),
        "checks": checks,
        "checkpoint_id": checkpoint_identity(resolved_pin),
    }


def checkpoint_identity(pin: dict[str, Any] | None = None) -> str:
    resolved = pin or load_victim_pin()
    identity = {
        "repository": resolved["source"]["repository"],
        "revision": resolved["source"]["revision"],
        "files": resolved["source"]["files"],
        "model_config": resolved["model_config"],
        "safetensors_contract": resolved["safetensors_contract"],
    }
    return "victim-ppo-" + canonical_json_sha256(identity)[:16]


def download_checkpoint(
    directory: Path | str,
    pin: dict[str, Any] | None = None,
    *,
    include_model_card: bool = True,
) -> dict[str, Any]:
    """Download exact revision files atomically, refusing to overwrite bad bytes."""

    resolved_pin = pin or load_victim_pin()
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository = resolved_pin["source"]["repository"]
    revision = resolved_pin["source"]["revision"]
    for name, specification in sorted(resolved_pin["source"]["files"].items()):
        if not specification["required_at_runtime"] and not include_model_card:
            continue
        destination = root / name
        if destination.exists():
            if (
                destination.is_file()
                and destination.stat().st_size == specification["size"]
                and sha256_file(destination) == specification["sha256"]
            ):
                continue
            raise VictimCheckpointError(
                f"refusing to overwrite checkpoint file with unexpected bytes: {destination}"
            )
        encoded_name = urllib.parse.quote(name)
        url = (
            f"https://huggingface.co/{repository}/resolve/{revision}/"
            f"{encoded_name}?download=true"
        )
        temporary = root / f".{name}.tmp-{uuid.uuid4().hex}"
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "gpudrive-adversary/0.1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open(
                "xb"
            ) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != specification["size"]:
                raise VictimCheckpointError(f"downloaded size mismatch for {name}")
            if sha256_file(temporary) != specification["sha256"]:
                raise VictimCheckpointError(f"downloaded SHA-256 mismatch for {name}")
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return verify_checkpoint(root, resolved_pin)
