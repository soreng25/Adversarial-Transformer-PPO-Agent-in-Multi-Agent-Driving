"""Safe, replayable Transformer-PPO checkpoint artifacts.

Model tensors use safetensors. Adam state uses numeric NPZ arrays loaded with
``allow_pickle=False`` plus an explicit JSON mapping from parameter names and
state fields to array keys. No Python object deserialization is used.
"""

from __future__ import annotations

import inspect
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..pins import canonical_json_sha256, sha256_file
from ..victim.checkpoint import (
    read_safetensors_contract,
    safetensors_state_sha256,
)
from ..victim.policy import torch_module_state_sha256
from .config import (
    AdversaryConfigError,
    adversary_config_sha256,
    validate_adversary_config,
)


class AdversaryCheckpointError(RuntimeError):
    """Raised when an adversary checkpoint violates its safe artifact contract."""


_SEQUENCE_TAG = "optimizer_sequence_type"
_SEQUENCE_ITEMS = "items"
_ADAM_GROUP_FIELDS = frozenset(
    {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        # A scheduler can add this scalar. It is restored after Adam creation.
        "initial_lr",
    }
)
_ADAM_STATE_FIELDS = frozenset(
    {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
)
_ADAM_PARAMETER_STATE_FIELDS = frozenset(
    {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
)


def _temporary_directory(output: Path) -> Path:
    resolved = output.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.parent / f".{resolved.name}.tmp-{uuid.uuid4().hex}"


def _json_without_duplicate_keys(payload: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdversaryCheckpointError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(payload, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise AdversaryCheckpointError(f"invalid checkpoint JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise AdversaryCheckpointError("checkpoint manifest must be a JSON object")
    return result


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        return _json_without_duplicate_keys(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise AdversaryCheckpointError(
            f"cannot read checkpoint manifest: {exc}"
        ) from exc


def _encode_optimizer_value(value: Any) -> Any:
    """Encode optimizer metadata without silently dropping sequence values."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AdversaryCheckpointError(
                "optimizer hyperparameters must be finite"
            )
        return value
    if isinstance(value, (tuple, list)):
        return {
            _SEQUENCE_TAG: "tuple" if isinstance(value, tuple) else "list",
            _SEQUENCE_ITEMS: [_encode_optimizer_value(item) for item in value],
        }
    raise AdversaryCheckpointError(
        "unsupported optimizer hyperparameter type: "
        f"{type(value).__name__}"
    )


def _decode_optimizer_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AdversaryCheckpointError(
                "optimizer hyperparameters must be finite"
            )
        return value
    if not isinstance(value, dict) or set(value) != {
        _SEQUENCE_TAG,
        _SEQUENCE_ITEMS,
    }:
        raise AdversaryCheckpointError("invalid typed optimizer hyperparameter")
    sequence_type = value[_SEQUENCE_TAG]
    items = value[_SEQUENCE_ITEMS]
    if sequence_type not in {"tuple", "list"} or not isinstance(items, list):
        raise AdversaryCheckpointError("invalid optimizer sequence encoding")
    decoded = [_decode_optimizer_value(item) for item in items]
    return tuple(decoded) if sequence_type == "tuple" else decoded


def _jsonable_optimizer_groups(
    optimizer: Any, names_by_id: dict[int, str]
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    if type(optimizer).__name__ != "Adam":
        raise AdversaryCheckpointError(
            f"only Adam checkpoints are supported, got {type(optimizer).__name__}"
        )
    for group_index, group in enumerate(optimizer.param_groups):
        if not isinstance(group, dict) or "params" not in group:
            raise AdversaryCheckpointError(
                f"optimizer parameter group {group_index} is malformed"
            )
        unknown = set(group) - _ADAM_GROUP_FIELDS - {"params"}
        if unknown:
            raise AdversaryCheckpointError(
                f"unsupported Adam group fields: {sorted(unknown)}"
            )
        encoded = {
            key: _encode_optimizer_value(value)
            for key, value in group.items()
            if key != "params"
        }
        try:
            encoded["parameter_names"] = [
                names_by_id[id(parameter)] for parameter in group["params"]
            ]
        except KeyError as exc:
            raise AdversaryCheckpointError(
                "optimizer contains a parameter not owned by the model"
            ) from exc
        groups.append(encoded)
    return groups


def _optimizer_state_array(value: Any, *, field: str) -> np.ndarray:
    if hasattr(value, "detach"):
        array = value.detach().cpu().contiguous().numpy()
    elif isinstance(value, (bool, int, float, np.generic)):
        array = np.asarray(value)
    else:
        raise AdversaryCheckpointError(
            f"unsupported Adam state type for {field}: {type(value).__name__}"
        )
    if array.dtype.kind == "O":
        raise AdversaryCheckpointError(f"Adam state {field} has object dtype")
    if not np.all(np.isfinite(array)):
        raise AdversaryCheckpointError(f"Adam state {field} is non-finite")
    # Unlike ascontiguousarray(), this preserves a scalar Adam step as 0-D.
    return np.array(array, copy=True, order="C")


def _save_optimizer_npz(path: Path, model: Any, optimizer: Any) -> dict[str, Any]:
    named_parameters = list(model.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in named_parameters}
    if len(names_by_id) != len(named_parameters):
        raise AdversaryCheckpointError("model exposes duplicate parameter objects")
    arrays: dict[str, np.ndarray] = {}
    states: list[dict[str, Any]] = []
    for parameter_index, (name, parameter) in enumerate(named_parameters):
        state = optimizer.state.get(parameter, {})
        if not isinstance(state, dict):
            raise AdversaryCheckpointError(f"Adam state for {name} is not a mapping")
        unknown = set(state) - _ADAM_STATE_FIELDS
        if unknown:
            raise AdversaryCheckpointError(
                f"unsupported Adam state fields for {name}: {sorted(unknown)}"
            )
        fields: dict[str, str] = {}
        for field, value in sorted(state.items()):
            key = f"state_{parameter_index:04d}_{field}"
            arrays[key] = _optimizer_state_array(value, field=field)
            fields[field] = key
        states.append({"parameter_name": name, "fields": fields})
    np.savez_compressed(path, **arrays)
    return {
        "class": "Adam",
        "format": "typed_npz_no_pickle",
        "parameter_groups": _jsonable_optimizer_groups(optimizer, names_by_id),
        "states": states,
        "array_keys": sorted(arrays),
    }


def _valid_hash(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_adam_hyperparameters(group: dict[str, Any]) -> bool:
    try:
        decoded = {
            key: _decode_optimizer_value(value)
            for key, value in group.items()
            if key != "parameter_names"
        }
    except AdversaryCheckpointError:
        return False
    if not set(decoded).issubset(_ADAM_GROUP_FIELDS):
        return False
    betas = decoded.get("betas")
    if not (
        isinstance(betas, (tuple, list))
        and len(betas) == 2
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0.0 <= float(value) < 1.0
            for value in betas
        )
    ):
        return False
    for positive in ("lr", "eps"):
        value = decoded.get(positive)
        if not (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            return False
    weight_decay = decoded.get("weight_decay")
    if not (
        isinstance(weight_decay, (int, float))
        and not isinstance(weight_decay, bool)
        and math.isfinite(float(weight_decay))
        and float(weight_decay) >= 0.0
    ):
        return False
    for boolean in ("amsgrad", "maximize", "capturable", "differentiable"):
        if boolean in decoded and not isinstance(decoded[boolean], bool):
            return False
    for optional_boolean in ("foreach", "fused", "decoupled_weight_decay"):
        if optional_boolean in decoded and decoded[optional_boolean] is not None:
            if not isinstance(decoded[optional_boolean], bool):
                return False
    return True


def _optimizer_contract_checks(
    optimizer_contract: Any,
    tensor_contract: dict[str, Any] | None,
    payload: Any | None,
) -> dict[str, bool]:
    checks = {
        "optimizer.format": False,
        "optimizer.groups": False,
        "optimizer.states": False,
        "optimizer.field_mapping": False,
        "optimizer.array_types": False,
    }
    if not isinstance(optimizer_contract, dict):
        return checks
    checks["optimizer.format"] = (
        optimizer_contract.get("class") == "Adam"
        and optimizer_contract.get("format") == "typed_npz_no_pickle"
    )
    groups = optimizer_contract.get("parameter_groups")
    states = optimizer_contract.get("states")
    declared_keys = optimizer_contract.get("array_keys")
    if not (
        isinstance(groups, list)
        and groups
        and all(isinstance(group, dict) for group in groups)
    ):
        return checks

    parameter_names: list[str] = []
    groups_ok = True
    amsgrad_by_parameter: dict[str, bool] = {}
    for group in groups:
        names = group.get("parameter_names")
        if not (
            isinstance(names, list)
            and names
            and all(isinstance(name, str) and name for name in names)
            and _validate_adam_hyperparameters(group)
        ):
            groups_ok = False
            continue
        decoded_amsgrad = bool(
            _decode_optimizer_value(group.get("amsgrad", False))
        )
        parameter_names.extend(names)
        amsgrad_by_parameter.update({name: decoded_amsgrad for name in names})
    groups_ok = groups_ok and len(parameter_names) == len(set(parameter_names))
    checks["optimizer.groups"] = groups_ok

    if not isinstance(states, list) or not isinstance(declared_keys, list):
        return checks
    state_names: list[str] = []
    mapped_keys: list[str] = []
    state_fields_by_name: dict[str, dict[str, str]] = {}
    states_ok = True
    for state in states:
        if not isinstance(state, dict) or set(state) != {"parameter_name", "fields"}:
            states_ok = False
            continue
        name = state["parameter_name"]
        fields = state["fields"]
        if not isinstance(name, str) or not isinstance(fields, dict):
            states_ok = False
            continue
        if not all(
            isinstance(field, str)
            and field in _ADAM_STATE_FIELDS
            and isinstance(key, str)
            and key
            for field, key in fields.items()
        ):
            states_ok = False
            continue
        field_set = set(fields)
        if field_set and not {"step", "exp_avg", "exp_avg_sq"}.issubset(field_set):
            states_ok = False
        if amsgrad_by_parameter.get(name, False) and field_set:
            states_ok = states_ok and "max_exp_avg_sq" in field_set
        if not amsgrad_by_parameter.get(name, False) and "max_exp_avg_sq" in field_set:
            states_ok = False
        state_names.append(name)
        state_fields_by_name[name] = fields
        mapped_keys.extend(fields.values())
    states_ok = (
        states_ok
        and len(state_names) == len(set(state_names))
        and set(state_names) == set(parameter_names)
    )
    checks["optimizer.states"] = states_ok

    declared_ok = (
        all(isinstance(key, str) and key for key in declared_keys)
        and len(declared_keys) == len(set(declared_keys))
        and sorted(declared_keys) == sorted(mapped_keys)
        and len(mapped_keys) == len(set(mapped_keys))
    )
    if payload is not None:
        declared_ok = declared_ok and sorted(payload.files) == sorted(declared_keys)
    checks["optimizer.field_mapping"] = declared_ok

    arrays_ok = payload is not None and declared_ok
    tensor_shapes = (
        tensor_contract.get("tensors", {})
        if isinstance(tensor_contract, dict)
        else {}
    )
    if tensor_shapes:
        arrays_ok = arrays_ok and set(parameter_names) == set(tensor_shapes)
    if arrays_ok:
        for parameter_name, fields in state_fields_by_name.items():
            shape = tuple(tensor_shapes.get(parameter_name, {}).get("shape", ()))
            for field, key in fields.items():
                array = payload[key]
                if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
                    arrays_ok = False
                if field == "step":
                    arrays_ok = arrays_ok and array.size == 1
                elif field in _ADAM_PARAMETER_STATE_FIELDS:
                    arrays_ok = arrays_ok and array.shape == shape
    checks["optimizer.array_types"] = bool(arrays_ok)
    return checks


def _model_config_from_research_config(config: dict[str, Any]) -> Any:
    from .model import AdversaryModelConfig

    try:
        token = config["token"]
        model = config["model"]
        prior = config["prior"]
        if model["activation"] != "relu":
            raise ValueError("only the pinned relu activation is supported")
        return AdversaryModelConfig(
            token_dim=int(token["token_dim"]),
            context_length=int(token["history_length"]),
            model_dim=int(model["d_model"]),
            pre_actor_dim=int(model["d_model"]),
            action_dim=int(model["action_dimensions"]),
            num_layers=int(model["num_layers"]),
            num_heads=int(model["nhead"]),
            feed_forward_dim=int(model["dim_feedforward"]),
            dropout=float(model["dropout"]),
            initial_log_std=float(model["initial_log_std"]),
            minimum_log_std=float(model["min_log_std"]),
            maximum_log_std=float(model["max_log_std"]),
            action_epsilon=float(prior["epsilon"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AdversaryCheckpointError(
            f"cannot reconstruct adversary architecture from config: {exc}"
        ) from exc


def save_adversary_checkpoint(
    output: Path,
    *,
    model: Any,
    optimizer: Any,
    config: dict[str, Any],
    iteration: int,
    total_transitions: int,
    metrics: dict[str, Any],
    fingerprints: dict[str, Any],
    parent_victim_checkpoint_id: str,
) -> dict[str, Any]:
    """Atomically save inference weights, typed Adam state, and provenance."""

    if output.exists():
        raise AdversaryCheckpointError(f"checkpoint output already exists: {output}")
    try:
        from safetensors.torch import save_file
    except Exception as exc:
        raise AdversaryCheckpointError(f"safetensors is unavailable: {exc}") from exc
    temporary = _temporary_directory(output)
    temporary.mkdir()
    try:
        weights_path = temporary / "model.safetensors"
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        save_file(state, str(weights_path))
        optimizer_path = temporary / "optimizer.npz"
        optimizer_contract = _save_optimizer_npz(optimizer_path, model, optimizer)
        manifest = {
            "schema": "gpudrive_adversary_checkpoint",
            "schema_version": 1,
            "artifact_id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": config["purpose"],
            "research_claims_allowed": config["research_claims_allowed"],
            "iteration": int(iteration),
            "total_transitions": int(total_transitions),
            "parent_victim_checkpoint_id": parent_victim_checkpoint_id,
            "config": config,
            "config_sha256": adversary_config_sha256(config),
            "model": {
                "class": type(model).__name__,
                "state_sha256": torch_module_state_sha256(model),
                "architecture": config["model"],
                "token": config["token"],
                "training": bool(model.training),
            },
            "optimizer": optimizer_contract,
            "metrics": metrics,
            "files": {
                "model.safetensors": {
                    "size": weights_path.stat().st_size,
                    "sha256": sha256_file(weights_path),
                },
                "optimizer.npz": {
                    "size": optimizer_path.stat().st_size,
                    "sha256": sha256_file(optimizer_path),
                },
            },
            "fingerprints": fingerprints,
        }
        manifest["artifact_id"] = (
            "adversary-ppo-" + canonical_json_sha256(manifest)[:16]
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        validation = validate_adversary_checkpoint(temporary)
        if not validation["ok"]:
            raise AdversaryCheckpointError(
                "generated checkpoint failed validation: "
                f"{validation['failed_checks']}"
            )
        temporary.replace(output.resolve())
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_adversary_checkpoint(path: Path) -> dict[str, Any]:
    manifest = _read_manifest(path)
    checks: dict[str, bool] = {}
    checks["schema"] = (
        manifest.get("schema") == "gpudrive_adversary_checkpoint"
        and manifest.get("schema_version") == 1
    )
    config = manifest.get("config")
    config_hash = adversary_config_sha256(config) if isinstance(config, dict) else None
    try:
        validate_adversary_config(config)
        approved_config = True
    except (AdversaryConfigError, TypeError, ValueError):
        approved_config = False
    checks["scope"] = (
        isinstance(config, dict)
        and manifest.get("research_claims_allowed")
        is config.get("research_claims_allowed")
        is False
        and manifest.get("purpose") == config.get("purpose")
        == "tiny_training_smoke_only"
    )
    checks["config"] = (
        approved_config
        and manifest.get("config_sha256") == config_hash
    )
    try:
        _model_config_from_research_config(config if isinstance(config, dict) else {})
        reconstructable = True
    except AdversaryCheckpointError:
        reconstructable = False
    model_contract = manifest.get("model", {})
    checks["model_config_consistency"] = (
        reconstructable
        and isinstance(model_contract, dict)
        and model_contract.get("class") == "CausalTransformerActorCritic"
        and model_contract.get("architecture") == config.get("model")
        and model_contract.get("token") == config.get("token")
        and isinstance(model_contract.get("training"), bool)
    )

    files = manifest.get("files", {})
    for name in ("model.safetensors", "optimizer.npz"):
        file_path = path / name
        specification = files.get(name, {}) if isinstance(files, dict) else {}
        checks[f"file.{name}"] = (
            file_path.is_file()
            and file_path.stat().st_size == specification.get("size")
            and sha256_file(file_path) == specification.get("sha256")
        )

    tensor_contract: dict[str, Any] | None = None
    try:
        tensor_contract = read_safetensors_contract(path / "model.safetensors")
        observed_state_hash = safetensors_state_sha256(
            path / "model.safetensors"
        )
        checks["safe_weights"] = (
            tensor_contract["tensor_count"] > 0
            and tensor_contract["parameter_count"] > 0
        )
        checks["model_state_sha256"] = (
            _valid_hash(model_contract.get("state_sha256"), 64)
            and observed_state_hash == model_contract.get("state_sha256")
        )
    except Exception:
        checks["safe_weights"] = False
        checks["model_state_sha256"] = False

    try:
        with np.load(path / "optimizer.npz", allow_pickle=False) as payload:
            checks.update(
                _optimizer_contract_checks(
                    manifest.get("optimizer"), tensor_contract, payload
                )
            )
    except (OSError, ValueError):
        checks.update(
            _optimizer_contract_checks(manifest.get("optimizer"), tensor_contract, None)
        )

    fingerprints = manifest.get("fingerprints", {})
    checks["victim_fingerprint"] = (
        isinstance(fingerprints, dict)
        and _valid_hash(fingerprints.get("victim_checkpoint_model_sha256"), 64)
        and _valid_hash(fingerprints.get("victim_config_sha256"), 64)
        and isinstance(manifest.get("parent_victim_checkpoint_id"), str)
        and manifest["parent_victim_checkpoint_id"].startswith("victim-ppo-")
    )
    checks["scene_fingerprint"] = isinstance(fingerprints, dict) and _valid_hash(
        fingerprints.get("scene_sha256"), 64
    ) and isinstance(fingerprints.get("scene_scenario_id"), str) and bool(
        fingerprints.get("scene_scenario_id")
    ) and isinstance(fingerprints.get("victim_stable_id"), int) and not isinstance(
        fingerprints.get("victim_stable_id"), bool
    )
    checks["gpudrive_fingerprint"] = (
        isinstance(fingerprints, dict)
        and _valid_hash(fingerprints.get("gpudrive_commit"), 40)
        and isinstance(fingerprints.get("gpudrive_submodules"), dict)
    )
    checks["native_fingerprint"] = isinstance(fingerprints, dict) and _valid_hash(
        fingerprints.get("native_extension_sha256"), 64
    )
    checks["kernel_cache_fingerprint"] = isinstance(
        fingerprints, dict
    ) and _valid_hash(fingerprints.get("madrona_kernel_cache_sha256"), 64)
    methodology = config.get("methodology_source", {}) if isinstance(config, dict) else {}
    checks["config_fingerprint_consistency"] = (
        isinstance(fingerprints, dict)
        and fingerprints.get("adversary_config_sha256") == config_hash
        and fingerprints.get("methodology_repository")
        == methodology.get("repository")
        and fingerprints.get("methodology_commit") == methodology.get("commit")
        and _valid_hash(fingerprints.get("source_verification_sha256"), 64)
        and _valid_hash(fingerprints.get("victim_verification_sha256"), 64)
    )
    manifest_for_id = dict(manifest)
    manifest_for_id["artifact_id"] = None
    checks["artifact_id"] = manifest.get("artifact_id") == (
        "adversary-ppo-" + canonical_json_sha256(manifest_for_id)[:16]
    )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "gpudrive_adversary_checkpoint_validation",
        "schema_version": 1,
        "ok": not failed,
        "artifact": str(path.resolve()),
        "checks": checks,
        "failed_checks": failed,
    }


@dataclass(slots=True)
class LoadedAdversaryCheckpoint:
    """Fully reconstructed model, Adam optimizer, and verified manifest."""

    model: Any
    optimizer: Any
    manifest: dict[str, Any]
    validation: dict[str, Any]


def _decoded_group(group: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _decode_optimizer_value(value)
        for key, value in group.items()
        if key != "parameter_names"
    }


def _restore_optimizer_state(
    *,
    optimizer: Any,
    model: Any,
    optimizer_contract: dict[str, Any],
    payload: Any,
    torch: Any,
) -> None:
    parameters = dict(model.named_parameters())
    group_for_name: dict[str, dict[str, Any]] = {}
    for serialized, restored in zip(
        optimizer_contract["parameter_groups"], optimizer.param_groups, strict=True
    ):
        for name in serialized["parameter_names"]:
            group_for_name[name] = restored
    for state_specification in optimizer_contract["states"]:
        name = state_specification["parameter_name"]
        parameter = parameters[name]
        group = group_for_name[name]
        restored_state: dict[str, Any] = {}
        for field, array_key in state_specification["fields"].items():
            expected = np.asarray(payload[array_key])
            tensor = torch.from_numpy(np.array(expected, copy=True))
            if field == "step":
                use_parameter_device = bool(group.get("capturable")) or bool(
                    group.get("fused")
                )
                if use_parameter_device:
                    tensor = tensor.to(device=parameter.device)
            else:
                tensor = tensor.to(device=parameter.device)
            restored_state[field] = tensor
        if restored_state:
            optimizer.state[parameter] = restored_state
        else:
            optimizer.state.pop(parameter, None)

    # Verify that every restored tensor is byte-identical after a CPU transfer.
    for state_specification in optimizer_contract["states"]:
        parameter = parameters[state_specification["parameter_name"]]
        actual_state = optimizer.state.get(parameter, {})
        if set(actual_state) != set(state_specification["fields"]):
            raise AdversaryCheckpointError("restored Adam state fields differ")
        for field, array_key in state_specification["fields"].items():
            actual = actual_state[field].detach().cpu().contiguous().numpy()
            expected = np.asarray(payload[array_key])
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise AdversaryCheckpointError(
                    f"restored Adam state differs for {parameter=} {field=}"
                )


def load_adversary_checkpoint(
    path: Path,
    device: str,
) -> LoadedAdversaryCheckpoint:
    """Safely reconstruct a strict model and Adam optimizer checkpoint.

    Validation, exact model-state hashing, optimizer field mapping, and typed
    optimizer-state verification all complete before this function returns.
    """

    validation = validate_adversary_checkpoint(path)
    if not validation["ok"]:
        raise AdversaryCheckpointError(
            f"checkpoint validation failed: {validation['failed_checks']}"
        )
    manifest = _read_manifest(path)
    try:
        import torch
        from safetensors.torch import load_file

        from .model import CausalTransformerActorCritic
    except Exception as exc:
        raise AdversaryCheckpointError(
            f"Torch/safetensors checkpoint dependencies are unavailable: {exc}"
        ) from exc

    try:
        torch_device = torch.device(device)
        model = CausalTransformerActorCritic(
            _model_config_from_research_config(manifest["config"])
        ).to(torch_device)
        state = load_file(str(path / "model.safetensors"), device=str(torch_device))
        model.load_state_dict(state, strict=True)
        if torch_module_state_sha256(model) != manifest["model"]["state_sha256"]:
            raise AdversaryCheckpointError(
                "loaded model state does not match its manifest hash"
            )
        model.train(manifest["model"]["training"])

        parameter_by_name = dict(model.named_parameters())
        serialized_groups = manifest["optimizer"]["parameter_groups"]
        adam_signature = inspect.signature(torch.optim.Adam.__init__)
        accepted_fields = set(adam_signature.parameters) - {"self", "params"}
        groups: list[dict[str, Any]] = []
        auxiliary: list[dict[str, Any]] = []
        for serialized in serialized_groups:
            decoded = _decoded_group(serialized)
            unsupported = set(decoded) - accepted_fields - {"initial_lr"}
            if unsupported:
                raise AdversaryCheckpointError(
                    "runtime Adam does not support checkpoint fields: "
                    f"{sorted(unsupported)}"
                )
            groups.append(
                {
                    "params": [
                        parameter_by_name[name]
                        for name in serialized["parameter_names"]
                    ],
                    **{
                        key: value
                        for key, value in decoded.items()
                        if key != "initial_lr"
                    },
                }
            )
            auxiliary.append(
                {key: value for key, value in decoded.items() if key == "initial_lr"}
            )
        optimizer = torch.optim.Adam(groups)
        for restored_group, metadata in zip(
            optimizer.param_groups, auxiliary, strict=True
        ):
            restored_group.update(metadata)

        names_by_id = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        if _jsonable_optimizer_groups(optimizer, names_by_id) != serialized_groups:
            raise AdversaryCheckpointError(
                "reconstructed Adam hyperparameters differ from the manifest"
            )
        with np.load(path / "optimizer.npz", allow_pickle=False) as payload:
            _restore_optimizer_state(
                optimizer=optimizer,
                model=model,
                optimizer_contract=manifest["optimizer"],
                payload=payload,
                torch=torch,
            )
    except AdversaryCheckpointError:
        raise
    except Exception as exc:
        raise AdversaryCheckpointError(
            f"cannot reconstruct adversary checkpoint: {type(exc).__name__}: {exc}"
        ) from exc
    return LoadedAdversaryCheckpoint(
        model=model,
        optimizer=optimizer,
        manifest=manifest,
        validation=validation,
    )
