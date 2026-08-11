"""Pure validation for an indivisible Milestone C training-run artifact."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..pins import canonical_json_sha256, load_pins, sha256_file
from ..victim.checkpoint import checkpoint_identity, load_victim_pin
from .checkpoint import validate_adversary_checkpoint
from .config import AdversaryConfigError, load_adversary_config, validate_adversary_config
from .failure import RAW_INFO_ORDER, assess_nominal_goal_eligibility


class AdversaryTrainingArtifactError(RuntimeError):
    """Raised when a training-run directory cannot be parsed as an artifact."""


_ROLLOUT_ARRAYS = {
    "victim_observations",
    "raw_info",
    "tokens",
    "history_masks",
    "victim_action_indices",
    "victim_logits",
    "victim_nominal_commands",
    "adversary_raw_actions",
    "disturbance_requested",
    "disturbance_effective",
    "applied_commands",
    "disturbance_saturated",
    "command_saturated",
    "prior_nll_exact",
    "prior_nll_penalty",
    "policy_log_probability",
    "adversary_values",
    "rewards",
    "pre_actor_latent",
    "failure_by_transition",
    "failure_kind_bits",
}

_NOMINAL_ARRAYS = {
    "victim_observations",
    "raw_info",
    "victim_action_indices",
    "victim_nominal_commands",
    "applied_commands",
}


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdversaryTrainingArtifactError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdversaryTrainingArtifactError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdversaryTrainingArtifactError(f"{path} must contain a JSON object")
    return value


def _verification_passed(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get("ok"), bool):
            return False
        if row.get("required", True) and not row["ok"]:
            return False
    return True


def _named_checks(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    rows = [row for row in value if isinstance(row, dict)]
    names = [row.get("name") for row in rows]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        return {}
    return {str(row["name"]): row for row in rows}


def _arrays_are_safe_and_finite(arrays: dict[str, np.ndarray]) -> bool:
    for value in arrays.values():
        if value.dtype.kind not in "biuf":
            return False
        if value.dtype.kind in "f" and not bool(np.all(np.isfinite(value))):
            return False
    return True


def _rollout_checks(
    path: Path,
    metadata: Any,
    config: dict[str, Any],
) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "file": path.is_file(),
        "sha256": False,
        "npz": False,
        "arrays": False,
        "safe_numeric": False,
        "shapes": False,
        "transition_count": False,
        "failure_index": False,
    }
    if not path.is_file() or not isinstance(metadata, dict):
        return checks
    checks["sha256"] = _is_hex(metadata.get("sha256"), 64) and sha256_file(
        path
    ) == metadata["sha256"]
    try:
        with np.load(path, allow_pickle=False) as payload:
            checks["arrays"] = set(payload.files) == _ROLLOUT_ARRAYS
            if not checks["arrays"]:
                return checks
            arrays = {name: np.asarray(payload[name]) for name in _ROLLOUT_ARRAYS}
        checks["npz"] = True
    except (OSError, ValueError, KeyError):
        return checks

    checks["safe_numeric"] = _arrays_are_safe_and_finite(arrays)
    transition_count = arrays["victim_action_indices"].shape[0]
    token = config.get("token", {})
    model = config.get("model", {})
    intervention = config.get("intervention", {})
    expected_shapes = {
        "victim_observations": (
            transition_count + 1,
            int(token.get("victim_observation_dim", -1)),
        ),
        "raw_info": (transition_count + 1, len(RAW_INFO_ORDER)),
        "tokens": (transition_count, int(token.get("token_dim", -1))),
        "history_masks": (
            transition_count,
            int(token.get("history_length", -1)),
        ),
        "victim_action_indices": (transition_count,),
        "victim_logits": (transition_count, 91),
        "victim_nominal_commands": (transition_count, 3),
        "adversary_raw_actions": (
            transition_count,
            int(intervention.get("dimensions", -1)),
        ),
        "disturbance_requested": (
            transition_count,
            int(intervention.get("dimensions", -1)),
        ),
        "disturbance_effective": (
            transition_count,
            int(intervention.get("dimensions", -1)),
        ),
        "applied_commands": (transition_count, 3),
        "disturbance_saturated": (
            transition_count,
            int(intervention.get("dimensions", -1)),
        ),
        "command_saturated": (
            transition_count,
            int(intervention.get("dimensions", -1)),
        ),
        "prior_nll_exact": (transition_count,),
        "prior_nll_penalty": (transition_count,),
        "policy_log_probability": (transition_count,),
        "adversary_values": (transition_count,),
        "rewards": (transition_count,),
        "pre_actor_latent": (
            transition_count,
            int(model.get("d_model", -1)),
        ),
        "failure_by_transition": (transition_count,),
        "failure_kind_bits": (transition_count, 3),
    }
    checks["shapes"] = transition_count > 0 and all(
        arrays[name].shape == shape for name, shape in expected_shapes.items()
    )
    checks["transition_count"] = (
        isinstance(metadata.get("transition_count"), int)
        and not isinstance(metadata.get("transition_count"), bool)
        and metadata["transition_count"] == transition_count
    )
    failure_timestep = metadata.get("failure_timestep")
    failures = np.flatnonzero(arrays["failure_by_transition"])
    if failure_timestep is None:
        checks["failure_index"] = failures.size == 0
    else:
        checks["failure_index"] = (
            isinstance(failure_timestep, int)
            and not isinstance(failure_timestep, bool)
            and failures.tolist() == [failure_timestep]
            and failure_timestep == transition_count - 1
        )
    return checks


def _nominal_checks(
    path: Path,
    expected_sha256: Any,
    eligibility: Any,
    config: dict[str, Any],
) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "file": path.is_file(),
        "sha256": False,
        "npz": False,
        "arrays": False,
        "safe_numeric": False,
        "shapes": False,
        "eligibility_evidence": False,
    }
    if not path.is_file():
        return checks
    checks["sha256"] = _is_hex(expected_sha256, 64) and sha256_file(path) == expected_sha256
    try:
        with np.load(path, allow_pickle=False) as payload:
            checks["arrays"] = set(payload.files) == _NOMINAL_ARRAYS
            if not checks["arrays"]:
                return checks
            arrays = {name: np.asarray(payload[name]) for name in _NOMINAL_ARRAYS}
        checks["npz"] = True
    except (OSError, ValueError, KeyError):
        return checks

    checks["safe_numeric"] = _arrays_are_safe_and_finite(arrays)
    transition_count = arrays["victim_action_indices"].shape[0]
    observation_dim = int(config.get("token", {}).get("victim_observation_dim", -1))
    checks["shapes"] = transition_count > 0 and (
        arrays["victim_observations"].shape
        == (transition_count + 1, observation_dim)
        and arrays["raw_info"].shape
        == (transition_count + 1, len(RAW_INFO_ORDER))
        and arrays["victim_action_indices"].shape == (transition_count,)
        and arrays["victim_nominal_commands"].shape == (transition_count, 3)
        and arrays["applied_commands"].shape == (transition_count, 3)
    )
    try:
        assessed = asdict(assess_nominal_goal_eligibility(arrays["raw_info"][1:]))
        checks["eligibility_evidence"] = (
            isinstance(eligibility, dict)
            and assessed["eligible"] is True
            and eligibility.get("eligible") is True
            and eligibility.get("reason") == assessed["reason"]
            and eligibility.get("goal_timestep") == assessed["goal_timestep"]
            and eligibility.get("failure_timestep") is None
            and list(eligibility.get("failure_kinds", []))
            == list(assessed["failure_kinds"])
        )
    except (ValueError, TypeError):
        checks["eligibility_evidence"] = False
    return checks


def validate_adversary_training_artifact(path: Path) -> dict[str, Any]:
    """Validate the complete run directory without importing Torch or GPUDrive."""

    root = Path(path).resolve()
    manifest = _load_json_object(root / "manifest.json")
    checks: dict[str, bool] = {}
    checks["schema"] = (
        manifest.get("schema") == "gpudrive_adversary_training_run"
        and manifest.get("schema_version") == 1
    )
    checks["scope"] = (
        manifest.get("purpose") == "tiny_training_smoke_only"
        and manifest.get("research_claims_allowed") is False
    )

    config = manifest.get("config")
    checks["config.object"] = isinstance(config, dict)
    if not isinstance(config, dict):
        config = {}
    config_sha256 = canonical_json_sha256(config)
    checks["config.sha256"] = (
        _is_hex(manifest.get("config_sha256"), 64)
        and manifest["config_sha256"] == config_sha256
    )
    try:
        validate_adversary_config(config)
        checks["config.contract"] = True
    except (AdversaryConfigError, TypeError, ValueError):
        checks["config.contract"] = False

    fingerprints = manifest.get("fingerprints")
    checks["fingerprints.object"] = isinstance(fingerprints, dict)
    if not isinstance(fingerprints, dict):
        fingerprints = {}
    checks["fingerprints.config"] = (
        fingerprints.get("adversary_config_sha256") == config_sha256
    )

    expected_config = load_adversary_config()
    expected_pins = load_pins()
    expected_victim = load_victim_pin()
    methodology = config.get("methodology_source", {})
    checks["fingerprints.methodology"] = (
        methodology == expected_config["methodology_source"]
        and fingerprints.get("methodology_repository")
        == methodology.get("repository")
        and fingerprints.get("methodology_commit") == methodology.get("commit")
    )
    expected_gpudrive = expected_pins["gpudrive"]
    checks["fingerprints.gpudrive"] = (
        fingerprints.get("gpudrive_commit") == expected_gpudrive["commit"]
        and fingerprints.get("gpudrive_submodules")
        == expected_gpudrive["submodules"]
    )
    scene = manifest.get("scene", {})
    expected_scene = expected_pins["smoke_scene"]
    checks["fingerprints.scene"] = (
        fingerprints.get("scene_sha256") == expected_scene["sha256"]
        and fingerprints.get("scene_scenario_id")
        == expected_scene["scenario_id"]
        and fingerprints.get("victim_stable_id")
        == expected_scene["sdc_object_id"]
        and scene.get("relative_path") == expected_scene["relative_path"]
        and scene.get("scenario_id") == expected_scene["scenario_id"]
        and scene.get("victim_slot") == 0
        and scene.get("victim_stable_id") == expected_scene["sdc_object_id"]
    )
    expected_victim_id = checkpoint_identity(expected_victim)
    checks["fingerprints.victim"] = (
        manifest.get("parent_victim_checkpoint_id") == expected_victim_id
        and fingerprints.get("victim_checkpoint_model_sha256")
        == expected_victim["source"]["files"]["model.safetensors"]["sha256"]
        and fingerprints.get("victim_config_sha256")
        == expected_victim["source"]["files"]["config.json"]["sha256"]
    )
    checks["fingerprints.native"] = _is_hex(
        fingerprints.get("native_extension_sha256"), 64
    )
    cache_sha256 = fingerprints.get("madrona_kernel_cache_sha256")
    checks["fingerprints.kernel_cache"] = _is_hex(cache_sha256, 64)
    port = fingerprints.get("port", {})
    checks["fingerprints.port"] = (
        isinstance(port, dict)
        and _is_hex(port.get("source_tree_sha256"), 64)
        and port.get("source_tree_matches_declared") is True
    )

    source_verification = manifest.get("source_verification")
    victim_verification = manifest.get("victim_verification")
    checks["source_verification.passed"] = _verification_passed(source_verification)
    checks["source_verification.sha256"] = (
        fingerprints.get("source_verification_sha256")
        == canonical_json_sha256(source_verification)
    )
    source_rows = _named_checks(source_verification)
    identity_expectations = {
        "source.commit": expected_gpudrive["commit"],
        "scene.sha256": expected_scene["sha256"],
        "scene.scenario_id": expected_scene["scenario_id"],
        "scene.sdc_object_id": expected_scene["sdc_object_id"],
        **{
            f"source.gitlink.{name}": revision
            for name, revision in expected_gpudrive["submodules"].items()
        },
    }
    checks["source_verification.identity"] = all(
        source_rows.get(name, {}).get("ok") is True
        and source_rows[name].get("expected") == expected
        and source_rows[name].get("observed") == expected
        for name, expected in identity_expectations.items()
    )
    checks["victim_verification.passed"] = (
        isinstance(victim_verification, dict)
        and victim_verification.get("ok") is True
        and victim_verification.get("checkpoint_id") == expected_victim_id
        and _verification_passed(victim_verification.get("checks"))
    )
    checks["victim_verification.sha256"] = (
        fingerprints.get("victim_verification_sha256")
        == canonical_json_sha256(victim_verification)
    )

    victim_before = manifest.get("victim_state_sha256_before")
    checks["victim.frozen"] = (
        _is_hex(victim_before, 64)
        and manifest.get("victim_state_sha256_after") == victim_before
    )
    runtime = manifest.get("runtime")
    checks["runtime.cublas_workspace_config"] = (
        isinstance(runtime, dict)
        and runtime.get("cublas_workspace_config") in {":4096:8", ":16:8"}
    )
    checks["runtime.deterministic_algorithms"] = (
        isinstance(runtime, dict)
        and runtime.get("deterministic_algorithms") is True
    )
    eligibility = manifest.get("eligibility")
    checks["eligibility"] = (
        isinstance(eligibility, dict)
        and eligibility.get("eligible") is True
        and eligibility.get("reason") == "clean_goal_success"
        and eligibility.get("failure_timestep") is None
        and not eligibility.get("failure_kinds")
    )

    expected_root_entries = {
        "manifest.json",
        "checkpoints",
        "last-training-rollout.npz",
        "last-evaluation-rollout.npz",
        "nominal-eligibility.npz",
    }
    try:
        checks["directory.indivisible"] = root.is_dir() and {
            child.name for child in root.iterdir()
        } == expected_root_entries
    except OSError:
        checks["directory.indivisible"] = False

    training_metadata = manifest.get("last_training_rollout")
    evaluation_metadata = manifest.get("last_deterministic_evaluation")
    for prefix, result in (
        (
            "rollout.training",
            _rollout_checks(
                root / "last-training-rollout.npz",
                training_metadata,
                config,
            ),
        ),
        (
            "rollout.evaluation",
            _rollout_checks(
                root / "last-evaluation-rollout.npz",
                evaluation_metadata,
                config,
            ),
        ),
        (
            "rollout.nominal",
            _nominal_checks(
                root / "nominal-eligibility.npz",
                manifest.get("nominal_eligibility_trace_sha256"),
                eligibility,
                config,
            ),
        ),
    ):
        checks.update({f"{prefix}.{name}": passed for name, passed in result.items()})

    iterations = config.get("training", {}).get("iterations")
    checkpoint_ids = manifest.get("checkpoint_ids")
    expected_names = (
        {f"iteration-{index:04d}" for index in range(1, iterations + 1)}
        if isinstance(iterations, int) and not isinstance(iterations, bool) and iterations > 0
        else set()
    )
    checkpoints_directory = root / "checkpoints"
    try:
        observed_names = {
            child.name for child in checkpoints_directory.iterdir() if child.is_dir()
        }
        only_directories = all(child.is_dir() for child in checkpoints_directory.iterdir())
    except OSError:
        observed_names = set()
        only_directories = False
    checks["checkpoints.layout"] = (
        bool(expected_names)
        and observed_names == expected_names
        and only_directories
        and isinstance(checkpoint_ids, list)
        and len(checkpoint_ids) == len(expected_names)
        and all(isinstance(value, str) for value in checkpoint_ids)
        and len(set(checkpoint_ids)) == len(checkpoint_ids)
    )

    child_reports: dict[str, Any] = {}
    child_model_states: dict[str, Any] = {}
    child_contracts_ok = checks["checkpoints.layout"]
    if isinstance(checkpoint_ids, list):
        for index, checkpoint_id in enumerate(checkpoint_ids, start=1):
            name = f"iteration-{index:04d}"
            child = checkpoints_directory / name
            try:
                child_manifest = _load_json_object(child / "manifest.json")
                child_report = validate_adversary_checkpoint(child)
                child_reports[name] = child_report
                child_model = child_manifest.get("model")
                if isinstance(checkpoint_id, str) and isinstance(child_model, dict):
                    child_model_states[checkpoint_id] = child_model.get("state_sha256")
                child_ok = (
                    child_report.get("ok") is True
                    and child_manifest.get("artifact_id") == checkpoint_id
                    and child_manifest.get("iteration") == index
                    and child_manifest.get("config_sha256") == config_sha256
                    and child_manifest.get("fingerprints") == fingerprints
                    and child_manifest.get("parent_victim_checkpoint_id")
                    == expected_victim_id
                )
            except Exception as exc:
                child_reports[name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                child_ok = False
            checks[f"checkpoint.{name}"] = child_ok
            child_contracts_ok = child_contracts_ok and child_ok
    checks["checkpoints.valid"] = child_contracts_ok
    checks["checkpoints.evaluation_binding"] = (
        isinstance(checkpoint_ids, list)
        and bool(checkpoint_ids)
        and isinstance(evaluation_metadata, dict)
        and evaluation_metadata.get("behavior_checkpoint_id") == checkpoint_ids[-1]
    )
    checks["rollout.training.behavior_state"] = _is_hex(
        training_metadata.get("behavior_model_state_sha256")
        if isinstance(training_metadata, dict)
        else None,
        64,
    )
    training_behavior_id = (
        training_metadata.get("behavior_checkpoint_id")
        if isinstance(training_metadata, dict)
        else None
    )
    checks["rollout.training.behavior_checkpoint_binding"] = (
        training_behavior_id is None
        or (
            isinstance(training_behavior_id, str)
            and isinstance(checkpoint_ids, list)
            and training_behavior_id in checkpoint_ids
            and child_model_states.get(training_behavior_id)
            == training_metadata.get("behavior_model_state_sha256")
        )
    )
    checks["metrics.iterations"] = (
        isinstance(manifest.get("metrics"), list)
        and isinstance(iterations, int)
        and not isinstance(iterations, bool)
        and len(manifest["metrics"]) == iterations
    )

    manifest_for_id = dict(manifest)
    manifest_for_id["artifact_id"] = None
    checks["artifact_id"] = manifest.get("artifact_id") == (
        "adversary-train-" + canonical_json_sha256(manifest_for_id)[:16]
    )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "gpudrive_adversary_training_run_validation",
        "schema_version": 1,
        "ok": not failed,
        "artifact": str(root),
        "artifact_id": manifest.get("artifact_id"),
        "checks": checks,
        "failed_checks": failed,
        "child_checkpoints": child_reports,
    }
