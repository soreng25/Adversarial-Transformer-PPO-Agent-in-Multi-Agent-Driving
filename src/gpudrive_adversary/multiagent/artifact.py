"""Fail-closed validation of a ten-agent highway training run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..adversary.checkpoint import validate_adversary_checkpoint
from ..adversary.config import validate_adversary_config
from ..pins import canonical_json_sha256, sha256_file
from .clearance import minimum_pairwise_clearance
from .environment import (
    ANY_CONTROLLED_FAILURE_SCOPE,
    NONFOCAL_SYSTEM_FAILURE_SCOPE,
    classify_multiagent_failure,
)
from .scene import (
    default_highway_config_path,
    default_nonfocal_highway_config_path,
    load_highway_experiment_config,
)


REQUIRED_ARRAYS = {
    "observations", "raw_info", "boxes", "done", "goal_ever", "minimum_clearance", "closest_pair",
    "tokens", "history_masks", "victim_action_indices", "victim_logits", "nominal_commands", "applied_commands",
    "disturbance_requested", "disturbance_effective", "disturbance_saturated", "command_saturated",
    "prior_nll_exact", "prior_nll_penalty", "policy_log_probability", "adversary_values", "rewards",
    "pre_actor_latent", "failure_by_transition", "failing_agents", "failure_kind_bits",
}


class MultiAgentArtifactError(ValueError):
    """Raised when a highway artifact cannot be validated or summarized."""


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(character in "0123456789abcdef" for character in value)


def _trace_checks(
    path: Path,
    metadata: dict[str, Any],
    *,
    nominal: bool,
    failure_scope: str,
) -> dict[str, bool]:
    checks = {
        "file": path.is_file(), "hash": False, "arrays": False, "safe": False,
        "shapes": False, "clearance": False, "failure_clock": False,
        "failure_evidence": False, "goal_clock": False, "focal_intervention": False,
        "nominal_goals": not nominal,
    }
    if not path.is_file(): return checks
    checks["hash"] = _hex(metadata.get("sha256"), 64) and sha256_file(path) == metadata["sha256"]
    try:
        with np.load(path, allow_pickle=False) as payload:
            checks["arrays"] = set(payload.files) == REQUIRED_ARRAYS
            if not checks["arrays"]: return checks
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
    except (OSError, ValueError): return checks
    checks["safe"] = all(value.dtype.kind in "biuf" and (value.dtype.kind != "f" or np.isfinite(value).all()) for value in arrays.values())
    transitions = arrays["victim_action_indices"].shape[0]; states = transitions + 1
    expected = {
        "observations": (states, 10, 2984), "raw_info": (states, 10, 5), "boxes": (states, 10, 5),
        "done": (states, 10), "goal_ever": (states, 10), "minimum_clearance": (states,), "closest_pair": (states, 2),
        "tokens": (transitions, 2989), "history_masks": (transitions, 50), "victim_action_indices": (transitions, 10),
        "victim_logits": (transitions, 10, 91), "nominal_commands": (transitions, 10, 3), "applied_commands": (transitions, 10, 3),
        "disturbance_requested": (transitions, 2), "disturbance_effective": (transitions, 2), "disturbance_saturated": (transitions, 2),
        "command_saturated": (transitions, 2), "prior_nll_exact": (transitions,), "prior_nll_penalty": (transitions,),
        "policy_log_probability": (transitions,), "adversary_values": (transitions,), "rewards": (transitions,),
        "failure_by_transition": (transitions,), "failing_agents": (transitions, 10), "failure_kind_bits": (transitions, 10, 3),
    }
    checks["shapes"] = transitions > 0 and metadata.get("transition_count") == transitions and all(arrays[name].shape == shape for name, shape in expected.items()) and arrays["pre_actor_latent"].ndim == 2 and arrays["pre_actor_latent"].shape[0] == transitions
    clearance_ok = True
    for timestep in range(states):
        active = ~arrays["done"][timestep].astype(bool)
        if failure_scope == NONFOCAL_SYSTEM_FAILURE_SCOPE:
            active[0] = False
        if active.sum() >= 2:
            clearance, pair = minimum_pairwise_clearance(arrays["boxes"][timestep], active)
            clearance_ok &= bool(np.isclose(clearance, arrays["minimum_clearance"][timestep], atol=1e-5, rtol=1e-5) and np.array_equal(pair, arrays["closest_pair"][timestep]))
    checks["clearance"] = clearance_ok
    failures = np.flatnonzero(arrays["failure_by_transition"])
    failure_timestep = metadata.get("failure_timestep")
    checks["failure_clock"] = (failures.size == 0 and failure_timestep is None) or (failures.tolist() == [failure_timestep] and failure_timestep == transitions - 1)
    expected_failure_bits = arrays["raw_info"][1:, :, :3].astype(np.int8)
    classified = [
        classify_multiagent_failure(
            arrays["raw_info"][timestep + 1],
            arrays["boxes"][timestep + 1],
            scope=failure_scope,
        )
        for timestep in range(transitions)
    ]
    expected_failure_clock = np.asarray([item[0] for item in classified], dtype=np.bool_)
    expected_failing_agents = np.stack([item[2] for item in classified])
    checks["failure_evidence"] = bool(
        np.array_equal(arrays["failure_kind_bits"], expected_failure_bits)
        and np.array_equal(arrays["failing_agents"].astype(bool), expected_failing_agents)
        and np.array_equal(arrays["failure_by_transition"].astype(bool), expected_failure_clock)
    )
    expected_goals = np.maximum.accumulate((arrays["raw_info"][:, :, 3] != 0).astype(np.int8), axis=0).astype(bool)
    checks["goal_clock"] = bool(np.array_equal(arrays["goal_ever"].astype(bool), expected_goals))
    checks["focal_intervention"] = bool(
        np.array_equal(arrays["applied_commands"][:, 1:], arrays["nominal_commands"][:, 1:])
        and np.array_equal(arrays["applied_commands"][:, 0, 2], arrays["nominal_commands"][:, 0, 2])
        and np.allclose(
            arrays["applied_commands"][:, 0, :2] - arrays["nominal_commands"][:, 0, :2],
            arrays["disturbance_effective"],
            atol=1e-6,
            rtol=1e-6,
        )
    )
    if nominal:
        checks["nominal_goals"] = failure_timestep is None and metadata.get("termination_reason") == "all_goals_reached" and bool(arrays["goal_ever"][-1].all())
    return checks


def validate_multiagent_training_artifact(path: Path | str) -> dict[str, Any]:
    root = Path(path).resolve(); checks: dict[str, bool] = {}
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema": "gpudrive_highway_10agent_training_validation", "schema_version": 1, "ok": False, "artifact": str(root), "checks": {"manifest": False}, "failed_checks": ["manifest"]}
    checks["schema"] = manifest.get("schema") == "gpudrive_highway_10agent_training_run" and manifest.get("schema_version") == 1 and manifest.get("research_claims_allowed") is False
    try:
        config = validate_adversary_config(manifest.get("config")); checks["config"] = config.get("purpose") in {"highway_10agent_training_pilot", "highway_10agent_nonfocal_system_training_pilot"}
    except Exception: checks["config"] = False; config = {}
    try:
        experiment = manifest.get("experiment")
        expected_path = (
            default_nonfocal_highway_config_path()
            if config.get("purpose") == "highway_10agent_nonfocal_system_training_pilot"
            else default_highway_config_path()
        )
        expected = load_highway_experiment_config(expected_path)
        checks["experiment"] = experiment == expected
    except Exception: checks["experiment"] = False; experiment = {}
    scene = manifest.get("scene_identity", {})
    checks["scene"] = scene.get("dynamic_object_count") == 10 and scene.get("background_dynamic_object_count") == 0 and scene.get("selected_object_ids") == experiment.get("scene", {}).get("selected_object_ids")
    checks["victim_frozen"] = _hex(manifest.get("victim_state_sha256_before"), 64) and manifest.get("victim_state_sha256_before") == manifest.get("victim_state_sha256_after")
    files = manifest.get("files", {})
    for name in ("derived_scene", "nominal", "last_training", "last_evaluation"):
        spec = files.get(name, {}); file_path = root / spec.get("relative_path", "missing")
        checks[f"file.{name}"] = file_path.is_file() and _hex(spec.get("sha256"), 64) and sha256_file(file_path) == spec["sha256"]
    try:
        derived = json.loads((root / files["derived_scene"]["relative_path"]).read_text(encoding="utf-8"))
        checks["derived_scene_contract"] = (
            [obj["id"] for obj in derived["objects"]] == experiment["scene"]["selected_object_ids"]
            and len(derived["objects"]) == 10
            and derived["metadata"]["sdc_track_index"] == 0
            and canonical_json_sha256(derived) == experiment["scene"]["derived_canonical_sha256"]
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        checks["derived_scene_contract"] = False
    configured_scope = config.get("failure", {}).get("scope", ANY_CONTROLLED_FAILURE_SCOPE)
    for name in ("nominal", "last_training", "last_evaluation"):
        trace_checks = _trace_checks(
            root / files.get(name, {}).get("relative_path", "missing"),
            files.get(name, {}),
            nominal=name == "nominal",
            failure_scope=(
                ANY_CONTROLLED_FAILURE_SCOPE if name == "nominal" else configured_scope
            ),
        )
        checks.update({f"trace.{name}.{key}": value for key, value in trace_checks.items()})
    metrics = manifest.get("metrics")
    checks["metrics"] = isinstance(metrics, list) and bool(metrics)
    if checks["metrics"] and configured_scope == NONFOCAL_SYSTEM_FAILURE_SCOPE:
        expected_slot_keys = {str(slot) for slot in range(1, 10)}
        checks["metrics"] = all(
            isinstance(row, dict)
            and set(row.get("qualifying_failures_by_slot", {})) == expected_slot_keys
            and all(
                isinstance(value, int) and value >= 0
                for value in row["qualifying_failures_by_slot"].values()
            )
            and isinstance(row.get("qualifying_failures_by_kind"), dict)
            and isinstance(row.get("qualifying_vehicle_collision_pairs"), dict)
            and isinstance(row.get("episodes_with_focal_safety_event"), int)
            for row in metrics
        )
    checkpoint_ids = manifest.get("checkpoint_ids", [])
    checks["checkpoints"] = isinstance(checkpoint_ids, list) and bool(checkpoint_ids)
    if checks["checkpoints"]:
        for index, artifact_id in enumerate(checkpoint_ids, start=1):
            checkpoint_path = root / "checkpoints" / f"iteration-{index:04d}"
            report = validate_adversary_checkpoint(checkpoint_path)
            try:
                checkpoint_manifest = json.loads((checkpoint_path / "manifest.json").read_text(encoding="utf-8"))
                exact_id = checkpoint_manifest.get("artifact_id") == artifact_id
            except (OSError, json.JSONDecodeError):
                exact_id = False
            checks["checkpoints"] &= report["ok"] and exact_id
    fingerprints = manifest.get("fingerprints", {})
    checks["fingerprints"] = all(_hex(fingerprints.get(name), length) for name, length in {
        "methodology_commit": 40, "gpudrive_commit": 40, "source_scene_sha256": 64, "derived_scene_canonical_sha256": 64,
        "derived_scene_byte_sha256": 64, "adversary_config_sha256": 64, "experiment_config_sha256": 64,
        "victim_checkpoint_model_sha256": 64, "native_extension_sha256": 64,
    }.items()) and (
        fingerprints.get("selected_object_ids") == scene.get("selected_object_ids")
        and fingerprints.get("dataset_repository") == experiment.get("dataset", {}).get("repository")
        and fingerprints.get("dataset_revision") == experiment.get("dataset", {}).get("revision")
        and fingerprints.get("source_scene_relative_path") == experiment.get("scene", {}).get("source_relative_path")
    )
    without_id = dict(manifest); without_id.pop("artifact_id", None)
    checks["artifact_id"] = manifest.get("artifact_id") == "highway-10ppo-" + canonical_json_sha256(without_id)[:16]
    failed = [name for name, passed in checks.items() if not passed]
    return {"schema": "gpudrive_highway_10agent_training_validation", "schema_version": 1, "ok": not failed, "artifact": str(root), "checks": checks, "failed_checks": failed}


def summarize_nonfocal_system_run(path: Path | str) -> dict[str, Any]:
    """Aggregate qualifying non-focal failures without reinterpreting episodes."""

    root = Path(path).resolve()
    validation = validate_multiagent_training_artifact(root)
    if not validation["ok"]:
        raise MultiAgentArtifactError(
            f"training artifact validation failed: {validation['failed_checks']}"
        )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"]["failure"]["scope"] != NONFOCAL_SYSTEM_FAILURE_SCOPE:
        raise MultiAgentArtifactError("run does not use the slots-1-through-9 failure objective")
    metrics = manifest["metrics"]
    slot_counts = {str(slot): 0 for slot in range(1, 10)}
    kind_counts = {
        name: 0
        for name in ("road_object_contact", "vehicle_collision", "nonvehicle_collision")
    }
    pair_counts: dict[str, int] = {}
    total_episodes = 0
    total_failures = 0
    focal_event_episodes = 0
    for row in metrics:
        total_episodes += int(row["episodes"])
        total_failures += int(row["failures"])
        focal_event_episodes += int(row["episodes_with_focal_safety_event"])
        for slot, count in row["qualifying_failures_by_slot"].items():
            slot_counts[slot] += int(count)
        for kind, count in row["qualifying_failures_by_kind"].items():
            kind_counts[kind] += int(count)
        for pair, count in row["qualifying_vehicle_collision_pairs"].items():
            pair_counts[pair] = pair_counts.get(pair, 0) + int(count)
    ranked_slots = sorted(
        ({"slot": int(slot), "qualifying_failure_episodes": count} for slot, count in slot_counts.items()),
        key=lambda item: (-item["qualifying_failure_episodes"], item["slot"]),
    )
    ranked_pairs = sorted(
        ({"slots": pair, "qualifying_collision_episodes": count} for pair, count in pair_counts.items()),
        key=lambda item: (-item["qualifying_collision_episodes"], item["slots"]),
    )
    return {
        "schema": "gpudrive_highway_nonfocal_system_summary",
        "schema_version": 1,
        "artifact": str(root),
        "artifact_id": manifest["artifact_id"],
        "iterations": len(metrics),
        "total_episodes": total_episodes,
        "total_qualifying_failure_episodes": total_failures,
        "qualifying_failure_rate": (
            float(total_failures / total_episodes) if total_episodes else 0.0
        ),
        "qualifying_failures_by_slot": slot_counts,
        "ranked_failing_slots": ranked_slots,
        "qualifying_failures_by_kind": kind_counts,
        "ranked_vehicle_collision_pairs": ranked_pairs,
        "episodes_with_focal_safety_event_no_automatic_credit": focal_event_episodes,
    }
