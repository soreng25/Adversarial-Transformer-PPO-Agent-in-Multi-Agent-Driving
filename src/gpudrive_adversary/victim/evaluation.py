"""Closed-loop deterministic evaluation of the pinned slot-0 PPO victim."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..pins import (
    canonical_json_sha256,
    load_pins,
    repository_root,
    required_checks_pass,
    sha256_file,
    tree_sha256,
    verify_source_tree,
)
from ..provenance import port_identity
from ..smoke import compare_sequences
from .checkpoint import (
    VictimCheckpointError,
    checkpoint_identity,
    load_victim_pin,
    verify_checkpoint,
)
from .policy import (
    assert_policy_frozen,
    deterministic_argmax,
    load_frozen_policy,
    pinned_action_table,
    select_deterministic,
    torch_module_state_sha256,
    validate_slot0_binding,
)


class VictimEvaluationError(RuntimeError):
    """Raised when deterministic victim evaluation violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VictimEvaluationError(message)


def _to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy().copy()


def _prepend_source_paths(source: Path) -> None:
    for path in (source / "build", source):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _temporary_directory(output: Path) -> Path:
    resolved = output.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.parent / f".{resolved.name}.tmp-{uuid.uuid4().hex}"


def _cleanup_temporary(path: Path) -> None:
    if path.exists() and path.name.startswith(".") and ".tmp-" in path.name:
        shutil.rmtree(path)


def _publish_directory(temporary: Path, output: Path) -> None:
    resolved = output.resolve()
    if resolved.exists():
        raise VictimEvaluationError(f"output already exists: {resolved}")
    temporary.replace(resolved)


def _nvidia_identity() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "returncode": result.returncode,
            "rows": result.stdout.strip().splitlines(),
            "stderr": result.stderr.strip() or None,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "rows": [], "stderr": str(exc)}


def _capture(env: Any, reward_weights: dict[str, float]) -> dict[str, np.ndarray]:
    observations = env.get_obs()
    return {
        "victim_observations": _to_numpy(observations[0, 0]),
        "absolute_state": _to_numpy(
            env.sim.absolute_self_observation_tensor().to_torch().clone()
        ),
        "raw_info": _to_numpy(env.sim.info_tensor().to_torch().clone()),
        "dones": _to_numpy(env.get_dones()),
        "rewards": _to_numpy(
            env.get_rewards(
                collision_weight=reward_weights["collision_weight"],
                goal_achieved_weight=reward_weights["goal_achieved_weight"],
                off_road_weight=reward_weights["off_road_weight"],
            )
        ),
    }


def _rollout(
    env: Any,
    policy: Any,
    torch: Any,
    *,
    model_config: dict[str, Any],
    environment_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    env.reset()
    reward_weights = {
        key: float(model_config[key])
        for key in (
            "collision_weight",
            "goal_achieved_weight",
            "off_road_weight",
        )
    }
    captures = [_capture(env, reward_weights)]
    logits: list[np.ndarray] = []
    values: list[np.ndarray] = []
    log_probabilities: list[np.ndarray] = []
    action_indices: list[np.ndarray] = []
    action_templates: list[np.ndarray] = []
    nominal_commands: list[np.ndarray] = []
    native_commands: list[np.ndarray] = []
    termination = {
        "reason": "max_steps",
        "action_timestep": None,
        "transition_count": 0,
    }

    max_steps = min(int(evaluation_config["max_steps"]), int(env.episode_len))
    for timestep in range(max_steps):
        observation = env.get_obs(env.cont_agent_mask)
        _require(
            tuple(observation.shape) == (1, int(model_config["obs_dim"])),
            f"masked victim observation shape is {tuple(observation.shape)}",
        )
        result = select_deterministic(policy, observation)
        action = result["action"]
        _require(tuple(action.shape) == (1,), f"victim action shape is {action.shape}")
        template = torch.full(
            (1, env.max_agent_count),
            int(environment_config["neutral_action_index"]),
            dtype=torch.long,
            device=device,
        )
        template[0, 0] = action[0]
        nominal = env.action_keys_tensor[action]
        env._apply_actions(template)
        native = env.sim.action_tensor().to_torch()[0, 0, :3].clone()
        _require(
            bool(torch.equal(native, nominal[0])),
            f"native victim command differs at timestep {timestep}",
        )

        logits.append(_to_numpy(result["logits"][0]))
        values.append(_to_numpy(result["value"][0]))
        log_probabilities.append(_to_numpy(result["log_probability"][0]))
        action_indices.append(_to_numpy(action[0]))
        action_templates.append(_to_numpy(template))
        nominal_commands.append(_to_numpy(nominal[0]))
        native_commands.append(_to_numpy(native))

        env.step_dynamics(None)
        successor = _capture(env, reward_weights)
        captures.append(successor)
        termination["transition_count"] = timestep + 1
        if bool(successor["dones"][0, 0]):
            termination["action_timestep"] = timestep
            if bool(successor["raw_info"][0, 0, 3]):
                termination["reason"] = "goal"
            elif timestep + 1 >= int(env.episode_len):
                termination["reason"] = "horizon"
            else:
                termination["reason"] = "victim_done_other"
            break
    else:
        if max_steps >= int(env.episode_len):
            termination["reason"] = "horizon_without_done"

    trace = {
        name: np.stack([capture[name] for capture in captures], axis=0)
        for name in captures[0]
    }
    trace.update(
        {
            "logits": np.stack(logits, axis=0),
            "values": np.stack(values, axis=0),
            "log_probabilities": np.stack(log_probabilities, axis=0),
            "action_indices": np.stack(action_indices, axis=0),
            "dense_action_templates": np.stack(action_templates, axis=0),
            "nominal_commands": np.stack(nominal_commands, axis=0),
            "native_commands": np.stack(native_commands, axis=0),
        }
    )
    return trace, termination


def compare_victim_sequences(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
    equal_nan: bool,
) -> dict[str, Any]:
    comparison = compare_sequences(
        left, right, rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    exact_fields = {
        "action_indices",
        "action_table",
        "controlled_mask",
        "dense_action_templates",
        "dones",
        "metadata",
        "native_commands",
        "nominal_commands",
        "raw_info",
    }
    for name in exact_fields:
        if name in comparison["fields"]:
            comparison["fields"][name]["ok"] = comparison["fields"][name][
                "exact"
            ]
    comparison["ok"] = all(
        field["ok"] for field in comparison["fields"].values()
    )
    return comparison


def _read_manifest(artifact: Path) -> dict[str, Any]:
    try:
        return json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VictimEvaluationError(f"cannot read victim manifest: {exc}") from exc


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


def validate_victim_artifact(
    artifact: Path,
    *,
    victim_pin_path: Path | None = None,
    gpudrive_pin_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _read_manifest(artifact)
    victim_pin = load_victim_pin(victim_pin_path)
    gpudrive_pins = load_pins(gpudrive_pin_path)
    trace_path = artifact / "trace.npz"
    fingerprints = manifest.get("fingerprints", {})
    runtime = manifest.get("runtime", {})
    policy = manifest.get("policy", {})
    checks: dict[str, bool] = {}
    checks["schema"] = (
        manifest.get("schema") == "gpudrive_victim_evaluation"
        and manifest.get("schema_version") == 1
    )
    checks["trace_hash"] = trace_path.is_file() and sha256_file(
        trace_path
    ) == fingerprints.get("trace_sha256")
    trace: dict[str, np.ndarray] = {}
    try:
        trace = _load_trace(trace_path)
        checks["trace_safe"] = all(
            value.dtype.kind in "biuf" for value in trace.values()
        )
    except (OSError, ValueError):
        checks["trace_safe"] = False
    required = {
        "absolute_state",
        "action_indices",
        "action_table",
        "controlled_mask",
        "dense_action_templates",
        "dones",
        "log_probabilities",
        "logits",
        "metadata",
        "native_commands",
        "nominal_commands",
        "raw_info",
        "rewards",
        "values",
        "victim_observations",
    }
    checks["trace_fields"] = required.issubset(trace)
    semantic_checks = (
        "trace_clock",
        "trace_dtypes",
        "trace_finite",
        "action_table",
        "argmax_actions",
        "decoded_actions",
        "native_actions",
        "action_templates",
        "slot0_binding",
    )
    if checks["trace_fields"] and checks["trace_safe"]:
        try:
            transitions = trace["action_indices"].shape[0]
            checks["trace_clock"] = (
                trace["victim_observations"].shape == (transitions + 1, 2984)
                and trace["absolute_state"].shape == (transitions + 1, 1, 64, 14)
                and trace["raw_info"].shape == (transitions + 1, 1, 64, 5)
                and trace["dones"].shape == (transitions + 1, 1, 64)
                and trace["rewards"].shape == (transitions + 1, 1, 64)
                and trace["logits"].shape == (transitions, 91)
                and trace["values"].shape == (transitions,)
                and trace["log_probabilities"].shape == (transitions,)
                and trace["action_indices"].shape == (transitions,)
                and trace["dense_action_templates"].shape
                == (transitions, 1, 64)
                and trace["nominal_commands"].shape == (transitions, 3)
                and trace["native_commands"].shape == (transitions, 3)
                and trace["action_table"].shape == (91, 3)
                and trace["controlled_mask"].shape == (1, 64)
                and trace["metadata"].shape == (1, 64, 4)
                and manifest.get("termination", {}).get("transition_count")
                == transitions
            )
            checks["trace_dtypes"] = (
                trace["action_indices"].dtype.kind in "iu"
                and trace["dense_action_templates"].dtype.kind in "iu"
                and trace["controlled_mask"].dtype.kind == "b"
                and all(
                    trace[name].dtype.kind in "f"
                    for name in (
                        "victim_observations",
                        "logits",
                        "values",
                        "log_probabilities",
                        "action_table",
                        "nominal_commands",
                        "native_commands",
                    )
                )
            )
            finite_fields = (
                "victim_observations",
                "raw_info",
                "rewards",
                "logits",
                "values",
                "log_probabilities",
                "action_table",
                "nominal_commands",
                "native_commands",
            )
            checks["trace_finite"] = all(
                np.isfinite(trace[name]).all() for name in finite_fields
            ) and np.isfinite(trace["absolute_state"][:, :, 0, :]).all()
            expected_table = pinned_action_table(victim_pin["environment"])
            checks["action_table"] = np.array_equal(
                np.asarray(trace["action_table"], dtype="<f4"), expected_table
            )
            expected_actions = deterministic_argmax(trace["logits"])
            checks["argmax_actions"] = np.array_equal(
                trace["action_indices"], expected_actions
            )
            checks["decoded_actions"] = np.array_equal(
                trace["nominal_commands"],
                expected_table[trace["action_indices"].astype(np.int64)],
            )
            checks["native_actions"] = np.array_equal(
                trace["nominal_commands"], trace["native_commands"]
            )
            templates = trace["dense_action_templates"]
            neutral = int(victim_pin["environment"]["neutral_action_index"])
            checks["action_templates"] = (
                np.array_equal(templates[:, 0, 0], trace["action_indices"])
                and np.all(templates[:, 0, 1:] == neutral)
            )
            validate_slot0_binding(
                trace["controlled_mask"],
                trace["metadata"][:, :, 0],
                trace["absolute_state"][0, :, :, 13],
                expected_id=int(gpudrive_pins["smoke_scene"]["sdc_object_id"]),
            )
            checks["slot0_binding"] = True
        except (IndexError, TypeError, ValueError, VictimCheckpointError):
            for name in semantic_checks:
                checks[name] = False
    else:
        for name in semantic_checks:
            checks[name] = False

    checks["checkpoint"] = (
        manifest.get("checkpoint_id") == checkpoint_identity(victim_pin)
        and fingerprints.get("checkpoint_model_sha256")
        == victim_pin["source"]["files"]["model.safetensors"]["sha256"]
        and fingerprints.get("checkpoint_config_sha256")
        == victim_pin["source"]["files"]["config.json"]["sha256"]
    )
    checks["scene"] = (
        manifest.get("scene", {}).get("scenario_id")
        == gpudrive_pins["smoke_scene"]["scenario_id"]
        and fingerprints.get("scene_sha256")
        == gpudrive_pins["smoke_scene"]["sha256"]
    )
    checks["gpudrive"] = (
        fingerprints.get("gpudrive_commit")
        == gpudrive_pins["gpudrive"]["commit"]
        and fingerprints.get("gpudrive_submodules")
        == gpudrive_pins["gpudrive"]["submodules"]
    )
    checks["configuration"] = fingerprints.get(
        "victim_pin_sha256"
    ) == canonical_json_sha256(victim_pin)
    checks["policy_frozen"] = (
        policy.get("training") is False
        and policy.get("any_parameter_requires_grad") is False
        and policy.get("state_sha256_before") == policy.get("state_sha256_after")
        and policy.get("state_sha256_before")
        == victim_pin["safetensors_contract"]["state_dict_sha256"]
    )
    embedded_source = manifest.get("source_verification")
    embedded_checkpoint = manifest.get("checkpoint_verification")
    checks["embedded_source_verification"] = (
        isinstance(embedded_source, list)
        and required_checks_pass(embedded_source)
        and fingerprints.get("source_verification_sha256")
        == canonical_json_sha256(embedded_source)
    )
    checks["embedded_checkpoint_verification"] = (
        isinstance(embedded_checkpoint, dict)
        and embedded_checkpoint.get("ok") is True
        and embedded_checkpoint.get("checkpoint_id")
        == checkpoint_identity(victim_pin)
        and isinstance(embedded_checkpoint.get("checks"), list)
        and required_checks_pass(embedded_checkpoint["checks"])
        and fingerprints.get("checkpoint_verification_sha256")
        == canonical_json_sha256(embedded_checkpoint)
    )
    checks["same_process"] = manifest.get("same_process_replay", {}).get("ok") is True
    checks["scope"] = (
        manifest.get("failure_definition") is None
        and manifest.get("failure_timestep") is None
        and manifest.get("eligibility", {}).get("status") == "not_assessed"
    )
    port = fingerprints.get("port", {})
    checks["port"] = (
        isinstance(port.get("commit"), str)
        and len(port["commit"]) == 40
        and isinstance(port.get("dirty"), bool)
        and isinstance(port.get("diff_sha256"), str)
        and len(port["diff_sha256"]) == 64
        and port.get("source_tree_matches_declared") is True
    )
    checks["reference_runtime"] = (
        runtime.get("reference_image_digest")
        == gpudrive_pins["reference_runtime"]["base_image_amd64_digest"]
        and runtime.get("torch_cuda") == gpudrive_pins["reference_runtime"]["cuda"]
        and runtime.get("cuda_available") is True
        and bool(runtime.get("gpu_names"))
        and runtime.get("nvidia_smi", {}).get("returncode") == 0
    )
    cache_hash = fingerprints.get("madrona_kernel_cache_tree_sha256")
    cache_reason = fingerprints.get("madrona_kernel_cache_reason")
    cache_hash_valid = isinstance(cache_hash, str) and len(cache_hash) == 64
    execution_device = runtime.get("execution_device")
    cache_fingerprint_valid = (
        execution_device == "cuda" and cache_hash_valid and cache_reason is None
    ) or (
        execution_device == "cpu"
        and (
            (cache_hash_valid and cache_reason is None)
            or (
                cache_hash is None
                and cache_reason == "not_created_by_cpu_execution_mode"
            )
        )
    )
    checks["native_fingerprints"] = (
        isinstance(fingerprints.get("native_extension_sha256"), str)
        and len(fingerprints["native_extension_sha256"]) == 64
        and cache_fingerprint_valid
    )
    manifest_for_id = dict(manifest)
    manifest_for_id["artifact_id"] = None
    expected_id = "victim-eval-" + canonical_json_sha256(manifest_for_id)[:16]
    checks["artifact_id"] = manifest.get("artifact_id") == expected_id
    return {
        "schema": "gpudrive_victim_evaluation_validation",
        "schema_version": 1,
        "ok": all(checks.values()),
        "artifact": str(artifact.resolve()),
        "checks": checks,
    }


def run_victim_evaluation(
    *,
    source: Path,
    checkpoint_directory: Path,
    output: Path,
    device: str,
    victim_pin_path: Path | None = None,
    gpudrive_pin_path: Path | None = None,
) -> dict[str, Any]:
    if device not in {"cpu", "cuda"}:
        raise VictimEvaluationError("device must be exactly 'cpu' or 'cuda'")
    if output.exists():
        raise VictimEvaluationError(f"output already exists: {output}")
    victim_pin = load_victim_pin(victim_pin_path)
    gpudrive_pins = load_pins(gpudrive_pin_path)
    source = source.resolve()
    source_checks = verify_source_tree(source, gpudrive_pins)
    if not required_checks_pass(source_checks):
        raise VictimEvaluationError("pinned GPUDrive source verification failed")
    checkpoint_report = verify_checkpoint(
        checkpoint_directory, victim_pin, gpudrive_source=source
    )
    if not checkpoint_report["ok"]:
        raise VictimEvaluationError("pinned victim checkpoint verification failed")

    _prepend_source_paths(source)
    try:
        import gpudrive
        import madrona_gpudrive
        import torch
        from gpudrive.datatypes.metadata import Metadata
        from gpudrive.datatypes.observation import GlobalEgoState
        from gpudrive.env.config import EnvConfig
        from gpudrive.env.dataset import SceneDataLoader
        from gpudrive.env.env_torch import GPUDriveTorchEnv
    except Exception as exc:
        raise VictimEvaluationError(
            f"native GPUDrive imports failed: {type(exc).__name__}: {exc}"
        ) from exc
    if device == "cuda":
        _require(torch.cuda.is_available(), "CUDA evaluation requested without CUDA")
    gpudrive_file = Path(gpudrive.__file__).resolve()
    _require(gpudrive_file.is_relative_to(source), "imported GPUDrive from wrong source")
    native_path = Path(madrona_gpudrive.__file__).resolve()
    _require(
        native_path.is_relative_to(source),
        "imported madrona_gpudrive from outside the pinned source build",
    )

    model = victim_pin["model_config"]
    environment = victim_pin["environment"]
    scene_pin = gpudrive_pins["smoke_scene"]
    scene_path = source / scene_pin["relative_path"]
    loader = SceneDataLoader(
        root=str(scene_path.parent),
        batch_size=1,
        dataset_size=1,
        sample_with_replacement=False,
        file_prefix=scene_path.name,
        seed=int(victim_pin["evaluation"]["loader_seed"]),
        shuffle=False,
    )
    env_config = EnvConfig(
        ego_state=bool(model["ego_state"]),
        road_map_obs=bool(model["road_map_obs"]),
        partner_obs=bool(model["partner_obs"]),
        norm_obs=bool(model["norm_obs"]),
        max_controlled_agents=int(model["max_controlled_agents"]),
        num_worlds=1,
        disable_classic_obs=bool(model["lidar_obs"]),
        lidar_obs=bool(model["lidar_obs"]),
        collision_weight=float(model["collision_weight"]),
        goal_achieved_weight=float(model["goal_achieved_weight"]),
        off_road_weight=float(model["off_road_weight"]),
        obs_radius=float(model["obs_radius"]),
        polyline_reduction_threshold=float(model["polyline_reduction_threshold"]),
        dynamics_model=str(model["dynamics_model"]),
        steer_actions=torch.tensor(environment["steering_values"]),
        accel_actions=torch.tensor(environment["acceleration_values"]),
        head_tilt_actions=torch.tensor(environment["head_angle_values"]),
        collision_behavior=str(model["collision_behavior"]),
        remove_non_vehicles=bool(model["remove_non_vehicles"]),
        init_steps=0,
        reward_type=str(model["reward_type"]),
        dist_to_goal_threshold=float(model["dist_to_goal_threshold"]),
        init_mode=str(model["init_mode"]),
        use_vbd=False,
        vbd_in_obs=bool(model["vbd_in_obs"]),
    )
    env = None
    try:
        env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=loader,
            max_cont_agents=int(environment["max_cont_agents"]),
            device=device,
            action_type=str(environment["action_type"]),
        )
        env.reset()
        metadata = Metadata.from_tensor(env.sim.metadata_tensor(), backend="torch")
        state = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(), backend="torch", device=device
        )
        binding = validate_slot0_binding(
            _to_numpy(env.cont_agent_mask),
            _to_numpy(metadata.is_sdc),
            _to_numpy(state.id),
            expected_id=int(scene_pin["sdc_object_id"]),
        )
        _require(env.get_scenario_ids()[0] == scene_pin["scenario_id"], "scenario mismatch")
        expected_table = pinned_action_table(environment)
        observed_table = _to_numpy(env.action_keys_tensor).astype("<f4", copy=False)
        _require(np.array_equal(observed_table, expected_table), "action table mismatch")

        policy = load_frozen_policy(
            checkpoint_directory, model, device=device
        )
        state_before = torch_module_state_sha256(policy)
        first, first_termination = _rollout(
            env,
            policy,
            torch,
            model_config=model,
            environment_config=environment,
            evaluation_config=victim_pin["evaluation"],
            device=device,
        )
        between = torch_module_state_sha256(policy)
        second, second_termination = _rollout(
            env,
            policy,
            torch,
            model_config=model,
            environment_config=environment,
            evaluation_config=victim_pin["evaluation"],
            device=device,
        )
        state_after = torch_module_state_sha256(policy)
        assert_policy_frozen(policy)
        _require(state_before == between == state_after, "victim policy state changed")
        first["controlled_mask"] = _to_numpy(env.cont_agent_mask)
        first["metadata"] = _to_numpy(env.sim.metadata_tensor().to_torch().clone())
        first["action_table"] = observed_table
        second["controlled_mask"] = first["controlled_mask"].copy()
        second["metadata"] = first["metadata"].copy()
        second["action_table"] = observed_table.copy()
        comparison_config = victim_pin["evaluation"]["comparison"]
        same_process = compare_victim_sequences(
            first,
            second,
            rtol=float(comparison_config["rtol"]),
            atol=float(comparison_config["atol"]),
            equal_nan=bool(comparison_config["equal_nan"]),
        )
        same_process["termination_equal"] = first_termination == second_termination
        same_process["ok"] = same_process["ok"] and same_process["termination_equal"]
        _require(same_process["ok"], "same-process victim evaluation mismatch")
    finally:
        if env is not None:
            env.close()

    temporary = _temporary_directory(output)
    temporary.mkdir()
    try:
        trace_path = temporary / "trace.npz"
        np.savez_compressed(trace_path, **first)
        cache_path = os.environ.get("MADRONA_MWGPU_KERNEL_CACHE")
        cache_hash = tree_sha256(cache_path) if cache_path else None
        cache_reason = (
            None
            if cache_hash is not None
            else (
                "not_created_by_cpu_execution_mode"
                if device == "cpu"
                else "missing_after_cuda_execution"
            )
        )
        manifest = {
            "schema": "gpudrive_victim_evaluation",
            "schema_version": 1,
            "artifact_id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": victim_pin["purpose"],
            "checkpoint_id": checkpoint_identity(victim_pin),
            "scene": {
                "relative_path": scene_pin["relative_path"],
                "scenario_id": scene_pin["scenario_id"],
                "victim_slot": binding["slot"],
                "victim_stable_id": binding["stable_id"],
                "loader_seed": victim_pin["evaluation"]["loader_seed"],
                "background_behavior": environment["background_behavior"],
            },
            "policy": {
                "deterministic_rule": victim_pin["evaluation"]["deterministic_rule"],
                "training": bool(policy.training),
                "any_parameter_requires_grad": any(
                    parameter.requires_grad for parameter in policy.parameters()
                ),
                "state_sha256_before": state_before,
                "state_sha256_after": state_after,
            },
            "termination": first_termination,
            "same_process_replay": same_process,
            "eligibility": victim_pin["eligibility"],
            "failure_definition": None,
            "failure_definition_explanation": victim_pin[
                "failure_definition_reason"
            ],
            "failure_timestep": None,
            "fingerprints": {
                "checkpoint_model_sha256": victim_pin["source"]["files"][
                    "model.safetensors"
                ]["sha256"],
                "checkpoint_config_sha256": victim_pin["source"]["files"][
                    "config.json"
                ]["sha256"],
                "victim_pin_sha256": canonical_json_sha256(victim_pin),
                "action_table_float32_sha256": environment[
                    "action_table_float32_sha256"
                ],
                "gpudrive_commit": gpudrive_pins["gpudrive"]["commit"],
                "gpudrive_submodules": gpudrive_pins["gpudrive"]["submodules"],
                "scene_sha256": scene_pin["sha256"],
                "trace_sha256": sha256_file(trace_path),
                "native_extension_sha256": sha256_file(native_path),
                "madrona_kernel_cache_tree_sha256": cache_hash,
                "madrona_kernel_cache_reason": cache_reason,
                "source_verification_sha256": canonical_json_sha256(source_checks),
                "checkpoint_verification_sha256": canonical_json_sha256(
                    checkpoint_report
                ),
                "port": port_identity(repository_root()),
            },
            "runtime": {
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "execution_device": device,
                "cuda_available": torch.cuda.is_available(),
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "nvidia_smi": _nvidia_identity(),
                "reference_image_digest": os.environ.get(
                    "GPUDRIVE_REFERENCE_IMAGE_DIGEST"
                ),
                "gpudrive_python_file": str(gpudrive_file),
                "native_extension_file": str(native_path),
            },
            "source_verification": source_checks,
            "checkpoint_verification": checkpoint_report,
        }
        manifest["artifact_id"] = "victim-eval-" + canonical_json_sha256(
            manifest
        )[:16]
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        validation = validate_victim_artifact(
            temporary,
            victim_pin_path=victim_pin_path,
            gpudrive_pin_path=gpudrive_pin_path,
        )
        if not validation["ok"]:
            failed = [name for name, ok in validation["checks"].items() if not ok]
            raise VictimEvaluationError(
                f"generated victim artifact failed validation: {failed}"
            )
        _publish_directory(temporary, output)
        return manifest
    except Exception:
        _cleanup_temporary(temporary)
        raise


def compare_victim_artifacts(
    left_artifact: Path,
    right_artifact: Path,
    *,
    output: Path | None = None,
    victim_pin_path: Path | None = None,
    gpudrive_pin_path: Path | None = None,
) -> dict[str, Any]:
    left_manifest = _read_manifest(left_artifact)
    right_manifest = _read_manifest(right_artifact)
    left_validation = validate_victim_artifact(
        left_artifact,
        victim_pin_path=victim_pin_path,
        gpudrive_pin_path=gpudrive_pin_path,
    )
    right_validation = validate_victim_artifact(
        right_artifact,
        victim_pin_path=victim_pin_path,
        gpudrive_pin_path=gpudrive_pin_path,
    )
    identity_names = (
        "checkpoint_id",
        "scene",
        "policy",
        "eligibility",
    )
    identity = {
        name: left_manifest.get(name) == right_manifest.get(name)
        for name in identity_names
    }
    fingerprint_names = (
        "checkpoint_model_sha256",
        "checkpoint_config_sha256",
        "victim_pin_sha256",
        "action_table_float32_sha256",
        "gpudrive_commit",
        "gpudrive_submodules",
        "scene_sha256",
        "native_extension_sha256",
        "madrona_kernel_cache_tree_sha256",
        "madrona_kernel_cache_reason",
        "port",
    )
    identity.update(
        {
            f"fingerprint.{name}": left_manifest.get("fingerprints", {}).get(name)
            == right_manifest.get("fingerprints", {}).get(name)
            for name in fingerprint_names
        }
    )
    identity["runtime"] = left_manifest.get("runtime") == right_manifest.get(
        "runtime"
    )
    comparison_config = load_victim_pin(victim_pin_path)["evaluation"][
        "comparison"
    ]
    trace_comparison = compare_victim_sequences(
        _load_trace(left_artifact / "trace.npz"),
        _load_trace(right_artifact / "trace.npz"),
        rtol=float(comparison_config["rtol"]),
        atol=float(comparison_config["atol"]),
        equal_nan=bool(comparison_config["equal_nan"]),
    )
    termination_equal = left_manifest.get("termination") == right_manifest.get(
        "termination"
    )
    report = {
        "schema": "gpudrive_victim_fresh_process_comparison",
        "schema_version": 1,
        "ok": left_validation["ok"]
        and right_validation["ok"]
        and all(identity.values())
        and trace_comparison["ok"]
        and termination_equal,
        "left_artifact": str(left_artifact.resolve()),
        "right_artifact": str(right_artifact.resolve()),
        "validation": {"left": left_validation, "right": right_validation},
        "identity": identity,
        "termination_equal": termination_equal,
        "trace": trace_comparison,
    }
    if output is not None:
        if output.exists():
            raise VictimEvaluationError(f"comparison output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(output)
        finally:
            if temporary.exists():
                temporary.unlink()
    return report


def run_fresh_victim_evaluation(
    *,
    source: Path,
    checkpoint_directory: Path,
    output: Path,
    device: str,
    victim_pin_path: Path | None = None,
    gpudrive_pin_path: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise VictimEvaluationError(f"output already exists: {output}")
    temporary = _temporary_directory(output)
    temporary.mkdir()
    try:
        runs = [temporary / "run-1", temporary / "run-2"]
        for run in runs:
            command = [
                sys.executable,
                "-m",
                "gpudrive_adversary",
                "victim-eval",
                "--source",
                str(source),
                "--checkpoint",
                str(checkpoint_directory),
                "--device",
                device,
                "--output",
                str(run),
            ]
            if victim_pin_path is not None:
                command.extend(["--victim-pin", str(victim_pin_path)])
            if gpudrive_pin_path is not None:
                command.extend(["--pins", str(gpudrive_pin_path)])
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise VictimEvaluationError(
                    f"fresh victim process failed with exit {result.returncode}"
                )
        report = compare_victim_artifacts(
            runs[0],
            runs[1],
            output=temporary / "comparison.json",
            victim_pin_path=victim_pin_path,
            gpudrive_pin_path=gpudrive_pin_path,
        )
        if not report["ok"]:
            raise VictimEvaluationError("fresh victim evaluations differ")
        report["left_artifact"] = str((output / "run-1").resolve())
        report["right_artifact"] = str((output / "run-2").resolve())
        report["validation"]["left"]["artifact"] = report["left_artifact"]
        report["validation"]["right"]["artifact"] = report["right_artifact"]
        (temporary / "comparison.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        _publish_directory(temporary, output)
        return report
    except Exception:
        _cleanup_temporary(temporary)
        raise
