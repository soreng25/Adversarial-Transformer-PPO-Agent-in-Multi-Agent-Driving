"""Linux/CUDA Transformer-PPO training for the ten-agent highway pilot."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..adversary.checkpoint import save_adversary_checkpoint
from ..adversary.config import load_adversary_config
from ..adversary.distribution import BoundedTanhNormal
from ..adversary.ppo import generalized_advantage_estimate_numpy
from ..adversary.training import (
    _TorchAdversaryAdapter,
    _ZeroAdversary,
    _combine_batches,
    _intervention,
    _model_config,
    _ppo_config,
    _ppo_update,
    _prepend_source_paths,
    _to_numpy,
)
from ..adversary.model import CausalTransformerActorCritic
from ..pins import canonical_json_sha256, load_pins, repository_root, required_checks_pass, sha256_file, verify_source_tree
from ..provenance import port_identity
from ..victim.checkpoint import checkpoint_identity, default_checkpoint_directory, load_victim_pin, verify_checkpoint
from ..victim.policy import assert_policy_frozen, load_frozen_policy, pinned_action_table, torch_module_state_sha256, validate_multiagent_binding
from .environment import AGENT_COUNT, MultiAgentRollout, MultiAgentState, MultiAgentVictimPolicy, run_multiagent_rollout
from .artifact import validate_multiagent_training_artifact
from .scene import (
    build_derived_scene,
    default_highway_source_path,
    load_highway_experiment_config,
    write_derived_scene,
)


class MultiAgentTrainingError(RuntimeError):
    """Raised when the native ten-agent experiment violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiAgentTrainingError(message)


class GPUDriveTenAgentBackend:
    def __init__(self, env: Any, torch: Any, *, device: str, horizon: int, expected_ids: list[int], scenario_id: str, agent_scale: float):
        self.env = env; self.torch = torch; self.device = device; self.horizon = horizon
        self.expected_ids = expected_ids; self.scenario_id = scenario_id; self.agent_scale = float(agent_scale); self.timestep = 0

    def _binding(self) -> None:
        metadata = _to_numpy(self.env.sim.metadata_tensor().to_torch())[:, :, 0]
        absolute = _to_numpy(self.env.sim.absolute_self_observation_tensor().to_torch())
        validate_multiagent_binding(_to_numpy(self.env.cont_agent_mask), metadata, absolute[:, :, 13], expected_ids=self.expected_ids)
        _require(self.env.get_scenario_ids()[0] == self.scenario_id, "GPUDrive loaded an unexpected scenario")

    def _state(self) -> MultiAgentState:
        observations = _to_numpy(self.env.get_obs(self.env.cont_agent_mask))
        raw_info = _to_numpy(self.env.sim.info_tensor().to_torch())[0, :AGENT_COUNT]
        absolute = _to_numpy(self.env.sim.absolute_self_observation_tensor().to_torch())[0, :AGENT_COUNT]
        boxes = np.stack(
            (absolute[:, 0], absolute[:, 1], absolute[:, 7], absolute[:, 10] * self.agent_scale, absolute[:, 11] * self.agent_scale), axis=1
        )
        done = _to_numpy(self.env.get_dones())[0, :AGENT_COUNT]
        return MultiAgentState(observations, raw_info, boxes, done, self.timestep >= self.horizon)

    def reset(self) -> MultiAgentState:
        self.env.reset(); self.timestep = 0; self._binding(); return self._state()

    def step(self, commands: np.ndarray) -> MultiAgentState:
        values = np.asarray(commands, dtype=np.float32)
        _require(values.shape == (AGENT_COUNT, 3), "ten physical commands are required")
        native = self.torch.zeros((1, self.env.max_agent_count, 3), dtype=self.torch.float32, device=self.device)
        native[0, :AGENT_COUNT] = self.torch.as_tensor(values, dtype=self.torch.float32, device=self.device)
        self.env.step_dynamics(native); self.timestep += 1
        return self._state()


def _episode_arrays(rollout: MultiAgentRollout, prior: BoundedTanhNormal, config: dict[str, Any]) -> dict[str, np.ndarray]:
    mean = np.asarray(config["prior"]["base_mean"], dtype=np.float64)
    penalty = np.asarray(prior.nll_excess_from_zero(rollout.disturbance_requested, mean), dtype=np.float64)
    rewards = -float(config["reward"]["nll_coefficient"]) * penalty
    if rollout.failure_timestep is not None:
        rewards[-1] += float(config["reward"]["failure_bonus"])
    else:
        initial = float(rollout.minimum_clearance[0])
        closest = max(0.0, rollout.episode_minimum_clearance)
        rewards[-1] -= float(np.clip(closest / initial, 0.0, 1.0))
    terminated = np.zeros(rollout.transition_count, dtype=np.bool_); terminated[-1] = True
    advantages, returns = generalized_advantage_estimate_numpy(
        rewards.astype(np.float32), rollout.adversary_values.astype(np.float32), terminated,
        np.asarray(0.0, dtype=np.float32), gamma=float(config["ppo"]["gamma"]), gae_lambda=float(config["ppo"]["gae_lambda"]),
    )
    return {
        "histories": rollout.histories.astype(np.float32), "history_masks": rollout.history_masks.astype(np.bool_),
        "actions": rollout.disturbance_requested.astype(np.float32), "old_log_probability": rollout.policy_log_probability.astype(np.float32),
        "old_value": rollout.adversary_values.astype(np.float32), "advantages": advantages.astype(np.float32),
        "returns": returns.astype(np.float32), "rewards": rewards.astype(np.float32), "nll_penalty": penalty.astype(np.float32),
    }


def _payload(rollout: MultiAgentRollout, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    failure_bits = np.zeros((rollout.transition_count, AGENT_COUNT, 3), dtype=np.int8)
    for timestep in range(rollout.transition_count):
        failure_bits[timestep] = rollout.raw_info[timestep + 1, :, :3].astype(np.int8)
    return {
        "observations": rollout.observations, "raw_info": rollout.raw_info, "boxes": rollout.boxes,
        "done": rollout.done, "goal_ever": rollout.goal_ever, "minimum_clearance": rollout.minimum_clearance,
        "closest_pair": rollout.closest_pair, "tokens": rollout.tokens, "history_masks": rollout.history_masks,
        "victim_action_indices": rollout.victim_action_indices, "victim_logits": rollout.victim_logits,
        "nominal_commands": rollout.nominal_commands, "applied_commands": rollout.applied_commands,
        "disturbance_requested": rollout.disturbance_requested, "disturbance_effective": rollout.disturbance_effective,
        "disturbance_saturated": rollout.disturbance_saturated, "command_saturated": rollout.command_saturated,
        "prior_nll_exact": rollout.prior_nll_exact, "prior_nll_penalty": arrays["nll_penalty"],
        "policy_log_probability": rollout.policy_log_probability, "adversary_values": rollout.adversary_values,
        "rewards": arrays["rewards"], "pre_actor_latent": rollout.pre_actor_latent,
        "failure_by_transition": rollout.failure_by_transition, "failing_agents": rollout.failing_agents,
        "failure_kind_bits": failure_bits,
    }


def _env_config(EnvConfig: Any, torch: Any, victim_pin: dict[str, Any]) -> Any:
    model = victim_pin["model_config"]; environment = victim_pin["environment"]
    return EnvConfig(
        ego_state=bool(model["ego_state"]), road_map_obs=bool(model["road_map_obs"]), partner_obs=bool(model["partner_obs"]),
        norm_obs=bool(model["norm_obs"]), max_controlled_agents=int(model["max_controlled_agents"]), num_worlds=1,
        disable_classic_obs=bool(model["lidar_obs"]), lidar_obs=bool(model["lidar_obs"]), collision_weight=float(model["collision_weight"]),
        goal_achieved_weight=float(model["goal_achieved_weight"]), off_road_weight=float(model["off_road_weight"]), obs_radius=float(model["obs_radius"]),
        polyline_reduction_threshold=float(model["polyline_reduction_threshold"]), dynamics_model=str(model["dynamics_model"]),
        steer_actions=torch.tensor(environment["steering_values"]), accel_actions=torch.tensor(environment["acceleration_values"]),
        head_tilt_actions=torch.tensor(environment["head_angle_values"]), collision_behavior="ignore", remove_non_vehicles=True,
        init_steps=0, reward_type=str(model["reward_type"]), dist_to_goal_threshold=float(model["dist_to_goal_threshold"]),
        init_mode=str(model["init_mode"]), use_vbd=False, vbd_in_obs=False,
    )


def train_highway_multiagent(
    *, source: Path, output: Path, checkpoint_directory: Path | None = None,
    scene_source: Path | None = None,
    adversary_config_path: Path | None = None, experiment_config_path: Path | None = None,
    victim_pin_path: Path | None = None, gpudrive_pin_path: Path | None = None,
) -> dict[str, Any]:
    _require(platform.system() == "Linux", "ten-agent training is Linux/CUDA only")
    _require(not output.exists(), f"training output already exists: {output}")
    adversary_path = adversary_config_path or repository_root() / "configs/adversary/highway_10agent_transformer_ppo.json"
    config = load_adversary_config(adversary_path); experiment = load_highway_experiment_config(experiment_config_path)
    _require(config["intervention"]["bounds"] == experiment["intervention"]["bounds"], "experiment and PPO disturbance bounds differ")
    pins = load_pins(gpudrive_pin_path); victim_pin = load_victim_pin(victim_pin_path); source = source.resolve()
    checkpoint = checkpoint_directory.resolve() if checkpoint_directory else default_checkpoint_directory(victim_pin).resolve()
    source_checks = verify_source_tree(source, pins); _require(required_checks_pass(source_checks), "pinned GPUDrive source verification failed")
    victim_report = verify_checkpoint(checkpoint, victim_pin, gpudrive_source=source); _require(victim_report["ok"], "victim checkpoint verification failed")
    source_scene = (scene_source or default_highway_source_path(experiment)).resolve()
    derived, scene_identity = build_derived_scene(source_scene, experiment)
    _prepend_source_paths(source)
    try:
        import gpudrive
        import madrona_gpudrive
        import torch
        from gpudrive.datatypes.observation import AGENT_SCALE
        from gpudrive.env.config import EnvConfig
        from gpudrive.env.dataset import SceneDataLoader
        from gpudrive.env.env_torch import GPUDriveTorchEnv
    except Exception as exc:
        raise MultiAgentTrainingError(f"native imports failed: {type(exc).__name__}: {exc}") from exc
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"}, "CUBLAS_WORKSPACE_CONFIG must be set before CUDA starts")
    _require(torch.cuda.is_available(), "CUDA is required")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = "cuda"; seed = int(config["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    native_path = Path(madrona_gpudrive.__file__).resolve(); _require(Path(gpudrive.__file__).resolve().is_relative_to(source), "imported GPUDrive from wrong source")
    _require(native_path.is_relative_to(source), "imported native extension from wrong source")

    temporary = output.resolve().parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; temporary.parent.mkdir(parents=True, exist_ok=True); temporary.mkdir()
    env = None
    try:
        derived_path = temporary / "derived-scene" / experiment["scene"]["derived_name"]
        derived_byte_hash = write_derived_scene(derived_path, derived)
        loader = SceneDataLoader(root=str(derived_path.parent), batch_size=1, dataset_size=1, sample_with_replacement=False, file_prefix=derived_path.name, seed=seed, shuffle=False)
        env = GPUDriveTorchEnv(config=_env_config(EnvConfig, torch, victim_pin), data_loader=loader, max_cont_agents=AGENT_COUNT, device=device, action_type="discrete")
        table = _to_numpy(env.action_keys_tensor).astype("<f4", copy=False); _require(np.array_equal(table, pinned_action_table(victim_pin["environment"])), "victim action table changed")
        victim_model = load_frozen_policy(checkpoint, victim_pin["model_config"], device=device); victim_state = torch_module_state_sha256(victim_model)
        victim = MultiAgentVictimPolicy(victim_model, table, torch, device=device)
        backend = GPUDriveTenAgentBackend(env, torch, device=device, horizon=91, expected_ids=experiment["scene"]["selected_object_ids"], scenario_id=experiment["scene"]["scenario_id"], agent_scale=float(AGENT_SCALE))
        intervention = _intervention(config)
        nominal = run_multiagent_rollout(backend=backend, victim=victim, adversary=_ZeroAdversary(), intervention=intervention, max_steps=91, adversary_deterministic=True)
        _require(nominal.failure_timestep is None, "clean ten-agent rollout has a safety failure")
        _require(nominal.all_goals_reached and nominal.termination_reason == "all_goals_reached", "not all ten PPO vehicles complete their goals cleanly")
        print(
            "Clean eligibility passed: all 10 PPO vehicles reached their goals "
            f"without a safety failure; minimum clearance={nominal.episode_minimum_clearance:.3f} m",
            flush=True,
        )

        model = CausalTransformerActorCritic(_model_config(config)).to(device); model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=_ppo_config(config).learning_rate)
        bounds = np.asarray(config["intervention"]["bounds"], dtype=np.float64)
        prior = BoundedTanhNormal(bounds, np.asarray(config["prior"]["base_std"], dtype=np.float64))
        adversary = _TorchAdversaryAdapter(model, prior, np.asarray(config["prior"]["base_mean"]), bounds.astype(np.float32), torch, device=device)
        cache_path = Path(os.environ["MADRONA_MWGPU_KERNEL_CACHE"]) if os.environ.get("MADRONA_MWGPU_KERNEL_CACHE") else None
        _require(cache_path is not None and cache_path.is_file(), "Madrona CUDA kernel cache was not materialized")
        fingerprints = {
            "methodology_repository": config["methodology_source"]["repository"], "methodology_commit": config["methodology_source"]["commit"], "gpudrive_commit": pins["gpudrive"]["commit"],
            "gpudrive_submodules": pins["gpudrive"]["submodules"], "source_scene_sha256": scene_identity["source_sha256"],
            "dataset_repository": scene_identity["dataset_repository"], "dataset_revision": scene_identity["dataset_revision"],
            "source_scene_relative_path": scene_identity["source_relative_path"],
            "derived_scene_canonical_sha256": scene_identity["derived_canonical_sha256"], "derived_scene_byte_sha256": derived_byte_hash,
            "scene_sha256": derived_byte_hash, "scene_scenario_id": experiment["scene"]["scenario_id"], "victim_stable_id": experiment["scene"]["focal_object_id"],
            "selected_object_ids": scene_identity["selected_object_ids"], "focal_object_id": scene_identity["focal_object_id"],
            "adversary_config_sha256": canonical_json_sha256(config), "experiment_config_sha256": canonical_json_sha256(experiment),
            "victim_checkpoint_model_sha256": victim_pin["source"]["files"]["model.safetensors"]["sha256"],
            "victim_config_sha256": victim_pin["source"]["files"]["config.json"]["sha256"],
            "native_extension_sha256": sha256_file(native_path), "source_verification_sha256": canonical_json_sha256(source_checks),
            "victim_verification_sha256": canonical_json_sha256(victim_report), "madrona_kernel_cache_sha256": sha256_file(cache_path), "port": port_identity(repository_root()),
        }
        metrics: list[dict[str, Any]] = []; checkpoint_ids: list[str] = []; total_transitions = 0
        last_training = last_training_arrays = last_evaluation = last_evaluation_arrays = None
        for iteration in range(1, int(config["training"]["iterations"]) + 1):
            episodes: list[MultiAgentRollout] = []; batches: list[dict[str, np.ndarray]] = []; collected = 0
            while collected < int(config["training"]["transitions_per_iteration"]):
                rollout = run_multiagent_rollout(backend=backend, victim=victim, adversary=adversary, intervention=intervention, max_steps=91)
                arrays = _episode_arrays(rollout, prior, config); episodes.append(rollout); batches.append(arrays); collected += rollout.transition_count
                last_training, last_training_arrays = rollout, arrays
            combined = _combine_batches(batches)
            update = _ppo_update(model=model, optimizer=optimizer, batch=combined, config=config, torch=torch, device=device)
            total_transitions += collected; assert_policy_frozen(victim_model); _require(torch_module_state_sha256(victim_model) == victim_state, "victim policy changed")
            evaluation = run_multiagent_rollout(backend=backend, victim=victim, adversary=adversary, intervention=intervention, max_steps=91, adversary_deterministic=True)
            evaluation_arrays = _episode_arrays(evaluation, prior, config); last_evaluation, last_evaluation_arrays = evaluation, evaluation_arrays
            row = {
                "iteration": iteration, "transitions": collected, "total_transitions": total_transitions, "episodes": len(episodes),
                "failures": sum(item.failure_timestep is not None for item in episodes),
                "failure_rate": float(np.mean([item.failure_timestep is not None for item in episodes])),
                "mean_episode_minimum_clearance": float(np.mean([item.episode_minimum_clearance for item in episodes])),
                "mean_reward": float(np.mean(combined["rewards"])), "mean_nll_penalty": float(np.mean(combined["nll_penalty"])),
                "deterministic_evaluation": {"failure_timestep": evaluation.failure_timestep, "termination_reason": evaluation.termination_reason, "minimum_clearance": evaluation.episode_minimum_clearance, "return": float(evaluation_arrays["rewards"].sum())},
                **update,
            }
            metrics.append(row)
            print(
                f"iteration {iteration:03d}/{int(config['training']['iterations']):03d} "
                f"episodes={row['episodes']} failures={row['failures']} "
                f"failure_rate={100.0 * row['failure_rate']:.2f}% "
                f"mean_min_clearance={row['mean_episode_minimum_clearance']:.3f} m",
                flush=True,
            )
            checkpoint_manifest = save_adversary_checkpoint(
                temporary / "checkpoints" / f"iteration-{iteration:04d}", model=model, optimizer=optimizer, config=config,
                iteration=iteration, total_transitions=total_transitions, metrics=row, fingerprints=fingerprints,
                parent_victim_checkpoint_id=checkpoint_identity(victim_pin),
            )
            checkpoint_ids.append(checkpoint_manifest["artifact_id"])

        _require(all(value is not None for value in (last_training, last_training_arrays, last_evaluation, last_evaluation_arrays)), "training produced no rollouts")
        nominal_path = temporary / "nominal-eligibility.npz"; training_path = temporary / "last-training-rollout.npz"; evaluation_path = temporary / "last-evaluation-rollout.npz"
        np.savez_compressed(nominal_path, **_payload(nominal, _episode_arrays(nominal, prior, config)))
        np.savez_compressed(training_path, **_payload(last_training, last_training_arrays))
        np.savez_compressed(evaluation_path, **_payload(last_evaluation, last_evaluation_arrays))
        manifest: dict[str, Any] = {
            "schema": "gpudrive_highway_10agent_training_run", "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "research_claims_allowed": False, "config": config, "experiment": experiment, "scene_identity": scene_identity,
            "parent_victim_checkpoint_id": checkpoint_identity(victim_pin), "victim_state_sha256_before": victim_state,
            "victim_state_sha256_after": torch_module_state_sha256(victim_model), "clean_eligibility": {"all_goals_reached": nominal.all_goals_reached, "termination_reason": nominal.termination_reason, "minimum_clearance": nominal.episode_minimum_clearance},
            "metrics": metrics, "checkpoint_ids": checkpoint_ids,
            "files": {
                "derived_scene": {"relative_path": f"derived-scene/{derived_path.name}", "sha256": derived_byte_hash},
                "nominal": {"relative_path": nominal_path.name, "sha256": sha256_file(nominal_path), "transition_count": nominal.transition_count, "termination_reason": nominal.termination_reason, "failure_timestep": nominal.failure_timestep},
                "last_training": {"relative_path": training_path.name, "sha256": sha256_file(training_path), "transition_count": last_training.transition_count, "termination_reason": last_training.termination_reason, "failure_timestep": last_training.failure_timestep},
                "last_evaluation": {"relative_path": evaluation_path.name, "sha256": sha256_file(evaluation_path), "transition_count": last_evaluation.transition_count, "termination_reason": last_evaluation.termination_reason, "failure_timestep": last_evaluation.failure_timestep},
            },
            "fingerprints": fingerprints, "source_verification": source_checks, "victim_verification": victim_report,
            "runtime": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        }
        manifest["artifact_id"] = "highway-10ppo-" + canonical_json_sha256(manifest)[:16]
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        _require(manifest["victim_state_sha256_after"] == victim_state, "victim state changed during training")
        validation = validate_multiagent_training_artifact(temporary)
        _require(validation["ok"], f"generated ten-agent run failed validation: {validation['failed_checks']}")
        temporary.replace(output.resolve()); return manifest
    except Exception:
        if temporary.exists(): shutil.rmtree(temporary)
        raise
    finally:
        if env is not None: env.close()
