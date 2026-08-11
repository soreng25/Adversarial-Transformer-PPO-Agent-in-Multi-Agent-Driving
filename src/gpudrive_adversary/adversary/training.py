"""Linux/CUDA tiny training pipeline for the frozen-victim Transformer adversary."""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import uuid
from dataclasses import asdict
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
from ..victim.checkpoint import (
    checkpoint_identity,
    default_checkpoint_directory,
    load_victim_pin,
    verify_checkpoint,
)
from ..victim.policy import (
    assert_policy_frozen,
    load_frozen_policy,
    pinned_action_table,
    torch_module_state_sha256,
    validate_slot0_binding,
)
from .checkpoint import save_adversary_checkpoint
from .config import load_adversary_config
from .distribution import BoundedTanhNormal
from .environment import (
    AdversaryContext,
    AdversaryDecision,
    BackendState,
    PinnedVictimPolicyAdapter,
    SequentialRollout,
    run_sequential_rollout,
)
from .failure import (
    RAW_INFO_ORDER,
    assess_nominal_goal_eligibility,
    classify_victim_post_step,
)
from .intervention import InterventionSpec, apply_intervention
from .model import AdversaryModelConfig, CausalTransformerActorCritic
from .ppo import PPOConfig, generalized_advantage_estimate_numpy, ppo_minibatch_update
from .training_artifact import validate_adversary_training_artifact


class AdversaryTrainingError(RuntimeError):
    """Raised when native adversary training violates its pinned contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdversaryTrainingError(message)


def _prepend_source_paths(source: Path) -> None:
    for path in (source / "build", source):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().contiguous().numpy().copy()


class _GPUDriveBackend:
    def __init__(
        self,
        env: Any,
        torch: Any,
        *,
        device: str,
        horizon: int,
        expected_sdc_id: int,
        expected_scenario_id: str,
    ) -> None:
        self.env = env
        self.torch = torch
        self.device = device
        self.horizon = int(horizon)
        self.expected_sdc_id = int(expected_sdc_id)
        self.expected_scenario_id = expected_scenario_id
        self.timestep = 0

    def _state(self) -> BackendState:
        observation = _to_numpy(self.env.get_obs(self.env.cont_agent_mask))[0]
        raw_info = _to_numpy(self.env.sim.info_tensor().to_torch())[0, 0]
        _require(
            observation.shape == (2984,),
            f"victim observation shape changed: {observation.shape}",
        )
        _require(raw_info.shape == (5,), f"raw info shape changed: {raw_info.shape}")
        done = bool(_to_numpy(self.env.get_dones())[0, 0])
        return BackendState(
            victim_observation=observation,
            evidence=raw_info,
            done=done,
            horizon_reached=self.timestep >= self.horizon,
        )

    def _assert_binding(self) -> None:
        metadata = _to_numpy(self.env.sim.metadata_tensor().to_torch())[:, :, 0]
        stable_ids = _to_numpy(
            self.env.sim.absolute_self_observation_tensor().to_torch()
        )[:, :, 13]
        validate_slot0_binding(
            _to_numpy(self.env.cont_agent_mask),
            metadata,
            stable_ids,
            expected_id=self.expected_sdc_id,
        )
        _require(
            self.env.get_scenario_ids()[0] == self.expected_scenario_id,
            "GPUDrive loaded an unexpected scenario",
        )

    def reset(self) -> BackendState:
        self.env.reset()
        self.timestep = 0
        self._assert_binding()
        return self._state()

    def step(self, applied_command: np.ndarray) -> BackendState:
        command = np.asarray(applied_command, dtype=np.float32)
        _require(command.shape == (3,), "applied command must have shape (3,)")
        action_values = self.torch.zeros(
            (1, self.env.max_agent_count, 3),
            dtype=self.torch.float32,
            device=self.device,
        )
        action_values[0, 0] = self.torch.as_tensor(
            command, dtype=self.torch.float32, device=self.device
        )
        self.env.step_dynamics(action_values)
        self.timestep += 1
        return self._state()


class _TorchAdversaryAdapter:
    def __init__(
        self,
        model: Any,
        prior: BoundedTanhNormal,
        prior_mean: np.ndarray,
        bounds: np.ndarray,
        torch: Any,
        *,
        device: str,
    ) -> None:
        self.model = model
        self.prior = prior
        self.prior_mean = np.asarray(prior_mean, dtype=np.float64)
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.torch = torch
        self.device = device

    def act(
        self, context: AdversaryContext, *, deterministic: bool
    ) -> AdversaryDecision:
        tokens = self.torch.as_tensor(
            context.history[None], dtype=self.torch.float32, device=self.device
        )
        mask = self.torch.as_tensor(
            context.history_mask[None], dtype=self.torch.bool, device=self.device
        )
        with self.torch.no_grad():
            decision = self.model.act(
                tokens,
                mask,
                deterministic=deterministic,
                bounds=self.bounds,
            )
        disturbance = _to_numpy(decision.action)[0]
        _require(
            bool(np.all(np.abs(disturbance) < self.bounds)),
            "sampled disturbance reached the open-support boundary",
        )
        exact_nll = float(
            np.asarray(
                self.prior.nll(disturbance, self.prior_mean)
            ).item()
        )
        return AdversaryDecision(
            requested_disturbance=disturbance,
            negative_log_likelihood=exact_nll,
            pre_actor_latent=_to_numpy(decision.pre_actor_features)[0],
            raw_action=_to_numpy(decision.raw_action)[0],
            policy_log_probability=float(
                decision.log_probability.detach().cpu().item()
            ),
            value=float(decision.value.detach().cpu().item()),
        )


class _ZeroAdversary:
    def act(
        self, context: AdversaryContext, *, deterministic: bool
    ) -> AdversaryDecision:
        del context, deterministic
        return AdversaryDecision(
            requested_disturbance=np.zeros(2, dtype=np.float32),
            negative_log_likelihood=0.0,
            pre_actor_latent=np.zeros(64, dtype=np.float32),
            raw_action=np.zeros(2, dtype=np.float32),
            policy_log_probability=0.0,
            value=0.0,
        )


def _intervention(config: dict[str, Any]):
    section = config["intervention"]
    spec = InterventionSpec(
        np.asarray(section["bounds"], dtype=np.float64),
        np.asarray(
            [
                section["final_acceleration_envelope"][0],
                section["final_steering_envelope"][0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                section["final_acceleration_envelope"][1],
                section["final_steering_envelope"][1],
            ],
            dtype=np.float64,
        ),
    )

    def apply(nominal: np.ndarray, disturbance: np.ndarray):
        return apply_intervention(nominal, disturbance, spec)

    return apply


def _nominal_eligibility(
    *,
    backend: _GPUDriveBackend,
    victim: PinnedVictimPolicyAdapter,
    intervention: Any,
    horizon: int,
) -> tuple[dict[str, Any], SequentialRollout]:
    rollout = run_sequential_rollout(
        backend=backend,
        victim=victim,
        adversary=_ZeroAdversary(),
        intervention=intervention,
        failure_classifier=classify_victim_post_step,
        max_steps=horizon,
        adversary_deterministic=True,
    )
    initial = np.asarray(rollout.evidence[0])
    initial_event = bool(np.any(initial[:3] != 0))
    assessed = assess_nominal_goal_eligibility(rollout.evidence[1:])
    result = asdict(assessed)
    if initial_event:
        result.update(
            {
                "eligible": False,
                "reason": "initial_safety_event",
                "failure_timestep": None,
                "failure_kinds": tuple(
                    RAW_INFO_ORDER[index]
                    for index in range(3)
                    if initial[index] != 0
                ),
            }
        )
    return result, rollout


def _model_config(config: dict[str, Any]) -> AdversaryModelConfig:
    model = config["model"]
    token = config["token"]
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
        action_epsilon=float(config["prior"]["epsilon"]),
    )


def _ppo_config(config: dict[str, Any]) -> PPOConfig:
    ppo = config["ppo"]
    return PPOConfig(
        learning_rate=float(ppo["learning_rate"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        policy_clip=float(ppo["clip_ratio"]),
        value_clip=float(ppo["value_clip"]),
        value_coefficient=float(ppo["value_coefficient"]),
        entropy_coefficient=float(ppo["entropy_coefficient"]),
        max_grad_norm=float(ppo["max_grad_norm"]),
        normalize_advantages=bool(ppo["normalize_advantages"]),
    )


def _episode_training_arrays(
    rollout: SequentialRollout,
    prior: BoundedTanhNormal,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    reward_config = config["reward"]
    prior_mean = np.asarray(config["prior"]["base_mean"], dtype=np.float64)
    nll_penalty = np.asarray(
        prior.nll_excess_from_zero(rollout.disturbance_requested, prior_mean),
        dtype=np.float64,
    )
    rewards = (
        float(reward_config["failure_bonus"])
        * rollout.failure_by_transition.astype(np.float64)
        - float(reward_config["nll_coefficient"]) * nll_penalty
    )
    if rollout.termination_reason == "horizon":
        rewards[-1] += float(reward_config["horizon_reward"])
    terminated = np.zeros(rollout.transition_count, dtype=np.bool_)
    terminated[-1] = True
    advantages, returns = generalized_advantage_estimate_numpy(
        rewards.astype(np.float32),
        rollout.adversary_values.astype(np.float32),
        terminated,
        np.asarray(0.0, dtype=np.float32),
        gamma=float(config["ppo"]["gamma"]),
        gae_lambda=float(config["ppo"]["gae_lambda"]),
    )
    return {
        "histories": rollout.histories.astype(np.float32, copy=False),
        "history_masks": rollout.history_masks.astype(np.bool_, copy=False),
        "actions": rollout.disturbance_requested.astype(np.float32, copy=False),
        "old_log_probability": rollout.policy_log_probability.astype(
            np.float32, copy=False
        ),
        "old_value": rollout.adversary_values.astype(np.float32, copy=False),
        "advantages": advantages.astype(np.float32, copy=False),
        "returns": returns.astype(np.float32, copy=False),
        "rewards": rewards.astype(np.float32, copy=False),
        "nll_penalty": nll_penalty.astype(np.float32, copy=False),
    }


def _combine_batches(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        name: np.concatenate([batch[name] for batch in batches], axis=0)
        for name in batches[0]
    }


def _ppo_update(
    *,
    model: Any,
    optimizer: Any,
    batch: dict[str, np.ndarray],
    config: dict[str, Any],
    torch: Any,
    device: str,
) -> dict[str, float]:
    settings = _ppo_config(config)
    count = batch["actions"].shape[0]
    minibatch_size = min(int(config["ppo"]["minibatch_size"]), count)
    bounds = torch.as_tensor(
        config["intervention"]["bounds"], dtype=torch.float32, device=device
    )
    metrics: list[dict[str, float]] = []
    model.train()
    for _ in range(int(config["ppo"]["update_epochs"])):
        permutation = torch.randperm(count, device="cpu").numpy()
        for start in range(0, count, minibatch_size):
            indices = permutation[start : start + minibatch_size]

            def tensor(name: str, dtype: Any = torch.float32):
                return torch.as_tensor(batch[name][indices], dtype=dtype, device=device)

            metrics.append(
                ppo_minibatch_update(
                    model=model,
                    optimizer=optimizer,
                    tokens=tensor("histories"),
                    valid_mask=tensor("history_masks", torch.bool),
                    actions=tensor("actions"),
                    old_log_probability=tensor("old_log_probability"),
                    old_value=tensor("old_value"),
                    advantages=tensor("advantages"),
                    returns=tensor("returns"),
                    bounds=bounds,
                    config=settings,
                )
            )
    model.eval()
    return {
        name: float(np.mean([item[name] for item in metrics]))
        for name in metrics[0]
    }


def _rollout_payload(
    rollout: SequentialRollout,
    training_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    kinds = np.zeros((rollout.transition_count, 3), dtype=np.int8)
    kind_names = RAW_INFO_ORDER[:3]
    for timestep, present in enumerate(rollout.failure_kinds):
        for kind in present:
            if kind in kind_names:
                kinds[timestep, kind_names.index(kind)] = 1
    return {
        "victim_observations": rollout.victim_observations,
        "raw_info": rollout.evidence,
        "tokens": rollout.tokens,
        "history_masks": rollout.history_masks,
        "victim_action_indices": rollout.victim_action_indices,
        "victim_logits": rollout.victim_logits,
        "victim_nominal_commands": rollout.victim_nominal_commands,
        "adversary_raw_actions": rollout.adversary_raw_actions,
        "disturbance_requested": rollout.disturbance_requested,
        "disturbance_effective": rollout.disturbance_effective,
        "applied_commands": rollout.applied_commands,
        "disturbance_saturated": rollout.disturbance_saturated,
        "command_saturated": rollout.command_saturated,
        "prior_nll_exact": rollout.negative_log_likelihood,
        "prior_nll_penalty": training_arrays["nll_penalty"],
        "policy_log_probability": rollout.policy_log_probability,
        "adversary_values": rollout.adversary_values,
        "rewards": training_arrays["rewards"],
        "pre_actor_latent": rollout.pre_actor_latent,
        "failure_by_transition": rollout.failure_by_transition,
        "failure_kind_bits": kinds,
    }


def _training_fingerprints(
    *,
    gpudrive_pins: dict[str, Any],
    victim_pin: dict[str, Any],
    native_path: Path,
    source_checks: list[dict[str, Any]],
    victim_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    cache_path = os.environ.get("MADRONA_MWGPU_KERNEL_CACHE")
    cache = Path(cache_path) if cache_path else None
    if cache is None or not cache.exists():
        cache_sha256 = None
    elif cache.is_file():
        cache_sha256 = sha256_file(cache)
    else:
        cache_sha256 = tree_sha256(cache)
    _require(
        cache_sha256 is not None,
        "CUDA training requires a materialized Madrona kernel-cache fingerprint",
    )
    return {
        "methodology_repository": config["methodology_source"]["repository"],
        "methodology_commit": config["methodology_source"]["commit"],
        "gpudrive_commit": gpudrive_pins["gpudrive"]["commit"],
        "gpudrive_submodules": gpudrive_pins["gpudrive"]["submodules"],
        "scene_sha256": gpudrive_pins["smoke_scene"]["sha256"],
        "scene_scenario_id": gpudrive_pins["smoke_scene"]["scenario_id"],
        "victim_stable_id": gpudrive_pins["smoke_scene"]["sdc_object_id"],
        "victim_checkpoint_model_sha256": victim_pin["source"]["files"][
            "model.safetensors"
        ]["sha256"],
        "victim_config_sha256": victim_pin["source"]["files"]["config.json"][
            "sha256"
        ],
        "adversary_config_sha256": canonical_json_sha256(config),
        "native_extension_sha256": sha256_file(native_path),
        "madrona_kernel_cache_sha256": cache_sha256,
        "source_verification_sha256": canonical_json_sha256(source_checks),
        "victim_verification_sha256": canonical_json_sha256(victim_report),
        "port": port_identity(repository_root()),
    }


def train_adversary_smoke(
    *,
    source: Path,
    output: Path,
    checkpoint_directory: Path | None = None,
    adversary_config_path: Path | None = None,
    victim_pin_path: Path | None = None,
    gpudrive_pin_path: Path | None = None,
) -> dict[str, Any]:
    """Train the approved adversary briefly on one replay fixture.

    This is intentionally a plumbing smoke. It refuses to train if the pinned
    scene does not satisfy the approved nominal clean-goal eligibility rule.
    """

    _require(platform.system() == "Linux", "adversary training is Linux/CUDA only")
    _require(not output.exists(), f"training output already exists: {output}")
    config = load_adversary_config(adversary_config_path)
    gpudrive_pins = load_pins(gpudrive_pin_path)
    victim_pin = load_victim_pin(victim_pin_path)
    source = source.resolve()
    checkpoint = (
        checkpoint_directory.resolve()
        if checkpoint_directory is not None
        else default_checkpoint_directory(victim_pin).resolve()
    )
    source_checks = verify_source_tree(source, gpudrive_pins)
    _require(required_checks_pass(source_checks), "pinned GPUDrive source verification failed")
    victim_report = verify_checkpoint(checkpoint, victim_pin, gpudrive_source=source)
    _require(victim_report["ok"], "pinned victim checkpoint verification failed")
    _prepend_source_paths(source)
    try:
        import gpudrive
        import madrona_gpudrive
        import torch
        from gpudrive.env.config import EnvConfig
        from gpudrive.env.dataset import SceneDataLoader
        from gpudrive.env.env_torch import GPUDriveTorchEnv
    except Exception as exc:
        raise AdversaryTrainingError(
            f"native training imports failed: {type(exc).__name__}: {exc}"
        ) from exc
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"},
        "CUBLAS_WORKSPACE_CONFIG must be set before the CUDA process starts",
    )
    torch.use_deterministic_algorithms(True)
    _require(torch.cuda.is_available(), "CUDA is required for adversary training")
    device = "cuda"
    gpudrive_path = Path(gpudrive.__file__).resolve()
    native_path = Path(madrona_gpudrive.__file__).resolve()
    _require(gpudrive_path.is_relative_to(source), "imported GPUDrive from wrong source")
    _require(native_path.is_relative_to(source), "imported native extension from wrong source")

    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    victim_model_config = victim_pin["model_config"]
    victim_environment = victim_pin["environment"]
    scene_pin = gpudrive_pins["smoke_scene"]
    scene_path = source / scene_pin["relative_path"]
    loader = SceneDataLoader(
        root=str(scene_path.parent),
        batch_size=1,
        dataset_size=1,
        sample_with_replacement=False,
        file_prefix=scene_path.name,
        seed=seed,
        shuffle=False,
    )
    _require(
        [Path(value).resolve() for value in loader.dataset] == [scene_path.resolve()],
        "scene loader did not resolve to exactly the pinned smoke scene",
    )
    env_config = EnvConfig(
        ego_state=bool(victim_model_config["ego_state"]),
        road_map_obs=bool(victim_model_config["road_map_obs"]),
        partner_obs=bool(victim_model_config["partner_obs"]),
        norm_obs=bool(victim_model_config["norm_obs"]),
        max_controlled_agents=int(victim_model_config["max_controlled_agents"]),
        num_worlds=1,
        disable_classic_obs=bool(victim_model_config["lidar_obs"]),
        lidar_obs=bool(victim_model_config["lidar_obs"]),
        collision_weight=float(victim_model_config["collision_weight"]),
        goal_achieved_weight=float(victim_model_config["goal_achieved_weight"]),
        off_road_weight=float(victim_model_config["off_road_weight"]),
        obs_radius=float(victim_model_config["obs_radius"]),
        polyline_reduction_threshold=float(
            victim_model_config["polyline_reduction_threshold"]
        ),
        dynamics_model=str(victim_model_config["dynamics_model"]),
        steer_actions=torch.tensor(victim_environment["steering_values"]),
        accel_actions=torch.tensor(victim_environment["acceleration_values"]),
        head_tilt_actions=torch.tensor(victim_environment["head_angle_values"]),
        collision_behavior="ignore",
        remove_non_vehicles=bool(victim_model_config["remove_non_vehicles"]),
        init_steps=0,
        reward_type=str(victim_model_config["reward_type"]),
        dist_to_goal_threshold=float(victim_model_config["dist_to_goal_threshold"]),
        init_mode=str(victim_model_config["init_mode"]),
        use_vbd=False,
        vbd_in_obs=False,
    )
    env = None
    temporary = output.resolve().parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=loader,
            max_cont_agents=1,
            device=device,
            action_type="discrete",
        )
        observed_table = _to_numpy(env.action_keys_tensor).astype("<f4", copy=False)
        _require(
            np.array_equal(observed_table, pinned_action_table(victim_environment)),
            "native victim action table changed",
        )
        victim_policy = load_frozen_policy(checkpoint, victim_model_config, device=device)
        victim_state_before = torch_module_state_sha256(victim_policy)
        victim = PinnedVictimPolicyAdapter(victim_policy, observed_table, device=device)
        backend = _GPUDriveBackend(
            env,
            torch,
            device=device,
            horizon=int(config["environment"]["episode_horizon"]),
            expected_sdc_id=int(scene_pin["sdc_object_id"]),
            expected_scenario_id=str(scene_pin["scenario_id"]),
        )
        intervention = _intervention(config)
        eligibility, nominal_rollout = _nominal_eligibility(
            backend=backend,
            victim=victim,
            intervention=intervention,
            horizon=int(config["environment"]["episode_horizon"]),
        )
        _require(
            bool(eligibility["eligible"]),
            f"pinned scene is not nominally eligible: {eligibility['reason']}",
        )

        model = CausalTransformerActorCritic(_model_config(config)).to(device)
        model.eval()
        ppo_settings = _ppo_config(config)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=ppo_settings.learning_rate
        )
        bounds = np.asarray(config["intervention"]["bounds"], dtype=np.float64)
        prior = BoundedTanhNormal(
            bounds,
            np.asarray(config["prior"]["base_std"], dtype=np.float64),
        )
        adversary = _TorchAdversaryAdapter(
            model,
            prior,
            np.asarray(config["prior"]["base_mean"], dtype=np.float64),
            bounds.astype(np.float32),
            torch,
            device=device,
        )
        fingerprints = _training_fingerprints(
            gpudrive_pins=gpudrive_pins,
            victim_pin=victim_pin,
            native_path=native_path,
            source_checks=source_checks,
            victim_report=victim_report,
            config=config,
        )
        all_metrics: list[dict[str, Any]] = []
        checkpoint_ids: list[str] = []
        total_transitions = 0
        last_training_rollout: SequentialRollout | None = None
        last_training_arrays: dict[str, np.ndarray] | None = None
        last_training_behavior_state_sha256: str | None = None
        last_training_behavior_checkpoint_id: str | None = None
        last_evaluation_rollout: SequentialRollout | None = None
        last_evaluation_arrays: dict[str, np.ndarray] | None = None
        for iteration in range(1, int(config["training"]["iterations"]) + 1):
            collection_state_sha256 = torch_module_state_sha256(model)
            collection_checkpoint_id = checkpoint_ids[-1] if checkpoint_ids else None
            episodes: list[SequentialRollout] = []
            batches: list[dict[str, np.ndarray]] = []
            collected = 0
            while collected < int(config["training"]["transitions_per_iteration"]):
                rollout = run_sequential_rollout(
                    backend=backend,
                    victim=victim,
                    adversary=adversary,
                    intervention=intervention,
                    failure_classifier=classify_victim_post_step,
                    max_steps=int(config["environment"]["episode_horizon"]),
                    adversary_deterministic=False,
                )
                arrays = _episode_training_arrays(rollout, prior, config)
                episodes.append(rollout)
                batches.append(arrays)
                collected += rollout.transition_count
                last_training_rollout, last_training_arrays = rollout, arrays
                last_training_behavior_state_sha256 = collection_state_sha256
                last_training_behavior_checkpoint_id = collection_checkpoint_id
            combined = _combine_batches(batches)
            update_metrics = _ppo_update(
                model=model,
                optimizer=optimizer,
                batch=combined,
                config=config,
                torch=torch,
                device=device,
            )
            total_transitions += collected
            assert_policy_frozen(victim_policy)
            _require(
                torch_module_state_sha256(victim_policy) == victim_state_before,
                "victim state changed during adversary optimization",
            )
            evaluation_rollout = run_sequential_rollout(
                backend=backend,
                victim=victim,
                adversary=adversary,
                intervention=intervention,
                failure_classifier=classify_victim_post_step,
                max_steps=int(config["environment"]["episode_horizon"]),
                adversary_deterministic=True,
            )
            evaluation_arrays = _episode_training_arrays(
                evaluation_rollout, prior, config
            )
            last_evaluation_rollout = evaluation_rollout
            last_evaluation_arrays = evaluation_arrays
            iteration_metrics = {
                "iteration": iteration,
                "transitions": collected,
                "total_transitions": total_transitions,
                "episodes": len(episodes),
                "failures": int(sum(item.failure_timestep is not None for item in episodes)),
                "failure_rate": float(
                    np.mean([item.failure_timestep is not None for item in episodes])
                ),
                "mean_episode_length": float(
                    np.mean([item.transition_count for item in episodes])
                ),
                "mean_prior_nll_exact": float(
                    np.mean(
                        np.concatenate(
                            [item.negative_log_likelihood for item in episodes]
                        )
                    )
                ),
                "mean_nll_penalty": float(np.mean(combined["nll_penalty"])),
                "mean_reward": float(np.mean(combined["rewards"])),
                "deterministic_evaluation": {
                    "transition_count": evaluation_rollout.transition_count,
                    "termination_reason": evaluation_rollout.termination_reason,
                    "failure_timestep": evaluation_rollout.failure_timestep,
                    "return": float(np.sum(evaluation_arrays["rewards"])),
                    "mean_nll_penalty": float(
                        np.mean(evaluation_arrays["nll_penalty"])
                    ),
                },
                **update_metrics,
            }
            all_metrics.append(iteration_metrics)
            checkpoint_manifest = save_adversary_checkpoint(
                temporary / "checkpoints" / f"iteration-{iteration:04d}",
                model=model,
                optimizer=optimizer,
                config=config,
                iteration=iteration,
                total_transitions=total_transitions,
                metrics=iteration_metrics,
                fingerprints=fingerprints,
                parent_victim_checkpoint_id=checkpoint_identity(victim_pin),
            )
            checkpoint_ids.append(checkpoint_manifest["artifact_id"])

        _require(
            last_training_rollout is not None
            and last_training_arrays is not None
            and last_training_behavior_state_sha256 is not None
            and last_evaluation_rollout is not None
            and last_evaluation_arrays is not None,
            "training produced no rollout",
        )
        training_trace_path = temporary / "last-training-rollout.npz"
        np.savez_compressed(
            training_trace_path,
            **_rollout_payload(last_training_rollout, last_training_arrays),
        )
        evaluation_trace_path = temporary / "last-evaluation-rollout.npz"
        np.savez_compressed(
            evaluation_trace_path,
            **_rollout_payload(last_evaluation_rollout, last_evaluation_arrays),
        )
        nominal_path = temporary / "nominal-eligibility.npz"
        np.savez_compressed(
            nominal_path,
            victim_observations=nominal_rollout.victim_observations,
            raw_info=nominal_rollout.evidence,
            victim_action_indices=nominal_rollout.victim_action_indices,
            victim_nominal_commands=nominal_rollout.victim_nominal_commands,
            applied_commands=nominal_rollout.applied_commands,
        )
        victim_state_after = torch_module_state_sha256(victim_policy)
        _require(victim_state_after == victim_state_before, "victim changed during training")
        run_manifest = {
            "schema": "gpudrive_adversary_training_run",
            "schema_version": 1,
            "artifact_id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": config["purpose"],
            "research_claims_allowed": False,
            "config": config,
            "config_sha256": canonical_json_sha256(config),
            "parent_victim_checkpoint_id": checkpoint_identity(victim_pin),
            "victim_state_sha256_before": victim_state_before,
            "victim_state_sha256_after": victim_state_after,
            "eligibility": eligibility,
            "scene": {
                "relative_path": scene_pin["relative_path"],
                "scenario_id": scene_pin["scenario_id"],
                "victim_slot": 0,
                "victim_stable_id": scene_pin["sdc_object_id"],
            },
            "metrics": all_metrics,
            "checkpoint_ids": checkpoint_ids,
            "last_training_rollout": {
                "sha256": sha256_file(training_trace_path),
                "behavior_model_state_sha256": last_training_behavior_state_sha256,
                "behavior_checkpoint_id": last_training_behavior_checkpoint_id,
                "transition_count": last_training_rollout.transition_count,
                "termination_reason": last_training_rollout.termination_reason,
                "failure_timestep": last_training_rollout.failure_timestep,
            },
            "last_deterministic_evaluation": {
                "sha256": sha256_file(evaluation_trace_path),
                "behavior_checkpoint_id": checkpoint_ids[-1],
                "transition_count": last_evaluation_rollout.transition_count,
                "termination_reason": last_evaluation_rollout.termination_reason,
                "failure_timestep": last_evaluation_rollout.failure_timestep,
            },
            "nominal_eligibility_trace_sha256": sha256_file(nominal_path),
            "fingerprints": fingerprints,
            "source_verification": source_checks,
            "victim_verification": victim_report,
            "runtime": {
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
                "deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "reference_image_digest": os.environ.get(
                    "GPUDRIVE_REFERENCE_IMAGE_DIGEST"
                ),
            },
        }
        run_manifest["artifact_id"] = "adversary-train-" + canonical_json_sha256(
            run_manifest
        )[:16]
        (temporary / "manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        validation = validate_adversary_training_artifact(temporary)
        _require(
            validation["ok"],
            "generated training run failed validation: "
            f"{validation['failed_checks']}",
        )
        temporary.replace(output.resolve())
        return run_manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        if env is not None:
            env.close()
