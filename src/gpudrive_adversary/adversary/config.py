"""Validated configuration for the approved Transformer-PPO attack contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..pins import canonical_json_sha256, repository_root


class AdversaryConfigError(ValueError):
    """Raised when a Transformer-PPO configuration changes the approved contract."""


def default_adversary_config_path() -> Path:
    return repository_root() / "configs/adversary/smoke_transformer_ppo.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdversaryConfigError(message)


def load_adversary_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_adversary_config_path()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdversaryConfigError(f"cannot load adversary config {config_path}: {exc}") from exc
    return validate_adversary_config(config)


def validate_adversary_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate an already parsed configuration against the approved contract."""

    _require(isinstance(config, dict), "adversary configuration must be an object")
    _require(
        config.get("schema") == "gpudrive_transformer_ppo_config"
        and config.get("schema_version") == 1,
        "unsupported adversary configuration schema",
    )
    environment = config.get("environment", {})
    intervention = config.get("intervention", {})
    prior = config.get("prior", {})
    failure = config.get("failure", {})
    eligibility = config.get("eligibility", {})
    token = config.get("token", {})
    model = config.get("model", {})
    reward = config.get("reward", {})
    ppo = config.get("ppo", {})
    training = config.get("training", {})

    methodology = config.get("methodology_source", {})
    purpose = config.get("purpose")
    _require(
        purpose
        in {
            "tiny_training_smoke_only",
            "highway_10agent_training_pilot",
            "highway_10agent_nonfocal_system_training_pilot",
        },
        "unsupported adversary configuration purpose",
    )
    highway = purpose in {
        "highway_10agent_training_pilot",
        "highway_10agent_nonfocal_system_training_pilot",
    }
    nonfocal_system = purpose == "highway_10agent_nonfocal_system_training_pilot"
    _require(config.get("research_claims_allowed") is False, "training config cannot authorize research claims")
    _require(
        methodology.get("repository")
        == "https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole.git"
        and methodology.get("commit")
        == "315b14a90b252ba416eb329e8003d5926806ba67",
        "methodology source pin changed",
    )
    _require(environment.get("victim_slot") == 0, "only slot 0 may be disturbed")
    _require(environment.get("max_controlled_agents") == (10 if highway else 1), "controlled-agent count changed")
    _require(environment.get("scene_source") == ("derived_highway_10ppo_scene" if highway else "pinned_smoke_scene"), "training scene source changed")
    _require(environment.get("episode_horizon") == 91, "pinned horizon must remain 91")
    _require(environment.get("decision_period_steps") == 1, "the adversary must act every simulator step")
    _require(environment.get("collision_behavior") == "ignore", "checkpoint-compatible collision behavior must be ignore")
    _require(environment.get("stop_on_first_failure") is True, "the wrapper must stop on the first failure")
    _require(intervention.get("dimensions") == 2, "approved intervention is two-dimensional")
    _require(intervention.get("names") == ["delta_acceleration", "delta_steering"], "intervention dimensions changed")
    _require(intervention.get("bounds") == [0.667, 0.262], "approved disturbance bounds changed")
    _require(intervention.get("final_acceleration_envelope") == [-4.0, 4.0], "approved acceleration envelope changed")
    _require(intervention.get("final_steering_envelope") == [-3.142, 3.142], "approved steering envelope changed")
    _require(intervention.get("head_angle_rule") == "preserve_nominal", "head angle must not be attacked")
    _require(intervention.get("apply_after_discrete_decode") is True, "residual must be applied after action decoding")
    _require(intervention.get("requantize") is False, "continuous residual must not be requantized")
    _require(prior.get("family") == "tanh_squashed_normal", "bounded prior must have no clipping atoms")
    _require(prior.get("base_mean") == [0.0, 0.0], "the smoke prior must remain zero-centered")
    _require(len(prior.get("base_std", [])) == 2 and all(float(value) > 0 for value in prior["base_std"]), "prior base_std must contain two positive values")
    _require(prior.get("density") == "exact_change_of_variables", "prior density accounting changed")
    _require(prior.get("reward_term") == "nll_excess_from_zero_disturbance", "likelihood penalty definition changed")
    expected_failure_scope = (
        "nonfocal_slots_1_through_9_only"
        if nonfocal_system
        else ("any_controlled_agent" if highway else "victim_only")
    )
    _require(failure.get("scope") == expected_failure_scope, "failure scope changed")
    _require(
        failure.get("post_step_raw_info_indices")
        == {
            "road_object_contact": 0,
            "vehicle_collision": 1,
            "nonvehicle_collision": 2,
            "goal": 3,
        },
        "raw failure-evidence indices changed",
    )
    _require(
        failure.get("failure_if_any")
        == ["road_object_contact", "vehicle_collision", "nonvehicle_collision"],
        "approved failure event set changed",
    )
    _require(failure.get("goal_is_failure") is False and failure.get("horizon_is_failure") is False, "goal/horizon cannot silently become failures")
    _require(failure.get("safety_wins_goal_ties") is True, "safety must win simultaneous goal ties")
    if nonfocal_system:
        _require(
            failure.get("focal_slot_events")
            == "diagnostic_only_no_reward_no_termination",
            "focal-only events must not count as system failures",
        )
        _require(
            failure.get("vehicle_collision_qualifier")
            == "both_participants_slots_1_through_9_with_raw_flags_and_obb_contact",
            "non-focal collision attribution changed",
        )
        _require(
            float(failure.get("obb_contact_tolerance_m", -1)) == 0.00001,
            "non-focal collision contact tolerance changed",
        )
    if highway:
        _require(eligibility.get("require_all_controlled_clean_goals") is True, "all ten clean goals are required")
        _require(eligibility.get("controlled_agent_count") == 10, "eligibility agent count changed")
    else:
        _require(eligibility.get("require_nominal_clean_goal") is True, "nominal clean-goal eligibility is required")
    expected_token_dim = int(token.get("victim_observation_dim", -1)) + int(token.get("previous_nominal_command_dim", -1)) + int(token.get("previous_applied_disturbance_dim", -1))
    _require(token.get("token_dim") == expected_token_dim == 2989, "adversary token dimensions are inconsistent")
    _require(token.get("history_length") == 50, "source-matched history length must be 50")
    _require(token.get("current_victim_action_visible") is False, "current victim action must not enter adversary input")
    _require(model.get("d_model") == 64 and model.get("num_layers") == 1 and model.get("nhead") == 1, "source-matched Transformer shape changed")
    _require(model.get("dim_feedforward") == 64 and model.get("dropout") == 0.0 and model.get("activation") == "relu", "Transformer feed-forward contract changed")
    _require(model.get("latent_name") == "pre_actor_features", "latent contract changed")
    _require(float(reward.get("failure_bonus", 0)) == 1.0, "smoke failure bonus changed")
    _require(float(reward.get("nll_coefficient", -1)) == 0.01, "smoke NLL coefficient changed")
    if highway:
        expected_shaping = (
            "terminal_minimum_signed_nonfocal_obb_clearance"
            if nonfocal_system
            else "terminal_minimum_signed_obb_clearance"
        )
        _require(reward.get("nonfailure_shaping") == expected_shaping, "highway clearance reward changed")
        _require(reward.get("normalization") == "divide_by_positive_initial_minimum_clearance_then_clip_0_1", "clearance normalization changed")
        _require(reward.get("calibration_status") == "single_scene_pilot_not_for_generalization", "pilot calibration status changed")
    else:
        _require(reward.get("calibration_status") == "smoke_only_not_approved_for_research_claims", "reward calibration status changed")
    _require(0 < float(ppo.get("clip_ratio", 0)) < 1, "PPO clip ratio must be in (0,1)")
    _require(int(ppo.get("minibatch_size", 0)) > 0, "PPO minibatch size must be positive")
    _require(int(ppo.get("update_epochs", 0)) > 0, "PPO update epochs must be positive")
    _require(int(training.get("iterations", 0)) > 0 and int(training.get("transitions_per_iteration", 0)) > 0, "training work must be positive")
    _require(training.get("checkpoint_every_iterations") == 1, "training must checkpoint every iteration")
    _require(training.get("deterministic_adversary_evaluation") is True, "deterministic post-update evaluation is required")
    _require(training.get("device") == "cuda", "the reference adversary training path is Linux/CUDA only")
    return config


def adversary_config_sha256(config: dict[str, Any] | None = None) -> str:
    return canonical_json_sha256(config or load_adversary_config())
