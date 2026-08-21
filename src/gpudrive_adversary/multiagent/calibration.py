"""Random-prior disturbance-bound calibration for the ten-agent highway scene."""

from __future__ import annotations

import copy
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

from ..adversary.config import load_adversary_config
from ..adversary.distribution import BoundedTanhNormal
from ..adversary.environment import AdversaryContext, AdversaryDecision
from ..adversary.training import _ZeroAdversary, _intervention, _prepend_source_paths, _to_numpy
from ..pins import canonical_json_sha256, load_pins, repository_root, required_checks_pass, sha256_file, verify_source_tree
from ..provenance import port_identity
from ..victim.checkpoint import checkpoint_identity, default_checkpoint_directory, load_victim_pin, verify_checkpoint
from ..victim.policy import assert_policy_frozen, load_frozen_policy, pinned_action_table, torch_module_state_sha256
from .environment import AGENT_COUNT, ANY_CONTROLLED_FAILURE_SCOPE, MultiAgentRollout, MultiAgentVictimPolicy, run_multiagent_rollout
from .scene import build_derived_scene, default_highway_source_path, load_highway_experiment_config, write_derived_scene
from .training import GPUDriveTenAgentBackend, _env_config, _episode_failure_diagnostics


class BoundCalibrationError(RuntimeError):
    """Raised when the bound sweep violates its pinned experiment contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundCalibrationError(message)


def load_bound_sweep_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or repository_root() / "configs/calibration/highway_nonfocal_bound_sweep.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundCalibrationError(f"cannot read bound-sweep config: {exc}") from exc
    _require(config.get("schema") == "gpudrive_highway_bound_sweep_config", "bound-sweep schema changed")
    _require(config.get("schema_version") == 1, "unsupported bound-sweep schema version")
    _require(isinstance(config.get("episodes_per_bound"), int) and config["episodes_per_bound"] > 0, "episodes_per_bound must be positive")
    _require(isinstance(config.get("seed"), int) and config["seed"] >= 0, "bound-sweep seed must be nonnegative")
    target = np.asarray(config.get("target_random_failure_rate"), dtype=np.float64)
    _require(target.shape == (2,) and 0.0 <= target[0] <= target[1] <= 1.0, "target failure-rate interval is invalid")
    sampling = config.get("sampling", {})
    _require(sampling.get("family") == "iid_zero_mean_tanh_normal", "unsupported calibration distribution")
    sigma = np.asarray(sampling.get("latent_standard_deviation"), dtype=np.float64)
    _require(sigma.shape == (2,) and np.all(sigma > 0), "calibration latent standard deviation is invalid")
    _require(sampling.get("common_random_latents_across_bounds") is True, "calibration must use common latent draws across bounds")
    candidates = config.get("bounds")
    _require(isinstance(candidates, list) and len(candidates) >= 2, "at least two bound candidates are required")
    names: set[str] = set()
    for candidate in candidates:
        _require(isinstance(candidate, dict), "bound candidate must be an object")
        name = candidate.get("name")
        _require(isinstance(name, str) and name and name not in names, "bound candidate names must be unique")
        names.add(name)
        values = np.asarray([candidate.get("acceleration"), candidate.get("steering")], dtype=np.float64)
        _require(values.shape == (2,) and np.all(np.isfinite(values)) and np.all(values > 0), f"bounds for {name} are invalid")
        _require(isinstance(candidate.get("scale"), (int, float)) and candidate["scale"] > 0, f"scale for {name} is invalid")
    return config


class RandomPriorAdversary:
    """IID zero-mean draws from the declared bounded calibration prior."""

    def __init__(self, prior: BoundedTanhNormal, rng: np.random.Generator) -> None:
        self.prior = prior
        self.rng = rng
        self.mean = np.zeros(prior.dimension, dtype=np.float64)

    def act(self, context: AdversaryContext, *, deterministic: bool) -> AdversaryDecision:
        del context
        _require(not deterministic, "random calibration adversary cannot be deterministic")
        sample = self.prior.sample(self.rng, self.mean)
        return AdversaryDecision(
            requested_disturbance=sample.value.astype(np.float32),
            negative_log_likelihood=float(-sample.log_prob),
            pre_actor_latent=np.zeros(64, dtype=np.float32),
            raw_action=sample.latent.astype(np.float32),
            policy_log_probability=float(sample.log_prob),
            value=0.0,
        )


def _summarize_bound(episodes: list[MultiAgentRollout], *, name: str, bounds: np.ndarray) -> dict[str, Any]:
    _require(bool(episodes), "a bound candidate produced no episodes")
    failures = [episode for episode in episodes if episode.failure_timestep is not None]
    diagnostics = _episode_failure_diagnostics(episodes)
    while_focal_active = 0
    after_focal_goal = 0
    signatures: set[tuple[Any, ...]] = set()
    for episode in failures:
        assert episode.failure_timestep is not None
        state = episode.failure_timestep + 1
        focal_goal = bool(episode.goal_ever[state, 0])
        while_focal_active += int(not focal_goal and not bool(episode.done[state, 0]))
        after_focal_goal += int(focal_goal)
        slots = tuple(np.flatnonzero(episode.failing_agents[episode.failure_timestep]).astype(int).tolist())
        signatures.add((tuple(episode.failure_kinds[episode.failure_timestep]), slots, episode.failure_timestep // 5))
    disturbances = np.concatenate([episode.disturbance_effective for episode in episodes], axis=0)
    return {
        "name": name,
        "bounds": bounds.astype(float).tolist(),
        "episodes": len(episodes),
        "transitions": int(sum(episode.transition_count for episode in episodes)),
        "failures": len(failures),
        "failure_rate": len(failures) / len(episodes),
        "failures_while_focal_active": while_focal_active,
        "failures_after_focal_goal": after_focal_goal,
        "coarse_distinct_failure_signatures": len(signatures),
        "mean_episode_length": float(np.mean([episode.transition_count for episode in episodes])),
        "mean_absolute_effective_disturbance": np.mean(np.abs(disturbances), axis=0).astype(float).tolist(),
        "maximum_absolute_effective_disturbance": np.max(np.abs(disturbances), axis=0).astype(float).tolist(),
        **diagnostics,
    }


def run_highway_bound_sweep(
    *,
    source: Path,
    output: Path,
    checkpoint_directory: Path | None = None,
    scene_source: Path | None = None,
    adversary_config_path: Path | None = None,
    experiment_config_path: Path | None = None,
    sweep_config_path: Path | None = None,
    victim_pin_path: Path | None = None,
    gpudrive_pin_path: Path | None = None,
    episodes_per_bound: int | None = None,
) -> dict[str, Any]:
    _require(platform.system() == "Linux", "bound calibration is Linux/CUDA only")
    _require(not output.exists(), f"calibration output already exists: {output}")
    sweep = load_bound_sweep_config(sweep_config_path)
    episodes = episodes_per_bound if episodes_per_bound is not None else int(sweep["episodes_per_bound"])
    _require(isinstance(episodes, int) and episodes > 0, "episodes per bound must be positive")
    adversary_path = adversary_config_path or repository_root() / "configs/adversary/highway_10agent_nonfocal_system_transformer_ppo.json"
    experiment_path = experiment_config_path or repository_root() / "configs/multiagent/highway_10agent_nonfocal_system.json"
    adversary_config = load_adversary_config(adversary_path)
    experiment = load_highway_experiment_config(experiment_path)
    _require(adversary_config["failure"]["scope"] == "nonfocal_slots_1_through_9_only", "calibration requires the nonfocal failure scope")
    pins = load_pins(gpudrive_pin_path)
    victim_pin = load_victim_pin(victim_pin_path)
    source = source.resolve()
    checkpoint = checkpoint_directory.resolve() if checkpoint_directory else default_checkpoint_directory(victim_pin).resolve()
    source_checks = verify_source_tree(source, pins)
    _require(required_checks_pass(source_checks), "pinned GPUDrive source verification failed")
    victim_report = verify_checkpoint(checkpoint, victim_pin, gpudrive_source=source)
    _require(victim_report["ok"], "victim checkpoint verification failed")
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
        raise BoundCalibrationError(f"native imports failed: {type(exc).__name__}: {exc}") from exc
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"}, "CUBLAS_WORKSPACE_CONFIG must be set before CUDA starts")
    _require(torch.cuda.is_available(), "CUDA is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = int(sweep["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    native_path = Path(madrona_gpudrive.__file__).resolve()
    _require(Path(gpudrive.__file__).resolve().is_relative_to(source), "imported GPUDrive from wrong source")
    _require(native_path.is_relative_to(source), "imported native extension from wrong source")

    temporary = output.resolve().parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    env = None
    try:
        derived_path = temporary / "derived-scene" / experiment["scene"]["derived_name"]
        derived_hash = write_derived_scene(derived_path, derived)
        loader = SceneDataLoader(root=str(derived_path.parent), batch_size=1, dataset_size=1, sample_with_replacement=False, file_prefix=derived_path.name, seed=seed, shuffle=False)
        env = GPUDriveTorchEnv(config=_env_config(EnvConfig, torch, victim_pin), data_loader=loader, max_cont_agents=AGENT_COUNT, device="cuda", action_type="discrete")
        table = _to_numpy(env.action_keys_tensor).astype("<f4", copy=False)
        _require(np.array_equal(table, pinned_action_table(victim_pin["environment"])), "victim action table changed")
        victim_model = load_frozen_policy(checkpoint, victim_pin["model_config"], device="cuda")
        victim_state = torch_module_state_sha256(victim_model)
        victim = MultiAgentVictimPolicy(victim_model, table, torch, device="cuda")
        backend = GPUDriveTenAgentBackend(env, torch, device="cuda", horizon=91, expected_ids=experiment["scene"]["selected_object_ids"], scenario_id=experiment["scene"]["scenario_id"], agent_scale=float(AGENT_SCALE))

        nominal = run_multiagent_rollout(
            backend=backend,
            victim=victim,
            adversary=_ZeroAdversary(),
            intervention=_intervention(adversary_config),
            max_steps=91,
            adversary_deterministic=True,
            failure_scope=ANY_CONTROLLED_FAILURE_SCOPE,
        )
        _require(nominal.failure_timestep is None and nominal.all_goals_reached, "clean ten-agent eligibility failed")
        print("Clean eligibility passed; starting random-prior bound sweep", flush=True)

        rows: list[dict[str, Any]] = []
        sigma = np.asarray(sweep["sampling"]["latent_standard_deviation"], dtype=np.float64)
        for candidate in sweep["bounds"]:
            bounds = np.asarray([candidate["acceleration"], candidate["steering"]], dtype=np.float64)
            prior = BoundedTanhNormal(bounds, sigma)
            candidate_config = copy.deepcopy(adversary_config)
            candidate_config["intervention"]["bounds"] = bounds.tolist()
            intervention = _intervention(candidate_config)
            rollouts: list[MultiAgentRollout] = []
            for episode_index in range(episodes):
                rng = np.random.default_rng(np.random.SeedSequence([seed, episode_index]))
                rollout = run_multiagent_rollout(
                    backend=backend,
                    victim=victim,
                    adversary=RandomPriorAdversary(prior, rng),
                    intervention=intervention,
                    max_steps=91,
                    failure_scope=adversary_config["failure"]["scope"],
                )
                rollouts.append(rollout)
                if (episode_index + 1) % 100 == 0 or episode_index + 1 == episodes:
                    observed = sum(item.failure_timestep is not None for item in rollouts)
                    print(f"bounds={candidate['name']} episodes={episode_index + 1}/{episodes} failures={observed}", flush=True)
            row = _summarize_bound(rollouts, name=candidate["name"], bounds=bounds)
            target = sweep["target_random_failure_rate"]
            row["within_target_random_failure_rate"] = bool(target[0] <= row["failure_rate"] <= target[1])
            rows.append(row)
            print(f"bounds={candidate['name']} failure_rate={100.0 * row['failure_rate']:.2f}% failures={row['failures']}/{row['episodes']}", flush=True)

        assert_policy_frozen(victim_model)
        _require(torch_module_state_sha256(victim_model) == victim_state, "victim policy changed during calibration")
        cache_path = Path(os.environ["MADRONA_MWGPU_KERNEL_CACHE"]) if os.environ.get("MADRONA_MWGPU_KERNEL_CACHE") else None
        _require(cache_path is not None and cache_path.is_file(), "Madrona CUDA kernel cache was not materialized")
        manifest: dict[str, Any] = {
            "schema": "gpudrive_highway_bound_sweep",
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "research_claims_allowed": False,
            "artifact_id": None,
            "sweep_config": sweep,
            "episodes_per_bound": episodes,
            "experiment": experiment,
            "adversary_reference_config": adversary_config,
            "clean_eligibility": {
                "all_goals_reached": nominal.all_goals_reached,
                "termination_reason": nominal.termination_reason,
                "minimum_clearance": nominal.episode_minimum_clearance,
            },
            "results": rows,
            "scene_identity": scene_identity,
            "fingerprints": {
                "gpudrive_commit": pins["gpudrive"]["commit"],
                "gpudrive_submodules": pins["gpudrive"]["submodules"],
                "dataset_repository": scene_identity["dataset_repository"],
                "dataset_revision": scene_identity["dataset_revision"],
                "source_scene_sha256": scene_identity["source_sha256"],
                "derived_scene_sha256": derived_hash,
                "victim_checkpoint_id": checkpoint_identity(victim_pin),
                "victim_state_sha256": victim_state,
                "native_extension_sha256": sha256_file(native_path),
                "madrona_kernel_cache_sha256": sha256_file(cache_path),
                "sweep_config_sha256": canonical_json_sha256(sweep),
                "experiment_config_sha256": canonical_json_sha256(experiment),
                "adversary_reference_config_sha256": canonical_json_sha256(adversary_config),
                "port": port_identity(repository_root()),
            },
            "source_verification": source_checks,
            "victim_verification": victim_report,
            "runtime": {
                "platform": platform.platform(),
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
        manifest["artifact_id"] = "highway-bound-sweep-" + canonical_json_sha256(manifest)[:16]
        (temporary / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        temporary.replace(output.resolve())
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        if env is not None:
            env.close()
