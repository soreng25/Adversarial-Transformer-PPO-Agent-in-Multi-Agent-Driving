"""Certified replay and visualization of a ten-agent highway failure."""

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

from ..adversary.checkpoint import load_adversary_checkpoint
from ..adversary.distribution import BoundedTanhNormal
from ..adversary.training import (
    _TorchAdversaryAdapter,
    _intervention,
    _prepend_source_paths,
    _to_numpy,
)
from ..pins import (
    canonical_json_sha256,
    load_pins,
    required_checks_pass,
    sha256_file,
    verify_source_tree,
)
from ..victim.checkpoint import (
    checkpoint_identity,
    default_checkpoint_directory,
    load_victim_pin,
    verify_checkpoint,
)
from ..victim.policy import (
    load_frozen_policy,
    pinned_action_table,
    torch_module_state_sha256,
)
from .artifact import validate_multiagent_training_artifact
from .clearance import oriented_box_corners
from .environment import AGENT_COUNT, MultiAgentRollout, MultiAgentVictimPolicy, run_multiagent_rollout
from .training import GPUDriveTenAgentBackend, _env_config, _episode_arrays, _payload


class MultiAgentReplayError(RuntimeError):
    """Raised when a selected checkpoint cannot produce a certified failure."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiAgentReplayError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultiAgentReplayError(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def resolve_checkpoint(run: Path, checkpoint: str | Path) -> Path:
    """Resolve ``iteration-0094`` beneath an indivisible training artifact."""

    root = run.resolve()
    supplied = Path(checkpoint)
    candidate = supplied if supplied.is_absolute() else root / "checkpoints" / supplied
    resolved = candidate.resolve()
    checkpoint_root = (root / "checkpoints").resolve()
    _require(
        resolved.parent == checkpoint_root,
        "checkpoint must be one direct child of RUN/checkpoints",
    )
    _require(resolved.is_dir(), f"checkpoint does not exist: {resolved}")
    return resolved


def failure_signature(rollout: MultiAgentRollout) -> dict[str, Any]:
    """Return the event identity that two certified replays must preserve."""

    timestep = rollout.failure_timestep
    if timestep is None:
        return {
            "is_failure": False,
            "failure_timestep": None,
            "failure_step_count": None,
            "failure_kinds": [],
            "failing_slots": [],
            "termination_reason": rollout.termination_reason,
        }
    return {
        "is_failure": True,
        "failure_timestep": int(timestep),
        "failure_step_count": int(timestep) + 1,
        "failure_kinds": list(rollout.failure_kinds[timestep]),
        "failing_slots": np.flatnonzero(rollout.failing_agents[timestep]).astype(int).tolist(),
        "termination_reason": rollout.termination_reason,
    }


def compare_failure_replays(
    first: MultiAgentRollout,
    second: MultiAgentRollout,
    *,
    absolute_tolerance: float = 1.0e-5,
    relative_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare two deterministic replays, including their first failure event."""

    _require(absolute_tolerance >= 0.0, "absolute tolerance must be nonnegative")
    _require(relative_tolerance >= 0.0, "relative tolerance must be nonnegative")
    first.validate()
    second.validate()
    signature_match = failure_signature(first) == failure_signature(second)
    transition_match = first.transition_count == second.transition_count
    failure_kind_trace_match = first.failure_kinds == second.failure_kinds

    exact_names = (
        "raw_info",
        "done",
        "goal_ever",
        "closest_pair",
        "history_masks",
        "victim_action_indices",
        "disturbance_saturated",
        "command_saturated",
        "failure_by_transition",
        "failing_agents",
    )
    numeric_names = (
        "observations",
        "boxes",
        "minimum_clearance",
        "tokens",
        "histories",
        "victim_logits",
        "nominal_commands",
        "applied_commands",
        "disturbance_requested",
        "disturbance_effective",
        "prior_nll_exact",
        "policy_log_probability",
        "adversary_values",
        "pre_actor_latent",
    )
    exact: dict[str, bool] = {}
    numeric: dict[str, dict[str, Any]] = {}
    if transition_match:
        exact = {
            name: bool(np.array_equal(getattr(first, name), getattr(second, name)))
            for name in exact_names
        }
        for name in numeric_names:
            left = np.asarray(getattr(first, name), dtype=np.float64)
            right = np.asarray(getattr(second, name), dtype=np.float64)
            difference = np.abs(left - right)
            numeric[name] = {
                "ok": bool(
                    np.allclose(
                        left,
                        right,
                        atol=absolute_tolerance,
                        rtol=relative_tolerance,
                        equal_nan=False,
                    )
                ),
                "maximum_absolute_difference": float(difference.max(initial=0.0)),
            }
    ok = (
        transition_match
        and signature_match
        and failure_kind_trace_match
        and all(exact.values())
        and all(item["ok"] for item in numeric.values())
    )
    return {
        "ok": bool(ok),
        "same_process": True,
        "transition_count_match": transition_match,
        "failure_signature_match": signature_match,
        "failure_kind_trace_match": failure_kind_trace_match,
        "failure_signature": failure_signature(first),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "exact_checks": exact,
        "numeric_checks": numeric,
    }


class _FrameCaptureBackend:
    """Decorate the native backend and render every returned simulator state."""

    def __init__(self, backend: GPUDriveTenAgentBackend, *, zoom_radius: int):
        self.backend = backend
        self.zoom_radius = int(zoom_radius)
        self.frames: list[np.ndarray] = []

    def _capture(self, state: Any) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise MultiAgentReplayError(f"Matplotlib visualization import failed: {exc}") from exc
        figures = self.backend.env.vis.plot_simulator_state(
            env_indices=[0],
            time_steps=[self.backend.timestep],
            center_agent_indices=[0],
            zoom_radius=max(
                self.zoom_radius,
                int(
                    np.ceil(
                        np.linalg.norm(
                            state.boxes[:, :2] - state.boxes[0, :2], axis=1
                        ).max()
                    )
                )
                + 10,
            ),
            plot_log_replay_trajectory=False,
        )
        _require(len(figures) == 1, "GPUDrive returned an unexpected number of figures")
        figure = figures[0]
        axis = figure.axes[0]
        focal_x, focal_y = state.boxes[0, :2]
        axis.scatter(
            [focal_x], [focal_y], s=500, facecolors="none", edgecolors="#d62728",
            linewidths=2.5, zorder=20,
        )
        axis.text(
            focal_x, focal_y, "  slot 0 (disturbed)", color="#d62728",
            fontsize=10, fontweight="bold", zorder=21,
        )
        failing = np.flatnonzero(np.any(state.raw_info[:, :3] != 0, axis=1))
        for slot in failing:
            x, y = state.boxes[int(slot), :2]
            axis.scatter([x], [y], marker="x", s=250, color="#ffbf00", linewidths=3, zorder=22)
            axis.text(x, y, f"  FAILURE slot {int(slot)}", color="#8c2d04", fontsize=10, fontweight="bold", zorder=23)
        label = f"t={self.backend.timestep} ({0.1 * self.backend.timestep:.1f} s)"
        if failing.size:
            label += " — first safety failure"
        figure.text(0.01, 0.99, label, va="top", ha="left", fontsize=12, fontweight="bold")
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
        self.frames.append(np.ascontiguousarray(rgba[..., :3]))
        plt.close(figure)

    def reset(self) -> Any:
        self.frames.clear()
        state = self.backend.reset()
        self._capture(state)
        return state

    def step(self, commands: np.ndarray) -> Any:
        state = self.backend.step(commands)
        self._capture(state)
        return state


def _write_diagnostic_plots(
    output: Path,
    rollout: MultiAgentRollout,
    *,
    object_ids: list[int],
    frames: list[np.ndarray],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as exc:
        raise MultiAgentReplayError(f"plotting dependencies are unavailable: {exc}") from exc

    state_time = np.arange(rollout.transition_count + 1, dtype=np.float64) * 0.1
    action_time = np.arange(rollout.transition_count, dtype=np.float64) * 0.1
    signature = failure_signature(rollout)
    failure_state = int(signature["failure_step_count"])

    fig, axis = plt.subplots(figsize=(11, 7))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, AGENT_COUNT))
    for slot in range(AGENT_COUNT):
        centers = rollout.boxes[:, slot, :2]
        width = 3.0 if slot == 0 else 1.4
        label = f"slot {slot}, id {object_ids[slot]}" + (" (disturbed)" if slot == 0 else "")
        axis.plot(centers[:, 0], centers[:, 1], color=colors[slot], linewidth=width, label=label)
        axis.scatter(centers[0, 0], centers[0, 1], color=colors[slot], marker="o", s=25)
    for slot in signature["failing_slots"]:
        center = rollout.boxes[failure_state, slot, :2]
        corners = oriented_box_corners(rollout.boxes[failure_state, slot])
        closed = np.vstack((corners, corners[0]))
        axis.plot(closed[:, 0], closed[:, 1], color="black", linewidth=3)
        axis.scatter(center[0], center[1], marker="x", color="black", s=130, linewidths=3)
    axis.set(title="Certified failure trajectories", xlabel="x (m)", ylabel="y (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(state_time, rollout.minimum_clearance, color="#1f77b4", linewidth=2)
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.axvline(state_time[failure_state], color="#d62728", linewidth=2, linestyle=":", label="failure")
    axis.set(title="Minimum pairwise oriented-box clearance", xlabel="simulation time (s)", ylabel="signed clearance (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "clearance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    labels = ((0, "acceleration", "m/s²"), (1, "steering", "rad"))
    for axis, (component, name, unit) in zip(axes, labels, strict=True):
        axis.plot(action_time, rollout.nominal_commands[:, 0, component], label="victim nominal", linewidth=1.8)
        axis.plot(action_time, rollout.applied_commands[:, 0, component], label="applied command", linewidth=1.8)
        axis.plot(action_time, rollout.disturbance_effective[:, component], label="effective disturbance", linewidth=1.4)
        axis.axvline(action_time[rollout.failure_timestep], color="#d62728", linestyle=":", linewidth=2)
        axis.set(ylabel=f"{name} ({unit})")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("simulation time before action (s)")
    fig.suptitle("Focal slot-0 controls and disturbance")
    fig.tight_layout()
    fig.savefig(output / "controls.png", dpi=180)
    plt.close(fig)

    _require(isinstance(frames, list) and len(frames) == rollout.transition_count + 1, "rendered frame clock does not match trace")
    images = [Image.fromarray(frame) for frame in frames]
    images[failure_state].save(output / "failure-frame.png")
    images[0].save(
        output / "failure.gif",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        optimize=False,
    )

def render_highway_failure(
    *,
    source: Path,
    run: Path,
    checkpoint: str | Path,
    output: Path,
    checkpoint_directory: Path | None = None,
    victim_pin_path: Path | None = None,
    gpudrive_pin_path: Path | None = None,
    zoom_radius: int = 70,
) -> dict[str, Any]:
    """Replay a deterministic learned failure twice and export visual evidence."""

    _require(platform.system() == "Linux", "failure replay/rendering is Linux/CUDA only")
    _require(zoom_radius > 0, "zoom radius must be positive")
    _require(not output.exists(), f"visualization output already exists: {output}")
    run = run.resolve()
    validation = validate_multiagent_training_artifact(run)
    _require(validation["ok"], f"training artifact validation failed: {validation['failed_checks']}")
    run_manifest = _read_json(run / "manifest.json")
    _require(checkpoint_identity(load_victim_pin(victim_pin_path)) == run_manifest["parent_victim_checkpoint_id"], "current victim pin differs from the training run")
    checkpoint_path = resolve_checkpoint(run, checkpoint)
    checkpoint_manifest = _read_json(checkpoint_path / "manifest.json")
    iteration = checkpoint_manifest.get("iteration")
    _require(isinstance(iteration, int) and 1 <= iteration <= len(run_manifest["metrics"]), "checkpoint iteration is outside run history")
    _require(run_manifest["checkpoint_ids"][iteration - 1] == checkpoint_manifest.get("artifact_id"), "checkpoint does not belong to this training run")
    _require(run_manifest["metrics"][iteration - 1] == checkpoint_manifest.get("metrics"), "checkpoint metrics differ from the parent run")
    _require(run_manifest["config"] == checkpoint_manifest.get("config"), "checkpoint and run configurations differ")
    expected_failure = checkpoint_manifest["metrics"]["deterministic_evaluation"].get("failure_timestep")
    _require(isinstance(expected_failure, int), "selected checkpoint was not recorded as a deterministic failure")

    source = source.resolve()
    pins = load_pins(gpudrive_pin_path)
    source_checks = verify_source_tree(source, pins)
    _require(required_checks_pass(source_checks), "pinned GPUDrive source verification failed")
    victim_pin = load_victim_pin(victim_pin_path)
    victim_checkpoint = checkpoint_directory.resolve() if checkpoint_directory else default_checkpoint_directory(victim_pin).resolve()
    victim_report = verify_checkpoint(victim_checkpoint, victim_pin, gpudrive_source=source)
    _require(victim_report["ok"], "victim checkpoint verification failed")
    _prepend_source_paths(source)
    try:
        import gpudrive
        import madrona_gpudrive
        import torch
        from gpudrive.datatypes.observation import AGENT_SCALE
        from gpudrive.env.config import EnvConfig, RenderConfig
        from gpudrive.env.dataset import SceneDataLoader
        from gpudrive.env.env_torch import GPUDriveTorchEnv
    except Exception as exc:
        raise MultiAgentReplayError(f"native imports failed: {type(exc).__name__}: {exc}") from exc

    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"}, "CUBLAS_WORKSPACE_CONFIG must be set before CUDA starts")
    _require(torch.cuda.is_available(), "CUDA is required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = int(run_manifest["config"]["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _require(Path(gpudrive.__file__).resolve().is_relative_to(source), "imported GPUDrive from wrong source")
    _require(Path(madrona_gpudrive.__file__).resolve().is_relative_to(source), "imported native extension from wrong source")

    temporary = output.resolve().parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    env = None
    try:
        scene_spec = run_manifest["files"]["derived_scene"]
        derived_scene = (run / scene_spec["relative_path"]).resolve()
        _require(derived_scene.is_relative_to(run), "derived scene path escapes the training artifact")
        _require(derived_scene.is_file() and sha256_file(derived_scene) == scene_spec["sha256"], "derived scene fingerprint changed")
        loader = SceneDataLoader(
            root=str(derived_scene.parent), batch_size=1, dataset_size=1,
            sample_with_replacement=False, file_prefix=derived_scene.name,
            seed=seed, shuffle=False,
        )
        env = GPUDriveTorchEnv(
            config=_env_config(EnvConfig, torch, victim_pin),
            data_loader=loader,
            max_cont_agents=AGENT_COUNT,
            device="cuda",
            action_type="discrete",
            render_config=RenderConfig(
                draw_expert_trajectories=False,
                draw_only_controllable_veh=True,
                obj_idx_font_size=10,
                render_3d=False,
            ),
        )
        env.vis.figsize = (8, 8)
        table = _to_numpy(env.action_keys_tensor).astype("<f4", copy=False)
        _require(np.array_equal(table, pinned_action_table(victim_pin["environment"])), "victim action table changed")
        victim_model = load_frozen_policy(victim_checkpoint, victim_pin["model_config"], device="cuda")
        victim_state_hash = torch_module_state_sha256(victim_model)
        _require(victim_state_hash == run_manifest["victim_state_sha256_after"], "victim model differs from the training run")
        victim = MultiAgentVictimPolicy(victim_model, table, torch, device="cuda")
        expected_ids = run_manifest["experiment"]["scene"]["selected_object_ids"]
        backend = GPUDriveTenAgentBackend(
            env, torch, device="cuda", horizon=91, expected_ids=expected_ids,
            scenario_id=run_manifest["experiment"]["scene"]["scenario_id"],
            agent_scale=float(AGENT_SCALE),
        )

        loaded = load_adversary_checkpoint(checkpoint_path, "cuda")
        loaded.model.eval()
        loaded.model.requires_grad_(False)
        config = loaded.manifest["config"]
        bounds = np.asarray(config["intervention"]["bounds"], dtype=np.float64)
        prior = BoundedTanhNormal(bounds, np.asarray(config["prior"]["base_std"], dtype=np.float64))
        adversary = _TorchAdversaryAdapter(
            loaded.model, prior, np.asarray(config["prior"]["base_mean"]),
            bounds.astype(np.float32), torch, device="cuda",
        )
        intervention = _intervention(config)
        first = run_multiagent_rollout(
            backend=backend, victim=victim, adversary=adversary,
            intervention=intervention, max_steps=91, adversary_deterministic=True,
        )
        _require(first.failure_timestep is not None, "selected checkpoint did not reproduce a deterministic failure")
        _require(first.failure_timestep == expected_failure, "replayed failure timestep differs from the training-time deterministic evaluation")

        capturing_backend = _FrameCaptureBackend(backend, zoom_radius=zoom_radius)
        second = run_multiagent_rollout(
            backend=capturing_backend, victim=victim, adversary=adversary,
            intervention=intervention, max_steps=91, adversary_deterministic=True,
        )
        comparison = compare_failure_replays(first, second)
        _require(comparison["ok"], "repeat deterministic replay did not match")

        arrays = _episode_arrays(second, prior, config)
        trace_path = temporary / "failure-trace.npz"
        np.savez_compressed(trace_path, **_payload(second, arrays))
        _write_diagnostic_plots(
            temporary,
            second,
            object_ids=expected_ids,
            frames=capturing_backend.frames,
        )

        files = {}
        for name in (
            "failure-trace.npz", "failure.gif", "failure-frame.png",
            "trajectories.png", "clearance.png", "controls.png",
        ):
            path = temporary / name
            _require(path.is_file() and path.stat().st_size > 0, f"renderer did not create {name}")
            files[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        manifest: dict[str, Any] = {
            "schema": "gpudrive_highway_failure_visualization",
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact_id": None,
            "parent_run_artifact_id": run_manifest["artifact_id"],
            "checkpoint_artifact_id": loaded.manifest["artifact_id"],
            "checkpoint_iteration": iteration,
            "victim_checkpoint_id": checkpoint_identity(victim_pin),
            "scene_identity": run_manifest["scene_identity"],
            "failure": failure_signature(second),
            "replay_certificate": comparison,
            "render": {"fps": 10, "frame_count": len(capturing_backend.frames), "zoom_radius_m": zoom_radius},
            "files": files,
            "runtime": {
                "platform": platform.platform(), "python": sys.version,
                "numpy": np.__version__, "torch": torch.__version__,
                "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            },
        }
        manifest["artifact_id"] = "highway-failure-viz-" + canonical_json_sha256(manifest)[:16]
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(output.resolve())
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        if env is not None:
            env.close()
