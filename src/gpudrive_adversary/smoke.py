"""Pinned one-scene GPUDrive smoke and deterministic replay comparison."""

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

from .pins import (
    PinError,
    canonical_json_sha256,
    load_pins,
    load_smoke_config,
    repository_root,
    required_checks_pass,
    sha256_file,
    tree_sha256,
    verify_source_tree,
)
from .provenance import port_identity


class SmokeError(RuntimeError):
    """Raised when the native one-scene smoke contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def _prepend_source_paths(source: Path) -> None:
    for path in (source / "build", source):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy().copy()


def _capture(env: Any) -> dict[str, np.ndarray]:
    info = env.get_infos()
    return {
        "observations": _to_numpy(env.get_obs()),
        "absolute_state": _to_numpy(
            env.sim.absolute_self_observation_tensor().to_torch().clone()
        ),
        "rewards": _to_numpy(env.get_rewards()),
        "dones": _to_numpy(env.get_dones()),
        "raw_info": _to_numpy(env.sim.info_tensor().to_torch().clone()),
        "info_off_road": _to_numpy(info.off_road),
        "info_collided": _to_numpy(info.collided),
        "info_goal_achieved": _to_numpy(info.goal_achieved),
    }


def _run_command_sequence(
    env: Any,
    torch: Any,
    *,
    commands: list[dict[str, Any]],
    device: str,
    controlled_slot: int,
    max_slots: int,
) -> dict[str, np.ndarray]:
    env.reset()
    captures = [_capture(env)]
    action_indices: list[np.ndarray] = []
    requested_physical: list[np.ndarray] = []
    native_physical: list[np.ndarray] = []
    command_names: list[str] = []
    command_kinds: list[int] = []
    for command in commands:
        kind = command["kind"]
        indices = torch.full(
            (env.num_worlds, max_slots),
            -1,
            dtype=torch.long,
            device=device,
        )
        if kind == "discrete":
            index = int(command["index"])
            action = torch.full_like(indices, index)
            indices.copy_(action)
            physical = env.action_keys_tensor[action]
            expected = torch.tensor(
                command["expected_accel_steer_head"],
                dtype=physical.dtype,
                device=device,
            )
            _require(
                bool(torch.equal(physical[0, controlled_slot], expected)),
                f"{command['name']} decoded to {physical[0, controlled_slot].tolist()}",
            )
            kind_code = 0
        elif kind == "physical":
            physical = torch.zeros(
                (env.num_worlds, max_slots, 3),
                dtype=env.action_keys_tensor.dtype,
                device=device,
            )
            physical[0, controlled_slot] = torch.tensor(
                command["accel_steer_head"], dtype=physical.dtype, device=device
            )
            action = physical
            kind_code = 1
        else:
            raise SmokeError(f"unsupported smoke command kind: {kind!r}")

        env._apply_actions(action)  # Pinned action-transport regression assertion.
        native = env.sim.action_tensor().to_torch()[:, :, :3].clone()
        _require(
            bool(torch.equal(native, physical)),
            f"{command['name']} was reordered during native action transport",
        )
        env.step_dynamics(None)
        action_indices.append(_to_numpy(indices))
        requested_physical.append(_to_numpy(physical))
        native_physical.append(_to_numpy(native))
        command_names.append(str(command["name"]))
        command_kinds.append(kind_code)
        captures.append(_capture(env))

    sequence = {
        name: np.stack([capture[name] for capture in captures], axis=0)
        for name in captures[0]
    }
    sequence["requested_action_indices"] = np.stack(action_indices, axis=0)
    sequence["requested_physical_actions"] = np.stack(
        requested_physical, axis=0
    )
    sequence["native_physical_actions"] = np.stack(native_physical, axis=0)
    sequence["command_names"] = np.asarray(command_names, dtype=np.str_)
    sequence["command_kind_codes"] = np.asarray(command_kinds, dtype=np.int8)
    return sequence


def _require_info_mapping(sequence: dict[str, np.ndarray]) -> None:
    raw = sequence["raw_info"]
    mappings = {
        "road": (raw[..., 0], sequence["info_off_road"]),
        "vehicle_plus_nonvehicle": (
            raw[..., 1] + raw[..., 2],
            sequence["info_collided"],
        ),
        "goal": (raw[..., 3], sequence["info_goal_achieved"]),
    }
    mismatched = [
        name for name, (left, right) in mappings.items() if not np.array_equal(left, right)
    ]
    _require(not mismatched, f"raw/high-level info mapping mismatch: {mismatched}")


def compare_sequences(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
    equal_nan: bool,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise SmokeError(
            f"trace fields differ: left={sorted(left)}, right={sorted(right)}"
        )
    event_fields = {
        "command_kind_codes",
        "command_names",
        "controlled_mask",
        "dones",
        "info_collided",
        "info_goal_achieved",
        "info_off_road",
        "metadata",
        "native_physical_actions",
        "raw_info",
        "requested_action_indices",
        "requested_physical_actions",
    }
    fields: dict[str, Any] = {}
    overall = True
    for name in sorted(left):
        left_value = np.asarray(left[name])
        right_value = np.asarray(right[name])
        same_shape = left_value.shape == right_value.shape
        numeric = left_value.dtype.kind in "biufc" and right_value.dtype.kind in "biufc"
        exact = same_shape and (
            np.array_equal(left_value, right_value, equal_nan=True)
            if numeric
            else np.array_equal(left_value, right_value)
        )
        if same_shape and numeric:
            with np.errstate(invalid="ignore"):
                difference = np.abs(left_value.astype(np.float64) - right_value.astype(np.float64))
            finite_difference = difference[np.isfinite(difference)]
            max_abs = float(finite_difference.max()) if finite_difference.size else 0.0
            close = bool(
                np.allclose(
                    left_value,
                    right_value,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=equal_nan,
                )
            )
        else:
            max_abs = None
            close = exact
        passes = exact if name in event_fields else close
        overall = overall and passes
        fields[name] = {
            "ok": passes,
            "shape_equal": same_shape,
            "exact": exact,
            "allclose": close,
            "max_abs_difference": max_abs,
            "left_shape": list(left_value.shape),
            "right_shape": list(right_value.shape),
            "left_dtype": str(left_value.dtype),
            "right_dtype": str(right_value.dtype),
        }
    return {"ok": overall, "fields": fields, "rtol": rtol, "atol": atol}


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


def _read_manifest(artifact: Path) -> dict[str, Any]:
    try:
        return json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeError(f"cannot read smoke manifest from {artifact}: {exc}") from exc


def compare_smoke_artifacts(
    left_artifact: Path,
    right_artifact: Path,
    *,
    output: Path | None = None,
    pin_path: Path | None = None,
) -> dict[str, Any]:
    left_manifest = _read_manifest(left_artifact)
    right_manifest = _read_manifest(right_artifact)
    left_validation = validate_smoke_artifact(left_artifact, pin_path=pin_path)
    right_validation = validate_smoke_artifact(right_artifact, pin_path=pin_path)

    identity_fields = {
        "device": ("device",),
        "scene": ("scene",),
        "gpudrive_commit": ("fingerprints", "gpudrive_commit"),
        "gpudrive_tree": ("fingerprints", "gpudrive_tree"),
        "gpudrive_submodules": ("fingerprints", "gpudrive_submodules"),
        "gpudrive_uv_lock_sha256": (
            "fingerprints",
            "gpudrive_uv_lock_sha256",
        ),
        "scene_sha256": ("fingerprints", "scene_sha256"),
        "config_sha256": ("fingerprints", "config_sha256"),
        "native_extension_sha256": (
            "fingerprints",
            "native_extension_sha256",
        ),
        "kernel_cache_sha256": (
            "fingerprints",
            "madrona_kernel_cache_tree_sha256",
        ),
        "source_verification_sha256": (
            "fingerprints",
            "source_verification_sha256",
        ),
        "port": ("fingerprints", "port"),
        "runtime_platform": ("runtime", "platform"),
        "runtime_python": ("runtime", "python"),
        "runtime_torch": ("runtime", "torch"),
        "runtime_torch_cuda": ("runtime", "torch_cuda"),
        "runtime_gpu_names": ("runtime", "gpu_names"),
        "runtime_nvidia_smi": ("runtime", "nvidia_smi"),
        "reference_image_digest": ("runtime", "reference_image_digest"),
    }

    def nested(manifest: dict[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = manifest
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        return value

    identity = {
        name: {
            "ok": nested(left_manifest, path) == nested(right_manifest, path),
            "left": nested(left_manifest, path),
            "right": nested(right_manifest, path),
        }
        for name, path in identity_fields.items()
    }
    comparison_config = left_manifest["resolved_config"]["comparison"]
    trace_comparison = compare_sequences(
        _load_trace(left_artifact / "trace.npz"),
        _load_trace(right_artifact / "trace.npz"),
        rtol=float(comparison_config["rtol"]),
        atol=float(comparison_config["atol"]),
        equal_nan=bool(comparison_config["equal_nan"]),
    )
    result = {
        "schema": "gpudrive_fresh_process_smoke_comparison",
        "schema_version": 1,
        "ok": left_validation["ok"]
        and right_validation["ok"]
        and all(item["ok"] for item in identity.values())
        and trace_comparison["ok"],
        "left_artifact": str(left_artifact.resolve()),
        "right_artifact": str(right_artifact.resolve()),
        "artifact_validation": {
            "left": left_validation,
            "right": right_validation,
        },
        "identity": identity,
        "trace": trace_comparison,
    }
    if output is not None:
        if output.exists():
            raise SmokeError(f"comparison output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(output)
        finally:
            if temporary.exists():
                temporary.unlink()
    return result


def _nvidia_smi_identity() -> dict[str, Any]:
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


def _temporary_directory(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"


def _publish_directory(temporary: Path, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise SmokeError(f"output already exists: {output}")
    temporary.replace(output)


def _cleanup_temporary(temporary: Path) -> None:
    if temporary.exists() and temporary.name.startswith(".") and ".tmp-" in temporary.name:
        shutil.rmtree(temporary)


def run_scene_smoke(
    *,
    source: Path,
    output: Path,
    device: str,
    pin_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run one pinned scene twice in one process and publish an artifact."""

    if device not in {"cpu", "cuda"}:
        raise SmokeError("device must be exactly 'cpu' or 'cuda'")
    if output.exists():
        raise SmokeError(f"output already exists: {output}")

    pins = load_pins(pin_path)
    config = load_smoke_config(config_path)
    source = source.resolve()
    source_checks = verify_source_tree(source, pins)
    if not required_checks_pass(source_checks):
        failed = [check["name"] for check in source_checks if check["required"] and not check["ok"]]
        raise SmokeError(f"pinned GPUDrive source verification failed: {failed}")

    _prepend_source_paths(source)
    try:
        import torch
        import madrona_gpudrive
        import gpudrive
        from gpudrive.datatypes.metadata import Metadata
        from gpudrive.datatypes.observation import GlobalEgoState
        from gpudrive.env.config import EnvConfig
        from gpudrive.env.dataset import SceneDataLoader
        from gpudrive.env.env_torch import GPUDriveTorchEnv
    except Exception as exc:
        raise SmokeError(
            "GPUDrive native imports failed. Build the pinned source in the reference "
            f"container first: {type(exc).__name__}: {exc}"
        ) from exc

    gpudrive_file = Path(gpudrive.__file__).resolve()
    _require(
        gpudrive_file.is_relative_to(source),
        f"imported gpudrive from the wrong checkout: {gpudrive_file}",
    )
    if device == "cuda":
        _require(torch.cuda.is_available(), "device=cuda but torch.cuda.is_available() is false")

    scene_pin = pins["smoke_scene"]
    scene_path = source / scene_pin["relative_path"]
    loader_config = config["loader"]
    environment_config = config["environment"]
    action_config = config["action"]
    expected = config["expected"]

    loader = SceneDataLoader(
        root=str(scene_path.parent),
        batch_size=int(loader_config["batch_size"]),
        dataset_size=int(loader_config["dataset_size"]),
        sample_with_replacement=bool(loader_config["sample_with_replacement"]),
        file_prefix=scene_path.name,
        seed=int(loader_config["seed"]),
        shuffle=bool(loader_config["shuffle"]),
    )
    _require(loader.dataset == [str(scene_path)], f"loader did not select only {scene_path}")

    env_config = EnvConfig(
        num_worlds=int(environment_config["num_worlds"]),
        max_controlled_agents=int(environment_config["max_cont_agents"]),
        dynamics_model=environment_config["dynamics_model"],
        collision_behavior=environment_config["collision_behavior"],
        init_mode=environment_config["init_mode"],
        init_steps=int(environment_config["init_steps"]),
        remove_non_vehicles=bool(environment_config["remove_non_vehicles"]),
        reward_type=environment_config["reward_type"],
        ego_state=bool(environment_config["ego_state"]),
        partner_obs=bool(environment_config["partner_obs"]),
        road_map_obs=bool(environment_config["road_map_obs"]),
        norm_obs=bool(environment_config["norm_obs"]),
    )
    env = None
    try:
        env = GPUDriveTorchEnv(
            config=env_config,
            data_loader=loader,
            max_cont_agents=int(environment_config["max_cont_agents"]),
            device=device,
            action_type=action_config["type"],
        )
        initial_obs = env.reset()
        controlled = env.cont_agent_mask
        max_slots = int(expected["max_agent_slots"])
        controlled_slot = int(expected["controlled_slot"])
        _require(tuple(controlled.shape) == (1, max_slots), f"control mask shape={controlled.shape}")
        _require(int(controlled.sum().item()) == int(expected["controlled_agents"]), "expected exactly one controlled agent")
        _require(bool(controlled[0, controlled_slot].item()), "expected controlled SDC in slot 0")

        metadata = Metadata.from_tensor(env.sim.metadata_tensor(), backend="torch")
        _require(int(metadata.is_sdc[0, controlled_slot].item()) == 1, "controlled slot is not the SDC")
        state = GlobalEgoState.from_tensor(
            env.sim.absolute_self_observation_tensor(),
            backend="torch",
            device=device,
        )
        observed_sdc_id = int(state.id[0, controlled_slot].item())
        _require(observed_sdc_id == int(scene_pin["sdc_object_id"]), f"SDC id={observed_sdc_id}")
        _require(env.get_scenario_ids()[0] == scene_pin["scenario_id"], "scenario ID mismatch")

        masked_obs = env.get_obs(controlled)
        observation_width = int(expected["observation_width"])
        _require(tuple(initial_obs.shape) == (1, max_slots, observation_width), f"obs shape={initial_obs.shape}")
        _require(tuple(masked_obs.shape) == (1, observation_width), f"masked obs shape={masked_obs.shape}")
        _require(bool(torch.isfinite(initial_obs).all().item()), "initial observations contain non-finite values")

        raw_info = env.sim.info_tensor().to_torch()
        _require(tuple(raw_info.shape) == (1, max_slots, int(expected["raw_info_width"])), f"raw info shape={raw_info.shape}")
        _require(env.action_space.n == int(expected["discrete_actions"]), f"action count={env.action_space.n}")
        neutral_index = int(action_config["neutral_index"])
        neutral_decoded = torch.tensor(
            action_config["neutral_decoded"], dtype=env.action_keys_tensor.dtype, device=device
        )
        _require(
            bool(torch.equal(env.action_keys_tensor[neutral_index], neutral_decoded)),
            f"neutral action {neutral_index} decodes to {env.action_keys_tensor[neutral_index].tolist()}",
        )

        commands = action_config["commands"]
        _require(bool(commands), "smoke command list must not be empty")
        first = _run_command_sequence(
            env,
            torch,
            commands=commands,
            device=device,
            controlled_slot=controlled_slot,
            max_slots=max_slots,
        )
        second = _run_command_sequence(
            env,
            torch,
            commands=commands,
            device=device,
            controlled_slot=controlled_slot,
            max_slots=max_slots,
        )
        _require_info_mapping(first)
        _require_info_mapping(second)
        comparison_config = config["comparison"]
        same_process = compare_sequences(
            first,
            second,
            rtol=float(comparison_config["rtol"]),
            atol=float(comparison_config["atol"]),
            equal_nan=bool(comparison_config["equal_nan"]),
        )
        _require(same_process["ok"], "same-process reset replay mismatch")
        initial_xy = first["absolute_state"][0, 0, controlled_slot, :2]
        final_xy = first["absolute_state"][-1, 0, controlled_slot, :2]
        _require(not np.array_equal(initial_xy, final_xy), "SDC position did not change")

        first["controlled_mask"] = _to_numpy(controlled)
        first["metadata"] = _to_numpy(env.sim.metadata_tensor().to_torch().clone())
    finally:
        if env is not None:
            env.close()

    native_file = Path(madrona_gpudrive.__file__).resolve()
    cache_path = os.environ.get("MADRONA_MWGPU_KERNEL_CACHE")
    port = port_identity(repository_root())
    transport = [
        {
            "name": str(first["command_names"][index]),
            "kind_code": int(first["command_kind_codes"][index]),
            "requested_accel_steer_head": first["requested_physical_actions"][
                index, 0, controlled_slot
            ].tolist(),
            "observed_native_accel_steer_head": first["native_physical_actions"][
                index, 0, controlled_slot
            ].tolist(),
        }
        for index in range(len(commands))
    ]

    temporary = _temporary_directory(output)
    temporary.mkdir()
    trace_path = temporary / "trace.npz"
    try:
        np.savez_compressed(trace_path, **first)
        manifest = {
            "schema": "gpudrive_scene_smoke_artifact",
            "schema_version": 1,
            "artifact_id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": config["purpose"],
            "device": device,
            "scene": {
                "relative_path": scene_pin["relative_path"],
                "scenario_id": scene_pin["scenario_id"],
                "sdc_object_id": scene_pin["sdc_object_id"],
                "controlled_slot": controlled_slot,
                "loader_seed": loader_config["seed"],
            },
            "resolved_config": config,
            "same_process_replay": same_process,
            "action_transport": {
                "ok": all(
                    item["requested_accel_steer_head"]
                    == item["observed_native_accel_steer_head"]
                    for item in transport
                ),
                "declared_order": expected["physical_action_order"],
                "commands": transport,
            },
            "raw_info_order": expected["raw_info_order"],
            "raw_info_mapping_verified": True,
            "termination_reason": None,
            "termination_reason_explanation": "Milestone A does not implement a research failure or termination wrapper.",
            "failure_timestep": None,
            "failure_definition": None,
            "failure_definition_explanation": config[
                "failure_definition_reason"
            ],
            "victim_checkpoint": None,
            "victim_checkpoint_explanation": "Victim policy work begins in Milestone B.",
            "adversary_checkpoint": None,
            "adversary_checkpoint_explanation": "Adversary work begins in Milestone C.",
            "fingerprints": {
                "gpudrive_commit": pins["gpudrive"]["commit"],
                "gpudrive_tree": pins["gpudrive"]["tree"],
                "gpudrive_submodules": pins["gpudrive"]["submodules"],
                "gpudrive_uv_lock_sha256": pins["gpudrive"][
                    "uv_lock_sha256"
                ],
                "scene_sha256": pins["smoke_scene"]["sha256"],
                "config_sha256": canonical_json_sha256(config),
                "trace_sha256": sha256_file(trace_path),
                "native_extension_sha256": sha256_file(native_file),
                "madrona_kernel_cache_tree_sha256": tree_sha256(cache_path)
                if cache_path
                else None,
                "source_verification_sha256": canonical_json_sha256(
                    source_checks
                ),
                "port": port,
            },
            "runtime": {
                "platform": platform.platform(),
                "python": sys.version,
                "python_executable": sys.executable,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "nvidia_smi": _nvidia_smi_identity(),
                "reference_image_digest": os.environ.get(
                    "GPUDRIVE_REFERENCE_IMAGE_DIGEST"
                ),
                "gpudrive_python_file": str(gpudrive_file),
                "native_extension_file": str(native_file),
                "madrona_kernel_cache": cache_path,
            },
            "source_verification": source_checks,
        }
        manifest["artifact_id"] = "scene-smoke-" + canonical_json_sha256(
            manifest
        )[:16]
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        validation = validate_smoke_artifact(temporary, pin_path=pin_path)
        if not validation["ok"]:
            failed = [name for name, ok in validation["checks"].items() if not ok]
            raise SmokeError(f"generated smoke artifact failed validation: {failed}")
        _publish_directory(temporary, output)
        return manifest
    except Exception:
        _cleanup_temporary(temporary)
        raise


def run_fresh_process_smoke(
    *,
    source: Path,
    output: Path,
    device: str,
    pin_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run two isolated smoke processes and compare their typed traces."""

    if output.exists():
        raise SmokeError(f"output already exists: {output}")
    temporary = _temporary_directory(output)
    temporary.mkdir()
    try:
        runs = [temporary / "run-1", temporary / "run-2"]
        for run_output in runs:
            command = [
                sys.executable,
                "-m",
                "gpudrive_adversary",
                "scene-smoke",
                "--source",
                str(source),
                "--output",
                str(run_output),
                "--device",
                device,
            ]
            if pin_path is not None:
                command.extend(["--pins", str(pin_path)])
            if config_path is not None:
                command.extend(["--config", str(config_path)])
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise SmokeError(
                    "fresh smoke process failed with exit code "
                    f"{result.returncode}: {command}"
                )
        comparison = compare_smoke_artifacts(
            runs[0],
            runs[1],
            output=temporary / "comparison.json",
            pin_path=pin_path,
        )
        if not comparison["ok"]:
            raise SmokeError("fresh-process smoke traces do not match")
        comparison["left_artifact"] = str((output / "run-1").resolve())
        comparison["right_artifact"] = str((output / "run-2").resolve())
        (temporary / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
        )
        _publish_directory(temporary, output)
        return comparison
    except Exception:
        _cleanup_temporary(temporary)
        raise


def validate_smoke_artifact(
    artifact: Path, *, pin_path: Path | None = None
) -> dict[str, Any]:
    manifest = _read_manifest(artifact)
    pins = load_pins(pin_path)
    trace_path = artifact / "trace.npz"

    def is_hex(value: Any, length: int) -> bool:
        return (
            isinstance(value, str)
            and len(value) == length
            and all(character in "0123456789abcdef" for character in value)
        )

    fingerprints = manifest.get("fingerprints", {})
    config = manifest.get("resolved_config")
    scene = manifest.get("scene", {})
    runtime = manifest.get("runtime", {})
    port = fingerprints.get("port", {})
    source_checks = manifest.get("source_verification", [])

    trace: dict[str, np.ndarray] = {}
    trace_readable = False
    if trace_path.is_file():
        try:
            trace = _load_trace(trace_path)
            trace_readable = True
        except (OSError, ValueError):
            pass
    required_trace_fields = {
        "absolute_state",
        "command_kind_codes",
        "command_names",
        "controlled_mask",
        "dones",
        "info_collided",
        "info_goal_achieved",
        "info_off_road",
        "metadata",
        "native_physical_actions",
        "observations",
        "raw_info",
        "requested_action_indices",
        "requested_physical_actions",
        "rewards",
    }
    trace_fields_present = trace_readable and required_trace_fields.issubset(trace)
    trace_shapes_valid = False
    trace_info_mapping = False
    trace_sdc_binding = False
    trace_command_contract = False
    trace_action_transport = False
    if trace_fields_present:
        state_count = trace["raw_info"].shape[0]
        action_count = trace["requested_physical_actions"].shape[0]
        trace_shapes_valid = (
            trace["raw_info"].ndim == 4
            and trace["raw_info"].shape[-1] == 5
            and state_count == action_count + 1
            and trace["requested_physical_actions"].shape
            == trace["native_physical_actions"].shape
            and trace["command_names"].shape == (action_count,)
            and trace["command_kind_codes"].shape == (action_count,)
        )
        try:
            _require_info_mapping(trace)
            trace_info_mapping = True
        except SmokeError:
            pass
        try:
            controlled_slot = int(scene["controlled_slot"])
            trace_sdc_binding = (
                int(trace["controlled_mask"].sum()) == 1
                and bool(trace["controlled_mask"][0, controlled_slot])
                and int(trace["absolute_state"][0, 0, controlled_slot, 13])
                == int(pins["smoke_scene"]["sdc_object_id"])
            )
        except (IndexError, KeyError, TypeError, ValueError):
            pass
        try:
            configured_commands = config["action"]["commands"]
            command_values_match = True
            for index, command in enumerate(configured_commands):
                expected_values = command.get(
                    "expected_accel_steer_head", command.get("accel_steer_head")
                )
                command_values_match = command_values_match and np.array_equal(
                    trace["requested_physical_actions"][index, 0, controlled_slot],
                    np.asarray(expected_values),
                )
                expected_kind = 0 if command["kind"] == "discrete" else 1
                command_values_match = command_values_match and int(
                    trace["command_kind_codes"][index]
                ) == expected_kind
                expected_index = command.get("index", -1)
                command_values_match = command_values_match and int(
                    trace["requested_action_indices"][index, 0, controlled_slot]
                ) == expected_index
            trace_command_contract = (
                trace["command_names"].tolist()
                == [command["name"] for command in configured_commands]
                and command_values_match
            )
            trace_action_transport = np.array_equal(
                trace["requested_physical_actions"],
                trace["native_physical_actions"],
            )
        except (KeyError, TypeError):
            pass

    artifact_id_manifest = dict(manifest)
    artifact_id_manifest["artifact_id"] = None
    expected_artifact_id = "scene-smoke-" + canonical_json_sha256(
        artifact_id_manifest
    )[:16]

    checks = {
        "schema": manifest.get("schema") == "gpudrive_scene_smoke_artifact"
        and manifest.get("schema_version") == 1,
        "artifact_id": manifest.get("artifact_id") == expected_artifact_id,
        "trace_exists": trace_path.is_file(),
        "trace_readable_without_pickle": trace_readable,
        "trace_fields_present": trace_fields_present,
        "trace_shapes_valid": trace_shapes_valid,
        "trace_info_mapping": trace_info_mapping,
        "trace_sdc_binding": trace_sdc_binding,
        "trace_command_contract": trace_command_contract,
        "trace_action_transport": trace_action_transport,
        "trace_sha256": trace_path.is_file()
        and sha256_file(trace_path) == fingerprints.get("trace_sha256"),
        "config_fingerprint": isinstance(config, dict)
        and canonical_json_sha256(config) == fingerprints.get("config_sha256"),
        "config_has_no_failure_definition": isinstance(config, dict)
        and config.get("failure_definition") is None,
        "scene_identity": scene.get("scenario_id")
        == pins["smoke_scene"]["scenario_id"]
        and scene.get("sdc_object_id") == pins["smoke_scene"]["sdc_object_id"]
        and scene.get("relative_path") == pins["smoke_scene"]["relative_path"],
        "scene_fingerprint": fingerprints.get("scene_sha256")
        == pins["smoke_scene"]["sha256"],
        "gpudrive_commit": fingerprints.get("gpudrive_commit")
        == pins["gpudrive"]["commit"],
        "gpudrive_tree": fingerprints.get("gpudrive_tree")
        == pins["gpudrive"]["tree"],
        "gpudrive_submodules": fingerprints.get("gpudrive_submodules")
        == pins["gpudrive"]["submodules"],
        "gpudrive_lock": fingerprints.get("gpudrive_uv_lock_sha256")
        == pins["gpudrive"]["uv_lock_sha256"],
        "native_extension_fingerprint": is_hex(
            fingerprints.get("native_extension_sha256"), 64
        ),
        "kernel_cache_fingerprint": is_hex(
            fingerprints.get("madrona_kernel_cache_tree_sha256"), 64
        ),
        "source_verification_complete": isinstance(source_checks, list)
        and bool(source_checks)
        and required_checks_pass(source_checks),
        "source_verification_fingerprint": isinstance(source_checks, list)
        and canonical_json_sha256(source_checks)
        == fingerprints.get("source_verification_sha256"),
        "port_commit": is_hex(port.get("commit"), 40),
        "port_dirty_state": isinstance(port.get("dirty"), bool),
        "port_diff_fingerprint": is_hex(port.get("diff_sha256"), 64),
        "port_source_tree_fingerprint": is_hex(
            port.get("source_tree_sha256"), 64
        ),
        "port_declared_tree_match": is_hex(
            port.get("declared_source_tree_sha256"), 64
        )
        and port.get("declared_source_tree_sha256")
        == port.get("source_tree_sha256")
        and port.get("source_tree_matches_declared") is True,
        "reference_image": runtime.get("reference_image_digest")
        == pins["reference_runtime"]["base_image_amd64_digest"],
        "reference_python": isinstance(runtime.get("python"), str)
        and runtime["python"].startswith(pins["reference_runtime"]["python"]),
        "reference_torch_cuda": runtime.get("torch_cuda")
        == pins["reference_runtime"]["cuda"],
        "cuda_runtime_available": runtime.get("cuda_available") is True
        and bool(runtime.get("gpu_names")),
        "driver_identity": runtime.get("nvidia_smi", {}).get("returncode") == 0
        and bool(runtime.get("nvidia_smi", {}).get("rows")),
        "same_process_replay": manifest.get("same_process_replay", {}).get("ok")
        is True,
        "action_transport": manifest.get("action_transport", {}).get("ok") is True,
        "action_order": isinstance(config, dict)
        and manifest.get("action_transport", {}).get("declared_order")
        == config.get("expected", {}).get("physical_action_order"),
        "raw_info_mapping": manifest.get("raw_info_mapping_verified") is True,
        "raw_info_order": isinstance(config, dict)
        and manifest.get("raw_info_order")
        == config.get("expected", {}).get("raw_info_order"),
        "failure_unset": manifest.get("failure_timestep") is None
        and manifest.get("failure_definition") is None
        and bool(manifest.get("failure_definition_explanation")),
        "termination_unset": manifest.get("termination_reason") is None
        and bool(manifest.get("termination_reason_explanation")),
        "victim_checkpoint_explicit_null": "victim_checkpoint" in manifest
        and manifest["victim_checkpoint"] is None
        and bool(manifest.get("victim_checkpoint_explanation")),
        "adversary_checkpoint_explicit_null": "adversary_checkpoint" in manifest
        and manifest["adversary_checkpoint"] is None
        and bool(manifest.get("adversary_checkpoint_explanation")),
    }
    return {
        "schema": "gpudrive_scene_smoke_validation",
        "schema_version": 1,
        "ok": all(checks.values()),
        "artifact": str(artifact.resolve()),
        "checks": checks,
    }
