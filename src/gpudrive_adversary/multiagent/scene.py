"""Pinned construction of a ten-vehicle scene with no logged actors."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from ..pins import canonical_json_sha256, repository_root, sha256_file


class MultiAgentSceneError(ValueError):
    """Raised when the derived-scene identity is not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiAgentSceneError(message)


def default_highway_config_path() -> Path:
    return repository_root() / "configs/multiagent/highway_10agent.json"


def load_highway_experiment_config(path: Path | str | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else default_highway_config_path()
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultiAgentSceneError(f"cannot load highway experiment config: {exc}") from exc
    _require(config.get("schema") == "gpudrive_highway_10agent_experiment", "unsupported highway experiment schema")
    _require(config.get("schema_version") == 1, "unsupported highway experiment version")
    scene = config.get("scene", {})
    dataset = config.get("dataset", {})
    _require(dataset.get("repository") == "EMERGE-lab/GPUDrive_mini", "unexpected scene dataset")
    _require(isinstance(dataset.get("revision"), str) and len(dataset["revision"]) == 40, "immutable dataset revision is required")
    _require(dataset.get("split") == "validation", "highway pilot must use the pinned validation split")
    _require(scene.get("source_relative_path", "").startswith("validation/"), "scene path must belong to the validation split")
    ids = scene.get("selected_object_ids")
    _require(isinstance(ids, list) and len(ids) == 10 and len(set(ids)) == 10, "exactly ten unique object IDs are required")
    _require(all(isinstance(value, int) and not isinstance(value, bool) for value in ids), "object IDs must be integers")
    _require(scene.get("focal_object_id") == ids[0], "the focal SDC must be first in selected IDs")
    _require(isinstance(scene.get("derived_canonical_sha256"), str) and len(scene["derived_canonical_sha256"]) == 64, "derived scene hash is required")
    _require(config.get("control", {}).get("controlled_agents") == 10, "exactly ten PPO agents are required")
    _require(config["control"].get("remove_all_other_dynamic_objects") is True, "logged actors must be removed")
    _require(config["control"].get("focal_slot") == 0, "the disturbed agent must be slot 0")
    _require(config.get("intervention", {}).get("bounds") == [0.667, 0.262], "approved disturbance bounds changed")
    _require(config.get("eligibility", {}).get("all_ten_clean_goal_success") is True, "all ten clean goals are required")
    _require(config.get("failure", {}).get("scope") == "any_controlled_agent", "global controlled-agent failure is required")
    _require(config.get("reward", {}).get("nonfailure_shaping") == "terminal_minimum_signed_obb_clearance", "clearance reward changed")
    return config


def default_highway_source_path(config: dict[str, Any] | None = None) -> Path:
    """Return the non-vendored local cache path for the pinned source scene."""

    values = config if config is not None else load_highway_experiment_config()
    return repository_root() / ".deps/datasets/GPUDrive_mini" / values["scene"]["source_relative_path"]


def _initial_speed(obj: dict[str, Any]) -> float:
    velocity = obj["velocity"][0]
    return math.hypot(float(velocity["x"]), float(velocity["y"]))


def _recorded_displacement(obj: dict[str, Any]) -> tuple[float, int]:
    valid_indices = [index for index, valid in enumerate(obj["valid"]) if valid]
    _require(bool(valid_indices), f"vehicle {obj.get('id')} has no valid states")
    last = valid_indices[-1]
    start_position = obj["position"][0]
    end_position = obj["position"][last]
    displacement = math.hypot(
        float(end_position["x"]) - float(start_position["x"]),
        float(end_position["y"]) - float(start_position["y"]),
    )
    return displacement, len(valid_indices)


def selected_highway_object_ids(source: dict[str, Any], config: dict[str, Any]) -> list[int]:
    """Recompute the declared SDC-plus-nearest-nine selection from source data."""

    scene_config = config["scene"]
    selection = scene_config["selection"]
    objects = source["objects"]
    sdc_index = source["metadata"]["sdc_track_index"]
    _require(isinstance(sdc_index, int) and 0 <= sdc_index < len(objects), "source SDC index is invalid")
    sdc = objects[sdc_index]
    _require(sdc.get("type") == "vehicle" and sdc.get("valid", [False])[0], "source SDC is not a reset-valid vehicle")
    origin = sdc["position"][0]
    candidates: list[tuple[float, int]] = []
    for obj in objects:
        valid = obj.get("valid")
        if (
            obj.get("type") != "vehicle"
            or obj.get("mark_as_expert") is not False
            or not isinstance(valid, list)
            or len(valid) != 91
            or not valid[0]
        ):
            continue
        displacement, valid_count = _recorded_displacement(obj)
        if (
            _initial_speed(obj) < float(selection["minimum_initial_speed_mps"])
            or displacement < float(selection["minimum_recorded_displacement_m"])
            or valid_count < int(selection["minimum_valid_samples"])
        ):
            continue
        position = obj["position"][0]
        distance = math.hypot(
            float(position["x"]) - float(origin["x"]),
            float(position["y"]) - float(origin["y"]),
        )
        candidates.append((distance, int(obj["id"])))
    candidate_ids = [object_id for _, object_id in sorted(candidates, key=lambda item: (item[0], item[1]))]
    sdc_id = int(sdc["id"])
    _require(sdc_id in candidate_ids, "source SDC does not satisfy the declared highway selection rule")
    nearest_others = [object_id for object_id in candidate_ids if object_id != sdc_id][:9]
    _require(len(nearest_others) == 9, "source scene has fewer than nine suitable neighboring vehicles")
    return [sdc_id, *nearest_others]


def build_derived_scene(source_path: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact ten-object JSON document and its identity record."""

    scene_config = config["scene"]
    _require(source_path.is_file(), f"source scene does not exist: {source_path}")
    observed_source_hash = sha256_file(source_path)
    _require(observed_source_hash == scene_config["source_sha256"], "source highway scene hash mismatch")
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultiAgentSceneError(f"cannot parse source scene: {exc}") from exc
    _require(source.get("scenario_id") == scene_config["scenario_id"], "source scenario ID mismatch")
    objects = source.get("objects")
    _require(isinstance(objects, list), "source scene objects are missing")
    by_id: dict[int, tuple[int, dict[str, Any]]] = {}
    for index, obj in enumerate(objects):
        if isinstance(obj, dict) and isinstance(obj.get("id"), int):
            _require(obj["id"] not in by_id, "source object IDs are not unique")
            by_id[obj["id"]] = (index, obj)
    computed_ids = selected_highway_object_ids(source, config)
    _require(computed_ids == scene_config["selected_object_ids"], "selected vehicles do not match the declared deterministic rule")
    selected: list[dict[str, Any]] = []
    source_indices: list[int] = []
    for object_id in scene_config["selected_object_ids"]:
        _require(object_id in by_id, f"selected object {object_id} is absent")
        source_index, obj = by_id[object_id]
        valid = obj.get("valid")
        _require(obj.get("type") == "vehicle", f"selected object {object_id} is not a vehicle")
        _require(isinstance(valid, list) and len(valid) == 91 and valid[0], f"selected vehicle {object_id} is not valid at reset")
        _require(obj.get("mark_as_expert") is False, f"selected vehicle {object_id} is not controllable")
        selected.append(copy.deepcopy(obj))
        source_indices.append(source_index)
    source_sdc_index = source.get("metadata", {}).get("sdc_track_index")
    _require(source_indices[0] == source_sdc_index, "first selected vehicle is not the source SDC")

    derived = copy.deepcopy(source)
    derived["name"] = scene_config["derived_name"]
    derived["objects"] = selected
    metadata = copy.deepcopy(source.get("metadata", {}))
    metadata["sdc_track_index"] = 0
    metadata["objects_of_interest"] = []
    metadata["tracks_to_predict"] = []
    derived["metadata"] = metadata
    payload_hash = canonical_json_sha256(derived)
    expected = scene_config.get("derived_canonical_sha256")
    if expected is not None:
        _require(payload_hash == expected, "derived scene canonical hash mismatch")
    identity = {
        "source_relative_path": scene_config["source_relative_path"],
        "dataset_repository": config["dataset"]["repository"],
        "dataset_revision": config["dataset"]["revision"],
        "source_sha256": observed_source_hash,
        "source_scenario_id": source["scenario_id"],
        "source_object_indices": source_indices,
        "selected_object_ids": list(scene_config["selected_object_ids"]),
        "focal_object_id": scene_config["focal_object_id"],
        "derived_name": derived["name"],
        "derived_canonical_sha256": payload_hash,
        "dynamic_object_count": len(derived["objects"]),
        "background_dynamic_object_count": 0,
    }
    return derived, identity


def write_derived_scene(path: Path, scene: dict[str, Any]) -> str:
    """Write canonical JSON for GPUDrive and return its byte hash."""

    if path.exists():
        raise MultiAgentSceneError(f"derived scene already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(scene, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")
    return sha256_file(path)
