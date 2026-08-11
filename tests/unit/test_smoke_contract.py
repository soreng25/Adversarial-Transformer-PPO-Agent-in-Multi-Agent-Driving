import json
from pathlib import Path

import numpy as np
import pytest

from gpudrive_adversary.pins import (
    canonical_json_sha256,
    load_pins,
    load_smoke_config,
    sha256_file,
)
from gpudrive_adversary.smoke import (
    compare_sequences,
    compare_smoke_artifacts,
    validate_smoke_artifact,
)


pytestmark = pytest.mark.unit


def _sequence(offset: float = 0.0) -> dict[str, np.ndarray]:
    states = 4
    actions = states - 1
    absolute_state = np.zeros((states, 1, 64, 14), dtype=np.float32)
    absolute_state[:, 0, 0, 13] = 271
    controlled = np.zeros((1, 64), dtype=np.bool_)
    controlled[0, 0] = True
    requested_indices = np.full((actions, 1, 64), -1, dtype=np.int64)
    requested_indices[0] = 45
    requested_physical = np.zeros((actions, 1, 64, 3), dtype=np.float32)
    requested_physical[1, 0, 0] = [1.25, 0.0, 0.0]
    requested_physical[2, 0, 0] = [0.0, 0.125, 0.0]
    return {
        "observations": np.full(
            (states, 1, 64, 2), 1.0 + offset, dtype=np.float32
        ),
        "absolute_state": absolute_state,
        "rewards": np.zeros((states, 1, 64), dtype=np.float32),
        "dones": np.zeros((states, 1, 64), dtype=np.float32),
        "raw_info": np.zeros((states, 1, 64, 5), dtype=np.float32),
        "info_off_road": np.zeros((states, 1, 64), dtype=np.float32),
        "info_collided": np.zeros((states, 1, 64), dtype=np.float32),
        "info_goal_achieved": np.zeros((states, 1, 64), dtype=np.float32),
        "requested_action_indices": requested_indices,
        "requested_physical_actions": requested_physical,
        "native_physical_actions": requested_physical.copy(),
        "command_names": np.asarray(
            ["neutral_discrete_45", "mild_acceleration", "mild_steering"]
        ),
        "command_kind_codes": np.asarray([0, 1, 1], dtype=np.int8),
        "controlled_mask": controlled,
        "metadata": np.zeros((1, 64, 4), dtype=np.int32),
    }


def test_compare_sequences_accepts_small_numeric_drift() -> None:
    result = compare_sequences(
        _sequence(), _sequence(1e-7), rtol=1e-6, atol=1e-6, equal_nan=True
    )
    assert result["ok"]
    assert not result["fields"]["observations"]["exact"]


def test_compare_sequences_rejects_event_change() -> None:
    right = _sequence()
    right["raw_info"][0, 0, 0, 1] = 1
    result = compare_sequences(
        _sequence(), right, rtol=1e-6, atol=1e-6, equal_nan=True
    )
    assert not result["ok"]
    assert not result["fields"]["raw_info"]["ok"]


def _artifact(path: Path, sequence: dict[str, np.ndarray]) -> None:
    pins = load_pins()
    config = load_smoke_config()
    path.mkdir()
    np.savez_compressed(path / "trace.npz", **sequence)
    source_verification = [{"name": "source.test", "ok": True, "required": True}]
    manifest = {
        "schema": "gpudrive_scene_smoke_artifact",
        "schema_version": 1,
        "artifact_id": None,
        "created_at": "2026-08-11T00:00:00+00:00",
        "purpose": "installation_and_determinism_only",
        "device": "cuda",
        "scene": {
            "relative_path": pins["smoke_scene"]["relative_path"],
            "scenario_id": pins["smoke_scene"]["scenario_id"],
            "sdc_object_id": pins["smoke_scene"]["sdc_object_id"],
            "controlled_slot": 0,
            "loader_seed": 42,
        },
        "resolved_config": config,
        "same_process_replay": {"ok": True},
        "action_transport": {
            "ok": True,
            "declared_order": config["expected"]["physical_action_order"],
        },
        "raw_info_mapping_verified": True,
        "raw_info_order": config["expected"]["raw_info_order"],
        "termination_reason": None,
        "termination_reason_explanation": "Not defined in Milestone A.",
        "failure_timestep": None,
        "failure_definition": None,
        "failure_definition_explanation": "Not defined in Milestone A.",
        "victim_checkpoint": None,
        "victim_checkpoint_explanation": "Milestone B.",
        "adversary_checkpoint": None,
        "adversary_checkpoint_explanation": "Milestone C.",
        "fingerprints": {
            "gpudrive_commit": pins["gpudrive"]["commit"],
            "gpudrive_tree": pins["gpudrive"]["tree"],
            "gpudrive_submodules": pins["gpudrive"]["submodules"],
            "gpudrive_uv_lock_sha256": pins["gpudrive"]["uv_lock_sha256"],
            "scene_sha256": pins["smoke_scene"]["sha256"],
            "config_sha256": canonical_json_sha256(config),
            "native_extension_sha256": "d" * 64,
            "madrona_kernel_cache_tree_sha256": "e" * 64,
            "trace_sha256": sha256_file(path / "trace.npz"),
            "source_verification_sha256": canonical_json_sha256(
                source_verification
            ),
            "port": {
                "commit": "a" * 40,
                "dirty": True,
                "diff_sha256": "b" * 64,
                "source_tree_sha256": "c" * 64,
                "declared_source_tree_sha256": "c" * 64,
                "source_tree_matches_declared": True,
            },
        },
        "runtime": {
            "platform": "Linux-test",
            "python": "3.11.9 (test)",
            "torch": "2.6.0+cu124",
            "torch_cuda": pins["reference_runtime"]["cuda"],
            "cuda_available": True,
            "gpu_names": ["test-gpu"],
            "nvidia_smi": {"returncode": 0, "rows": ["test-gpu, test-driver"]},
            "reference_image_digest": pins["reference_runtime"][
                "base_image_amd64_digest"
            ],
        },
        "source_verification": source_verification,
    }
    manifest["artifact_id"] = "scene-smoke-" + canonical_json_sha256(manifest)[:16]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_artifact_validation_and_fresh_comparison(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _artifact(left, _sequence())
    _artifact(right, _sequence())
    assert validate_smoke_artifact(left)["ok"]
    assert compare_smoke_artifacts(left, right)["ok"]


def test_artifact_validation_rejects_config_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _artifact(artifact, _sequence())
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resolved_config"]["loader"]["seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_smoke_artifact(artifact)
    assert not validation["ok"]
    assert not validation["checks"]["config_fingerprint"]
