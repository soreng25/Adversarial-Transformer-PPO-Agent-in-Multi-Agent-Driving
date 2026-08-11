import json
from pathlib import Path

import numpy as np
import pytest

from gpudrive_adversary.adversary.config import load_adversary_config
from gpudrive_adversary.adversary.training_artifact import (
    AdversaryTrainingArtifactError,
    validate_adversary_training_artifact,
)
from gpudrive_adversary.pins import (
    canonical_json_sha256,
    load_pins,
    sha256_file,
)
from gpudrive_adversary.victim.checkpoint import (
    checkpoint_identity,
    load_victim_pin,
)


pytestmark = pytest.mark.unit


def _rollout_arrays(transitions: int = 2) -> dict[str, np.ndarray]:
    return {
        "victim_observations": np.zeros((transitions + 1, 2984), np.float32),
        "raw_info": np.zeros((transitions + 1, 5), np.float32),
        "tokens": np.zeros((transitions, 2989), np.float32),
        "history_masks": np.ones((transitions, 50), np.bool_),
        "victim_action_indices": np.zeros((transitions,), np.int64),
        "victim_logits": np.zeros((transitions, 91), np.float32),
        "victim_nominal_commands": np.zeros((transitions, 3), np.float32),
        "adversary_raw_actions": np.zeros((transitions, 2), np.float32),
        "disturbance_requested": np.zeros((transitions, 2), np.float32),
        "disturbance_effective": np.zeros((transitions, 2), np.float32),
        "applied_commands": np.zeros((transitions, 3), np.float32),
        "disturbance_saturated": np.zeros((transitions, 2), np.bool_),
        "command_saturated": np.zeros((transitions, 2), np.bool_),
        "prior_nll_exact": np.zeros((transitions,), np.float64),
        "prior_nll_penalty": np.zeros((transitions,), np.float64),
        "policy_log_probability": np.zeros((transitions,), np.float64),
        "adversary_values": np.zeros((transitions,), np.float32),
        "rewards": np.zeros((transitions,), np.float64),
        "pre_actor_latent": np.zeros((transitions, 64), np.float32),
        "failure_by_transition": np.zeros((transitions,), np.bool_),
        "failure_kind_bits": np.zeros((transitions, 3), np.int8),
    }


def _rewrite_manifest(path: Path, mutate=None) -> dict:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(manifest)
    manifest["artifact_id"] = None
    manifest["artifact_id"] = (
        "adversary-train-" + canonical_json_sha256(manifest)[:16]
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _make_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict]:
    root = tmp_path / "run"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    config = load_adversary_config()
    pins = load_pins()
    victim = load_victim_pin()
    victim_id = checkpoint_identity(victim)
    scene = pins["smoke_scene"]

    source_checks = [
        {
            "name": "source.commit",
            "ok": True,
            "required": True,
            "expected": pins["gpudrive"]["commit"],
            "observed": pins["gpudrive"]["commit"],
        },
        {
            "name": "scene.sha256",
            "ok": True,
            "required": True,
            "expected": scene["sha256"],
            "observed": scene["sha256"],
        },
        {
            "name": "scene.scenario_id",
            "ok": True,
            "required": True,
            "expected": scene["scenario_id"],
            "observed": scene["scenario_id"],
        },
        {
            "name": "scene.sdc_object_id",
            "ok": True,
            "required": True,
            "expected": scene["sdc_object_id"],
            "observed": scene["sdc_object_id"],
        },
    ]
    source_checks.extend(
        {
            "name": f"source.gitlink.{name}",
            "ok": True,
            "required": True,
            "expected": revision,
            "observed": revision,
        }
        for name, revision in pins["gpudrive"]["submodules"].items()
    )
    victim_verification = {
        "schema": "gpudrive_victim_checkpoint_verification",
        "schema_version": 1,
        "ok": True,
        "checkpoint_id": victim_id,
        "checks": [{"name": "checkpoint", "ok": True, "required": True}],
    }
    fingerprints = {
        "methodology_repository": config["methodology_source"]["repository"],
        "methodology_commit": config["methodology_source"]["commit"],
        "gpudrive_commit": pins["gpudrive"]["commit"],
        "gpudrive_submodules": pins["gpudrive"]["submodules"],
        "scene_sha256": scene["sha256"],
        "scene_scenario_id": scene["scenario_id"],
        "victim_stable_id": scene["sdc_object_id"],
        "victim_checkpoint_model_sha256": victim["source"]["files"]
        ["model.safetensors"]["sha256"],
        "victim_config_sha256": victim["source"]["files"]["config.json"]
        ["sha256"],
        "adversary_config_sha256": canonical_json_sha256(config),
        "native_extension_sha256": "1" * 64,
        "madrona_kernel_cache_sha256": "2" * 64,
        "source_verification_sha256": canonical_json_sha256(source_checks),
        "victim_verification_sha256": canonical_json_sha256(victim_verification),
        "port": {
            "source_tree_sha256": "3" * 64,
            "source_tree_matches_declared": True,
        },
    }
    checkpoint_ids = ["adversary-ppo-first", "adversary-ppo-second"]
    for index, checkpoint_id in enumerate(checkpoint_ids, start=1):
        directory = checkpoints / f"iteration-{index:04d}"
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_id": checkpoint_id,
                    "iteration": index,
                    "config_sha256": canonical_json_sha256(config),
                    "fingerprints": fingerprints,
                    "parent_victim_checkpoint_id": victim_id,
                    "model": {
                        "state_sha256": "5" * 64 if index == 1 else "6" * 64
                    },
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "gpudrive_adversary.adversary.training_artifact.validate_adversary_checkpoint",
        lambda _: {"ok": True, "failed_checks": []},
    )
    training_path = root / "last-training-rollout.npz"
    evaluation_path = root / "last-evaluation-rollout.npz"
    np.savez_compressed(training_path, **_rollout_arrays())
    np.savez_compressed(evaluation_path, **_rollout_arrays())
    nominal = {
        "victim_observations": np.zeros((3, 2984), np.float32),
        "raw_info": np.zeros((3, 5), np.float32),
        "victim_action_indices": np.zeros((2,), np.int64),
        "victim_nominal_commands": np.zeros((2, 3), np.float32),
        "applied_commands": np.zeros((2, 3), np.float32),
    }
    nominal["raw_info"][-1, 3] = 1
    nominal_path = root / "nominal-eligibility.npz"
    np.savez_compressed(nominal_path, **nominal)

    manifest = {
        "schema": "gpudrive_adversary_training_run",
        "schema_version": 1,
        "artifact_id": None,
        "created_at": "2026-08-11T00:00:00+00:00",
        "purpose": "tiny_training_smoke_only",
        "research_claims_allowed": False,
        "config": config,
        "config_sha256": canonical_json_sha256(config),
        "parent_victim_checkpoint_id": victim_id,
        "victim_state_sha256_before": "4" * 64,
        "victim_state_sha256_after": "4" * 64,
        "eligibility": {
            "eligible": True,
            "reason": "clean_goal_success",
            "goal_timestep": 1,
            "failure_timestep": None,
            "failure_kinds": [],
        },
        "scene": {
            "relative_path": scene["relative_path"],
            "scenario_id": scene["scenario_id"],
            "victim_slot": 0,
            "victim_stable_id": scene["sdc_object_id"],
        },
        "metrics": [{"iteration": 1}, {"iteration": 2}],
        "checkpoint_ids": checkpoint_ids,
        "last_training_rollout": {
            "sha256": sha256_file(training_path),
            "behavior_model_state_sha256": "5" * 64,
            "behavior_checkpoint_id": checkpoint_ids[0],
            "transition_count": 2,
            "termination_reason": "horizon",
            "failure_timestep": None,
        },
        "last_deterministic_evaluation": {
            "sha256": sha256_file(evaluation_path),
            "behavior_checkpoint_id": checkpoint_ids[-1],
            "transition_count": 2,
            "termination_reason": "horizon",
            "failure_timestep": None,
        },
        "nominal_eligibility_trace_sha256": sha256_file(nominal_path),
        "fingerprints": fingerprints,
        "source_verification": source_checks,
        "victim_verification": victim_verification,
        "runtime": {
            "platform": "Linux",
            "gpu": "test",
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
        },
    }
    manifest["artifact_id"] = (
        "adversary-train-" + canonical_json_sha256(manifest)[:16]
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root, manifest


def test_complete_training_run_is_validated_as_one_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = _make_run(tmp_path, monkeypatch)

    report = validate_adversary_training_artifact(root)

    assert report["ok"]
    assert report["artifact_id"] == manifest["artifact_id"]
    assert len(report["child_checkpoints"]) == 2


def test_validator_detects_changed_frozen_victim_even_with_new_artifact_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_run(tmp_path, monkeypatch)
    _rewrite_manifest(
        root,
        lambda manifest: manifest.__setitem__(
            "victim_state_sha256_after", "9" * 64
        ),
    )

    report = validate_adversary_training_artifact(root)

    assert not report["ok"]
    assert not report["checks"]["victim.frozen"]
    assert report["checks"]["artifact_id"]


def test_validator_detects_rollout_shape_tampering_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_run(tmp_path, monkeypatch)
    trace_path = root / "last-training-rollout.npz"
    arrays = _rollout_arrays()
    arrays["victim_observations"] = arrays["victim_observations"][:-1]
    np.savez_compressed(trace_path, **arrays)

    def update_hash(manifest: dict) -> None:
        manifest["last_training_rollout"]["sha256"] = sha256_file(trace_path)

    _rewrite_manifest(root, update_hash)
    report = validate_adversary_training_artifact(root)

    assert not report["ok"]
    assert report["checks"]["rollout.training.sha256"]
    assert not report["checks"]["rollout.training.shapes"]


def test_validator_requires_child_checkpoint_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_run(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "gpudrive_adversary.adversary.training_artifact.validate_adversary_checkpoint",
        lambda _: {"ok": False, "failed_checks": ["weights"]},
    )

    report = validate_adversary_training_artifact(root)

    assert not report["ok"]
    assert not report["checks"]["checkpoints.valid"]


def test_validator_requires_exact_gpudrive_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_run(tmp_path, monkeypatch)

    def change_pin(manifest: dict) -> None:
        manifest["fingerprints"]["gpudrive_commit"] = "0" * 40
        for checkpoint in (root / "checkpoints").iterdir():
            child_path = checkpoint / "manifest.json"
            child = json.loads(child_path.read_text(encoding="utf-8"))
            child["fingerprints"] = manifest["fingerprints"]
            child_path.write_text(json.dumps(child), encoding="utf-8")

    _rewrite_manifest(root, change_pin)
    report = validate_adversary_training_artifact(root)

    assert not report["ok"]
    assert not report["checks"]["fingerprints.gpudrive"]


def test_missing_manifest_is_a_parse_error(tmp_path: Path) -> None:
    with pytest.raises(AdversaryTrainingArtifactError, match="cannot read"):
        validate_adversary_training_artifact(tmp_path / "missing")


def test_training_validates_temporary_directory_before_atomic_publication() -> None:
    training_source = (
        Path(__file__).parents[2]
        / "src/gpudrive_adversary/adversary/training.py"
    ).read_text(encoding="utf-8")

    validation = training_source.index(
        "validate_adversary_training_artifact(temporary)"
    )
    publication = training_source.index("temporary.replace(output.resolve())")
    assert validation < publication
