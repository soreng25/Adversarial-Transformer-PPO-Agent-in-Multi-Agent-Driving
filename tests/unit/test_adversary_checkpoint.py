import copy
import json
import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from gpudrive_adversary.adversary.checkpoint import (
    AdversaryCheckpointError,
    _decode_optimizer_value,
    _encode_optimizer_value,
    load_adversary_checkpoint,
    save_adversary_checkpoint,
    validate_adversary_checkpoint,
)
from gpudrive_adversary.adversary.config import (
    adversary_config_sha256,
    load_adversary_config,
)
from gpudrive_adversary.pins import canonical_json_sha256, sha256_file
from gpudrive_adversary.victim.checkpoint import safetensors_state_sha256
from gpudrive_adversary.victim.policy import torch_module_state_sha256


pytestmark = pytest.mark.unit


def _write_small_safetensors(path: Path) -> None:
    values = np.asarray([1.25, -2.5], dtype="<f4")
    header = {
        "weight": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [0, values.nbytes],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + values.tobytes())


def _fingerprints(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "methodology_repository": config["methodology_source"]["repository"],
        "methodology_commit": config["methodology_source"]["commit"],
        "gpudrive_commit": "a" * 40,
        "gpudrive_submodules": {"external/madrona": "b" * 40},
        "scene_sha256": "c" * 64,
        "scene_scenario_id": "ef3a8f65142f41ac",
        "victim_stable_id": 271,
        "victim_checkpoint_model_sha256": "d" * 64,
        "victim_config_sha256": "e" * 64,
        "adversary_config_sha256": adversary_config_sha256(config),
        "native_extension_sha256": "f" * 64,
        "madrona_kernel_cache_sha256": "1" * 64,
        "source_verification_sha256": "2" * 64,
        "victim_verification_sha256": "3" * 64,
    }


def _write_pure_checkpoint(path: Path) -> dict[str, Any]:
    path.mkdir()
    weights_path = path / "model.safetensors"
    _write_small_safetensors(weights_path)
    arrays = {
        "state_0000_step": np.asarray(4.0, dtype=np.float32),
        "state_0000_exp_avg": np.asarray([0.1, 0.2], dtype=np.float32),
        "state_0000_exp_avg_sq": np.asarray([0.01, 0.04], dtype=np.float32),
    }
    optimizer_path = path / "optimizer.npz"
    np.savez_compressed(optimizer_path, **arrays)
    config = load_adversary_config()
    manifest = {
        "schema": "gpudrive_adversary_checkpoint",
        "schema_version": 1,
        "artifact_id": None,
        "created_at": "2026-08-11T00:00:00+00:00",
        "purpose": config["purpose"],
        "research_claims_allowed": config["research_claims_allowed"],
        "iteration": 1,
        "total_transitions": 32,
        "parent_victim_checkpoint_id": "victim-ppo-0123456789abcdef",
        "config": config,
        "config_sha256": adversary_config_sha256(config),
        "model": {
            "class": "CausalTransformerActorCritic",
            "state_sha256": safetensors_state_sha256(weights_path),
            "architecture": config["model"],
            "token": config["token"],
            "training": False,
        },
        "optimizer": {
            "class": "Adam",
            "format": "typed_npz_no_pickle",
            "parameter_groups": [
                {
                    "parameter_names": ["weight"],
                    "lr": 3e-4,
                    "betas": _encode_optimizer_value((0.8, 0.95)),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "maximize": False,
                    "capturable": False,
                    "differentiable": False,
                    "foreach": None,
                    "fused": None,
                }
            ],
            "states": [
                {
                    "parameter_name": "weight",
                    "fields": {
                        "step": "state_0000_step",
                        "exp_avg": "state_0000_exp_avg",
                        "exp_avg_sq": "state_0000_exp_avg_sq",
                    },
                }
            ],
            "array_keys": sorted(arrays),
        },
        "metrics": {"loss": 0.5},
        "files": {
            "model.safetensors": {
                "size": weights_path.stat().st_size,
                "sha256": sha256_file(weights_path),
            },
            "optimizer.npz": {
                "size": optimizer_path.stat().st_size,
                "sha256": sha256_file(optimizer_path),
            },
        },
        "fingerprints": _fingerprints(config),
    }
    manifest["artifact_id"] = (
        "adversary-ppo-" + canonical_json_sha256(manifest)[:16]
    )
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _rewrite_manifest(
    path: Path, mutation: Callable[[dict[str, Any]], None]
) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    mutation(manifest)
    manifest["artifact_id"] = None
    manifest["artifact_id"] = (
        "adversary-ppo-" + canonical_json_sha256(manifest)[:16]
    )
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def test_optimizer_sequences_preserve_tuple_and_list_types() -> None:
    tuple_value = (0.8, [0.9, 0.95])
    list_value = [0.8, (0.9, 0.95)]

    decoded_tuple = _decode_optimizer_value(_encode_optimizer_value(tuple_value))
    decoded_list = _decode_optimizer_value(_encode_optimizer_value(list_value))

    assert decoded_tuple == tuple_value and isinstance(decoded_tuple, tuple)
    assert isinstance(decoded_tuple[1], list)
    assert decoded_list == list_value and isinstance(decoded_list, list)
    assert isinstance(decoded_list[1], tuple)


def test_optimizer_metadata_rejects_unsafe_arbitrary_objects() -> None:
    with pytest.raises(AdversaryCheckpointError, match="unsupported"):
        _encode_optimizer_value({"would": "require object decoding"})


def test_pure_checkpoint_contract_validates_state_and_mapping(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_pure_checkpoint(checkpoint)

    validation = validate_adversary_checkpoint(checkpoint)

    assert validation["ok"], validation["failed_checks"]
    assert validation["checks"]["model_state_sha256"]
    assert validation["checks"]["optimizer.field_mapping"]
    assert validation["checks"]["config_fingerprint_consistency"]


def test_validation_rejects_manifest_model_state_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_pure_checkpoint(checkpoint)
    _rewrite_manifest(
        checkpoint,
        lambda manifest: manifest["model"].update({"state_sha256": "0" * 64}),
    )

    validation = validate_adversary_checkpoint(checkpoint)

    assert not validation["ok"]
    assert not validation["checks"]["model_state_sha256"]


def test_validation_rejects_duplicate_optimizer_array_mapping(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_pure_checkpoint(checkpoint)

    def duplicate_mapping(manifest: dict[str, Any]) -> None:
        fields = manifest["optimizer"]["states"][0]["fields"]
        fields["exp_avg_sq"] = fields["exp_avg"]

    _rewrite_manifest(checkpoint, duplicate_mapping)
    validation = validate_adversary_checkpoint(checkpoint)

    assert not validation["ok"]
    assert not validation["checks"]["optimizer.field_mapping"]


def test_validation_rejects_config_fingerprint_disagreement(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_pure_checkpoint(checkpoint)
    _rewrite_manifest(
        checkpoint,
        lambda manifest: manifest["fingerprints"].update(
            {"adversary_config_sha256": "0" * 64}
        ),
    )

    validation = validate_adversary_checkpoint(checkpoint)

    assert not validation["ok"]
    assert not validation["checks"]["config_fingerprint_consistency"]


def test_torch_checkpoint_round_trip_restores_model_and_adam(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    from gpudrive_adversary.adversary.model import (
        AdversaryModelConfig,
        CausalTransformerActorCritic,
    )

    config = load_adversary_config()
    model = CausalTransformerActorCritic(
        AdversaryModelConfig(
            token_dim=config["token"]["token_dim"],
            context_length=config["token"]["history_length"],
            model_dim=config["model"]["d_model"],
            pre_actor_dim=config["model"]["d_model"],
            action_dim=config["model"]["action_dimensions"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["nhead"],
            feed_forward_dim=config["model"]["dim_feedforward"],
            dropout=config["model"]["dropout"],
            initial_log_std=config["model"]["initial_log_std"],
            minimum_log_std=config["model"]["min_log_std"],
            maximum_log_std=config["model"]["max_log_std"],
            action_epsilon=config["prior"]["epsilon"],
        )
    )
    named = list(model.named_parameters())
    optimizer = torch.optim.Adam(
        [
            {
                "params": [named[0][1]],
                "lr": 2e-4,
                "betas": (0.8, 0.95),
            },
            {
                "params": [parameter for _, parameter in named[1:]],
                "lr": 3e-4,
                "betas": (0.9, 0.999),
            },
        ],
        eps=1e-7,
    )
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    expected_model_hash = torch_module_state_sha256(model)
    expected_state = {
        name: {
            field: value.detach().cpu().clone()
            for field, value in optimizer.state[parameter].items()
        }
        for name, parameter in model.named_parameters()
    }

    checkpoint = tmp_path / "checkpoint"
    manifest = save_adversary_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        iteration=2,
        total_transitions=64,
        metrics={"loss": float(loss.detach())},
        fingerprints=_fingerprints(config),
        parent_victim_checkpoint_id="victim-ppo-0123456789abcdef",
    )
    loaded = load_adversary_checkpoint(checkpoint, "cpu")

    assert loaded.validation["ok"]
    assert loaded.manifest == manifest
    assert torch_module_state_sha256(loaded.model) == expected_model_hash
    assert not loaded.model.training
    assert loaded.optimizer.param_groups[0]["betas"] == (0.8, 0.95)
    assert loaded.optimizer.param_groups[1]["betas"] == (0.9, 0.999)
    for name, parameter in loaded.model.named_parameters():
        actual = loaded.optimizer.state[parameter]
        assert set(actual) == set(expected_state[name])
        for field, expected in expected_state[name].items():
            torch.testing.assert_close(actual[field].cpu(), expected)
