import hashlib
import json
from pathlib import Path

import pytest

from gpudrive_adversary.pins import (
    canonical_json_sha256,
    load_pins,
    load_smoke_config,
    required_checks_pass,
    tree_sha256,
    verify_scene,
)


pytestmark = pytest.mark.unit


def test_repository_pin_is_exact() -> None:
    pins = load_pins()
    assert pins["gpudrive"]["commit"] == "aa48a431ed127a37610cc2176db30ec73d0c55df"
    assert pins["gpudrive"]["tree"] == "33240941cc9e2504f2cbc9f61f7169b2a7d5ac25"
    assert pins["gpudrive"]["submodules"] == {
        "external/json": "0457de21cffb298c22b629e538036bfeb96130b7",
        "external/madrona": "4bda33465340fabc2e61fb27f95aa04795a15466",
    }
    assert (
        pins["reference_runtime"]["base_image_amd64_digest"]
        == "sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4"
    )


def test_smoke_config_does_not_define_failure() -> None:
    config = load_smoke_config()
    assert config["failure_definition"] is None
    assert config["environment"]["max_cont_agents"] == 1
    assert config["action"]["neutral_index"] == 45
    assert [command["name"] for command in config["action"]["commands"]] == [
        "neutral_discrete_45",
        "mild_acceleration",
        "mild_steering",
    ]


def test_canonical_json_hash_is_order_independent() -> None:
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )


def test_tree_sha256_hashes_a_cache_file_bytes(tmp_path: Path) -> None:
    cache = tmp_path / "megakernel.bin"
    cache.write_bytes(b"compiled-kernel")
    assert tree_sha256(cache) == hashlib.sha256(b"compiled-kernel").hexdigest()


def test_scene_verification_checks_identity(tmp_path: Path) -> None:
    scene = {
        "scenario_id": "scenario-a",
        "metadata": {"sdc_track_index": 1},
        "objects": [
            {"id": 10, "type": "vehicle", "mark_as_expert": False},
            {"id": 271, "type": "vehicle", "mark_as_expert": False},
        ],
    }
    path = tmp_path / "scene.json"
    payload = json.dumps(scene).encode("utf-8")
    path.write_bytes(payload)
    pin = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "scenario_id": "scenario-a",
        "source_sdc_track_index": 1,
        "sdc_object_id": 271,
    }
    checks = verify_scene(path, pin)
    assert required_checks_pass(checks)


def test_scene_verification_fails_changed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "scene.json"
    path.write_text(
        json.dumps(
            {
                "scenario_id": "wrong",
                "metadata": {"sdc_track_index": 0},
                "objects": [
                    {"id": 271, "type": "vehicle", "mark_as_expert": False}
                ],
            }
        ),
        encoding="utf-8",
    )
    checks = verify_scene(
        path,
        {
            "sha256": "0" * 64,
            "scenario_id": "expected",
            "source_sdc_track_index": 0,
            "sdc_object_id": 271,
        },
    )
    assert not required_checks_pass(checks)
