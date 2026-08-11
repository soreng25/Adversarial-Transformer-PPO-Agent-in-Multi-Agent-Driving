import copy

import pytest

from gpudrive_adversary.adversary.config import (
    AdversaryConfigError,
    adversary_config_sha256,
    load_adversary_config,
)


pytestmark = pytest.mark.unit


def test_approved_adversary_contract_is_explicit_and_smoke_only() -> None:
    config = load_adversary_config()
    assert config["research_claims_allowed"] is False
    assert config["intervention"]["bounds"] == [0.667, 0.262]
    assert config["failure"]["failure_if_any"] == [
        "road_object_contact",
        "vehicle_collision",
        "nonvehicle_collision",
    ]
    assert config["token"]["token_dim"] == 2989
    assert len(adversary_config_sha256(config)) == 64


def test_config_rejects_current_victim_action_leak(tmp_path) -> None:
    config = copy.deepcopy(load_adversary_config())
    config["token"]["current_victim_action_visible"] = True
    path = tmp_path / "bad.json"
    import json

    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(AdversaryConfigError, match="current victim action"):
        load_adversary_config(path)
