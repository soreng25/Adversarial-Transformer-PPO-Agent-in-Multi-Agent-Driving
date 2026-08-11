import json
import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from gpudrive_adversary.victim.checkpoint import (
    VictimCheckpointError,
    checkpoint_identity,
    load_victim_pin,
    read_safetensors_contract,
    safetensors_state_sha256,
)
from gpudrive_adversary.victim.policy import (
    deterministic_argmax,
    pinned_action_table,
    validate_slot0_binding,
)


pytestmark = pytest.mark.unit


def test_published_victim_pin_is_immutable_and_failure_free() -> None:
    pin = load_victim_pin()
    assert pin["source"]["revision"] == "1532950cad84dafc6e9d976a2bcc524ee481a1a1"
    assert (
        pin["source"]["files"]["model.safetensors"]["sha256"]
        == "f3f26475def35f375c6c72d8f8f20b2b091f77175010345dc3fa968a860521b7"
    )
    assert pin["environment"]["max_cont_agents"] == 1
    assert pin["environment"]["model_agent_layout"] == 64
    assert pin["failure_definition"] is None
    assert checkpoint_identity(pin).startswith("victim-ppo-")
    action_table = pinned_action_table(pin["environment"])
    assert action_table.shape == (91, 3)
    assert action_table[45].tolist() == [0.0, 0.0, 0.0]


def test_deterministic_argmax_uses_lowest_index_on_tie() -> None:
    logits = np.zeros((1, 91), dtype=np.float32)
    logits[0, 7] = 2.0
    logits[0, 9] = 2.0
    assert deterministic_argmax(logits).tolist() == [7]


def test_deterministic_argmax_rejects_nonfinite_logits() -> None:
    logits = np.zeros((1, 91), dtype=np.float32)
    logits[0, 0] = np.nan
    with pytest.raises(VictimCheckpointError, match="non-finite"):
        deterministic_argmax(logits)


def test_slot0_binding_requires_one_sdc_with_stable_id() -> None:
    controlled = np.zeros((1, 64), dtype=bool)
    controlled[0, 0] = True
    is_sdc = np.zeros((1, 64), dtype=np.int32)
    is_sdc[0, 0] = 1
    ids = np.zeros((1, 64), dtype=np.int32)
    ids[0, 0] = 271
    binding = validate_slot0_binding(controlled, is_sdc, ids, expected_id=271)
    assert binding == {"world": 0, "slot": 0, "stable_id": 271, "is_sdc": True}

    controlled[0, 1] = True
    with pytest.raises(VictimCheckpointError, match="exactly slot 0"):
        validate_slot0_binding(controlled, is_sdc, ids, expected_id=271)


def test_safetensors_header_is_inspected_without_pickle(tmp_path: Path) -> None:
    header = {
        "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * 8)
    contract = read_safetensors_contract(path)
    assert contract["tensor_count"] == 1
    assert contract["parameter_count"] == 2
    assert contract["tensors"] == {
        "weight": {"dtype": "F32", "shape": [2]}
    }
    expected = hashlib.sha256(
        b"weight\0torch.float32\0(2,)\0" + b"\0" * 8 + b"\0"
    ).hexdigest()
    assert safetensors_state_sha256(path) == expected
