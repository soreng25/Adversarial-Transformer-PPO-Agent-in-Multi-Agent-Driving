import numpy as np
import pytest

from gpudrive_adversary.smoke import SmokeError
from gpudrive_adversary.victim.evaluation import compare_victim_sequences


pytestmark = pytest.mark.unit


def _sequence() -> dict[str, np.ndarray]:
    states = 3
    actions = states - 1
    controlled_mask = np.zeros((1, 64), dtype=np.bool_)
    controlled_mask[0, 0] = True
    action_table = np.zeros((91, 3), dtype=np.float32)
    action_table[45] = [0.0, 0.0, 0.0]
    action_table[46] = [0.0, 0.1, 0.0]
    action_indices = np.asarray([45, 46], dtype=np.int64)
    commands = action_table[action_indices].copy()
    templates = np.zeros((actions, 1, 64), dtype=np.int64)
    templates[:, 0, 0] = action_indices
    return {
        "victim_observations": np.ones((states, 2984), dtype=np.float32),
        "absolute_state": np.ones((states, 1, 64, 14), dtype=np.float32),
        "rewards": np.zeros((states, 1, 64), dtype=np.float32),
        "logits": np.ones((actions, 91), dtype=np.float32),
        "values": np.ones((actions,), dtype=np.float32),
        "log_probabilities": np.ones((actions,), dtype=np.float32),
        "action_indices": action_indices,
        "action_table": action_table,
        "controlled_mask": controlled_mask,
        "dense_action_templates": templates,
        "dones": np.zeros((states, 1, 64), dtype=np.bool_),
        "metadata": np.zeros((1, 64, 4), dtype=np.int32),
        "native_commands": commands.copy(),
        "nominal_commands": commands.copy(),
        "raw_info": np.zeros((states, 1, 64, 5), dtype=np.float32),
    }


def _copy(sequence: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in sequence.items()}


def test_compare_victim_sequences_accepts_tolerant_numeric_drift() -> None:
    left = _sequence()
    right = _copy(left)
    tolerant_fields = (
        "victim_observations",
        "absolute_state",
        "rewards",
        "logits",
        "values",
        "log_probabilities",
    )
    for name in tolerant_fields:
        right[name].flat[0] += np.float32(1e-7)

    result = compare_victim_sequences(
        left, right, rtol=1e-6, atol=1e-6, equal_nan=True
    )

    assert result["ok"]
    for name in tolerant_fields:
        assert not result["fields"][name]["exact"]
        assert result["fields"][name]["allclose"]
        assert result["fields"][name]["ok"]


@pytest.mark.parametrize(
    "field",
    (
        "action_indices",
        "action_table",
        "controlled_mask",
        "dense_action_templates",
        "dones",
        "metadata",
        "native_commands",
        "nominal_commands",
        "raw_info",
    ),
)
def test_compare_victim_sequences_requires_exact_action_and_event_fields(
    field: str,
) -> None:
    left = _sequence()
    right = _copy(left)
    if right[field].dtype == np.bool_:
        right[field].flat[0] = not bool(right[field].flat[0])
    else:
        right[field].flat[0] += 1

    result = compare_victim_sequences(
        left, right, rtol=0.0, atol=2.0, equal_nan=True
    )

    assert result["fields"][field]["allclose"]
    assert not result["fields"][field]["exact"]
    assert not result["fields"][field]["ok"]
    assert not result["ok"]


def test_compare_victim_sequences_rejects_different_trace_fields() -> None:
    left = _sequence()
    right = _copy(left)
    del right["logits"]

    with pytest.raises(SmokeError, match="trace fields differ"):
        compare_victim_sequences(
            left, right, rtol=1e-6, atol=1e-6, equal_nan=True
        )
