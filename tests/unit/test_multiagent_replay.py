from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gpudrive_adversary.cli import build_parser
from gpudrive_adversary.multiagent.environment import MultiAgentRollout
from gpudrive_adversary.multiagent.replay import (
    _FrameCaptureBackend,
    _camera_center_agent,
    _camera_zoom_radius,
    _visible_position_mask,
    MultiAgentReplayError,
    compare_failure_replays,
    failure_signature,
    resolve_checkpoint,
)


pytestmark = pytest.mark.unit


def _failure_rollout() -> MultiAgentRollout:
    transitions = 1
    states = 2
    boxes = np.zeros((states, 10, 5), dtype=np.float64)
    for slot in range(10):
        boxes[:, slot] = [slot * 10.0, 0.0, 0.0, 4.0, 2.0]
    raw_info = np.zeros((states, 10, 5), dtype=np.float32)
    raw_info[1, 2, 1] = 1.0
    failing_agents = np.zeros((transitions, 10), dtype=np.bool_)
    failing_agents[0, 2] = True
    mask = np.zeros((transitions, 50), dtype=np.bool_)
    mask[:, -1] = True
    return MultiAgentRollout(
        observations=np.zeros((states, 10, 2984), dtype=np.float32),
        raw_info=raw_info,
        boxes=boxes,
        done=np.zeros((states, 10), dtype=np.bool_),
        goal_ever=np.zeros((states, 10), dtype=np.bool_),
        minimum_clearance=np.asarray([6.0, 6.0]),
        closest_pair=np.asarray([[0, 1], [0, 1]], dtype=np.int64),
        tokens=np.zeros((transitions, 2989), dtype=np.float32),
        histories=np.zeros((transitions, 50, 2989), dtype=np.float32),
        history_masks=mask,
        victim_action_indices=np.zeros((transitions, 10), dtype=np.int64),
        victim_logits=np.zeros((transitions, 10, 91), dtype=np.float32),
        nominal_commands=np.zeros((transitions, 10, 3), dtype=np.float32),
        applied_commands=np.zeros((transitions, 10, 3), dtype=np.float32),
        disturbance_requested=np.zeros((transitions, 2), dtype=np.float32),
        disturbance_effective=np.zeros((transitions, 2), dtype=np.float32),
        disturbance_saturated=np.zeros((transitions, 2), dtype=np.bool_),
        command_saturated=np.zeros((transitions, 2), dtype=np.bool_),
        prior_nll_exact=np.zeros(transitions, dtype=np.float32),
        policy_log_probability=np.zeros(transitions, dtype=np.float32),
        adversary_values=np.zeros(transitions, dtype=np.float32),
        pre_actor_latent=np.zeros((transitions, 64), dtype=np.float32),
        failure_by_transition=np.asarray([True]),
        failing_agents=failing_agents,
        failure_kinds=(("vehicle_collision",),),
        failure_timestep=0,
        termination_reason="failure",
    )


def test_failure_signature_uses_successor_event_and_zero_based_action() -> None:
    signature = failure_signature(_failure_rollout())
    assert signature == {
        "is_failure": True,
        "failure_timestep": 0,
        "failure_step_count": 1,
        "failure_kinds": ["vehicle_collision"],
        "failing_slots": [2],
        "termination_reason": "failure",
    }


def test_repeat_replay_comparison_checks_event_and_numeric_trace() -> None:
    first = _failure_rollout()
    assert compare_failure_replays(first, _failure_rollout())["ok"]
    changed_boxes = first.boxes.copy()
    changed_boxes[1, 0, 0] += 0.1
    changed = replace(first, boxes=changed_boxes)
    report = compare_failure_replays(first, changed)
    assert not report["ok"]
    assert not report["numeric_checks"]["boxes"]["ok"]


def test_checkpoint_resolution_stays_inside_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "iteration-0094"
    checkpoint.mkdir(parents=True)
    assert resolve_checkpoint(run, "iteration-0094") == checkpoint.resolve()
    with pytest.raises(MultiAgentReplayError, match="direct child"):
        resolve_checkpoint(run, "../outside")


def test_render_cli_contract() -> None:
    args = build_parser().parse_args(
        [
            "render-highway-failure",
            "--run",
            "run",
            "--checkpoint",
            "iteration-0094",
            "--output",
            "visualization",
        ]
    )
    assert args.checkpoint == "iteration-0094"
    assert args.zoom_radius == 70
    assert args.fps == 10


def test_camera_ignores_done_and_out_of_bounds_agents() -> None:
    class State:
        boxes = np.asarray(
            [
                [0.0, 0.0, 0.0, 4.0, 2.0],
                [30.0, 0.0, 0.0, 4.0, 2.0],
                [1000.0, 1000.0, 0.0, 4.0, 2.0],
            ]
        )
        done = np.asarray([False, False, True])

    assert _visible_position_mask(State.boxes).tolist() == [True, True, False]
    assert _camera_zoom_radius(State(), 20) == 40
    assert callable(_FrameCaptureBackend.step)


def test_camera_follows_failure_agent_after_focal_agent_is_padded() -> None:
    class State:
        boxes = np.asarray(
            [
                [-11000.0, -11000.0, 0.0, 4.0, 2.0],
                [10.0, 30.0, 0.0, 4.0, 2.0],
                [35.0, 30.0, 0.0, 4.0, 2.0],
            ]
        )
        done = np.asarray([True, False, False])

    assert _camera_center_agent(State(), 1) == 1
    assert _camera_center_agent(State(), 0) == 1
    assert _camera_zoom_radius(State(), 20, center_agent_idx=1) == 35
