from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpudrive_adversary.multiagent.calibration import (
    RandomPriorAdversary,
    _summarize_bound,
    load_bound_sweep_config,
)
from gpudrive_adversary.adversary.distribution import BoundedTanhNormal


pytestmark = pytest.mark.unit


def test_committed_bound_sweep_contract() -> None:
    config = load_bound_sweep_config(Path("configs/calibration/highway_nonfocal_bound_sweep.json"))
    assert config["episodes_per_bound"] == 500
    assert [(item["name"], item["scale"]) for item in config["bounds"]] == [
        ("current", 1.0),
        ("half", 0.5),
        ("quarter", 0.25),
    ]


def test_random_prior_adversary_is_reproducible() -> None:
    prior = BoundedTanhNormal(np.asarray([0.3335, 0.131]), np.asarray([0.5, 0.5]))
    first = RandomPriorAdversary(prior, np.random.default_rng(42)).act(SimpleNamespace(), deterministic=False)
    second = RandomPriorAdversary(prior, np.random.default_rng(42)).act(SimpleNamespace(), deterministic=False)
    assert np.array_equal(first.requested_disturbance, second.requested_disturbance)
    assert np.all(np.abs(first.requested_disturbance) < prior.bounds)


def test_bound_summary_separates_failures_after_focal_goal() -> None:
    def episode(*, failure: bool, focal_goal: bool):
        raw = np.zeros((3, 10, 5), dtype=np.float32)
        failing = np.zeros((2, 10), dtype=bool)
        kinds = ((), ())
        failure_timestep = None
        if failure:
            raw[2, 8, 0] = 1.0
            failing[1, 8] = True
            kinds = ((), ("road_object_contact",))
            failure_timestep = 1
        goals = np.zeros((3, 10), dtype=bool)
        goals[2, 0] = focal_goal
        done = np.zeros((3, 10), dtype=bool)
        done[2, 0] = focal_goal
        return SimpleNamespace(
            failure_timestep=failure_timestep,
            transition_count=2,
            goal_ever=goals,
            done=done,
            failing_agents=failing,
            failure_kinds=kinds,
            raw_info=raw,
            boxes=np.zeros((3, 10, 5)),
            disturbance_effective=np.asarray([[0.1, 0.02], [-0.1, -0.02]]),
            episode_minimum_clearance=1.0,
        )

    row = _summarize_bound(
        [episode(failure=True, focal_goal=False), episode(failure=True, focal_goal=True), episode(failure=False, focal_goal=False)],
        name="half",
        bounds=np.asarray([0.3335, 0.131]),
    )
    assert row["failures"] == 2
    assert row["failures_while_focal_active"] == 1
    assert row["failures_after_focal_goal"] == 1
    assert row["qualifying_failures_by_slot"]["8"] == 2
