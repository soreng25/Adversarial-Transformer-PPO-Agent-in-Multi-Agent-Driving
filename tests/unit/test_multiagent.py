from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import gpudrive_adversary.multiagent.artifact as multiagent_artifact
from gpudrive_adversary.adversary.config import load_adversary_config
from gpudrive_adversary.adversary.environment import AdversaryDecision
from gpudrive_adversary.adversary.intervention import InterventionSpec, apply_intervention
from gpudrive_adversary.multiagent.clearance import minimum_pairwise_clearance, oriented_box_clearance
from gpudrive_adversary.multiagent.environment import (
    AGENT_COUNT,
    NONFOCAL_SYSTEM_FAILURE_SCOPE,
    MultiAgentState,
    MultiAgentVictimDecision,
    classify_multiagent_failure,
    qualifying_nonfocal_collision_pairs,
    run_multiagent_rollout,
)
from gpudrive_adversary.multiagent.scene import build_derived_scene, load_highway_experiment_config
from gpudrive_adversary.pins import sha256_file
from gpudrive_adversary.victim.policy import validate_multiagent_binding


pytestmark = pytest.mark.unit


def test_signed_oriented_box_clearance() -> None:
    first = [0.0, 0.0, 0.0, 4.0, 2.0]
    assert oriented_box_clearance(first, [5.0, 0.0, 0.0, 4.0, 2.0]) == pytest.approx(1.0)
    assert oriented_box_clearance(first, [4.0, 0.0, 0.0, 4.0, 2.0]) == pytest.approx(0.0)
    assert oriented_box_clearance(first, [3.0, 0.0, 0.0, 4.0, 2.0]) == pytest.approx(-1.0)
    assert oriented_box_clearance(first, [0.0, 3.0, 0.0, 4.0, 2.0]) == pytest.approx(1.0)


def test_minimum_clearance_reports_stable_pair_and_honors_active_mask() -> None:
    boxes = np.asarray([[index * 10.0, 0.0, 0.0, 4.0, 2.0] for index in range(10)])
    clearance, pair = minimum_pairwise_clearance(boxes)
    assert clearance == pytest.approx(6.0)
    assert pair == (0, 1)
    active = np.ones(10, dtype=bool); active[1] = False
    assert minimum_pairwise_clearance(boxes, active)[1] == (2, 3)


def test_highway_configs_pin_ten_agents_and_existing_bounds() -> None:
    experiment = load_highway_experiment_config()
    adversary = load_adversary_config("configs/adversary/highway_10agent_transformer_ppo.json")
    assert experiment["scene"]["selected_object_ids"] == [1460, 844, 846, 845, 843, 850, 858, 862, 857, 859]
    assert experiment["control"]["remove_all_other_dynamic_objects"] is True
    assert adversary["environment"]["max_controlled_agents"] == 10
    assert adversary["intervention"]["bounds"] == [0.667, 0.262]
    assert adversary["failure"]["scope"] == "any_controlled_agent"
    system_experiment = load_highway_experiment_config(
        "configs/multiagent/highway_10agent_nonfocal_system.json"
    )
    system_adversary = load_adversary_config(
        "configs/adversary/highway_10agent_nonfocal_system_transformer_ppo.json"
    )
    assert system_experiment["failure"]["scope"] == NONFOCAL_SYSTEM_FAILURE_SCOPE
    assert system_adversary["failure"]["scope"] == NONFOCAL_SYSTEM_FAILURE_SCOPE
    assert system_adversary["reward"]["nonfailure_shaping"] == "terminal_minimum_signed_nonfocal_obb_clearance"


def test_derived_scene_removes_every_unselected_dynamic_object(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    objects = []
    for index in range(12):
        objects.append({
            "id": index,
            "type": "vehicle",
            "valid": [True] * 91,
            "mark_as_expert": False,
            "position": [{"x": float(index * 10 + step), "y": 0.0, "z": 0.0} for step in range(91)],
            "velocity": [{"x": 10.0, "y": 0.0} for _ in range(91)],
        })
    document = {
        "name": "source",
        "scenario_id": "scenario",
        "objects": objects,
        "roads": [{"id": 1}],
        "tl_states": {},
        "metadata": {
            "sdc_track_index": 0,
            "objects_of_interest": [11],
            "tracks_to_predict": [{"track_index": 2, "difficulty": 0}],
        },
    }
    source.write_text(json.dumps(document), encoding="utf-8")
    config = copy.deepcopy(load_highway_experiment_config())
    config["scene"].update({"source_sha256": sha256_file(source), "scenario_id": "scenario", "selected_object_ids": list(range(10)), "focal_object_id": 0, "derived_canonical_sha256": None})
    derived, identity = build_derived_scene(source, config)
    assert [item["id"] for item in derived["objects"]] == list(range(10))
    assert derived["roads"] == [{"id": 1}]
    assert derived["metadata"]["sdc_track_index"] == 0
    assert derived["metadata"]["tracks_to_predict"] == []
    assert identity["background_dynamic_object_count"] == 0


def test_exact_ten_agent_binding() -> None:
    controlled = np.zeros((1, 64), dtype=bool); controlled[0, :10] = True
    sdc = np.zeros((1, 64)); sdc[0, 0] = 1
    ids = np.zeros((1, 64)); ids[0, :10] = np.arange(10) + 100
    result = validate_multiagent_binding(controlled, sdc, ids, expected_ids=list(range(100, 110)))
    assert result["sdc_slot"] == 0
    controlled[0, 10] = True
    with pytest.raises(Exception, match="controlled mask"):
        validate_multiagent_binding(controlled, sdc, ids, expected_ids=list(range(100, 110)))


class _Victim:
    def assert_frozen(self) -> None:
        return None

    def act_deterministic(self, observations: np.ndarray) -> MultiAgentVictimDecision:
        logits = np.zeros((AGENT_COUNT, 91), dtype=np.float32)
        commands = np.zeros((AGENT_COUNT, 3), dtype=np.float32)
        return MultiAgentVictimDecision(np.zeros(AGENT_COUNT, dtype=np.int64), logits, commands)


class _Adversary:
    def __init__(self, disturbance: tuple[float, float]): self.disturbance = disturbance
    def act(self, context, *, deterministic: bool) -> AdversaryDecision:
        return AdversaryDecision(np.asarray(self.disturbance), 0.5, np.zeros(4), np.zeros(2), -0.5, 0.0)


class _Backend:
    def __init__(self, failure_slot: int | None = None): self.failure_slot = failure_slot; self.last_commands = None; self.steps = 0
    def _state(self, successor: bool) -> MultiAgentState:
        observations = np.zeros((10, 2984), dtype=np.float32)
        raw = np.zeros((10, 5), dtype=np.float32)
        if successor and self.failure_slot is None: raw[:, 3] = 1
        if successor and self.failure_slot is not None: raw[self.failure_slot, 1] = 1
        boxes = np.asarray([[index * 10.0, 0.0, 0.0, 4.0, 2.0] for index in range(10)])
        return MultiAgentState(observations, raw, boxes, np.zeros(10, dtype=bool), False)
    def reset(self) -> MultiAgentState: self.steps = 0; return self._state(False)
    def step(self, commands: np.ndarray) -> MultiAgentState: self.steps += 1; self.last_commands = commands.copy(); return self._state(True)


def _intervention():
    spec = InterventionSpec(np.asarray([0.667, 0.262]), np.asarray([-4.0, -3.142]), np.asarray([4.0, 3.142]))
    return lambda nominal, disturbance: apply_intervention(nominal, disturbance, spec)


def test_rollout_perturbs_only_slot_zero_and_requires_all_goals() -> None:
    backend = _Backend()
    rollout = run_multiagent_rollout(backend=backend, victim=_Victim(), adversary=_Adversary((0.5, 0.1)), intervention=_intervention(), max_steps=5)
    assert rollout.termination_reason == "all_goals_reached"
    assert rollout.all_goals_reached
    assert np.allclose(backend.last_commands[0, :2], [0.5, 0.1])
    assert np.array_equal(backend.last_commands[1:], np.zeros((9, 3)))


def test_failure_of_nondisturbed_agent_is_global_failure() -> None:
    rollout = run_multiagent_rollout(backend=_Backend(failure_slot=7), victim=_Victim(), adversary=_Adversary((0.0, 0.0)), intervention=_intervention(), max_steps=5)
    assert rollout.failure_timestep == 0
    assert rollout.failing_agents[0, 7]
    assert rollout.failure_kinds[0] == ("vehicle_collision",)


def test_nonfocal_system_failure_excludes_focal_events_and_focal_collisions() -> None:
    boxes = np.asarray([[slot * 10.0, 0.0, 0.0, 4.0, 2.0] for slot in range(10)])
    raw = np.zeros((10, 5), dtype=np.float32)
    raw[0, 0] = 1
    failed, kinds, agents = classify_multiagent_failure(
        raw, boxes, scope=NONFOCAL_SYSTEM_FAILURE_SCOPE
    )
    assert not failed and kinds == () and not agents.any()

    raw[:] = 0
    raw[[0, 1], 1] = 1
    boxes[1, 0] = boxes[0, 0]
    failed, _, agents = classify_multiagent_failure(
        raw, boxes, scope=NONFOCAL_SYSTEM_FAILURE_SCOPE
    )
    assert not failed and not agents.any()
    assert qualifying_nonfocal_collision_pairs(raw, boxes) == ()


def test_nonfocal_system_failure_requires_nonfocal_road_event_or_pair() -> None:
    boxes = np.asarray([[slot * 10.0, 0.0, 0.0, 4.0, 2.0] for slot in range(10)])
    raw = np.zeros((10, 5), dtype=np.float32)
    raw[4, 0] = 1
    failed, kinds, agents = classify_multiagent_failure(
        raw, boxes, scope=NONFOCAL_SYSTEM_FAILURE_SCOPE
    )
    assert failed and kinds == ("road_object_contact",)
    assert np.flatnonzero(agents).tolist() == [4]

    raw[:] = 0
    raw[[2, 3], 1] = 1
    boxes[3, :2] = boxes[2, :2]
    failed, kinds, agents = classify_multiagent_failure(
        raw, boxes, scope=NONFOCAL_SYSTEM_FAILURE_SCOPE
    )
    assert failed and kinds == ("vehicle_collision",)
    assert np.flatnonzero(agents).tolist() == [2, 3]
    assert qualifying_nonfocal_collision_pairs(raw, boxes) == ((2, 3),)


def test_nonfocal_rollout_continues_after_focal_event_until_system_failure() -> None:
    class Backend:
        def __init__(self):
            self.steps = 0

        def state(self) -> MultiAgentState:
            raw = np.zeros((10, 5), dtype=np.float32)
            boxes = np.asarray(
                [[slot * 10.0, 0.0, 0.0, 4.0, 2.0] for slot in range(10)]
            )
            if self.steps == 1:
                raw[0, 0] = 1
            elif self.steps == 2:
                raw[[2, 3], 1] = 1
                boxes[3, :2] = boxes[2, :2]
            return MultiAgentState(
                np.zeros((10, 2984), dtype=np.float32),
                raw,
                boxes,
                np.zeros(10, dtype=np.bool_),
                False,
            )

        def reset(self) -> MultiAgentState:
            self.steps = 0
            return self.state()

        def step(self, commands: np.ndarray) -> MultiAgentState:
            self.steps += 1
            return self.state()

    rollout = run_multiagent_rollout(
        backend=Backend(),
        victim=_Victim(),
        adversary=_Adversary((0.0, 0.0)),
        intervention=_intervention(),
        max_steps=5,
        failure_scope=NONFOCAL_SYSTEM_FAILURE_SCOPE,
    )
    assert rollout.failure_timestep == 1
    assert rollout.failure_by_transition.tolist() == [False, True]
    assert np.flatnonzero(rollout.failing_agents[-1]).tolist() == [2, 3]
    assert rollout.raw_info[1, 0, 0] == 1


def test_nonfocal_run_summary_ranks_slots_and_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        multiagent_artifact,
        "validate_multiagent_training_artifact",
        lambda path: {"ok": True, "failed_checks": []},
    )
    metrics = []
    for failures, slots, pairs in (
        (2, {"1": 2, "2": 1}, {"1-2": 1}),
        (1, {"2": 1}, {"2-3": 1}),
    ):
        slot_counts = {str(slot): 0 for slot in range(1, 10)}
        slot_counts.update(slots)
        metrics.append(
            {
                "episodes": 10,
                "failures": failures,
                "episodes_with_focal_safety_event": 3,
                "qualifying_failures_by_slot": slot_counts,
                "qualifying_failures_by_kind": {
                    "road_object_contact": 1,
                    "vehicle_collision": 1,
                    "nonvehicle_collision": 0,
                },
                "qualifying_vehicle_collision_pairs": pairs,
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": "run-id",
                "config": {"failure": {"scope": NONFOCAL_SYSTEM_FAILURE_SCOPE}},
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )
    summary = multiagent_artifact.summarize_nonfocal_system_run(tmp_path)
    assert summary["total_episodes"] == 20
    assert summary["total_qualifying_failure_episodes"] == 3
    assert summary["qualifying_failure_rate"] == pytest.approx(0.15)
    assert summary["ranked_failing_slots"][:2] == [
        {"slot": 1, "qualifying_failure_episodes": 2},
        {"slot": 2, "qualifying_failure_episodes": 2},
    ]
    assert summary["episodes_with_focal_safety_event_no_automatic_credit"] == 6
