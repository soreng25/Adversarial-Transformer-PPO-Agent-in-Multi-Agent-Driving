"""Causal rollout for ten frozen PPO vehicles and one disturbed focal car."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..adversary.environment import (
    AdversaryContext,
    AdversaryDecision,
    HISTORY_LENGTH,
    TOKEN_DIM,
    VICTIM_OBSERVATION_DIM,
    build_adversary_token,
)
from ..adversary.failure import RAW_INFO_ORDER
from ..victim.policy import assert_policy_frozen, select_deterministic
from .clearance import minimum_pairwise_clearance, oriented_box_clearance


AGENT_COUNT = 10
ACTION_DIM = 91
COMMAND_DIM = 3
DISTURBANCE_DIM = 2
ANY_CONTROLLED_FAILURE_SCOPE = "any_controlled_agent"
NONFOCAL_SYSTEM_FAILURE_SCOPE = "nonfocal_slots_1_through_9_only"
NONFOCAL_COLLISION_CONTACT_TOLERANCE_M = 1.0e-5


class MultiAgentEnvironmentError(RuntimeError):
    """Raised when the ten-agent causal or shape contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiAgentEnvironmentError(message)


def _array(value: Any, shape: tuple[int, ...], name: str, dtype: Any = np.float32) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    _require(result.shape == shape, f"{name} must have shape {shape}; got {result.shape}")
    if result.dtype.kind == "f":
        _require(bool(np.isfinite(result).all()), f"{name} contains non-finite values")
    return np.ascontiguousarray(result).copy()


@dataclass(frozen=True)
class MultiAgentState:
    observations: np.ndarray
    raw_info: np.ndarray
    boxes: np.ndarray
    done: np.ndarray
    horizon_reached: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", _array(self.observations, (AGENT_COUNT, VICTIM_OBSERVATION_DIM), "observations"))
        object.__setattr__(self, "raw_info", _array(self.raw_info, (AGENT_COUNT, 5), "raw info"))
        object.__setattr__(self, "boxes", _array(self.boxes, (AGENT_COUNT, 5), "oriented boxes", np.float64))
        object.__setattr__(self, "done", _array(self.done, (AGENT_COUNT,), "done", np.bool_))


@dataclass(frozen=True)
class MultiAgentVictimDecision:
    action_indices: np.ndarray
    logits: np.ndarray
    nominal_commands: np.ndarray

    def __post_init__(self) -> None:
        actions = _array(self.action_indices, (AGENT_COUNT,), "victim actions", np.int64)
        logits = _array(self.logits, (AGENT_COUNT, ACTION_DIM), "victim logits")
        commands = _array(self.nominal_commands, (AGENT_COUNT, COMMAND_DIM), "nominal commands")
        _require(np.array_equal(actions, np.argmax(logits, axis=1)), "victim actions are not deterministic argmax")
        object.__setattr__(self, "action_indices", actions)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "nominal_commands", commands)


class MultiAgentVictimPolicy:
    """One frozen shared network evaluated as a batch of ten observations."""

    def __init__(self, policy: Any, action_table: np.ndarray, torch: Any, *, device: str):
        self.policy = policy
        self.action_table = _array(action_table, (ACTION_DIM, COMMAND_DIM), "action table")
        self.torch = torch
        self.device = device

    def assert_frozen(self) -> None:
        assert_policy_frozen(self.policy)

    def act_deterministic(self, observations: np.ndarray) -> MultiAgentVictimDecision:
        values = _array(observations, (AGENT_COUNT, VICTIM_OBSERVATION_DIM), "policy observations")
        tensor = self.torch.as_tensor(values, dtype=self.torch.float32, device=self.device)
        result = select_deterministic(self.policy, tensor)
        actions = result["action"].detach().cpu().numpy().astype(np.int64, copy=True)
        logits = result["logits"].detach().cpu().numpy().astype(np.float32, copy=True)
        return MultiAgentVictimDecision(actions, logits, self.action_table[actions])


class Backend(Protocol):
    def reset(self) -> MultiAgentState: ...
    def step(self, commands: np.ndarray) -> MultiAgentState: ...


class Victim(Protocol):
    def assert_frozen(self) -> None: ...
    def act_deterministic(self, observations: np.ndarray) -> MultiAgentVictimDecision: ...


class Adversary(Protocol):
    def act(self, context: AdversaryContext, *, deterministic: bool) -> AdversaryDecision: ...


def _history(tokens: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    history = np.zeros((HISTORY_LENGTH, TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(HISTORY_LENGTH, dtype=np.bool_)
    suffix = tokens[-HISTORY_LENGTH:]
    history[-len(suffix):] = np.stack(suffix)
    mask[-len(suffix):] = True
    return history, mask


def _global_failure(raw_info: np.ndarray) -> tuple[bool, tuple[str, ...], np.ndarray]:
    raw = _array(raw_info, (AGENT_COUNT, 5), "post-step raw info")
    flags = raw[:, :3]
    _require(bool(np.isin(flags, [0.0, 1.0]).all()), "safety flags must be binary")
    failed_agents = np.any(flags != 0, axis=1)
    kinds = tuple(RAW_INFO_ORDER[index] for index in range(3) if np.any(flags[:, index] != 0))
    return bool(np.any(failed_agents)), kinds, failed_agents


def qualifying_nonfocal_collision_pairs(
    raw_info: np.ndarray,
    boxes: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """Attribute vehicle contacts only when both participants are slots 1-9."""

    raw = _array(raw_info, (AGENT_COUNT, 5), "post-step raw info")
    geometry = _array(boxes, (AGENT_COUNT, 5), "post-step oriented boxes", np.float64)
    flags = raw[:, :3]
    _require(bool(np.isin(flags, [0.0, 1.0]).all()), "safety flags must be binary")
    vehicle_flags = raw[:, 1] != 0
    pairs: list[tuple[int, int]] = []
    for first in range(1, AGENT_COUNT):
        if not vehicle_flags[first]:
            continue
        for second in range(first + 1, AGENT_COUNT):
            if vehicle_flags[second] and oriented_box_clearance(
                geometry[first], geometry[second]
            ) <= NONFOCAL_COLLISION_CONTACT_TOLERANCE_M:
                pairs.append((first, second))
    return tuple(pairs)


def classify_multiagent_failure(
    raw_info: np.ndarray,
    boxes: np.ndarray,
    *,
    scope: str,
) -> tuple[bool, tuple[str, ...], np.ndarray]:
    """Classify either the legacy any-agent event or the non-focal objective."""

    if scope == ANY_CONTROLLED_FAILURE_SCOPE:
        return _global_failure(raw_info)
    _require(
        scope == NONFOCAL_SYSTEM_FAILURE_SCOPE,
        f"unsupported multi-agent failure scope: {scope}",
    )
    raw = _array(raw_info, (AGENT_COUNT, 5), "post-step raw info")
    flags = raw[:, :3]
    _require(bool(np.isin(flags, [0.0, 1.0]).all()), "safety flags must be binary")
    agents = np.zeros(AGENT_COUNT, dtype=np.bool_)
    road_agents = np.flatnonzero(raw[1:, 0] != 0) + 1
    nonvehicle_agents = np.flatnonzero(raw[1:, 2] != 0) + 1
    collision_pairs = qualifying_nonfocal_collision_pairs(raw, boxes)
    agents[road_agents] = True
    agents[nonvehicle_agents] = True
    for first, second in collision_pairs:
        agents[first] = True
        agents[second] = True
    kinds: list[str] = []
    if road_agents.size:
        kinds.append(RAW_INFO_ORDER[0])
    if collision_pairs:
        kinds.append(RAW_INFO_ORDER[1])
    if nonvehicle_agents.size:
        kinds.append(RAW_INFO_ORDER[2])
    return bool(agents.any()), tuple(kinds), agents


def _clearance_active_mask(done: np.ndarray, *, scope: str) -> np.ndarray:
    active = ~np.asarray(done, dtype=np.bool_)
    if scope == NONFOCAL_SYSTEM_FAILURE_SCOPE:
        active[0] = False
    return active


@dataclass(frozen=True)
class MultiAgentRollout:
    observations: np.ndarray
    raw_info: np.ndarray
    boxes: np.ndarray
    done: np.ndarray
    goal_ever: np.ndarray
    minimum_clearance: np.ndarray
    closest_pair: np.ndarray
    tokens: np.ndarray
    histories: np.ndarray
    history_masks: np.ndarray
    victim_action_indices: np.ndarray
    victim_logits: np.ndarray
    nominal_commands: np.ndarray
    applied_commands: np.ndarray
    disturbance_requested: np.ndarray
    disturbance_effective: np.ndarray
    disturbance_saturated: np.ndarray
    command_saturated: np.ndarray
    prior_nll_exact: np.ndarray
    policy_log_probability: np.ndarray
    adversary_values: np.ndarray
    pre_actor_latent: np.ndarray
    failure_by_transition: np.ndarray
    failing_agents: np.ndarray
    failure_kinds: tuple[tuple[str, ...], ...]
    failure_timestep: int | None
    termination_reason: str

    @property
    def transition_count(self) -> int:
        return int(self.victim_action_indices.shape[0])

    @property
    def all_goals_reached(self) -> bool:
        return bool(self.goal_ever[-1].all())

    @property
    def episode_minimum_clearance(self) -> float:
        return float(np.min(self.minimum_clearance))

    def validate(self) -> None:
        transitions = self.transition_count
        states = transitions + 1
        shapes = {
            "observations": (states, AGENT_COUNT, VICTIM_OBSERVATION_DIM),
            "raw_info": (states, AGENT_COUNT, 5), "boxes": (states, AGENT_COUNT, 5),
            "done": (states, AGENT_COUNT), "goal_ever": (states, AGENT_COUNT),
            "minimum_clearance": (states,), "closest_pair": (states, 2),
            "tokens": (transitions, TOKEN_DIM), "histories": (transitions, HISTORY_LENGTH, TOKEN_DIM),
            "history_masks": (transitions, HISTORY_LENGTH), "victim_action_indices": (transitions, AGENT_COUNT),
            "victim_logits": (transitions, AGENT_COUNT, ACTION_DIM), "nominal_commands": (transitions, AGENT_COUNT, COMMAND_DIM),
            "applied_commands": (transitions, AGENT_COUNT, COMMAND_DIM), "disturbance_requested": (transitions, DISTURBANCE_DIM),
            "disturbance_effective": (transitions, DISTURBANCE_DIM), "disturbance_saturated": (transitions, DISTURBANCE_DIM),
            "command_saturated": (transitions, DISTURBANCE_DIM), "prior_nll_exact": (transitions,),
            "policy_log_probability": (transitions,), "adversary_values": (transitions,),
            "failure_by_transition": (transitions,), "failing_agents": (transitions, AGENT_COUNT),
        }
        for name, shape in shapes.items():
            _require(np.asarray(getattr(self, name)).shape == shape, f"{name} trace shape changed")
        _require(self.pre_actor_latent.ndim == 2 and self.pre_actor_latent.shape[0] == transitions, "latent trace shape changed")
        _require(len(self.failure_kinds) == transitions, "failure-kind clock changed")
        failures = np.flatnonzero(self.failure_by_transition)
        if self.failure_timestep is None:
            _require(failures.size == 0, "failure trace lacks failure timestep")
        else:
            _require(failures.tolist() == [self.failure_timestep] and self.failure_timestep == transitions - 1, "failure timestep is not the causing final action")


def run_multiagent_rollout(
    *,
    backend: Backend,
    victim: Victim,
    adversary: Adversary,
    intervention: Any,
    max_steps: int,
    adversary_deterministic: bool = False,
    failure_scope: str = ANY_CONTROLLED_FAILURE_SCOPE,
) -> MultiAgentRollout:
    """Run ten deterministic PPO agents, perturbing only slot zero."""

    _require(max_steps > 0, "max_steps must be positive")
    victim.assert_frozen()
    state = backend.reset()
    _require(
        failure_scope in {ANY_CONTROLLED_FAILURE_SCOPE, NONFOCAL_SYSTEM_FAILURE_SCOPE},
        "unsupported multi-agent failure scope",
    )
    initial_failed, _, _ = _global_failure(state.raw_info)
    _require(not initial_failed, "derived scene begins with a safety event")
    initial_clearance, initial_pair = minimum_pairwise_clearance(
        state.boxes, _clearance_active_mask(state.done, scope=failure_scope)
    )
    _require(np.isfinite(initial_clearance) and initial_clearance > 0, "derived scene begins touching or overlapping")

    observations = [state.observations]; raw_infos = [state.raw_info]; boxes = [state.boxes]; dones = [state.done]
    goals = [state.raw_info[:, 3].astype(np.bool_)]; clearances = [initial_clearance]; pairs = [initial_pair]
    tokens: list[np.ndarray] = []; histories: list[np.ndarray] = []; masks: list[np.ndarray] = []
    actions: list[np.ndarray] = []; logits: list[np.ndarray] = []; nominal: list[np.ndarray] = []; applied: list[np.ndarray] = []
    requested: list[np.ndarray] = []; effective: list[np.ndarray] = []; disturbance_clip: list[np.ndarray] = []; command_clip: list[np.ndarray] = []
    nll: list[float] = []; log_prob: list[float] = []; values: list[float] = []; latents: list[np.ndarray] = []
    failed_clock: list[bool] = []; failing_agents: list[np.ndarray] = []; failure_kinds: list[tuple[str, ...]] = []
    previous_command = np.zeros(3, dtype=np.float32); previous_disturbance = np.zeros(2, dtype=np.float32)
    failure_timestep: int | None = None; termination = "horizon"

    for timestep in range(max_steps):
        token = build_adversary_token(state.observations[0], previous_command, previous_disturbance)
        tokens.append(token); history, mask = _history(tokens)
        adversary_decision = adversary.act(AdversaryContext(timestep, token, history, mask), deterministic=adversary_deterministic)
        victim_decision = victim.act_deterministic(state.observations)
        composed = intervention(victim_decision.nominal_commands[0], adversary_decision.requested_disturbance)
        commands = victim_decision.nominal_commands.copy()
        commands[0] = composed.applied_command
        successor = backend.step(commands)
        failed, kinds, agent_flags = classify_multiagent_failure(
            successor.raw_info, successor.boxes, scope=failure_scope
        )
        goal_ever = goals[-1] | (successor.raw_info[:, 3] != 0)
        active = _clearance_active_mask(successor.done, scope=failure_scope)
        clearance, pair = minimum_pairwise_clearance(successor.boxes, active)
        if not np.isfinite(clearance):
            clearance, pair = clearances[-1], pairs[-1]

        histories.append(history); masks.append(mask); actions.append(victim_decision.action_indices); logits.append(victim_decision.logits)
        nominal.append(victim_decision.nominal_commands); applied.append(commands); requested.append(np.asarray(composed.requested_disturbance))
        effective.append(np.asarray(composed.effective_disturbance)); disturbance_clip.append(np.asarray(composed.disturbance_saturated)); command_clip.append(np.asarray(composed.command_saturated))
        nll.append(float(adversary_decision.negative_log_likelihood)); log_prob.append(float(adversary_decision.policy_log_probability)); values.append(float(adversary_decision.value)); latents.append(np.asarray(adversary_decision.pre_actor_latent))
        failed_clock.append(failed); failing_agents.append(agent_flags); failure_kinds.append(kinds)
        observations.append(successor.observations); raw_infos.append(successor.raw_info); boxes.append(successor.boxes); dones.append(successor.done); goals.append(goal_ever); clearances.append(clearance); pairs.append(pair)
        previous_command = victim_decision.nominal_commands[0]; previous_disturbance = np.asarray(composed.effective_disturbance); state = successor
        if failed:
            failure_timestep = timestep; termination = "failure"; break
        if bool(goal_ever.all()):
            termination = "all_goals_reached"; break
        if successor.horizon_reached or timestep + 1 >= max_steps:
            termination = "horizon"; break

    victim.assert_frozen()
    stack = lambda values: np.stack(values, axis=0)
    trace = MultiAgentRollout(
        stack(observations), stack(raw_infos), stack(boxes), stack(dones), stack(goals), np.asarray(clearances), np.asarray(pairs, dtype=np.int64),
        stack(tokens), stack(histories), stack(masks), stack(actions), stack(logits), stack(nominal), stack(applied),
        stack(requested), stack(effective), stack(disturbance_clip), stack(command_clip), np.asarray(nll), np.asarray(log_prob), np.asarray(values, dtype=np.float32), stack(latents),
        np.asarray(failed_clock, dtype=np.bool_), stack(failing_agents), tuple(failure_kinds), failure_timestep, termination,
    )
    trace.validate()
    return trace
