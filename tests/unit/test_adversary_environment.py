from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gpudrive_adversary.adversary.environment import (
    HISTORY_LENGTH,
    TOKEN_DIM,
    VICTIM_OBSERVATION_DIM,
    AdversaryContext,
    AdversaryDecision,
    AdversaryEnvironmentError,
    BackendState,
    VictimDecision,
    run_sequential_rollout,
)
from gpudrive_adversary.adversary.failure import classify_victim_post_step
from gpudrive_adversary.adversary.intervention import (
    APPROVED_COMMAND_LOWER,
    APPROVED_COMMAND_UPPER,
    APPROVED_DISTURBANCE_BOUNDS,
    InterventionSpec,
    apply_intervention,
)


pytestmark = pytest.mark.unit


def _observation(state_index: int) -> np.ndarray:
    return np.full(VICTIM_OBSERVATION_DIM, state_index, dtype=np.float32)


class FakeBackend:
    def __init__(self, events: list[str], *, failure_on: int | None) -> None:
        self.events = events
        self.failure_on = failure_on
        self.transition = 0
        self.commands: list[np.ndarray] = []

    def reset(self) -> BackendState:
        self.transition = 0
        self.commands.clear()
        return BackendState(
            _observation(0), np.asarray([0, 0, 0, 0, 1], dtype=np.float32)
        )

    def step(self, applied_command: np.ndarray) -> BackendState:
        timestep = self.transition
        self.events.append(f"step:{timestep}")
        self.commands.append(np.asarray(applied_command).copy())
        self.transition += 1
        raw_info = np.asarray([0, 0, 0, 0, 1], dtype=np.float32)
        if self.failure_on == timestep:
            raw_info[1] = 1  # victim vehicle collision after this action
        return BackendState(_observation(self.transition), raw_info)


class FakeVictim:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.frozen_assertions = 0

    def assert_frozen(self) -> None:
        self.frozen_assertions += 1

    def act_deterministic(self, observation: np.ndarray) -> VictimDecision:
        timestep = int(observation[0])
        self.events.append(f"victim:{timestep}")
        action = timestep % 3
        logits = np.zeros(3, dtype=np.float32)
        logits[action] = 1.0
        command = np.asarray(
            [1.0 + timestep, 0.2 * timestep, 0.0], dtype=np.float32
        )
        return VictimDecision(action, logits, command)


class FakeAdversary:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.contexts: list[AdversaryContext] = []

    def act(
        self, context: AdversaryContext, *, deterministic: bool
    ) -> AdversaryDecision:
        del deterministic
        self.events.append(f"adversary:{context.timestep}")
        self.contexts.append(context)
        disturbance = np.asarray(
            [0.1 * (context.timestep + 1), -0.05], dtype=np.float32
        )
        return AdversaryDecision(
            requested_disturbance=disturbance,
            negative_log_likelihood=0.5 + context.timestep,
            pre_actor_latent=np.full(64, context.timestep, dtype=np.float32),
            raw_action=disturbance / np.asarray(
                APPROVED_DISTURBANCE_BOUNDS, dtype=np.float32
            ),
            policy_log_probability=-0.25 - context.timestep,
            value=10.0 + context.timestep,
        )


class LoggedIntervention:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.spec = InterventionSpec(
            APPROVED_DISTURBANCE_BOUNDS,
            APPROVED_COMMAND_LOWER,
            APPROVED_COMMAND_UPPER,
        )

    def __call__(
        self, nominal_command: np.ndarray, requested_disturbance: np.ndarray
    ):
        timestep = int(round(float(nominal_command[0] - 1.0)))
        self.events.append(f"intervention:{timestep}")
        return apply_intervention(
            nominal_command, requested_disturbance, self.spec
        )


class LoggedFailureClassifier:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(
        self,
        raw_info: np.ndarray,
        action_timestep: int,
        *,
        horizon_reached: bool = False,
        done: bool = False,
    ):
        self.events.append(f"failure:{action_timestep}")
        return classify_victim_post_step(
            raw_info,
            action_timestep,
            horizon_reached=horizon_reached,
            done=done,
        )


def _run(*, failure_on: int | None, max_steps: int):
    events: list[str] = []
    backend = FakeBackend(events, failure_on=failure_on)
    victim = FakeVictim(events)
    adversary = FakeAdversary(events)
    trace = run_sequential_rollout(
        backend=backend,
        victim=victim,
        adversary=adversary,
        intervention=LoggedIntervention(events),
        failure_classifier=LoggedFailureClassifier(events),
        max_steps=max_steps,
        adversary_deterministic=False,
    )
    return trace, events, backend, victim, adversary


def test_rollout_enforces_causal_order_and_previous_step_token() -> None:
    trace, events, backend, victim, adversary = _run(failure_on=1, max_steps=8)

    assert events == [
        "adversary:0",
        "victim:0",
        "intervention:0",
        "step:0",
        "failure:0",
        "adversary:1",
        "victim:1",
        "intervention:1",
        "step:1",
        "failure:1",
    ]
    assert victim.frozen_assertions == 2
    assert trace.victim_observations.shape == (3, VICTIM_OBSERVATION_DIM)
    assert trace.evidence.shape == (3, 5)
    assert trace.victim_action_indices.shape == (2,)
    assert trace.disturbance_requested.shape == (2, 2)
    assert trace.negative_log_likelihood.tolist() == [0.5, 1.5]
    assert len(backend.commands) == 2

    first = adversary.contexts[0]
    np.testing.assert_array_equal(first.token[-5:], np.zeros(5, dtype=np.float32))
    assert first.history.shape == (HISTORY_LENGTH, TOKEN_DIM)
    assert first.history_mask.sum() == 1
    assert first.history_mask[-1]

    second = adversary.contexts[1]
    np.testing.assert_array_equal(
        second.token[:VICTIM_OBSERVATION_DIM], _observation(1)
    )
    np.testing.assert_allclose(
        second.token[-5:],
        np.asarray([1.0, 0.0, 0.0, 0.1, -0.05], dtype=np.float32),
    )
    # The same-state victim command is [2.0, 0.2, 0.0], and is deliberately
    # absent: the token carries only the previous nominal command.
    assert not np.array_equal(
        second.token[-5:-2], trace.victim_nominal_commands[1]
    )
    assert second.history_mask.sum() == 2
    np.testing.assert_array_equal(second.history[-2], trace.tokens[0])
    np.testing.assert_array_equal(second.history[-1], trace.tokens[1])
    assert not second.token.flags.writeable
    assert not second.history.flags.writeable


def test_failure_index_selects_latent_immediately_before_causing_action() -> None:
    trace, _, _, _, _ = _run(failure_on=2, max_steps=8)

    assert trace.failure_timestep == 2
    assert trace.failure_step_count == 3
    assert trace.transition_count == 3
    assert trace.failure_by_transition.tolist() == [False, False, True]
    np.testing.assert_array_equal(
        trace.final_pre_failure_latent,
        np.full(64, 2, dtype=np.float32),
    )
    assert not np.array_equal(
        trace.final_pre_failure_latent, trace.pre_actor_latent[1]
    )

    off_by_one = replace(
        trace,
        failure_timestep=1,
        failure_step_count=2,
    )
    with pytest.raises(
        AdversaryEnvironmentError,
        match="failure_timestep does not index the failure-causing action",
    ):
        off_by_one.validate()


def test_nonfailure_horizon_has_null_failure_and_no_latent_selection() -> None:
    trace, _, _, _, _ = _run(failure_on=None, max_steps=2)

    assert trace.termination_reason == "horizon"
    assert trace.failure_timestep is None
    assert trace.failure_step_count is None
    assert not trace.failure_by_transition.any()
    with pytest.raises(
        AdversaryEnvironmentError,
        match="non-failure rollout has no final pre-failure latent",
    ):
        _ = trace.final_pre_failure_latent

