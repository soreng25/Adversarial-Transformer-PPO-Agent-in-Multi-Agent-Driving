"""Causal sequential-adversary rollout plumbing.

This module intentionally depends on neither Torch nor GPUDrive at import time.
Native integrations implement the small protocols below; unit tests can use
in-memory fakes.  The transition clock is normative::

    state/token -> adversary -> frozen victim -> intervention -> simulator
                -> post-step failure classification

In particular, the adversary is called before the current victim decision is
computed and its context type contains no victim-action field.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # Imports are only for static integration with sibling modules.
    from .failure import FailureEvidence
    from .intervention import InterventionResult


VICTIM_OBSERVATION_DIM = 2_984
NOMINAL_COMMAND_DIM = 3
DISTURBANCE_DIM = 2
TOKEN_DIM = (
    VICTIM_OBSERVATION_DIM + NOMINAL_COMMAND_DIM + DISTURBANCE_DIM
)
HISTORY_LENGTH = 50


class AdversaryEnvironmentError(RuntimeError):
    """Raised when a rollout component violates the causal trace contract."""


def _vector(
    value: Any,
    size: int | None,
    *,
    name: str,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
    finite: bool = True,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1 or (size is not None and array.shape != (size,)):
        expected = "a vector" if size is None else f"shape ({size},)"
        raise AdversaryEnvironmentError(
            f"{name} must have {expected}; got {array.shape}"
        )
    if finite and not bool(np.isfinite(array).all()):
        raise AdversaryEnvironmentError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array).copy()


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _scalar(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise AdversaryEnvironmentError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class BackendState:
    """The state slice needed by the policies and the failure classifier.

    ``evidence`` is deliberately backend-defined, but it must be a fixed-shape
    non-object NumPy value so a rollout can save all ``T + 1`` states without
    pickle.  For GPUDrive this is the victim's immediately captured raw
    post-step evidence (and can include a wider, fixed-shape state tensor).
    """

    victim_observation: np.ndarray
    evidence: np.ndarray
    done: bool = False
    horizon_reached: bool = False
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        observation = _vector(
            self.victim_observation,
            VICTIM_OBSERVATION_DIM,
            name="victim observation",
        )
        evidence = np.asarray(self.evidence)
        if evidence.dtype.hasobject:
            raise AdversaryEnvironmentError("backend evidence cannot use object dtype")
        if evidence.ndim == 0:
            evidence = evidence.reshape(1)
        reason = self.termination_reason
        if reason is not None and not str(reason).strip():
            raise AdversaryEnvironmentError("termination reason cannot be empty")
        object.__setattr__(self, "victim_observation", _readonly(observation))
        object.__setattr__(self, "evidence", _readonly(evidence))
        object.__setattr__(self, "done", bool(self.done))
        object.__setattr__(self, "horizon_reached", bool(self.horizon_reached))
        object.__setattr__(self, "termination_reason", reason)


@dataclass(frozen=True)
class VictimDecision:
    """Deterministic native victim output and its decoded physical command."""

    action_index: int
    logits: np.ndarray
    nominal_command: np.ndarray

    def __post_init__(self) -> None:
        logits = _vector(self.logits, None, name="victim logits")
        action = int(self.action_index)
        if logits.size == 0:
            raise AdversaryEnvironmentError("victim logits cannot be empty")
        if action < 0 or action >= logits.size:
            raise AdversaryEnvironmentError(
                f"victim action index {action} is outside {logits.size} logits"
            )
        if action != int(np.argmax(logits)):
            raise AdversaryEnvironmentError(
                "victim decision is not deterministic argmax (lowest-index tie rule)"
            )
        command = _vector(
            self.nominal_command,
            NOMINAL_COMMAND_DIM,
            name="nominal victim command",
        )
        object.__setattr__(self, "action_index", action)
        object.__setattr__(self, "logits", _readonly(logits))
        object.__setattr__(self, "nominal_command", _readonly(command))


@dataclass(frozen=True)
class AdversaryContext:
    """The adversary's complete information set at one decision.

    There is intentionally no current victim action or command here.  A token
    is ``[observation_t, nominal_command_(t-1), effective_disturbance_(t-1)]``.
    Histories are chronological and right aligned; ``history_mask`` marks the
    valid suffix.
    """

    timestep: int
    token: np.ndarray
    history: np.ndarray
    history_mask: np.ndarray

    def __post_init__(self) -> None:
        timestep = int(self.timestep)
        if timestep < 0:
            raise AdversaryEnvironmentError("adversary timestep cannot be negative")
        token = _vector(self.token, TOKEN_DIM, name="adversary token")
        history = np.asarray(self.history, dtype=np.float32)
        mask = np.asarray(self.history_mask, dtype=np.bool_)
        if history.shape != (HISTORY_LENGTH, TOKEN_DIM):
            raise AdversaryEnvironmentError(
                f"history must have shape ({HISTORY_LENGTH}, {TOKEN_DIM}); "
                f"got {history.shape}"
            )
        if mask.shape != (HISTORY_LENGTH,):
            raise AdversaryEnvironmentError(
                f"history mask must have shape ({HISTORY_LENGTH},); got {mask.shape}"
            )
        if not bool(mask[-1]):
            raise AdversaryEnvironmentError("current token must be history-valid")
        valid_count = int(mask.sum())
        expected = np.zeros(HISTORY_LENGTH, dtype=np.bool_)
        expected[-valid_count:] = True
        if not np.array_equal(mask, expected):
            raise AdversaryEnvironmentError(
                "history mask must be one right-aligned contiguous valid suffix"
            )
        if not np.array_equal(history[-1], token):
            raise AdversaryEnvironmentError("current token is not last in history")
        if not bool(np.isfinite(history).all()):
            raise AdversaryEnvironmentError("adversary history contains non-finite values")
        object.__setattr__(self, "timestep", timestep)
        object.__setattr__(self, "token", _readonly(token))
        object.__setattr__(self, "history", _readonly(history))
        object.__setattr__(self, "history_mask", _readonly(mask))


@dataclass(frozen=True)
class AdversaryDecision:
    """Backend-neutral result of one sequential adversary call."""

    requested_disturbance: np.ndarray
    negative_log_likelihood: float
    pre_actor_latent: np.ndarray
    raw_action: np.ndarray | None = None
    policy_log_probability: float = 0.0
    value: float = 0.0

    def __post_init__(self) -> None:
        disturbance = _vector(
            self.requested_disturbance,
            DISTURBANCE_DIM,
            name="requested disturbance",
        )
        latent = _vector(
            self.pre_actor_latent, None, name="pre-actor latent"
        )
        if latent.size == 0:
            raise AdversaryEnvironmentError("pre-actor latent cannot be empty")
        raw = disturbance if self.raw_action is None else self.raw_action
        raw_array = _vector(raw, None, name="raw adversary action")
        if raw_array.size == 0:
            raise AdversaryEnvironmentError("raw adversary action cannot be empty")
        object.__setattr__(self, "requested_disturbance", _readonly(disturbance))
        object.__setattr__(self, "negative_log_likelihood", _scalar(
            self.negative_log_likelihood, name="negative log likelihood"
        ))
        object.__setattr__(self, "pre_actor_latent", _readonly(latent))
        object.__setattr__(self, "raw_action", _readonly(raw_array))
        object.__setattr__(self, "policy_log_probability", _scalar(
            self.policy_log_probability, name="policy log probability"
        ))
        object.__setattr__(self, "value", _scalar(self.value, name="adversary value"))


@runtime_checkable
class RolloutBackend(Protocol):
    def reset(self) -> BackendState:
        """Reset the exact scene/config and return state zero."""

    def step(self, applied_command: np.ndarray) -> BackendState:
        """Apply one physical ``[acceleration, steering, head]`` command."""


@runtime_checkable
class FrozenDeterministicVictim(Protocol):
    def assert_frozen(self) -> None:
        """Fail unless inference mode/frozen-parameter invariants hold."""

    def act_deterministic(self, observation: np.ndarray) -> VictimDecision:
        """Return stable argmax output for exactly this pre-action state."""


@runtime_checkable
class SequentialAdversary(Protocol):
    def act(
        self, context: AdversaryContext, *, deterministic: bool
    ) -> AdversaryDecision:
        """Choose from the permitted context and expose the pre-actor latent."""


class InterventionLike(Protocol):
    nominal_command: np.ndarray
    requested_disturbance: np.ndarray
    effective_disturbance: np.ndarray
    applied_command: np.ndarray
    disturbance_saturated: np.ndarray
    command_saturated: np.ndarray


@runtime_checkable
class Intervention(Protocol):
    def __call__(
        self, nominal_command: np.ndarray, requested_disturbance: np.ndarray
    ) -> InterventionLike:
        """Compose through the one shared bounded intervention adapter."""


class FailureEvidenceLike(Protocol):
    is_failure: bool
    failure_timestep: int | None
    failure_kinds: tuple[str, ...]
    termination_reason: str | None


@runtime_checkable
class FailureClassifier(Protocol):
    def __call__(
        self,
        raw_info: np.ndarray,
        action_timestep: int,
        *,
        horizon_reached: bool = False,
        done: bool = False,
    ) -> FailureEvidenceLike:
        """Classify immediate victim post-step evidence."""


class PinnedVictimPolicyAdapter:
    """Lazy Torch adapter for the frozen policy API used by Milestone B.

    Constructing this object does not import Torch.  The supplied action table
    is the already verified 91-by-3 physical table from ``victim.policy``.
    """

    def __init__(self, policy: Any, action_table: np.ndarray, *, device: str) -> None:
        table = np.asarray(action_table, dtype=np.float32)
        if table.ndim != 2 or table.shape[1] != NOMINAL_COMMAND_DIM:
            raise AdversaryEnvironmentError(
                f"victim action table must have shape (A, 3); got {table.shape}"
            )
        if not bool(np.isfinite(table).all()):
            raise AdversaryEnvironmentError("victim action table is non-finite")
        self._policy = policy
        self._action_table = _readonly(table)
        self._device = str(device)

    def assert_frozen(self) -> None:
        from ..victim.policy import assert_policy_frozen

        assert_policy_frozen(self._policy)

    def act_deterministic(self, observation: np.ndarray) -> VictimDecision:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - native integration only
            raise AdversaryEnvironmentError(f"Torch is unavailable: {exc}") from exc
        from ..victim.policy import select_deterministic

        self.assert_frozen()
        vector = _vector(
            observation,
            VICTIM_OBSERVATION_DIM,
            name="victim policy observation",
        )
        tensor = torch.as_tensor(
            vector.reshape(1, -1), dtype=torch.float32, device=self._device
        )
        result = select_deterministic(self._policy, tensor)
        action = int(result["action"].detach().cpu().item())
        logits = (
            result["logits"][0].detach().cpu().contiguous().numpy().copy()
        )
        if action >= self._action_table.shape[0]:
            raise AdversaryEnvironmentError(
                f"victim action {action} has no physical action-table row"
            )
        return VictimDecision(action, logits, self._action_table[action])


def build_adversary_token(
    victim_observation: np.ndarray,
    previous_nominal_command: np.ndarray,
    previous_applied_disturbance: np.ndarray,
) -> np.ndarray:
    """Build the exact 2,989-D causal token in canonical float32."""

    observation = _vector(
        victim_observation,
        VICTIM_OBSERVATION_DIM,
        name="token victim observation",
    )
    command = _vector(
        previous_nominal_command,
        NOMINAL_COMMAND_DIM,
        name="previous nominal command",
    )
    disturbance = _vector(
        previous_applied_disturbance,
        DISTURBANCE_DIM,
        name="previous applied disturbance",
    )
    token = np.concatenate((observation, command, disturbance)).astype(
        np.float32, copy=False
    )
    if token.shape != (TOKEN_DIM,):  # Defensive if constants are edited.
        raise AdversaryEnvironmentError(f"built token has shape {token.shape}")
    return token


def _history(tokens: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((HISTORY_LENGTH, TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(HISTORY_LENGTH, dtype=np.bool_)
    suffix = tokens[-HISTORY_LENGTH:]
    values[-len(suffix) :] = np.stack(suffix, axis=0)
    mask[-len(suffix) :] = True
    return values, mask


def _stack(values: list[np.ndarray], *, name: str) -> np.ndarray:
    try:
        return np.stack(values, axis=0)
    except ValueError as exc:
        raise AdversaryEnvironmentError(
            f"{name} changed shape during rollout: {exc}"
        ) from exc


@dataclass(frozen=True)
class SequentialRollout:
    """A validated in-memory trajectory with one explicit transition clock."""

    victim_observations: np.ndarray
    evidence: np.ndarray
    backend_done: np.ndarray
    tokens: np.ndarray
    histories: np.ndarray
    history_masks: np.ndarray
    victim_action_indices: np.ndarray
    victim_logits: np.ndarray
    victim_nominal_commands: np.ndarray
    adversary_raw_actions: np.ndarray
    disturbance_requested: np.ndarray
    disturbance_effective: np.ndarray
    applied_commands: np.ndarray
    disturbance_saturated: np.ndarray
    command_saturated: np.ndarray
    negative_log_likelihood: np.ndarray
    policy_log_probability: np.ndarray
    adversary_values: np.ndarray
    pre_actor_latent: np.ndarray
    failure_by_transition: np.ndarray
    failure_kinds: tuple[tuple[str, ...], ...]
    failure_timestep: int | None
    failure_step_count: int | None
    termination_reason: str

    @property
    def transition_count(self) -> int:
        return int(self.victim_action_indices.shape[0])

    @property
    def final_pre_failure_latent(self) -> np.ndarray:
        """Return a copy of the latent captured before the causing action."""

        self.validate()
        if self.failure_timestep is None:
            raise AdversaryEnvironmentError(
                "a non-failure rollout has no final pre-failure latent"
            )
        return self.pre_actor_latent[self.failure_timestep].copy()

    def validate(self) -> None:
        transitions = self.transition_count
        states = transitions + 1
        if self.victim_observations.shape != (
            states,
            VICTIM_OBSERVATION_DIM,
        ):
            raise AdversaryEnvironmentError(
                "victim observations must be state-aligned with shape "
                f"({states}, {VICTIM_OBSERVATION_DIM})"
            )
        if self.evidence.shape[0] != states:
            raise AdversaryEnvironmentError("evidence must contain T+1 states")
        if self.backend_done.shape != (states,):
            raise AdversaryEnvironmentError("backend_done must contain T+1 values")

        fixed_shapes = {
            "tokens": (transitions, TOKEN_DIM),
            "histories": (transitions, HISTORY_LENGTH, TOKEN_DIM),
            "history_masks": (transitions, HISTORY_LENGTH),
            "victim_action_indices": (transitions,),
            "victim_nominal_commands": (transitions, NOMINAL_COMMAND_DIM),
            "disturbance_requested": (transitions, DISTURBANCE_DIM),
            "disturbance_effective": (transitions, DISTURBANCE_DIM),
            "applied_commands": (transitions, NOMINAL_COMMAND_DIM),
            "disturbance_saturated": (transitions, DISTURBANCE_DIM),
            "command_saturated": (transitions, DISTURBANCE_DIM),
            "negative_log_likelihood": (transitions,),
            "policy_log_probability": (transitions,),
            "adversary_values": (transitions,),
            "failure_by_transition": (transitions,),
        }
        for name, shape in fixed_shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise AdversaryEnvironmentError(
                    f"{name} must have shape {shape}; got "
                    f"{np.asarray(getattr(self, name)).shape}"
                )
        for name in ("victim_logits", "adversary_raw_actions", "pre_actor_latent"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or value.shape[0] != transitions:
                raise AdversaryEnvironmentError(f"{name} must be a T-by-width array")
        if len(self.failure_kinds) != transitions:
            raise AdversaryEnvironmentError("failure kinds must be transition-aligned")
        if transitions and (
            self.victim_logits.shape[1] == 0
            or self.adversary_raw_actions.shape[1] == 0
            or self.pre_actor_latent.shape[1] == 0
        ):
            raise AdversaryEnvironmentError("transition feature widths cannot be zero")

        zero_command = np.zeros(NOMINAL_COMMAND_DIM, dtype=np.float32)
        zero_disturbance = np.zeros(DISTURBANCE_DIM, dtype=np.float32)
        for timestep in range(transitions):
            previous_command = (
                zero_command
                if timestep == 0
                else self.victim_nominal_commands[timestep - 1]
            )
            previous_disturbance = (
                zero_disturbance
                if timestep == 0
                else self.disturbance_effective[timestep - 1]
            )
            expected_token = build_adversary_token(
                self.victim_observations[timestep],
                previous_command,
                previous_disturbance,
            )
            if not np.array_equal(self.tokens[timestep], expected_token):
                raise AdversaryEnvironmentError(
                    f"token {timestep} violates previous-step causal construction"
                )
            first = max(0, timestep + 1 - HISTORY_LENGTH)
            expected_history = np.zeros(
                (HISTORY_LENGTH, TOKEN_DIM), dtype=np.float32
            )
            suffix = self.tokens[first : timestep + 1]
            expected_history[-len(suffix) :] = suffix
            expected_mask = np.zeros(HISTORY_LENGTH, dtype=np.bool_)
            expected_mask[-len(suffix) :] = True
            if not np.array_equal(self.histories[timestep], expected_history):
                raise AdversaryEnvironmentError(
                    f"history {timestep} is not the causal 50-token suffix"
                )
            if not np.array_equal(self.history_masks[timestep], expected_mask):
                raise AdversaryEnvironmentError(
                    f"history mask {timestep} is not the valid suffix mask"
                )

        failures = np.flatnonzero(self.failure_by_transition)
        if failures.size:
            if failures.tolist() != [transitions - 1]:
                raise AdversaryEnvironmentError(
                    "rollout must stop immediately at the first classified failure"
                )
            expected_timestep = int(failures[0])
            if self.failure_timestep != expected_timestep:
                raise AdversaryEnvironmentError(
                    "failure_timestep does not index the failure-causing action"
                )
            if self.failure_step_count != expected_timestep + 1:
                raise AdversaryEnvironmentError(
                    "failure_step_count must equal failure_timestep + 1"
                )
        elif self.failure_timestep is not None or self.failure_step_count is not None:
            raise AdversaryEnvironmentError(
                "non-failure trace must use null failure indices"
            )
        if not str(self.termination_reason).strip():
            raise AdversaryEnvironmentError("termination reason cannot be empty")


def _termination_reason(result: FailureEvidenceLike) -> str | None:
    value = getattr(result, "termination_reason", None)
    if value is None:
        return None
    # ``failure.TerminationReason`` is a ``str, Enum`` (not ``StrEnum``), so
    # ``str(member)`` is its qualified name.  Prefer the serialized value.
    reason = str(getattr(value, "value", value)).strip()
    if not reason or reason == "continue":
        return None
    return reason


def run_sequential_rollout(
    *,
    backend: RolloutBackend,
    victim: FrozenDeterministicVictim,
    adversary: SequentialAdversary,
    intervention: Intervention,
    failure_classifier: FailureClassifier,
    max_steps: int,
    adversary_deterministic: bool = False,
) -> SequentialRollout:
    """Run one causal rollout and return a self-validating typed trace.

    The victim is asserted frozen before reset and after the last transition.
    Initial previous-command and disturbance fields are canonical zero vectors.
    Failure is evaluated only on successor evidence, and wins over a simultaneous
    horizon or backend-done condition.
    """

    limit = int(max_steps)
    if limit <= 0:
        raise AdversaryEnvironmentError("max_steps must be positive")
    victim.assert_frozen()
    state = backend.reset()
    if not isinstance(state, BackendState):
        raise AdversaryEnvironmentError("backend.reset() must return BackendState")
    if state.done or state.horizon_reached:
        raise AdversaryEnvironmentError(
            "backend reset returned an already-terminal state"
        )

    observations = [state.victim_observation.copy()]
    evidence = [state.evidence.copy()]
    backend_done = [state.done]
    tokens: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    action_indices: list[int] = []
    victim_logits: list[np.ndarray] = []
    nominal_commands: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    requested_disturbances: list[np.ndarray] = []
    effective_disturbances: list[np.ndarray] = []
    applied_commands: list[np.ndarray] = []
    disturbance_saturated: list[np.ndarray] = []
    command_saturated: list[np.ndarray] = []
    nlls: list[float] = []
    policy_log_probabilities: list[float] = []
    adversary_values: list[float] = []
    latents: list[np.ndarray] = []
    failure_by_transition: list[bool] = []
    failure_kinds: list[tuple[str, ...]] = []

    previous_command = np.zeros(NOMINAL_COMMAND_DIM, dtype=np.float32)
    previous_disturbance = np.zeros(DISTURBANCE_DIM, dtype=np.float32)
    failure_timestep: int | None = None
    termination_reason = "external_limit"

    for timestep in range(limit):
        # This scheduling makes the information boundary mechanically auditable:
        # no current victim decision exists when the adversary is called.
        token = build_adversary_token(
            state.victim_observation,
            previous_command,
            previous_disturbance,
        )
        tokens.append(token.copy())
        history, history_mask = _history(tokens)
        context = AdversaryContext(timestep, token, history, history_mask)
        adversary_decision = adversary.act(
            context, deterministic=bool(adversary_deterministic)
        )
        if not isinstance(adversary_decision, AdversaryDecision):
            raise AdversaryEnvironmentError(
                "adversary.act() must return AdversaryDecision"
            )

        victim_decision = victim.act_deterministic(
            state.victim_observation.copy()
        )
        if not isinstance(victim_decision, VictimDecision):
            raise AdversaryEnvironmentError(
                "victim.act_deterministic() must return VictimDecision"
            )
        composed = intervention(
            victim_decision.nominal_command.copy(),
            adversary_decision.requested_disturbance.copy(),
        )
        nominal = _vector(
            composed.nominal_command,
            NOMINAL_COMMAND_DIM,
            name="intervention nominal command",
        )
        requested = _vector(
            composed.requested_disturbance,
            DISTURBANCE_DIM,
            name="intervention requested disturbance",
        )
        effective = _vector(
            composed.effective_disturbance,
            DISTURBANCE_DIM,
            name="effective disturbance",
        )
        applied = _vector(
            composed.applied_command,
            NOMINAL_COMMAND_DIM,
            name="applied command",
        )
        disturbance_clip = _vector(
            composed.disturbance_saturated,
            DISTURBANCE_DIM,
            name="disturbance saturation",
            dtype=np.bool_,
            finite=False,
        )
        command_clip = _vector(
            composed.command_saturated,
            DISTURBANCE_DIM,
            name="command saturation",
            dtype=np.bool_,
            finite=False,
        )
        if not np.array_equal(nominal, victim_decision.nominal_command):
            raise AdversaryEnvironmentError(
                "intervention changed the recorded nominal victim command"
            )
        if not np.array_equal(
            requested, adversary_decision.requested_disturbance
        ):
            raise AdversaryEnvironmentError(
                "intervention changed the recorded requested disturbance"
            )

        successor = backend.step(applied.copy())
        if not isinstance(successor, BackendState):
            raise AdversaryEnvironmentError("backend.step() must return BackendState")
        if successor.evidence.shape != evidence[0].shape:
            raise AdversaryEnvironmentError(
                "backend evidence shape changed after a transition"
            )
        horizon_reached = bool(
            successor.horizon_reached or timestep + 1 >= limit
        )
        classified = failure_classifier(
            successor.evidence.copy(),
            timestep,
            horizon_reached=horizon_reached,
            done=successor.done,
        )
        failed = bool(classified.is_failure)
        classified_timestep = getattr(classified, "failure_timestep", None)
        if failed and classified_timestep != timestep:
            raise AdversaryEnvironmentError(
                "failure classifier did not identify the current action timestep"
            )
        if not failed and classified_timestep is not None:
            raise AdversaryEnvironmentError(
                "non-failure classification carried a failure timestep"
            )

        histories.append(history)
        masks.append(history_mask)
        action_indices.append(victim_decision.action_index)
        victim_logits.append(victim_decision.logits.copy())
        nominal_commands.append(nominal)
        raw_actions.append(adversary_decision.raw_action.copy())
        requested_disturbances.append(requested)
        effective_disturbances.append(effective)
        applied_commands.append(applied)
        disturbance_saturated.append(disturbance_clip)
        command_saturated.append(command_clip)
        nlls.append(adversary_decision.negative_log_likelihood)
        policy_log_probabilities.append(
            adversary_decision.policy_log_probability
        )
        adversary_values.append(adversary_decision.value)
        latents.append(adversary_decision.pre_actor_latent.copy())
        failure_by_transition.append(failed)
        kinds = tuple(str(value) for value in classified.failure_kinds)
        failure_kinds.append(kinds)
        observations.append(successor.victim_observation.copy())
        evidence.append(successor.evidence.copy())
        backend_done.append(successor.done)

        state = successor
        previous_command = nominal
        previous_disturbance = effective
        classified_reason = _termination_reason(classified)
        if failed:
            failure_timestep = timestep
            termination_reason = classified_reason or "safety_failure"
            break
        if classified_reason is not None:
            termination_reason = classified_reason
            break
        if successor.done:
            termination_reason = successor.termination_reason or "backend_done"
            break
        if horizon_reached:
            termination_reason = successor.termination_reason or "horizon"
            break

    victim.assert_frozen()
    transitions = len(action_indices)

    # Non-empty rollouts are guaranteed by a positive limit and nonterminal reset.
    trace = SequentialRollout(
        victim_observations=_stack(observations, name="victim observations"),
        evidence=_stack(evidence, name="backend evidence"),
        backend_done=np.asarray(backend_done, dtype=np.bool_),
        tokens=_stack(tokens, name="adversary tokens"),
        histories=_stack(histories, name="adversary histories"),
        history_masks=_stack(masks, name="history masks"),
        victim_action_indices=np.asarray(action_indices, dtype=np.int64),
        victim_logits=_stack(victim_logits, name="victim logits"),
        victim_nominal_commands=_stack(
            nominal_commands, name="nominal commands"
        ),
        adversary_raw_actions=_stack(raw_actions, name="raw adversary actions"),
        disturbance_requested=_stack(
            requested_disturbances, name="requested disturbances"
        ),
        disturbance_effective=_stack(
            effective_disturbances, name="effective disturbances"
        ),
        applied_commands=_stack(applied_commands, name="applied commands"),
        disturbance_saturated=_stack(
            disturbance_saturated, name="disturbance saturation"
        ),
        command_saturated=_stack(command_saturated, name="command saturation"),
        negative_log_likelihood=np.asarray(nlls, dtype=np.float64),
        policy_log_probability=np.asarray(
            policy_log_probabilities, dtype=np.float64
        ),
        adversary_values=np.asarray(adversary_values, dtype=np.float32),
        pre_actor_latent=_stack(latents, name="pre-actor latents"),
        failure_by_transition=np.asarray(
            failure_by_transition, dtype=np.bool_
        ),
        failure_kinds=tuple(failure_kinds),
        failure_timestep=failure_timestep,
        failure_step_count=(
            None if failure_timestep is None else failure_timestep + 1
        ),
        termination_reason=termination_reason,
    )
    if trace.transition_count != transitions:  # Defensive construction check.
        raise AdversaryEnvironmentError("transition count changed during assembly")
    trace.validate()
    return trace


def with_failure_index(
    trace: SequentialRollout, failure_timestep: int | None
) -> SequentialRollout:
    """Testing/validation helper that revalidates a replaced failure index."""

    updated = replace(
        trace,
        failure_timestep=failure_timestep,
        failure_step_count=(
            None if failure_timestep is None else int(failure_timestep) + 1
        ),
    )
    updated.validate()
    return updated
