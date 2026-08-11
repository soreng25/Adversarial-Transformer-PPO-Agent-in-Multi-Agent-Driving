"""Victim-only failure classification and nominal-scene eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


RAW_INFO_ORDER = (
    "road_object_contact",
    "vehicle_collision",
    "nonvehicle_collision",
    "goal_achieved",
    "actor_type",
)


class FailureContractError(ValueError):
    """Raised when raw evidence cannot be interpreted without ambiguity."""


class TerminationReason(str, Enum):
    """Derived wrapper outcome, ordered by the classifier's precedence."""

    FAILURE = "failure"
    GOAL = "goal"
    HORIZON = "horizon"
    DONE_OTHER = "done_other"
    CONTINUE = "continue"


def _binary_flag(value: float, *, name: str) -> bool:
    if not np.isfinite(value) or value not in (0.0, 1.0):
        raise FailureContractError(f"{name} must be exactly 0 or 1; got {value!r}")
    return bool(value)


@dataclass(frozen=True)
class FailureEvidence:
    """Post-step evidence aligned with the action that caused the transition."""

    action_timestep: int
    road_object_contact: bool
    vehicle_collision: bool
    nonvehicle_collision: bool
    goal_achieved: bool
    horizon_reached: bool
    done: bool
    is_failure: bool
    failure_timestep: int | None
    failure_kinds: tuple[str, ...]
    termination_reason: TerminationReason


def classify_victim_post_step(
    raw_info: Any,
    action_timestep: int,
    *,
    horizon_reached: bool = False,
    done: bool = False,
) -> FailureEvidence:
    """Classify slot-0 raw info captured immediately after action ``t``.

    A vehicle collision, non-vehicle collision, or road-object contact is a
    failure.  Goal and horizon are never failures.  If a safety flag coincides
    with goal or horizon, the derived termination reason is ``failure`` while
    every raw flag remains available in the result.
    """

    if isinstance(action_timestep, bool) or not isinstance(
        action_timestep, (int, np.integer)
    ):
        raise FailureContractError("action_timestep must be an integer")
    if int(action_timestep) < 0:
        raise FailureContractError("action_timestep must be non-negative")
    if not isinstance(horizon_reached, (bool, np.bool_)):
        raise FailureContractError("horizon_reached must be boolean")
    if not isinstance(done, (bool, np.bool_)):
        raise FailureContractError("done must be boolean")

    raw = np.asarray(raw_info, dtype=np.float64)
    if raw.shape != (len(RAW_INFO_ORDER),):
        raise FailureContractError(
            f"raw_info must have shape ({len(RAW_INFO_ORDER)},); got {raw.shape}"
        )
    if not np.all(np.isfinite(raw)):
        raise FailureContractError("raw_info must contain only finite values")

    road_object_contact = _binary_flag(raw[0], name=RAW_INFO_ORDER[0])
    vehicle_collision = _binary_flag(raw[1], name=RAW_INFO_ORDER[1])
    nonvehicle_collision = _binary_flag(raw[2], name=RAW_INFO_ORDER[2])
    goal_achieved = _binary_flag(raw[3], name=RAW_INFO_ORDER[3])
    kinds = tuple(
        name
        for name, present in (
            ("road_object_contact", road_object_contact),
            ("vehicle_collision", vehicle_collision),
            ("nonvehicle_collision", nonvehicle_collision),
        )
        if present
    )
    is_failure = bool(kinds)

    if is_failure:
        reason = TerminationReason.FAILURE
    elif goal_achieved:
        reason = TerminationReason.GOAL
    elif bool(horizon_reached):
        reason = TerminationReason.HORIZON
    elif bool(done):
        reason = TerminationReason.DONE_OTHER
    else:
        reason = TerminationReason.CONTINUE

    timestep = int(action_timestep)
    return FailureEvidence(
        action_timestep=timestep,
        road_object_contact=road_object_contact,
        vehicle_collision=vehicle_collision,
        nonvehicle_collision=nonvehicle_collision,
        goal_achieved=goal_achieved,
        horizon_reached=bool(horizon_reached),
        done=bool(done),
        is_failure=is_failure,
        failure_timestep=timestep if is_failure else None,
        failure_kinds=kinds,
        termination_reason=reason,
    )


@dataclass(frozen=True)
class NominalEligibility:
    """Whether an undisturbed victim safely reached its goal."""

    eligible: bool
    reason: str
    goal_timestep: int | None
    failure_timestep: int | None
    failure_kinds: tuple[str, ...]


def assess_nominal_goal_eligibility(
    post_step_raw_info: Any,
    *,
    first_action_timestep: int = 0,
) -> NominalEligibility:
    """Require a clean nominal prefix ending in a victim goal event.

    Row ``i`` must be the slot-0 raw info observed immediately after action
    ``first_action_timestep + i``.  The first safety event makes the scene
    ineligible; a simultaneous safety/goal event is therefore ineligible.
    A clean first goal makes it eligible.  A trace with neither is ineligible
    because nominal goal success was not verified.
    """

    if isinstance(first_action_timestep, bool) or not isinstance(
        first_action_timestep, (int, np.integer)
    ):
        raise FailureContractError("first_action_timestep must be an integer")
    if int(first_action_timestep) < 0:
        raise FailureContractError("first_action_timestep must be non-negative")
    raw = np.asarray(post_step_raw_info, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(RAW_INFO_ORDER):
        raise FailureContractError(
            "post_step_raw_info must have shape "
            f"(transitions, {len(RAW_INFO_ORDER)}); got {raw.shape}"
        )
    if raw.shape[0] == 0:
        return NominalEligibility(False, "empty_trace", None, None, ())

    for offset, row in enumerate(raw):
        evidence = classify_victim_post_step(
            row, int(first_action_timestep) + offset
        )
        if evidence.is_failure:
            return NominalEligibility(
                eligible=False,
                reason="nominal_failure",
                goal_timestep=(
                    evidence.action_timestep if evidence.goal_achieved else None
                ),
                failure_timestep=evidence.failure_timestep,
                failure_kinds=evidence.failure_kinds,
            )
        if evidence.goal_achieved:
            return NominalEligibility(
                eligible=True,
                reason="clean_goal_success",
                goal_timestep=evidence.action_timestep,
                failure_timestep=None,
                failure_kinds=(),
            )
    return NominalEligibility(False, "goal_not_reached", None, None, ())
