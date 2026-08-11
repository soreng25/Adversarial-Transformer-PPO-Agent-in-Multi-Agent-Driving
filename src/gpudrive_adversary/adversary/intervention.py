"""Deterministic composition of victim commands and adversarial residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# These are the intervention limits approved for Milestone C.  They are tuples
# so callers cannot mutate process-global scientific configuration.
APPROVED_DISTURBANCE_BOUNDS = (0.667, 0.262)
APPROVED_COMMAND_LOWER = (-4.0, -3.142)
APPROVED_COMMAND_UPPER = (4.0, 3.142)
DISTURBANCE_COMPONENTS = ("delta_accel", "delta_steer")
COMMAND_COMPONENTS = ("accel", "steer", "head")


class InterventionContractError(ValueError):
    """Raised when an intervention input or configuration is invalid."""


def _immutable_pair(value: Any, *, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise InterventionContractError(f"{name} must have shape (2,); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise InterventionContractError(f"{name} must contain only finite values")
    if positive and not np.all(array > 0.0):
        raise InterventionContractError(f"{name} must be strictly positive")
    result = array.copy()
    result.setflags(write=False)
    return result


def _immutable(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class InterventionSpec:
    """Explicit residual and actuator bounds for a two-control intervention."""

    disturbance_bounds: np.ndarray
    command_lower: np.ndarray
    command_upper: np.ndarray

    def __post_init__(self) -> None:
        disturbance_bounds = _immutable_pair(
            self.disturbance_bounds, name="disturbance_bounds", positive=True
        )
        command_lower = _immutable_pair(self.command_lower, name="command_lower")
        command_upper = _immutable_pair(self.command_upper, name="command_upper")
        if not np.all(command_lower < command_upper):
            raise InterventionContractError(
                "command_lower must be strictly below command_upper componentwise"
            )
        object.__setattr__(self, "disturbance_bounds", disturbance_bounds)
        object.__setattr__(self, "command_lower", command_lower)
        object.__setattr__(self, "command_upper", command_upper)


@dataclass(frozen=True)
class InterventionResult:
    """Lossless record of requested, bounded, and physically applied control."""

    nominal_command: np.ndarray
    requested_disturbance: np.ndarray
    requested_command: np.ndarray
    applied_disturbance: np.ndarray
    bounded_command: np.ndarray
    applied_command: np.ndarray
    effective_disturbance: np.ndarray
    disturbance_saturated: np.ndarray
    command_saturated: np.ndarray
    saturated: np.ndarray


def apply_intervention(
    nominal_command: Any,
    requested_disturbance: Any,
    spec: InterventionSpec,
) -> InterventionResult:
    """Apply a bounded residual after victim discrete-action decoding.

    Arrays may have arbitrary matching leading dimensions.  Commands have
    trailing shape ``(..., 3)`` in ``[accel, steer, head]`` order, while the
    disturbance has trailing shape ``(..., 2)``.  The residual is first clipped
    to its approved disturbance bounds, then acceleration and steering are
    clipped to actuator bounds.  The head component is copied bit-for-bit from
    the nominal command and is never adversary-controlled.
    """

    if not isinstance(spec, InterventionSpec):
        raise InterventionContractError("spec must be InterventionSpec")
    nominal = np.asarray(nominal_command, dtype=np.float64)
    requested = np.asarray(requested_disturbance, dtype=np.float64)
    if nominal.ndim < 1 or nominal.shape[-1] != 3:
        raise InterventionContractError(
            f"nominal_command must have trailing dimension 3; got {nominal.shape}"
        )
    if requested.ndim < 1 or requested.shape[-1] != 2:
        raise InterventionContractError(
            "requested_disturbance must have trailing dimension 2; "
            f"got {requested.shape}"
        )
    if nominal.shape[:-1] != requested.shape[:-1]:
        raise InterventionContractError(
            "nominal command and disturbance leading shapes differ: "
            f"{nominal.shape[:-1]} != {requested.shape[:-1]}"
        )
    if not np.all(np.isfinite(nominal)):
        raise InterventionContractError("nominal_command must be finite")
    if not np.all(np.isfinite(requested)):
        raise InterventionContractError("requested_disturbance must be finite")

    requested_command = nominal.copy()
    requested_command[..., :2] += requested

    applied_disturbance = np.clip(
        requested, -spec.disturbance_bounds, spec.disturbance_bounds
    )
    disturbance_saturated = (requested < -spec.disturbance_bounds) | (
        requested > spec.disturbance_bounds
    )

    bounded_command = nominal.copy()
    bounded_command[..., :2] += applied_disturbance
    command_saturated = (bounded_command[..., :2] < spec.command_lower) | (
        bounded_command[..., :2] > spec.command_upper
    )
    applied_command = bounded_command.copy()
    applied_command[..., :2] = np.clip(
        bounded_command[..., :2], spec.command_lower, spec.command_upper
    )
    # Assign explicitly as a guard against future changes that might apply a
    # vectorized command clip across all three native command components.
    applied_command[..., 2] = nominal[..., 2]
    effective_disturbance = applied_command[..., :2] - nominal[..., :2]
    saturated = disturbance_saturated | command_saturated

    return InterventionResult(
        nominal_command=_immutable(nominal),
        requested_disturbance=_immutable(requested),
        requested_command=_immutable(requested_command),
        applied_disturbance=_immutable(applied_disturbance),
        bounded_command=_immutable(bounded_command),
        applied_command=_immutable(applied_command),
        effective_disturbance=_immutable(effective_disturbance),
        disturbance_saturated=_immutable(disturbance_saturated),
        command_saturated=_immutable(command_saturated),
        saturated=_immutable(saturated),
    )
