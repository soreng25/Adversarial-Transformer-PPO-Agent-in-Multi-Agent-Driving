"""Bounded continuous distributions for the two-dimensional disturbance.

The policy and the reference prior use the same change of variables::

    latent ~ Normal(mean, sigma)
    disturbance = bounds * tanh(latent)

Consequently, samples have support on the *open* interval ``(-bounds,
bounds)``.  In particular, this is not a clipped Gaussian and it has no point
mass at either disturbance bound.  ``log_prob`` includes the exact Jacobian
of this transformation, so it can be used both for PPO ratios and for the
negative-log-likelihood penalty under an explicitly configured prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class DistributionContractError(ValueError):
    """Raised when a bounded-distribution contract is violated."""


def _immutable_vector(value: Any, *, name: str, positive: bool) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise DistributionContractError(f"{name} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(array)):
        raise DistributionContractError(f"{name} must contain only finite values")
    if positive and not np.all(array > 0.0):
        raise DistributionContractError(f"{name} must be strictly positive")
    result = array.copy()
    result.setflags(write=False)
    return result


def _event_array(value: Any, *, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != dimension:
        raise DistributionContractError(
            f"{name} must have trailing dimension {dimension}; got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise DistributionContractError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class BoundedSample:
    """One reparameterized draw and its density in disturbance space."""

    value: np.ndarray
    latent: np.ndarray
    log_prob: np.ndarray


@dataclass(frozen=True)
class BoundedTanhNormal:
    """Independent Normal components transformed into symmetric open bounds.

    Both ``bounds`` and ``sigma`` are required constructor arguments.  This is
    intentional: policy exploration scales and prior scales are research
    configuration, not implementation defaults.
    """

    bounds: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        bounds = _immutable_vector(self.bounds, name="bounds", positive=True)
        sigma = _immutable_vector(self.sigma, name="sigma", positive=True)
        if bounds.shape != sigma.shape:
            raise DistributionContractError(
                f"bounds and sigma shapes differ: {bounds.shape} != {sigma.shape}"
            )
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "sigma", sigma)

    @property
    def dimension(self) -> int:
        """Number of independent disturbance components."""

        return int(self.bounds.size)

    def transform(self, latent: Any) -> np.ndarray:
        """Map a latent vector to the open bounded disturbance support.

        No clipping is performed.  Extremely large finite floating-point
        latents can make ``tanh`` round to exactly +/-1; those values are
        rejected rather than manufacturing a boundary atom.
        """

        latent_array = _event_array(
            latent, dimension=self.dimension, name="latent"
        )
        normalized = np.tanh(latent_array)
        if np.any(np.abs(normalized) >= 1.0):
            raise DistributionContractError(
                "latent numerically saturates tanh at a disturbance boundary"
            )
        return self.bounds * normalized

    def inverse(self, value: Any) -> np.ndarray:
        """Invert an interior disturbance value to its pre-tanh latent."""

        value_array = _event_array(value, dimension=self.dimension, name="value")
        normalized = value_array / self.bounds
        if np.any(np.abs(normalized) >= 1.0):
            raise DistributionContractError(
                "value must lie strictly inside every disturbance bound"
            )
        return np.arctanh(normalized)

    def _broadcast_mean(self, mean: Any, shape: tuple[int, ...]) -> np.ndarray:
        mean_array = _event_array(mean, dimension=self.dimension, name="mean")
        try:
            result = np.broadcast_to(mean_array, shape)
        except ValueError as exc:
            raise DistributionContractError(
                f"mean shape {mean_array.shape} cannot broadcast to {shape}"
            ) from exc
        return result

    def log_prob_from_latent(self, latent: Any, mean: Any) -> np.ndarray:
        """Return exact disturbance-space log density for a pre-tanh latent."""

        latent_array = _event_array(
            latent, dimension=self.dimension, name="latent"
        )
        mean_array = self._broadcast_mean(mean, latent_array.shape)
        # Reject numerical boundary saturation consistently with transform().
        self.transform(latent_array)

        standardized = (latent_array - mean_array) / self.sigma
        normal_log_prob = -0.5 * standardized**2 - np.log(self.sigma) - 0.5 * np.log(
            2.0 * np.pi
        )
        # log(1 - tanh(z)^2), written stably for either sign of z.
        log_sech_squared = 2.0 * (
            np.log(2.0) - latent_array - np.logaddexp(0.0, -2.0 * latent_array)
        )
        log_abs_jacobian = np.log(self.bounds) + log_sech_squared
        return np.sum(normal_log_prob - log_abs_jacobian, axis=-1)

    def log_prob(self, value: Any, mean: Any) -> np.ndarray:
        """Return exact log density, with ``-inf`` outside the open support."""

        value_array = np.asarray(value, dtype=np.float64)
        if value_array.ndim < 1 or value_array.shape[-1] != self.dimension:
            raise DistributionContractError(
                f"value must have trailing dimension {self.dimension}; "
                f"got {value_array.shape}"
            )
        mean_array = self._broadcast_mean(mean, value_array.shape)
        normalized = value_array / self.bounds
        valid = np.all(np.isfinite(value_array) & (np.abs(normalized) < 1.0), axis=-1)

        # Invalid rows use a harmless interior placeholder and are replaced by
        # -inf below.  This avoids warnings while preserving standard density
        # semantics for boundary/out-of-support values.
        safe_normalized = np.where(valid[..., np.newaxis], normalized, 0.0)
        latent = np.arctanh(safe_normalized)
        standardized = (latent - mean_array) / self.sigma
        normal_log_prob = -0.5 * standardized**2 - np.log(self.sigma) - 0.5 * np.log(
            2.0 * np.pi
        )
        log_sech_squared = 2.0 * (
            np.log(2.0) - latent - np.logaddexp(0.0, -2.0 * latent)
        )
        density = np.sum(
            normal_log_prob - np.log(self.bounds) - log_sech_squared, axis=-1
        )
        return np.where(valid, density, -np.inf)

    def negative_log_likelihood(self, value: Any, mean: Any) -> np.ndarray:
        """Return ``-log_prob(value, mean)`` for a policy or prior penalty."""

        return -self.log_prob(value, mean)

    def nll(self, value: Any, mean: Any) -> np.ndarray:
        """Short alias for :meth:`negative_log_likelihood`."""

        return self.negative_log_likelihood(value, mean)

    def nll_excess(
        self,
        value: Any,
        mean: Any,
        reference_value: Any,
    ) -> np.ndarray:
        """Return NLL relative to an explicitly supplied reference action.

        This subtraction preserves the exact transformed-space density while
        allowing the no-disturbance reference to carry zero penalty.  It is not
        clipped: with a configuration whose reference is not a density mode,
        negative values correctly reveal that scientific inconsistency.
        """

        return self.nll(value, mean) - self.nll(reference_value, mean)

    def nll_excess_from_zero(self, value: Any, mean: Any) -> np.ndarray:
        """Return exact NLL excess over the zero-disturbance reference."""

        return self.nll_excess(value, mean, np.zeros(self.dimension, dtype=np.float64))

    def sample(self, rng: np.random.Generator, mean: Any) -> BoundedSample:
        """Draw a reparameterized sample using the caller-owned generator."""

        if not isinstance(rng, np.random.Generator):
            raise DistributionContractError("rng must be numpy.random.Generator")
        mean_array = _event_array(mean, dimension=self.dimension, name="mean")
        latent = rng.normal(loc=mean_array, scale=self.sigma)
        value = self.transform(latent)
        log_prob = self.log_prob_from_latent(latent, mean_array)
        return BoundedSample(value=value, latent=latent, log_prob=log_prob)
