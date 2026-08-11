"""Small, explicit PPO update used by the Transformer adversary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Torch is supplied by the reference Linux/CUDA training image.
    import torch
    from torch import Tensor
except ModuleNotFoundError:  # Pure GAE tests remain available without Torch.
    torch = None
    Tensor = Any

from .model import CausalTransformerActorCritic, TorchRequiredError


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO optimization settings; environment rewards are configured elsewhere."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.policy_clip < 0.0 or self.value_clip < 0.0:
            raise ValueError("PPO clip widths must be non-negative")
        if self.value_coefficient < 0.0 or self.entropy_coefficient < 0.0:
            raise ValueError("loss coefficients must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.advantage_epsilon <= 0.0:
            raise ValueError("advantage_epsilon must be positive")


def _validate_gae_shapes(
    rewards: Any,
    values: Any,
    terminated: Any,
    bootstrap_value: Any,
) -> None:
    if rewards.shape != values.shape or rewards.shape != terminated.shape:
        raise ValueError("rewards, values, and terminated must have equal shapes")
    if rewards.ndim < 1 or rewards.shape[0] < 1:
        raise ValueError("GAE requires a non-empty leading time dimension")
    if bootstrap_value.shape != rewards.shape[1:]:
        raise ValueError(
            f"bootstrap_value must have shape {rewards.shape[1:]}, got "
            f"{bootstrap_value.shape}"
        )


def generalized_advantage_estimate_numpy(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    bootstrap_value: np.ndarray | float,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE/returns with true-terminal masks using NumPy.

    ``bootstrap_value`` is used after the final transition only when that
    transition is not terminal. Truncations should therefore be passed as
    ``terminated=False`` and supplied with their critic bootstrap value.
    """

    rewards_array = np.asarray(rewards)
    values_array = np.asarray(values)
    terminated_array = np.asarray(terminated, dtype=np.bool_)
    bootstrap_array = np.asarray(bootstrap_value)
    _validate_gae_shapes(
        rewards_array, values_array, terminated_array, bootstrap_array
    )
    if not np.isfinite(rewards_array).all() or not np.isfinite(values_array).all():
        raise ValueError("GAE inputs contain non-finite rewards or values")
    if not np.isfinite(bootstrap_array).all():
        raise ValueError("bootstrap_value contains non-finite values")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")

    dtype = np.result_type(rewards_array.dtype, values_array.dtype, np.float32)
    advantages = np.zeros(rewards_array.shape, dtype=dtype)
    next_advantage = np.zeros(rewards_array.shape[1:], dtype=dtype)
    bootstrap = bootstrap_array.astype(dtype, copy=False)
    for time_index in range(rewards_array.shape[0] - 1, -1, -1):
        nonterminal = (~terminated_array[time_index]).astype(dtype, copy=False)
        next_value = (
            bootstrap
            if time_index == rewards_array.shape[0] - 1
            else values_array[time_index + 1]
        )
        delta = (
            rewards_array[time_index]
            + gamma * next_value * nonterminal
            - values_array[time_index]
        )
        next_advantage = (
            delta + gamma * gae_lambda * nonterminal * next_advantage
        )
        advantages[time_index] = next_advantage
    returns = advantages + values_array
    return advantages, returns


def clipped_ppo_loss_numpy(
    *,
    new_log_probability: np.ndarray,
    old_log_probability: np.ndarray,
    new_value: np.ndarray,
    old_value: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    entropy: np.ndarray,
    config: PPOConfig | None = None,
) -> dict[str, float]:
    """Pure numerical mirror of the PPO objective for contract testing."""

    settings = config or PPOConfig()
    arrays = {
        "new_log_probability": np.asarray(new_log_probability),
        "old_log_probability": np.asarray(old_log_probability),
        "new_value": np.asarray(new_value),
        "old_value": np.asarray(old_value),
        "advantages": np.asarray(advantages),
        "returns": np.asarray(returns),
        "entropy": np.asarray(entropy),
    }
    expected_shape = arrays["old_log_probability"].shape
    if not expected_shape:
        raise ValueError("PPO loss requires at least one batch dimension")
    for name, value in arrays.items():
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    objective_advantages = arrays["advantages"]
    if settings.normalize_advantages and objective_advantages.size > 1:
        objective_advantages = (
            objective_advantages - objective_advantages.mean()
        ) / (objective_advantages.std() + settings.advantage_epsilon)
    log_ratio = arrays["new_log_probability"] - arrays["old_log_probability"]
    ratio = np.exp(log_ratio)
    unclipped_policy = ratio * objective_advantages
    clipped_policy = np.clip(
        ratio, 1.0 - settings.policy_clip, 1.0 + settings.policy_clip
    ) * objective_advantages
    policy_loss = -float(np.minimum(unclipped_policy, clipped_policy).mean())

    clipped_value = arrays["old_value"] + np.clip(
        arrays["new_value"] - arrays["old_value"],
        -settings.value_clip,
        settings.value_clip,
    )
    value_loss = 0.5 * float(
        np.maximum(
            (arrays["new_value"] - arrays["returns"]) ** 2,
            (clipped_value - arrays["returns"]) ** 2,
        ).mean()
    )
    entropy_mean = float(arrays["entropy"].mean())
    total_loss = (
        policy_loss
        + settings.value_coefficient * value_loss
        - settings.entropy_coefficient * entropy_mean
    )
    approximate_kl = float(((ratio - 1.0) - log_ratio).mean())
    clip_fraction = float(
        (np.abs(ratio - 1.0) > settings.policy_clip).mean()
    )
    return {
        "loss_total": total_loss,
        "loss_policy": policy_loss,
        "loss_value": value_loss,
        "entropy": entropy_mean,
        "approximate_kl": approximate_kl,
        "clip_fraction": clip_fraction,
    }


if torch is not None:

    @dataclass(slots=True)
    class PPOLossTerms:
        """Differentiable PPO objective and detached diagnostics."""

        total: Tensor
        policy: Tensor
        value: Tensor
        entropy: Tensor
        approximate_kl: Tensor
        clip_fraction: Tensor


    def generalized_advantage_estimate(
        rewards: Tensor,
        values: Tensor,
        terminated: Tensor,
        bootstrap_value: Tensor,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[Tensor, Tensor]:
        """Torch GAE equivalent that preserves device and tensor dtype."""

        _validate_gae_shapes(rewards, values, terminated, bootstrap_value)
        if not bool(torch.isfinite(rewards).all().item()):
            raise ValueError("rewards contain non-finite values")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("values contain non-finite values")
        if not bool(torch.isfinite(bootstrap_value).all().item()):
            raise ValueError("bootstrap_value contains non-finite values")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")

        advantages = torch.zeros_like(values)
        next_advantage = torch.zeros_like(bootstrap_value)
        terminal_mask = terminated.to(dtype=torch.bool)
        for time_index in range(rewards.shape[0] - 1, -1, -1):
            nonterminal = (~terminal_mask[time_index]).to(dtype=values.dtype)
            next_value = (
                bootstrap_value
                if time_index == rewards.shape[0] - 1
                else values[time_index + 1]
            )
            delta = (
                rewards[time_index]
                + gamma * next_value * nonterminal
                - values[time_index]
            )
            next_advantage = (
                delta + gamma * gae_lambda * nonterminal * next_advantage
            )
            advantages[time_index] = next_advantage
        return advantages, advantages + values


    def clipped_ppo_loss(
        *,
        new_log_probability: Tensor,
        old_log_probability: Tensor,
        new_value: Tensor,
        old_value: Tensor,
        advantages: Tensor,
        returns: Tensor,
        entropy: Tensor,
        config: PPOConfig,
    ) -> PPOLossTerms:
        """Compute clipped policy/value objectives and entropy regularization."""

        expected_shape = old_log_probability.shape
        named = {
            "new_log_probability": new_log_probability,
            "new_value": new_value,
            "old_value": old_value,
            "advantages": advantages,
            "returns": returns,
            "entropy": entropy,
        }
        for name, value in named.items():
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} contains non-finite values")
        if not bool(torch.isfinite(old_log_probability).all().item()):
            raise ValueError("old_log_probability contains non-finite values")

        normalized_advantages = advantages
        if config.normalize_advantages and advantages.numel() > 1:
            normalized_advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + config.advantage_epsilon)

        log_ratio = new_log_probability - old_log_probability
        ratio = log_ratio.exp()
        unclipped_policy = ratio * normalized_advantages
        clipped_policy = ratio.clamp(
            1.0 - config.policy_clip, 1.0 + config.policy_clip
        ) * normalized_advantages
        policy_loss = -torch.minimum(unclipped_policy, clipped_policy).mean()

        value_delta = new_value - old_value
        clipped_value = old_value + value_delta.clamp(
            -config.value_clip, config.value_clip
        )
        value_unclipped_error = (new_value - returns).square()
        value_clipped_error = (clipped_value - returns).square()
        value_loss = 0.5 * torch.maximum(
            value_unclipped_error, value_clipped_error
        ).mean()
        entropy_mean = entropy.mean()
        total_loss = (
            policy_loss
            + config.value_coefficient * value_loss
            - config.entropy_coefficient * entropy_mean
        )
        with torch.no_grad():
            approximate_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                (ratio - 1.0).abs() > config.policy_clip
            ).to(dtype=ratio.dtype).mean()
        return PPOLossTerms(
            total=total_loss,
            policy=policy_loss,
            value=value_loss,
            entropy=entropy_mean,
            approximate_kl=approximate_kl,
            clip_fraction=clip_fraction,
        )


    def ppo_minibatch_update(
        *,
        model: CausalTransformerActorCritic,
        optimizer: Any,
        tokens: Tensor,
        actions: Tensor,
        old_log_probability: Tensor,
        old_value: Tensor,
        advantages: Tensor,
        returns: Tensor,
        valid_mask: Tensor | None = None,
        bounds: Tensor | Any | None = None,
        config: PPOConfig | None = None,
    ) -> dict[str, float]:
        """Apply one small PPO minibatch update and return scalar diagnostics."""

        settings = config or PPOConfig()
        evaluated = model.evaluate_actions(
            tokens, actions, valid_mask, bounds=bounds
        )
        terms = clipped_ppo_loss(
            new_log_probability=evaluated.log_probability,
            old_log_probability=old_log_probability,
            new_value=evaluated.value,
            old_value=old_value,
            advantages=advantages,
            returns=returns,
            entropy=evaluated.entropy,
            config=settings,
        )
        optimizer.zero_grad(set_to_none=True)
        terms.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings.max_grad_norm
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            optimizer.zero_grad(set_to_none=True)
            raise ValueError("PPO gradient norm is non-finite")
        optimizer.step()
        return {
            "loss_total": float(terms.total.detach().cpu().item()),
            "loss_policy": float(terms.policy.detach().cpu().item()),
            "loss_value": float(terms.value.detach().cpu().item()),
            "entropy": float(terms.entropy.detach().cpu().item()),
            "approximate_kl": float(terms.approximate_kl.detach().cpu().item()),
            "clip_fraction": float(terms.clip_fraction.detach().cpu().item()),
            "gradient_norm": float(gradient_norm.detach().cpu().item()),
        }

else:

    class PPOLossTerms:  # pragma: no cover - dependency guard only
        pass


    def generalized_advantage_estimate(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise TorchRequiredError(
            "Torch GAE requires the pinned Linux/CUDA training environment; "
            "use generalized_advantage_estimate_numpy for pure contract tests"
        )


    def clipped_ppo_loss(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise TorchRequiredError(
            "clipped_ppo_loss requires Torch from the pinned Linux/CUDA "
            "training environment"
        )


    def ppo_minibatch_update(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise TorchRequiredError(
            "ppo_minibatch_update requires Torch from the pinned Linux/CUDA "
            "training environment"
        )


__all__ = [
    "PPOConfig",
    "PPOLossTerms",
    "clipped_ppo_loss",
    "clipped_ppo_loss_numpy",
    "generalized_advantage_estimate",
    "generalized_advantage_estimate_numpy",
    "ppo_minibatch_update",
]
