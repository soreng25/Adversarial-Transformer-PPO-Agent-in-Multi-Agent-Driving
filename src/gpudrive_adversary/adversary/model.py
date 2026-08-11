"""One-layer causal Transformer actor-critic for bounded disturbances.

The module deliberately keeps normalized actions in ``[-1, 1]^2``. Converting
those normalized values to physical acceleration and steering residuals belongs
to the environment adapter, where the approved, dimensioned bounds live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Torch is supplied by the reference Linux/CUDA training image.
    import torch
    from torch import Tensor, nn
    from torch.distributions import Normal
except ModuleNotFoundError:  # Keep pure contract tests usable without Torch.
    torch = None
    Tensor = Any
    nn = None
    Normal = None


class TorchRequiredError(RuntimeError):
    """Raised when a neural-network operation is requested without Torch."""


def torch_available() -> bool:
    """Return whether the optional training dependency is importable."""

    return torch is not None


@dataclass(frozen=True, slots=True)
class AdversaryModelConfig:
    """Pinned compatibility architecture for the sequential adversary."""

    token_dim: int
    context_length: int = 50
    model_dim: int = 64
    pre_actor_dim: int = 64
    action_dim: int = 2
    num_layers: int = 1
    num_heads: int = 1
    feed_forward_dim: int = 64
    dropout: float = 0.0
    initial_log_std: float = -0.5
    minimum_log_std: float = -5.0
    maximum_log_std: float = 1.0
    action_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if self.context_length != 50:
            raise ValueError("the methodology port requires a 50-step context")
        if self.model_dim != 64 or self.pre_actor_dim != 64:
            raise ValueError("the methodology port requires 64-D model/latent features")
        if self.action_dim != 2:
            raise ValueError("the GPUDrive disturbance has two normalized components")
        if self.num_layers != 1 or self.num_heads != 1:
            raise ValueError("the compatibility architecture has one layer and one head")
        if self.feed_forward_dim <= 0:
            raise ValueError("feed_forward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.minimum_log_std > self.initial_log_std:
            raise ValueError("initial_log_std is below minimum_log_std")
        if self.initial_log_std > self.maximum_log_std:
            raise ValueError("initial_log_std is above maximum_log_std")
        if not 0.0 < self.action_epsilon < 0.5:
            raise ValueError("action_epsilon must be in (0, 0.5)")


def causal_attention_mask(
    length: int, *, context_length: int = 50
) -> np.ndarray:
    """Return a boolean mask where ``True`` entries are forbidden attention.

    This pure NumPy seam makes the no-future-information contract testable on
    machines that intentionally do not carry the training stack.
    """

    if length < 1:
        raise ValueError("sequence length must be positive")
    if length > context_length:
        raise ValueError(
            f"sequence length {length} exceeds context length {context_length}"
        )
    return np.triu(np.ones((length, length), dtype=np.bool_), k=1)


if torch is not None:

    @dataclass(slots=True)
    class ActorCriticOutput:
        """Distribution parameters, value, and the stable pre-action latent."""

        mean: Tensor
        log_std: Tensor
        value: Tensor
        pre_actor_features: Tensor


    @dataclass(slots=True)
    class BoundedActionOutput:
        """A normalized bounded action and its policy statistics."""

        action: Tensor
        normalized_action: Tensor
        raw_action: Tensor
        log_probability: Tensor
        entropy: Tensor
        value: Tensor
        pre_actor_features: Tensor
        mean: Tensor
        log_std: Tensor

        @property
        def pre_tanh(self) -> Tensor:
            """Alias used by trajectory serializers and policy replay."""

            return self.raw_action

        @property
        def log_prob(self) -> Tensor:
            return self.log_probability

        @property
        def latent(self) -> Tensor:
            return self.pre_actor_features


    @dataclass(slots=True)
    class ActionEvaluation:
        """Policy statistics for a supplied normalized action."""

        log_probability: Tensor
        entropy: Tensor
        value: Tensor
        pre_actor_features: Tensor
        mean: Tensor
        log_std: Tensor

        @property
        def log_prob(self) -> Tensor:
            return self.log_probability

        @property
        def latent(self) -> Tensor:
            return self.pre_actor_features


    class CausalTransformerActorCritic(nn.Module):
        """Single-layer causal Transformer with two-dimensional bounded actor."""

        def __init__(self, config: AdversaryModelConfig) -> None:
            super().__init__()
            self.config = config
            self.input_projection = nn.Linear(config.token_dim, config.model_dim)
            self.position_embedding = nn.Parameter(
                torch.zeros(config.context_length, config.model_dim)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feed_forward_dim,
                dropout=config.dropout,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.num_layers,
                enable_nested_tensor=False,
            )
            self.pre_actor = nn.Sequential(
                nn.Linear(config.model_dim, config.pre_actor_dim),
                nn.ReLU(),
            )
            self.actor_mean = nn.Linear(config.pre_actor_dim, config.action_dim)
            self.actor_log_std = nn.Parameter(
                torch.full((config.action_dim,), config.initial_log_std)
            )
            self.critic = nn.Linear(config.model_dim, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            """Use small actor outputs while retaining standard layer init."""

            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
            nn.init.zeros_(self.actor_mean.bias)
            nn.init.orthogonal_(self.critic.weight, gain=1.0)
            nn.init.zeros_(self.critic.bias)

        def _validate_inputs(
            self, tokens: Tensor, valid_mask: Tensor | None
        ) -> Tensor | None:
            if tokens.ndim != 3:
                raise ValueError(
                    f"tokens must have shape [batch, time, token_dim], got {tokens.shape}"
                )
            if tokens.shape[-1] != self.config.token_dim:
                raise ValueError(
                    f"token dimension is {tokens.shape[-1]}, expected "
                    f"{self.config.token_dim}"
                )
            if tokens.shape[1] < 1 or tokens.shape[1] > self.config.context_length:
                raise ValueError(
                    f"time dimension must be in [1, {self.config.context_length}]"
                )
            if not bool(torch.isfinite(tokens).all().item()):
                raise ValueError("tokens contain non-finite values")
            if valid_mask is None:
                return None
            if valid_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"valid_mask must have shape {tokens.shape[:2]}, got {valid_mask.shape}"
                )
            valid_mask = valid_mask.to(device=tokens.device, dtype=torch.bool)
            if not bool(valid_mask[:, -1].all().item()):
                raise ValueError("the current (last) token must be valid in every batch row")
            counts = valid_mask.sum(dim=1)
            positions = torch.arange(
                tokens.shape[1], device=tokens.device
            ).unsqueeze(0)
            expected = positions >= (tokens.shape[1] - counts).unsqueeze(1)
            if not bool(torch.equal(valid_mask, expected)):
                raise ValueError("valid_mask must be one right-aligned contiguous suffix")
            return valid_mask

        @staticmethod
        def _compact_valid_suffix(
            tokens: Tensor, valid_mask: Tensor | None
        ) -> tuple[Tensor, Tensor | None]:
            """Move each valid suffix left so no causal query is fully masked."""

            if valid_mask is None:
                return tokens, None
            batch, length = valid_mask.shape
            positions = torch.arange(length, device=tokens.device).expand(batch, -1)
            # Every valid key sorts ahead of every padded key while preserving
            # chronological order within both groups.
            sort_keys = torch.where(valid_mask, positions, positions + length)
            order = torch.argsort(sort_keys, dim=1)
            compacted = tokens.gather(
                1, order.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            )
            counts = valid_mask.sum(dim=1)
            compact_mask = positions < counts.unsqueeze(1)
            return compacted, compact_mask

        @staticmethod
        def _current_encoding(
            encoded: Tensor, valid_mask: Tensor | None
        ) -> Tensor:
            if valid_mask is None:
                return encoded[:, -1]
            current_indices = valid_mask.sum(dim=1) - 1
            batch_indices = torch.arange(encoded.shape[0], device=encoded.device)
            return encoded[batch_indices, current_indices]

        def encode_sequence(
            self, tokens: Tensor, valid_mask: Tensor | None = None
        ) -> Tensor:
            """Encode every prefix causally; output position ``t`` sees no future."""

            valid_mask = self._validate_inputs(tokens, valid_mask)
            tokens, compact_mask = self._compact_valid_suffix(tokens, valid_mask)
            length = tokens.shape[1]
            hidden = self.input_projection(tokens)
            hidden = hidden + self.position_embedding[:length].unsqueeze(0)
            blocked = torch.as_tensor(
                causal_attention_mask(
                    length, context_length=self.config.context_length
                ),
                device=tokens.device,
            )
            padding_mask = None if compact_mask is None else ~compact_mask
            return self.transformer(
                hidden,
                mask=blocked,
                src_key_padding_mask=padding_mask,
                is_causal=True,
            )

        def get_pre_actor_features(
            self, tokens: Tensor, valid_mask: Tensor | None = None
        ) -> Tensor:
            """Return the 64-D latent immediately before the action head.

            Calling this with the history ending at timestep ``t`` is the stable
            extraction point required for a latent associated with action ``t``.
            """

            encoded = self.encode_sequence(tokens, valid_mask)
            return self.pre_actor(self._current_encoding(encoded, valid_mask))

        def forward(
            self, tokens: Tensor, valid_mask: Tensor | None = None
        ) -> ActorCriticOutput:
            encoded = self.encode_sequence(tokens, valid_mask)
            current = self._current_encoding(encoded, valid_mask)
            features = self.pre_actor(current)
            mean = self.actor_mean(features)
            bounded_log_std = self.actor_log_std.clamp(
                min=self.config.minimum_log_std,
                max=self.config.maximum_log_std,
            )
            log_std = bounded_log_std.expand_as(mean)
            value = self.critic(current).squeeze(-1)
            return ActorCriticOutput(
                mean=mean,
                log_std=log_std,
                value=value,
                pre_actor_features=features,
            )

        @staticmethod
        def _base_distribution(output: ActorCriticOutput) -> Normal:
            return Normal(output.mean, output.log_std.exp())

        def _squashed_log_probability(
            self,
            distribution: Normal,
            raw_action: Tensor,
        ) -> Tensor:
            base_log_probability = distribution.log_prob(raw_action)
            # Stable exact log(1 - tanh(z)^2), shared with the pure NumPy
            # bounded-distribution contract. This avoids an epsilon-biased PPO
            # density when the action approaches a bound.
            log_jacobian = 2.0 * (
                np.log(2.0)
                - raw_action
                - torch.nn.functional.softplus(-2.0 * raw_action)
            )
            return (base_log_probability - log_jacobian).sum(dim=-1)

        def _validated_bounds(
            self, bounds: Tensor | Any | None, reference: Tensor
        ) -> Tensor | None:
            if bounds is None:
                return None
            bound_tensor = torch.as_tensor(
                bounds, device=reference.device, dtype=reference.dtype
            )
            if bound_tensor.shape != (self.config.action_dim,):
                raise ValueError(
                    f"bounds must have shape ({self.config.action_dim},), got "
                    f"{bound_tensor.shape}"
                )
            if not bool(torch.isfinite(bound_tensor).all().item()):
                raise ValueError("bounds contain non-finite values")
            if not bool((bound_tensor > 0.0).all().item()):
                raise ValueError("symmetric disturbance bounds must be positive")
            return bound_tensor

        def act(
            self,
            tokens: Tensor,
            valid_mask: Tensor | None = None,
            *,
            deterministic: bool = False,
            bounds: Tensor | Any | None = None,
        ) -> BoundedActionOutput:
            """Sample a bounded action, optionally scaled to physical bounds.

            With ``bounds=None``, ``action`` and ``normalized_action`` are the
            same value in ``[-1, 1]^2``. With positive symmetric ``bounds[2]``,
            ``action = normalized_action * bounds`` and log probability is the
            density in those physical coordinates.
            """

            output = self(tokens, valid_mask)
            distribution = self._base_distribution(output)
            raw_action = output.mean if deterministic else distribution.rsample()
            normalized_action = torch.tanh(raw_action)
            log_probability = self._squashed_log_probability(
                distribution, raw_action
            )
            physical_bounds = self._validated_bounds(bounds, normalized_action)
            action = normalized_action
            if physical_bounds is not None:
                action = normalized_action * physical_bounds
                log_probability = log_probability - physical_bounds.log().sum()
            # The exact entropy of a tanh-Normal has no simple closed form. The
            # base entropy is the stable PPO exploration diagnostic/regularizer.
            entropy = distribution.entropy().sum(dim=-1)
            return BoundedActionOutput(
                action=action,
                normalized_action=normalized_action,
                raw_action=raw_action,
                log_probability=log_probability,
                entropy=entropy,
                value=output.value,
                pre_actor_features=output.pre_actor_features,
                mean=output.mean,
                log_std=output.log_std,
            )

        def evaluate_actions(
            self,
            tokens: Tensor,
            actions: Tensor,
            valid_mask: Tensor | None = None,
            *,
            bounds: Tensor | Any | None = None,
        ) -> ActionEvaluation:
            """Evaluate stored normalized or physically bounded PPO actions.

            Pass the same ``bounds`` used by :meth:`act` when ``actions`` are
            physical disturbances. Omit bounds when storing normalized actions.
            """

            output = self(tokens, valid_mask)
            if actions.shape != output.mean.shape:
                raise ValueError(
                    f"actions must have shape {output.mean.shape}, got {actions.shape}"
                )
            if not bool(torch.isfinite(actions).all().item()):
                raise ValueError("actions contain non-finite values")
            physical_bounds = self._validated_bounds(bounds, actions)
            normalized_action = (
                actions if physical_bounds is None else actions / physical_bounds
            )
            if bool((normalized_action.abs() > 1.0 + self.config.action_epsilon).any().item()):
                raise ValueError("actions exceed their declared symmetric bounds")
            limit = 1.0 - self.config.action_epsilon
            bounded = normalized_action.clamp(min=-limit, max=limit)
            raw_action = torch.atanh(bounded)
            distribution = self._base_distribution(output)
            log_probability = self._squashed_log_probability(
                distribution, raw_action
            )
            if physical_bounds is not None:
                log_probability = log_probability - physical_bounds.log().sum()
            entropy = distribution.entropy().sum(dim=-1)
            return ActionEvaluation(
                log_probability=log_probability,
                entropy=entropy,
                value=output.value,
                pre_actor_features=output.pre_actor_features,
                mean=output.mean,
                log_std=output.log_std,
            )

else:

    class ActorCriticOutput:  # pragma: no cover - dependency guard only
        pass


    class BoundedActionOutput:  # pragma: no cover - dependency guard only
        pass


    class ActionEvaluation:  # pragma: no cover - dependency guard only
        pass


    class CausalTransformerActorCritic:
        """Dependency guard used by pure-test and documentation environments."""

        def __init__(self, config: AdversaryModelConfig) -> None:
            del config
            raise TorchRequiredError(
                "CausalTransformerActorCritic requires Torch from the pinned "
                "Linux/CUDA training environment"
            )


__all__ = [
    "ActionEvaluation",
    "ActorCriticOutput",
    "AdversaryModelConfig",
    "BoundedActionOutput",
    "CausalTransformerActorCritic",
    "TorchRequiredError",
    "causal_attention_mask",
    "torch_available",
]
