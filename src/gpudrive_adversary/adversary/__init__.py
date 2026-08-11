"""Causal adversary policy and PPO optimization primitives."""

from .model import (
    AdversaryModelConfig,
    CausalTransformerActorCritic,
    TorchRequiredError,
    causal_attention_mask,
    torch_available,
)
from .ppo import (
    PPOConfig,
    clipped_ppo_loss_numpy,
    generalized_advantage_estimate_numpy,
)

__all__ = [
    "AdversaryModelConfig",
    "CausalTransformerActorCritic",
    "PPOConfig",
    "TorchRequiredError",
    "causal_attention_mask",
    "clipped_ppo_loss_numpy",
    "generalized_advantage_estimate_numpy",
    "torch_available",
]
