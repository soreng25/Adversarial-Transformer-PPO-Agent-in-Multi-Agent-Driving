import numpy as np
import pytest

from gpudrive_adversary.adversary.model import (
    AdversaryModelConfig,
    CausalTransformerActorCritic,
    TorchRequiredError,
    causal_attention_mask,
    torch_available,
)
from gpudrive_adversary.adversary.ppo import (
    PPOConfig,
    clipped_ppo_loss_numpy,
    generalized_advantage_estimate_numpy,
)


pytestmark = pytest.mark.unit


def test_compatibility_architecture_is_explicit() -> None:
    config = AdversaryModelConfig(token_dim=17)

    assert config.context_length == 50
    assert config.model_dim == 64
    assert config.pre_actor_dim == 64
    assert config.num_layers == 1
    assert config.num_heads == 1
    assert config.action_dim == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("context_length", 49, "50-step"),
        ("model_dim", 32, "64-D"),
        ("pre_actor_dim", 32, "64-D"),
        ("num_layers", 2, "one layer"),
        ("num_heads", 2, "one layer"),
        ("action_dim", 1, "two normalized"),
    ),
)
def test_incompatible_architecture_is_rejected(
    field: str, value: int, message: str
) -> None:
    arguments = {"token_dim": 4, field: value}

    with pytest.raises(ValueError, match=message):
        AdversaryModelConfig(**arguments)


def test_causal_attention_mask_forbids_only_future_positions() -> None:
    mask = causal_attention_mask(5)

    assert mask.dtype == np.bool_
    assert mask.shape == (5, 5)
    for query in range(5):
        for key in range(5):
            assert bool(mask[query, key]) is (key > query)


def test_causal_attention_mask_enforces_context_limit() -> None:
    with pytest.raises(ValueError, match="exceeds context length"):
        causal_attention_mask(51)


def test_numpy_gae_matches_hand_computed_unterminated_rollout() -> None:
    rewards = np.asarray([1.0, 2.0], dtype=np.float32)
    values = np.asarray([0.5, 0.25], dtype=np.float32)
    terminated = np.asarray([False, False])

    advantages, returns = generalized_advantage_estimate_numpy(
        rewards,
        values,
        terminated,
        np.asarray(0.75, dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.8,
    )

    expected_last = 2.0 + 0.9 * 0.75 - 0.25
    expected_first = 1.0 + 0.9 * 0.25 - 0.5 + 0.9 * 0.8 * expected_last
    np.testing.assert_allclose(advantages, [expected_first, expected_last])
    np.testing.assert_allclose(returns, advantages + values)


def test_numpy_gae_terminal_transition_blocks_future_bootstrap() -> None:
    rewards = np.asarray([1.0, 7.0], dtype=np.float32)
    values = np.asarray([0.5, 3.0], dtype=np.float32)
    terminated = np.asarray([True, False])

    advantages, _ = generalized_advantage_estimate_numpy(
        rewards,
        values,
        terminated,
        np.asarray(100.0, dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages[0] == pytest.approx(1.0 - 0.5)


def test_ppo_defaults_match_methodology_starting_point() -> None:
    config = PPOConfig()

    assert config.learning_rate == pytest.approx(3e-4)
    assert config.gamma == pytest.approx(0.99)
    assert config.policy_clip == pytest.approx(0.2)
    assert config.value_clip == pytest.approx(0.2)
    assert config.entropy_coefficient == 0.0


def test_numpy_ppo_objective_clips_policy_and_value() -> None:
    metrics = clipped_ppo_loss_numpy(
        new_log_probability=np.log(np.asarray([1.5, 0.5])),
        old_log_probability=np.zeros(2),
        new_value=np.asarray([1.0, -1.0]),
        old_value=np.zeros(2),
        advantages=np.asarray([1.0, -1.0]),
        returns=np.zeros(2),
        entropy=np.full(2, 2.0),
        config=PPOConfig(
            normalize_advantages=False,
            entropy_coefficient=0.01,
        ),
    )

    assert metrics["loss_policy"] == pytest.approx(-0.2)
    assert metrics["loss_value"] == pytest.approx(0.5)
    assert metrics["loss_total"] == pytest.approx(0.03)
    assert metrics["entropy"] == pytest.approx(2.0)
    assert metrics["clip_fraction"] == pytest.approx(1.0)


def test_missing_torch_has_an_explicit_dependency_error() -> None:
    if torch_available():
        pytest.skip("Torch is installed; neural tests cover construction")

    with pytest.raises(TorchRequiredError, match="Linux/CUDA"):
        CausalTransformerActorCritic(AdversaryModelConfig(token_dim=4))


def test_torch_model_is_causal_bounded_and_exposes_stable_latent() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    model = CausalTransformerActorCritic(
        AdversaryModelConfig(token_dim=6, dropout=0.0)
    ).eval()
    prefix = torch.randn(2, 3, 6)
    first = torch.cat((prefix, torch.zeros(2, 2, 6)), dim=1)
    second = torch.cat((prefix, torch.full((2, 2, 6), 100.0)), dim=1)

    with torch.inference_mode():
        first_encoded = model.encode_sequence(first)
        second_encoded = model.encode_sequence(second)
        decision = model.act(prefix, bounds=torch.tensor([0.667, 0.262]))
        direct_latent = model.get_pre_actor_features(prefix)

    torch.testing.assert_close(first_encoded[:, :3], second_encoded[:, :3])
    torch.testing.assert_close(decision.pre_actor_features, direct_latent)
    assert decision.pre_actor_features.shape == (2, 64)
    assert decision.normalized_action.shape == (2, 2)
    assert bool((decision.normalized_action.abs() < 1.0).all().item())
    assert bool((decision.action[:, 0].abs() < 0.667).all().item())
    assert bool((decision.action[:, 1].abs() < 0.262).all().item())
    torch.testing.assert_close(decision.pre_tanh, decision.raw_action)
    torch.testing.assert_close(decision.log_prob, decision.log_probability)
    assert decision.log_probability.shape == (2,)
    assert decision.value.shape == (2,)


def test_torch_one_minibatch_ppo_update_changes_parameters() -> None:
    torch = pytest.importorskip("torch")
    from gpudrive_adversary.adversary.ppo import ppo_minibatch_update

    torch.manual_seed(11)
    model = CausalTransformerActorCritic(
        AdversaryModelConfig(token_dim=5, dropout=0.0)
    )
    settings = PPOConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    tokens = torch.randn(8, 4, 5)
    with torch.no_grad():
        behavior = model.act(tokens)
    before = model.actor_mean.weight.detach().clone()

    metrics = ppo_minibatch_update(
        model=model,
        optimizer=optimizer,
        tokens=tokens,
        actions=behavior.normalized_action.detach(),
        old_log_probability=behavior.log_probability.detach(),
        old_value=behavior.value.detach(),
        advantages=torch.linspace(-1.0, 1.0, 8),
        returns=behavior.value.detach() + 0.25,
        config=settings,
    )

    assert not torch.equal(before, model.actor_mean.weight.detach())
    assert all(np.isfinite(value) for value in metrics.values())


def test_torch_right_aligned_history_has_finite_forward_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(19)
    model = CausalTransformerActorCritic(
        AdversaryModelConfig(token_dim=2989, dropout=0.0)
    )
    tokens = torch.zeros(3, 50, 2989)
    valid_mask = torch.zeros(3, 50, dtype=torch.bool)
    for row, count in enumerate((1, 17, 50)):
        tokens[row, -count:] = torch.randn(count, 2989)
        valid_mask[row, -count:] = True

    output = model(tokens, valid_mask)
    loss = output.mean.square().mean() + output.value.square().mean()
    loss.backward()

    assert bool(torch.isfinite(output.mean).all().item())
    assert bool(torch.isfinite(output.value).all().item())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
