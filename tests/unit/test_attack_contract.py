import numpy as np
import pytest

from gpudrive_adversary.adversary.distribution import (
    BoundedTanhNormal,
    DistributionContractError,
)
from gpudrive_adversary.adversary.failure import (
    TerminationReason,
    assess_nominal_goal_eligibility,
    classify_victim_post_step,
)
from gpudrive_adversary.adversary.intervention import (
    APPROVED_COMMAND_LOWER,
    APPROVED_COMMAND_UPPER,
    APPROVED_DISTURBANCE_BOUNDS,
    InterventionSpec,
    apply_intervention,
)


pytestmark = pytest.mark.unit


def _distribution() -> BoundedTanhNormal:
    return BoundedTanhNormal(
        bounds=np.asarray(APPROVED_DISTURBANCE_BOUNDS),
        sigma=np.asarray([0.5, 0.5]),
    )


def _intervention_spec() -> InterventionSpec:
    return InterventionSpec(
        disturbance_bounds=np.asarray(APPROVED_DISTURBANCE_BOUNDS),
        command_lower=np.asarray(APPROVED_COMMAND_LOWER),
        command_upper=np.asarray(APPROVED_COMMAND_UPPER),
    )


def test_tanh_normal_round_trip_stays_strictly_inside_bounds() -> None:
    distribution = _distribution()
    latent = np.asarray([[0.0, 0.0], [3.0, -4.0], [-8.0, 8.0]])

    value = distribution.transform(latent)

    assert np.all(np.abs(value) < distribution.bounds)
    np.testing.assert_allclose(distribution.inverse(value), latent, rtol=1e-9)


def test_tanh_normal_log_density_includes_exact_jacobian() -> None:
    distribution = _distribution()
    latent = np.asarray([0.3, -0.4])
    mean = np.asarray([0.1, -0.2])
    value = distribution.transform(latent)
    sigma = distribution.sigma
    normal = np.sum(
        -0.5 * ((latent - mean) / sigma) ** 2
        - np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
    )
    jacobian = np.sum(
        np.log(distribution.bounds) + np.log(1.0 - np.tanh(latent) ** 2)
    )

    actual = distribution.log_prob(value, mean)

    assert actual == pytest.approx(normal - jacobian)
    assert distribution.negative_log_likelihood(value, mean) == pytest.approx(
        -actual
    )


def test_prior_nll_excess_is_zero_at_no_disturbance() -> None:
    distribution = _distribution()
    mean = np.zeros(2)

    assert distribution.nll_excess_from_zero(np.zeros(2), mean) == pytest.approx(0.0)
    assert distribution.nll_excess_from_zero(
        distribution.transform(np.asarray([0.4, -0.2])), mean
    ) > 0.0


def test_tanh_normal_sample_is_reproducible_and_reports_its_density() -> None:
    left = _distribution().sample(np.random.default_rng(37), np.zeros(2))
    right = _distribution().sample(np.random.default_rng(37), np.zeros(2))

    np.testing.assert_array_equal(left.latent, right.latent)
    np.testing.assert_array_equal(left.value, right.value)
    assert left.log_prob == pytest.approx(right.log_prob)
    assert left.log_prob == pytest.approx(
        _distribution().log_prob(left.value, np.zeros(2))
    )


def test_tanh_normal_has_no_boundary_density_or_clipping() -> None:
    distribution = _distribution()
    boundary = distribution.bounds.copy()

    assert np.isneginf(distribution.log_prob(boundary, np.zeros(2)))
    with pytest.raises(DistributionContractError, match="strictly inside"):
        distribution.inverse(boundary)
    with pytest.raises(DistributionContractError, match="numerically saturates"):
        distribution.transform(np.asarray([1000.0, 0.0]))


def test_distribution_requires_explicit_valid_sigma() -> None:
    with pytest.raises(TypeError):
        BoundedTanhNormal(bounds=np.asarray([0.667, 0.262]))  # type: ignore[call-arg]
    with pytest.raises(DistributionContractError, match="strictly positive"):
        BoundedTanhNormal(
            bounds=np.asarray([0.667, 0.262]), sigma=np.asarray([0.2, 0.0])
        )


def test_intervention_records_each_saturation_stage_and_preserves_head() -> None:
    nominal = np.asarray([3.8, 3.142 - 0.05, 0.75])
    requested = np.asarray([0.9, 0.4])

    result = apply_intervention(nominal, requested, _intervention_spec())

    np.testing.assert_array_equal(result.requested_disturbance, requested)
    np.testing.assert_allclose(result.applied_disturbance, [0.667, 0.262])
    np.testing.assert_allclose(result.requested_command, [4.7, 3.142 + 0.35, 0.75])
    np.testing.assert_allclose(
        result.bounded_command, [4.467, 3.142 + 0.212, 0.75]
    )
    np.testing.assert_allclose(result.applied_command, [4.0, 3.142, 0.75])
    np.testing.assert_allclose(result.effective_disturbance, [0.2, 0.05])
    np.testing.assert_array_equal(result.disturbance_saturated, [True, True])
    np.testing.assert_array_equal(result.command_saturated, [True, True])
    np.testing.assert_array_equal(result.saturated, [True, True])
    assert result.applied_command[2] == nominal[2]


def test_intervention_supports_batches_without_mutating_inputs() -> None:
    nominal = np.asarray([[0.0, 0.0, 0.2], [1.0, -1.0, -0.4]])
    requested = np.asarray([[0.1, -0.1], [-0.2, 0.2]])
    nominal_before = nominal.copy()
    requested_before = requested.copy()

    result = apply_intervention(nominal, requested, _intervention_spec())

    np.testing.assert_array_equal(nominal, nominal_before)
    np.testing.assert_array_equal(requested, requested_before)
    np.testing.assert_allclose(
        result.applied_command,
        [[0.1, -0.1, 0.2], [0.8, -0.8, -0.4]],
    )
    assert not np.any(result.saturated)
    assert not result.applied_command.flags.writeable


@pytest.mark.parametrize(
    ("index", "kind"),
    ((0, "road_object_contact"), (1, "vehicle_collision"), (2, "nonvehicle_collision")),
)
def test_each_approved_victim_event_is_failure(index: int, kind: str) -> None:
    raw = np.zeros(5)
    raw[index] = 1.0

    evidence = classify_victim_post_step(raw, 7)

    assert evidence.is_failure
    assert evidence.failure_timestep == 7
    assert evidence.failure_kinds == (kind,)
    assert evidence.termination_reason is TerminationReason.FAILURE


def test_safety_wins_goal_and_horizon_ties_but_raw_flags_are_preserved() -> None:
    raw = np.asarray([0.0, 1.0, 0.0, 1.0, 1.0])

    evidence = classify_victim_post_step(
        raw, 12, horizon_reached=True, done=True
    )

    assert evidence.is_failure
    assert evidence.goal_achieved
    assert evidence.horizon_reached
    assert evidence.failure_timestep == 12
    assert evidence.termination_reason is TerminationReason.FAILURE


def test_goal_and_horizon_are_not_failures() -> None:
    goal = classify_victim_post_step(np.asarray([0, 0, 0, 1, 1]), 3, done=True)
    horizon = classify_victim_post_step(
        np.asarray([0, 0, 0, 0, 1]), 9, horizon_reached=True, done=True
    )

    assert not goal.is_failure and goal.failure_timestep is None
    assert goal.termination_reason is TerminationReason.GOAL
    assert not horizon.is_failure and horizon.failure_timestep is None
    assert horizon.termination_reason is TerminationReason.HORIZON


def test_nominal_eligibility_requires_clean_goal_success() -> None:
    clean_goal = np.zeros((4, 5))
    clean_goal[:, 4] = 1.0
    clean_goal[3, 3] = 1.0
    eligibility = assess_nominal_goal_eligibility(
        clean_goal, first_action_timestep=5
    )

    assert eligibility.eligible
    assert eligibility.reason == "clean_goal_success"
    assert eligibility.goal_timestep == 8
    assert eligibility.failure_timestep is None


def test_nominal_failure_and_simultaneous_goal_is_ineligible() -> None:
    trace = np.zeros((2, 5))
    trace[:, 4] = 1.0
    trace[1, 0] = 1.0
    trace[1, 3] = 1.0

    eligibility = assess_nominal_goal_eligibility(trace)

    assert not eligibility.eligible
    assert eligibility.reason == "nominal_failure"
    assert eligibility.failure_timestep == 1
    assert eligibility.goal_timestep == 1


def test_nominal_trace_without_goal_is_not_verified() -> None:
    trace = np.zeros((3, 5))
    trace[:, 4] = 1.0

    eligibility = assess_nominal_goal_eligibility(trace)

    assert not eligibility.eligible
    assert eligibility.reason == "goal_not_reached"
