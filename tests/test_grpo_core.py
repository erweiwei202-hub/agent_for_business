import math

import pytest

from agent_for_business.grpo_core import (
    clipped_objective,
    compute_group_advantages,
    masked_mean,
    reference_kl,
)


def test_compute_group_advantages_standardizes_rewards_within_group():
    advantages = compute_group_advantages([1.0, 2.0, 3.0])

    scale = math.sqrt(2.0 / 3.0)
    assert advantages == pytest.approx([-1.0 / scale, 0.0, 1.0 / scale])


def test_compute_group_advantages_returns_zero_for_constant_rewards():
    advantages = compute_group_advantages([4.2, 4.2, 4.2])

    assert advantages == [0.0, 0.0, 0.0]
    assert all(math.isfinite(value) for value in advantages)


def test_compute_group_advantages_rejects_empty_group():
    with pytest.raises(ValueError, match="empty group"):
        compute_group_advantages([])


def test_masked_mean_ignores_observation_tokens():
    result = masked_mean([2.0, 100.0, -50.0], [1, 0, 0])

    assert result == 2.0


def test_masked_mean_rejects_misaligned_mask():
    with pytest.raises(ValueError, match="same length"):
        masked_mean([1.0, 2.0], [1])


def test_clipped_objective_uses_only_action_tokens():
    result = clipped_objective(
        old_logprobs=[[0.0, 0.0]],
        new_logprobs=[[1.0, 100.0]],
        advantages=[1.0],
        action_mask=[[1, 0]],
        clip_ratio=10.0,
    )

    assert result == pytest.approx(math.exp(1.0))


def test_clipped_objective_clips_positive_and_negative_advantage_ratios():
    result = clipped_objective(
        old_logprobs=[[math.log(0.5)], [math.log(0.5)]],
        new_logprobs=[[math.log(0.75)], [math.log(0.25)]],
        advantages=[2.0, -2.0],
        action_mask=[[1], [1]],
        clip_ratio=0.2,
    )

    assert result == pytest.approx((2.0 * 1.2 - 2.0 * 0.8) / 2.0)


def test_reference_kl_is_zero_when_current_matches_reference():
    result = reference_kl(
        current_logprobs=[[0.0, 100.0]],
        reference_logprobs=[[0.0, -100.0]],
        action_mask=[[1, 0]],
    )

    assert result == 0.0


def test_reference_kl_uses_only_action_tokens():
    result = reference_kl(
        current_logprobs=[[math.log(0.5), 0.0]],
        reference_logprobs=[[math.log(0.25), 100.0]],
        action_mask=[[1, 0]],
    )

    expected_delta = math.log(0.25) - math.log(0.5)
    assert result == pytest.approx(math.exp(expected_delta) - expected_delta - 1.0)


def test_objectives_reject_empty_groups_and_misaligned_sequences():
    with pytest.raises(ValueError, match="empty group"):
        clipped_objective([], [], [], [])

    with pytest.raises(ValueError, match="same number"):
        clipped_objective([[0.0]], [[0.0], [0.0]], [1.0], [[1]])

    with pytest.raises(ValueError, match="same length"):
        reference_kl([[0.0, 0.0]], [[0.0]], [[1, 1]])


def test_objectives_reject_groups_without_action_tokens():
    with pytest.raises(ValueError, match="action token"):
        clipped_objective([[0.0]], [[0.0]], [1.0], [[0]])

    with pytest.raises(ValueError, match="action token"):
        reference_kl([[0.0]], [[0.0]], [[0]])
