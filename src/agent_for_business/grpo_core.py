"""GRPO group objective 的纯 Python 数学核心，不依赖模型训练框架。"""

import math
from collections.abc import Sequence
from typing import List


def compute_group_advantages(
    rewards: Sequence[float], epsilon: float = 1e-8
) -> List[float]:
    """在一个 prompt group 内标准化 reward，生成 rollout-level advantage。

    这里使用总体标准差，因为输入就是该 prompt 的完整采样组；如果组内
    reward 没有方差，则返回全零 advantage，避免人为放大无信息的差异。
    """
    _require_sequence(rewards, "rewards")
    if not rewards:
        raise ValueError("empty group")

    epsilon_value = _finite_float(epsilon, "epsilon")
    if epsilon_value <= 0.0:
        raise ValueError("epsilon must be positive")

    reward_values = [_finite_float(reward, "reward") for reward in rewards]
    mean = sum(reward_values) / len(reward_values)
    variance = sum((reward - mean) ** 2 for reward in reward_values) / len(
        reward_values
    )
    standard_deviation = math.sqrt(variance)
    if standard_deviation == 0.0:
        # 所有 rollout 得分相同，组内没有可学习的相对偏好。
        return [0.0 for _ in reward_values]

    denominator = standard_deviation + epsilon_value
    return [(reward - mean) / denominator for reward in reward_values]


def masked_mean(values: Sequence[float], mask: Sequence[object]) -> float:
    """只对 action mask 选中的 token 求均值。"""
    _require_sequence(values, "values")
    _require_sequence(mask, "mask")
    if len(values) != len(mask):
        raise ValueError("values and mask must have the same length")

    selected_values = []
    for value, flag in zip(values, mask):
        if _mask_value(flag):
            selected_values.append(_finite_float(value, "value"))

    if not selected_values:
        # Action-only objective 没有 token 时无法定义均值，直接暴露输入错误。
        raise ValueError("at least one action token is required")
    return sum(selected_values) / len(selected_values)


def clipped_objective(
    old_logprobs: Sequence[Sequence[float]],
    new_logprobs: Sequence[Sequence[float]],
    advantages: Sequence[float],
    action_mask: Sequence[Sequence[object]],
    clip_ratio: float = 0.2,
) -> float:
    """计算 clipped GRPO surrogate objective 的 action-token 均值。

    ``advantages`` 每条 rollout 一个值，并广播到该 rollout 的 action token。
    这是最大化目标，因此高 reward rollout 的 action 概率上升时贡献为正。
    """
    group_size = _validate_group_inputs(
        old_logprobs, new_logprobs, advantages, action_mask
    )
    clip = _finite_float(clip_ratio, "clip_ratio")
    if clip < 0.0:
        raise ValueError("clip_ratio must be non-negative")

    lower = 1.0 - clip
    upper = 1.0 + clip
    total = 0.0
    action_count = 0

    for index in range(group_size):
        old_row = old_logprobs[index]
        new_row = new_logprobs[index]
        mask_row = action_mask[index]
        _require_sequence(old_row, "old_logprobs row")
        _require_sequence(new_row, "new_logprobs row")
        _require_sequence(mask_row, "action_mask row")
        _require_aligned_lengths(old_row, new_row, mask_row)

        advantage = _finite_float(advantages[index], "advantage")
        for old_logprob, new_logprob, flag in zip(old_row, new_row, mask_row):
            if not _mask_value(flag):
                continue

            old_value = _finite_float(old_logprob, "old logprob")
            new_value = _finite_float(new_logprob, "new logprob")
            # ratio 比较新旧 policy；clip 限制单步更新对 surrogate 的影响。
            ratio = math.exp(new_value - old_value)
            clipped = min(max(ratio, lower), upper)
            if advantage == 0.0:
                contribution = 0.0
            else:
                # min 同时处理正/负 advantage，选择 clipped surrogate 的保守值。
                contribution = min(ratio * advantage, clipped * advantage)
            total += contribution
            action_count += 1

    if action_count == 0:
        raise ValueError("at least one action token is required")
    return total / action_count


def reference_kl(
    current_logprobs: Sequence[Sequence[float]],
    reference_logprobs: Sequence[Sequence[float]],
    action_mask: Sequence[Sequence[object]],
) -> float:
    """用 action-token mask 计算 sampled-action reference KL 近似。

    每个采样 token 使用非负近似 ``exp(reference - current) -
    (reference - current) - 1``；``expm1`` 在两个 log-probability 接近时更稳定。
    """
    group_size = _validate_pair_inputs(
        current_logprobs, reference_logprobs, action_mask
    )
    total = 0.0
    action_count = 0

    for index in range(group_size):
        current_row = current_logprobs[index]
        reference_row = reference_logprobs[index]
        mask_row = action_mask[index]
        _require_sequence(current_row, "current_logprobs row")
        _require_sequence(reference_row, "reference_logprobs row")
        _require_sequence(mask_row, "action_mask row")
        _require_aligned_lengths(current_row, reference_row, mask_row)

        for current_logprob, reference_logprob, flag in zip(
            current_row, reference_row, mask_row
        ):
            if not _mask_value(flag):
                continue

            current_value = _finite_float(current_logprob, "current logprob")
            reference_value = _finite_float(
                reference_logprob, "reference logprob"
            )
            delta = reference_value - current_value
            try:
                # 只在 action token 上约束当前 policy 偏离 reference policy 的程度。
                term = math.expm1(delta) - delta
            except OverflowError:
                term = math.inf
            total += max(0.0, term)
            action_count += 1

    if action_count == 0:
        raise ValueError("at least one action token is required")
    return total / action_count


def _validate_group_inputs(
    old_logprobs: Sequence[Sequence[float]],
    new_logprobs: Sequence[Sequence[float]],
    advantages: Sequence[float],
    action_mask: Sequence[Sequence[object]],
) -> int:
    """校验 group-level 输入的行数，返回 rollout 数量。"""
    _require_sequence(old_logprobs, "old_logprobs")
    _require_sequence(new_logprobs, "new_logprobs")
    _require_sequence(advantages, "advantages")
    _require_sequence(action_mask, "action_mask")
    if not old_logprobs:
        raise ValueError("empty group")
    if len(new_logprobs) != len(old_logprobs):
        raise ValueError("logprob groups must have the same number of rows")
    if len(advantages) != len(old_logprobs):
        raise ValueError("advantages must have the same number of rows")
    if len(action_mask) != len(old_logprobs):
        raise ValueError("action mask must have the same number of rows")
    return len(old_logprobs)


def _validate_pair_inputs(
    current_logprobs: Sequence[Sequence[float]],
    reference_logprobs: Sequence[Sequence[float]],
    action_mask: Sequence[Sequence[object]],
) -> int:
    """校验 current/reference/action mask 的 rollout 行数。"""
    _require_sequence(current_logprobs, "current_logprobs")
    _require_sequence(reference_logprobs, "reference_logprobs")
    _require_sequence(action_mask, "action_mask")
    if not current_logprobs:
        raise ValueError("empty group")
    if len(reference_logprobs) != len(current_logprobs):
        raise ValueError("logprob groups must have the same number of rows")
    if len(action_mask) != len(current_logprobs):
        raise ValueError("action mask must have the same number of rows")
    return len(current_logprobs)


def _require_aligned_lengths(*sequences: Sequence[object]) -> None:
    """确保同一 rollout 的 logprob 和 mask 按 token 对齐。"""
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError("values and mask must have the same length")


def _require_sequence(value: object, name: str) -> None:
    """拒绝字符串等可迭代但不是数值序列的输入。"""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise ValueError("{} must be a sequence".format(name))


def _finite_float(value: object, name: str) -> float:
    """把输入转成有限浮点数，避免 NaN/Inf 污染训练目标。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be numeric".format(name))
    if not math.isfinite(number):
        raise ValueError("{} must be finite".format(name))
    return number


def _mask_value(value: object) -> bool:
    """将 bool 或严格的 0/1 值转换为 mask 标志。"""
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("action mask values must be 0 or 1")
    if not math.isfinite(number) or number not in (0.0, 1.0):
        raise ValueError("action mask values must be 0 or 1")
    return bool(number)
