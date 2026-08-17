"""把结构化 Verifier 结果映射为可统计的 Badcase 类别。"""

from dataclasses import dataclass
from typing import Optional

from .policy_verifier import RetailPolicyVerifier, VerificationResult
from .trajectory import Trajectory


@dataclass(frozen=True)
class BadcaseRecord:
    """一个轨迹的分类结果和审计所需核心字段。"""

    task_id: str
    category: str
    first_error: Optional[str]
    task_success: bool
    policy_violation: bool
    reward_valid: bool
    reward: float


class BadcaseAnalyzer:
    """复用 RetailPolicyVerifier，避免 Badcase 分类重新实现策略判断。"""

    def __init__(self, verifier: Optional[RetailPolicyVerifier] = None) -> None:
        self._verifier = verifier or RetailPolicyVerifier()

    def analyze(self, trajectory: Trajectory) -> BadcaseRecord:
        """验证一条轨迹并生成稳定的 BadcaseRecord。"""
        verification = self._verifier.verify(trajectory)
        return BadcaseRecord(
            task_id=trajectory.task_id,
            category=self._category(verification),
            first_error=verification.first_error,
            task_success=verification.task_success,
            policy_violation=verification.policy_violation,
            reward_valid=verification.reward_valid,
            reward=verification.reward,
        )

    @staticmethod
    def _category(verification: VerificationResult) -> str:
        """按从基础设施有效性到模型行为的优先级选择一个类别。"""
        if not verification.reward_valid:
            # 无效 reward 无法公平归因给模型，因此优先隔离为基础设施问题。
            return "infrastructure_invalid"
        if verification.first_error == "authentication_failure":
            return "authentication_failure"
        if verification.first_error in {
            "missing_confirmation",
            "missing_action_summary",
        }:
            return "missing_confirmation"
        if verification.first_error == "multiple_tool_calls":
            return "tool_loop"
        if verification.first_error == "tool_error":
            return "tool_error"
        if verification.policy_violation:
            return "policy_violation"
        if not verification.task_success:
            return "task_failure"
        return "not_badcase"
