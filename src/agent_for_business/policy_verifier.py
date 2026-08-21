"""验证 Retail Agent 轨迹中的认证、确认、工具调用和 reward 约束。"""

import re
from dataclasses import dataclass
from typing import Optional, Set

from .trajectory import Trajectory


@dataclass
class VerificationResult:
    """Verifier 输出的结构化结果，供筛选、Badcase 和 benchmark 共用。"""

    task_success: bool
    policy_violation: bool
    first_error: Optional[str]
    reward: float
    reward_valid: bool = True
    db_match: Optional[bool] = None
    communication_ok: Optional[bool] = None
    tool_error_count: int = 0


class RetailPolicyVerifier:
    """以事件顺序回放轨迹，并把策略违规与任务 reward 分开判定。"""

    AUTHENTICATION_TOOLS: Set[str] = {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
    }
    READ_ONLY_TOOLS: Set[str] = {
        "get_user_details",
        "get_order_details",
        "get_product_details",
    }
    MUTATING_TOOLS: Set[str] = {
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
    }
    POLICY_ERRORS = {
        "authentication_failure",
        "missing_confirmation",
        "missing_action_summary",
        "multiple_tool_calls",
    }

    def verify(self, trajectory: Trajectory) -> VerificationResult:
        """检查轨迹状态机并计算最终的合规 reward。"""
        authenticated = False
        confirmation_received = False
        confirmation_ready = False
        pending_tool_names = {}
        tool_error_count = 0
        first_error: Optional[str] = None

        for event in trajectory.events:
            if event.kind == "user_message":
                # 明确的 yes/confirm 是一次性的；普通对话不会消耗或刷新确认状态。
                if self._is_explicit_confirmation(str(event.content or "")):
                    confirmation_received = True
            elif event.kind == "assistant_message":
                # 操作摘要由 assistant 文本提供，必须先出现才允许 mutation。
                confirmation_ready = confirmation_ready or self._has_action_summary(
                    str(event.content or "")
                )
            elif event.kind == "tool_result":
                # pending_tool_names 同时承担调用关联和错误分类的作用。
                tool_name = pending_tool_names.pop(event.tool_call_id, None)
                if self._is_failed_tool_result(event.content):
                    if tool_name in self.AUTHENTICATION_TOOLS:
                        authenticated = False
                        if first_error is None:
                            first_error = "authentication_failure"
                    elif tool_name is not None:
                        tool_error_count += 1
                        if first_error is None:
                            first_error = "tool_error"
            elif event.kind == "tool_call":
                if event.tool_name in self.AUTHENTICATION_TOOLS:
                    # 认证工具本身是进入已认证状态的边界；失败结果稍后会撤销它。
                    authenticated = True
                    if event.tool_call_id:
                        pending_tool_names[event.tool_call_id] = event.tool_name
                    continue
                if not authenticated:
                    first_error = "authentication_failure"
                    break
                if event.tool_name in self.MUTATING_TOOLS:
                    if not confirmation_received:
                        first_error = "missing_confirmation"
                        break
                    if not confirmation_ready:
                        first_error = "missing_action_summary"
                        break
                    # 每次 mutation 消耗一次确认，下一次 mutation 必须重新确认。
                    confirmation_received = False
                    confirmation_ready = False

                # 只读查询可以并行；mutation 和其它 action 仍必须串行。
                pending_actions = [
                    name
                    for name in pending_tool_names.values()
                    if name not in self.AUTHENTICATION_TOOLS
                ]
                if pending_actions and (
                    event.tool_name not in self.READ_ONLY_TOOLS
                    or any(name not in self.READ_ONLY_TOOLS for name in pending_actions)
                ):
                    first_error = "multiple_tool_calls"
                    break
                if event.tool_call_id:
                    pending_tool_names[event.tool_call_id] = event.tool_name

        policy_violation = first_error in self.POLICY_ERRORS
        task_success = bool(trajectory.evaluation.get("task_success", False))
        # 策略违规直接 hard penalty；普通工具错误只在有效 reward 上做有限扣分。
        reward = -1.0 if policy_violation else self._reward_with_tool_penalty(
            trajectory.evaluation.get("reward", 0.0),
            tool_error_count,
        )

        return VerificationResult(
            task_success=task_success,
            policy_violation=policy_violation,
            first_error=first_error,
            reward=reward,
            reward_valid=bool(trajectory.evaluation.get("reward_valid", True)),
            db_match=trajectory.evaluation.get("db_match"),
            communication_ok=trajectory.evaluation.get("communication_ok"),
            tool_error_count=tool_error_count,
        )

    @staticmethod
    def _reward_with_tool_penalty(reward: object, tool_error_count: int) -> float:
        """把可恢复工具错误映射为最多 0.2 的小额扣分。"""
        try:
            base_reward = float(reward)
        except (TypeError, ValueError):
            base_reward = 0.0
        penalty = min(0.2, 0.1 * tool_error_count)
        return round(base_reward - penalty, 6)

    @staticmethod
    def _is_explicit_confirmation(content: str) -> bool:
        """识别大小写不敏感且可带动作详情的明确确认。"""
        normalized = re.sub(r"[^a-z]+", " ", content.lower()).strip()
        exact_matches = {
            "yes",
            "yes please",
            "yes proceed",
            "i confirm",
            "confirmed",
        }
        if normalized in exact_matches:
            return True

        confirmation_prefixes = ("yes ", "i confirm ", "confirmed ")
        if not normalized.startswith(confirmation_prefixes):
            return False

        return not re.search(
            r"\b(?:no|not|don t|do not|cancel|decline|rather|instead)\b",
            normalized,
        )

    @staticmethod
    def _has_action_summary(content: str) -> bool:
        """判断 assistant 是否说明了 mutation 动作并请求确认/继续。"""
        normalized = content.lower()
        action_word_present = any(
            word in normalized for word in ("cancel", "modify", "return", "exchange")
        )
        return action_word_present and (
            "?" in content
            or "confirm" in normalized
            or "proceed" in normalized
            or "yes" in normalized
        )

    @staticmethod
    def _is_failed_tool_result(content: object) -> bool:
        """兼容 τ³ 结构化 error 和文本 ``Error:`` 两种工具失败格式。"""
        if isinstance(content, dict):
            return bool(content.get("error"))
        return str(content or "").strip().lower().startswith("error:")
