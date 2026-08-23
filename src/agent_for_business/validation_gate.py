"""可复用的综合 benchmark 记录和指标汇总。"""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class BenchmarkRecord:
    """一条 τ² 运行及其项目 Verifier 结果的 JSON-safe 中间表示。"""

    task_id: str
    trial: Any = None
    tau_reward: Optional[float] = None
    tau_reward_valid: bool = False
    verifier_reward: Optional[float] = None
    verifier_reward_valid: bool = False
    verifier_valid: bool = False
    task_success: Optional[bool] = None
    policy_violation: Optional[bool] = None
    first_error: Optional[str] = None
    tool_error_count: Optional[int] = None
    db_match: Optional[bool] = None
    communication_ok: Optional[bool] = None
    termination_reason: Optional[str] = None
    verifier_error: Optional[str] = None

    @classmethod
    def from_verification(
        cls,
        *,
        task_id: str,
        trial: Any = None,
        tau_reward: Optional[float] = None,
        termination_reason: Optional[str] = None,
        result: Any,
    ) -> "BenchmarkRecord":
        """把项目 Verifier 结果和官方 τ² reward 合并成一条记录。"""
        verifier_reward = getattr(result, "reward", None)
        if verifier_reward is not None:
            verifier_reward = float(verifier_reward)

        tool_error_count = getattr(result, "tool_error_count", 0)
        return cls(
            task_id=task_id,
            trial=trial,
            tau_reward=tau_reward,
            tau_reward_valid=tau_reward is not None,
            verifier_reward=verifier_reward,
            verifier_reward_valid=bool(
                getattr(result, "reward_valid", True)
            ),
            verifier_valid=True,
            task_success=bool(getattr(result, "task_success", False)),
            policy_violation=bool(
                getattr(result, "policy_violation", False)
            ),
            first_error=getattr(result, "first_error", None),
            tool_error_count=int(tool_error_count),
            db_match=getattr(result, "db_match", None),
            communication_ok=getattr(result, "communication_ok", None),
            termination_reason=termination_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSummary:
    """综合 benchmark 的独立 τ²、Verifier 和诊断指标。"""

    expected_runs: int
    completed_runs: int
    incomplete_runs: int
    tau_reward_valid_count: int
    tau_reward_invalid_count: int
    tau_reward_valid_rate: float
    tau_success_count: int
    tau_success_rate: float
    tau_reward_mean: Optional[float]
    db_match_true_count: int
    db_match_present_count: int
    db_match_missing_count: int
    db_match_rate: Optional[float]
    communication_true_count: int
    communication_present_count: int
    communication_missing_count: int
    communication_rate: Optional[float]
    termination_counts: dict[str, int]
    verifier_evaluated_count: int
    verifier_invalid_count: int
    verifier_reward_valid_count: int
    verifier_reward_mean: Optional[float]
    policy_violation_count: int
    policy_violation_rate: float
    tool_error_run_count: int
    tool_error_rate: float
    tool_error_total: int
    first_error_counts: dict[str, int]

    @classmethod
    def from_records(
        cls,
        records: Iterable[BenchmarkRecord],
        *,
        expected_runs: Optional[int] = None,
    ) -> "BenchmarkSummary":
        """从运行记录聚合指标，并为缺少分母的指标返回 ``None``。"""
        rows = list(records)
        completed_runs = len(rows)
        expected = completed_runs if expected_runs is None else expected_runs
        expected = max(int(expected), completed_runs)

        tau_valid = [
            row.tau_reward
            for row in rows
            if row.tau_reward_valid and row.tau_reward is not None
        ]
        verifier_rows = [row for row in rows if row.verifier_valid]
        verifier_rewards = [
            row.verifier_reward
            for row in verifier_rows
            if row.verifier_reward_valid and row.verifier_reward is not None
        ]
        db_rows = [row.db_match for row in rows if row.db_match is not None]
        communication_rows = [
            row.communication_ok
            for row in rows
            if row.communication_ok is not None
        ]

        policy_violation_count = sum(
            bool(row.policy_violation) for row in verifier_rows
        )
        tool_error_run_count = sum(
            int(row.tool_error_count or 0) > 0 for row in verifier_rows
        )
        first_error_counts = Counter(
            row.first_error for row in verifier_rows if row.first_error
        )
        termination_counts = Counter(
            row.termination_reason or "unknown" for row in rows
        )

        return cls(
            expected_runs=expected,
            completed_runs=completed_runs,
            incomplete_runs=max(expected - completed_runs, 0),
            tau_reward_valid_count=len(tau_valid),
            tau_reward_invalid_count=completed_runs - len(tau_valid),
            tau_reward_valid_rate=_rate(len(tau_valid), completed_runs),
            tau_success_count=sum(reward >= 1.0 for reward in tau_valid),
            tau_success_rate=_rate(
                sum(reward >= 1.0 for reward in tau_valid), len(tau_valid)
            ),
            tau_reward_mean=_mean(tau_valid),
            db_match_true_count=sum(bool(value) for value in db_rows),
            db_match_present_count=len(db_rows),
            db_match_missing_count=completed_runs - len(db_rows),
            db_match_rate=_rate(sum(bool(value) for value in db_rows), len(db_rows)),
            communication_true_count=sum(
                bool(value) for value in communication_rows
            ),
            communication_present_count=len(communication_rows),
            communication_missing_count=completed_runs - len(communication_rows),
            communication_rate=_rate(
                sum(bool(value) for value in communication_rows),
                len(communication_rows),
            ),
            termination_counts=dict(termination_counts),
            verifier_evaluated_count=len(verifier_rows),
            verifier_invalid_count=completed_runs - len(verifier_rows),
            verifier_reward_valid_count=len(verifier_rewards),
            verifier_reward_mean=_mean(verifier_rewards),
            policy_violation_count=policy_violation_count,
            policy_violation_rate=_rate(
                policy_violation_count, len(verifier_rows)
            ),
            tool_error_run_count=tool_error_run_count,
            tool_error_rate=_rate(tool_error_run_count, len(verifier_rows)),
            tool_error_total=sum(
                int(row.tool_error_count or 0) for row in verifier_rows
            ),
            first_error_counts=dict(first_error_counts),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接写入 JSON 的字典。"""
        return asdict(self)


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else None


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
