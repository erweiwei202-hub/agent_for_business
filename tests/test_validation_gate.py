from agent_for_business.validation_gate import BenchmarkSummary, SFTValidationGate
from agent_for_business.policy_verifier import VerificationResult


def test_blocks_grpo_when_sft_success_rate_regresses():
    raw = BenchmarkSummary(
        task_count=14,
        success_rate=0.50,
        policy_violation_rate=0.0,
        tool_error_rate=0.10,
    )
    sft = BenchmarkSummary(
        task_count=14,
        success_rate=0.40,
        policy_violation_rate=0.0,
        tool_error_rate=0.10,
    )

    decision = SFTValidationGate().decide(raw=raw, sft=sft)

    assert decision.passed is False
    assert decision.reason == "sft_success_rate_regressed"


def test_blocks_grpo_when_validation_benchmark_is_empty():
    empty = BenchmarkSummary(
        task_count=0,
        success_rate=0.0,
        policy_violation_rate=0.0,
        tool_error_rate=0.0,
    )

    decision = SFTValidationGate().decide(raw=empty, sft=empty)

    assert decision.passed is False
    assert decision.reason == "benchmark_empty"


def test_allows_grpo_when_sft_improves_without_new_safety_regression():
    raw = BenchmarkSummary(
        task_count=14,
        success_rate=0.50,
        policy_violation_rate=0.01,
        tool_error_rate=0.10,
    )
    sft = BenchmarkSummary(
        task_count=14,
        success_rate=0.57,
        policy_violation_rate=0.0,
        tool_error_rate=0.08,
    )

    decision = SFTValidationGate().decide(raw=raw, sft=sft)

    assert decision.passed is True
    assert decision.reason == "sft_ready_for_grpo"


def test_builds_benchmark_summary_from_verification_results():
    results = [
        VerificationResult(
            task_success=True,
            policy_violation=False,
            first_error=None,
            reward=1.0,
            tool_error_count=0,
        ),
        VerificationResult(
            task_success=False,
            policy_violation=False,
            first_error="tool_error",
            reward=-0.1,
            tool_error_count=1,
        ),
    ]

    summary = BenchmarkSummary.from_results(results)

    assert summary.task_count == 2
    assert summary.success_rate == 0.5
    assert summary.policy_violation_rate == 0.0
    assert summary.tool_error_rate == 0.5


def test_blocks_grpo_when_validation_contains_invalid_rewards():
    raw = BenchmarkSummary(
        task_count=1,
        success_rate=0.5,
        policy_violation_rate=0.0,
        tool_error_rate=0.0,
    )
    invalid_sft = BenchmarkSummary(
        task_count=1,
        success_rate=0.5,
        policy_violation_rate=0.0,
        tool_error_rate=0.0,
        valid_rate=0.0,
    )

    decision = SFTValidationGate().decide(raw=raw, sft=invalid_sft)

    assert decision.passed is False
    assert decision.reason == "benchmark_contains_invalid_rewards"
