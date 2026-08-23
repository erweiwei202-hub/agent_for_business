import json

from agent_for_business.policy_verifier import VerificationResult
from agent_for_business.validation_gate import BenchmarkRecord, BenchmarkSummary


def test_summary_keeps_tau_and_verifier_rewards_separate_and_counts_missing_checks():
    records = [
        BenchmarkRecord(
            task_id="5", trial=0, tau_reward=1.0, tau_reward_valid=True,
            verifier_reward=0.9, verifier_reward_valid=True,
            verifier_valid=True, task_success=True, policy_violation=False,
            first_error=None, tool_error_count=1, db_match=True,
            communication_ok=True, termination_reason="agent_end",
        ),
        BenchmarkRecord(
            task_id="9", trial=0, tau_reward=0.0, tau_reward_valid=True,
            verifier_reward=-1.0, verifier_reward_valid=True,
            verifier_valid=True, task_success=False, policy_violation=True,
            first_error="missing_confirmation", tool_error_count=0,
            db_match=False, communication_ok=None,
            termination_reason="agent_end",
        ),
    ]

    summary = BenchmarkSummary.from_records(records, expected_runs=3)

    assert summary.to_dict() == {
        "expected_runs": 3, "completed_runs": 2, "incomplete_runs": 1,
        "tau_reward_valid_count": 2, "tau_reward_invalid_count": 0,
        "tau_reward_valid_rate": 1.0, "tau_success_count": 1,
        "tau_success_rate": 0.5, "tau_reward_mean": 0.5,
        "db_match_true_count": 1, "db_match_present_count": 2,
        "db_match_missing_count": 0, "db_match_rate": 0.5,
        "communication_true_count": 1, "communication_present_count": 1,
        "communication_missing_count": 1, "communication_rate": 1.0,
        "termination_counts": {"agent_end": 2},
        "verifier_evaluated_count": 2, "verifier_invalid_count": 0,
        "verifier_reward_valid_count": 2, "verifier_reward_mean": -0.05,
        "policy_violation_count": 1, "policy_violation_rate": 0.5,
        "tool_error_run_count": 1, "tool_error_rate": 0.5,
        "tool_error_total": 1,
        "first_error_counts": {"missing_confirmation": 1},
    }


def test_summary_excludes_invalid_tau_and_verifier_rows_from_reward_metrics():
    records = [
        BenchmarkRecord(
            task_id="5", tau_reward=None, tau_reward_valid=False,
            verifier_valid=False, verifier_error="message_conversion_failed",
        ),
        BenchmarkRecord(
            task_id="9", tau_reward=1.0, tau_reward_valid=True,
            verifier_reward=0.25, verifier_reward_valid=False,
            verifier_valid=True, policy_violation=False, tool_error_count=0,
        ),
    ]

    summary = BenchmarkSummary.from_records(records, expected_runs=2)

    assert summary.tau_reward_invalid_count == 1
    assert summary.verifier_invalid_count == 1
    assert summary.verifier_reward_valid_count == 0
    assert summary.verifier_reward_mean is None
    json.dumps(summary.to_dict())


def test_record_from_verification_preserves_grpo_reward_and_diagnostics():
    result = VerificationResult(
        task_success=True,
        policy_violation=False,
        first_error="tool_error",
        reward=0.8,
        reward_valid=True,
        db_match=True,
        communication_ok=False,
        tool_error_count=1,
    )

    record = BenchmarkRecord.from_verification(
        task_id="5",
        trial=1,
        tau_reward=1.0,
        termination_reason="agent_end",
        result=result,
    )

    assert record.to_dict() == {
        "task_id": "5",
        "trial": 1,
        "tau_reward": 1.0,
        "tau_reward_valid": True,
        "verifier_reward": 0.8,
        "verifier_reward_valid": True,
        "verifier_valid": True,
        "task_success": True,
        "policy_violation": False,
        "first_error": "tool_error",
        "tool_error_count": 1,
        "db_match": True,
        "communication_ok": False,
        "termination_reason": "agent_end",
        "verifier_error": None,
    }
