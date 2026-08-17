from agent_for_business.badcase import BadcaseAnalyzer
from agent_for_business.trajectory import TrajectoryRecorder


def test_classifies_missing_confirmation_badcase():
    recorder = TrajectoryRecorder(task_id="retail-bad-confirm", seed=101)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-badcase",
    )
    recorder.append_tool_call(
        name="cancel_pending_order",
        arguments={"order_id": "W804", "reason": "no longer needed"},
        call_id="cancel-badcase",
    )
    trajectory = recorder.finish(
        terminal_state={"order_status": "cancelled"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    record = BadcaseAnalyzer().analyze(trajectory)

    assert record.task_id == "retail-bad-confirm"
    assert record.category == "missing_confirmation"
    assert record.policy_violation is True


def test_classifies_invalid_reward_as_infrastructure_issue():
    recorder = TrajectoryRecorder(task_id="retail-invalid", seed=103)
    recorder.append_user("The simulator disconnected.")
    trajectory = recorder.finish(
        terminal_state={},
        evaluation={
            "task_success": False,
            "reward": 0.0,
            "reward_valid": False,
        },
    )

    record = BadcaseAnalyzer().analyze(trajectory)

    assert record.category == "infrastructure_invalid"
    assert record.reward_valid is False
