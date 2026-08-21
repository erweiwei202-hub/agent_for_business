from agent_for_business.policy_verifier import RetailPolicyVerifier
from agent_for_business.trajectory import TrajectoryRecorder


def test_rejects_mutation_without_explicit_confirmation():
    recorder = TrajectoryRecorder(task_id="retail-005", seed=23)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-1",
    )
    recorder.append_user("Please cancel my pending order W789.")
    recorder.append_tool_call(
        name="cancel_pending_order",
        arguments={"order_id": "W789", "reason": "no longer needed"},
        call_id="call-3",
    )
    trajectory = recorder.finish(
        terminal_state={"order_status": "cancelled"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.policy_violation is True
    assert result.first_error == "missing_confirmation"
    assert result.reward == -1.0


def test_accepts_mutation_after_explicit_confirmation():
    recorder = TrajectoryRecorder(task_id="retail-006", seed=29)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-2",
    )
    recorder.append_user("Please cancel my pending order W790.")
    recorder.append_assistant("The order and cancellation reason are ready. Yes?")
    recorder.append_user("Yes, please.")
    recorder.append_tool_call(
        name="cancel_pending_order",
        arguments={"order_id": "W790", "reason": "no longer needed"},
        call_id="call-4",
    )
    trajectory = recorder.finish(
        terminal_state={"order_status": "cancelled"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.policy_violation is False
    assert result.first_error is None
    assert result.reward == 1.0


def test_accepts_case_insensitive_confirmation_with_action_details():
    recorder = TrajectoryRecorder(task_id="retail-006-details", seed=30)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-details",
    )
    recorder.append_assistant(
        "The desk lamp exchange is ready. Please confirm that you want to proceed."
    )
    recorder.append_user("YeS, please proceed with the desk lamp exchange.")
    recorder.append_tool_call(
        name="exchange_delivered_order_items",
        arguments={
            "item_ids": ["lamp-old"],
            "new_item_ids": ["lamp-new"],
            "order_id": "W800",
            "payment_method_id": "paypal-1",
        },
        call_id="exchange-details",
    )
    trajectory = recorder.finish(
        terminal_state={"order_status": "exchange requested"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.policy_violation is False
    assert result.first_error is None
    assert result.reward == 1.0


def test_rejects_confirmation_without_action_summary():
    recorder = TrajectoryRecorder(task_id="retail-007", seed=31)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-3",
    )
    recorder.append_user("Please cancel my pending order W791.")
    recorder.append_assistant("Should I proceed?")
    recorder.append_user("Yes.")
    recorder.append_tool_call(
        name="cancel_pending_order",
        arguments={"order_id": "W791", "reason": "no longer needed"},
        call_id="call-5",
    )
    trajectory = recorder.finish(
        terminal_state={"order_status": "cancelled"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.policy_violation is True
    assert result.first_error == "missing_action_summary"
    assert result.reward == -1.0


def test_rejects_mutation_before_user_authentication():
    recorder = TrajectoryRecorder(task_id="retail-008", seed=37)
    recorder.append_user("Please cancel my pending order W792.")
    recorder.append_assistant(
        "I will cancel order W792 for no longer needed. Please confirm."
    )
    recorder.append_user("Yes, please.")
    recorder.append_tool_call(
        name="cancel_pending_order",
        arguments={"order_id": "W792", "reason": "no longer needed"},
        call_id="call-6",
    )
    trajectory = recorder.finish(
        terminal_state={"order_status": "cancelled"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.policy_violation is True
    assert result.first_error == "authentication_failure"
    assert result.reward == -1.0


def test_requires_a_new_confirmation_for_each_mutation():
    recorder = TrajectoryRecorder(task_id="retail-009", seed=41)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-4",
    )
    recorder.append_assistant("I will cancel order W793. Please confirm.")
    recorder.append_user("Yes.")
    recorder.append_tool_call(
        name="cancel_pending_order",
        arguments={"order_id": "W793", "reason": "no longer needed"},
        call_id="call-7",
    )
    recorder.append_tool_call(
        name="modify_pending_order_address",
        arguments={"order_id": "W793", "address": "New address"},
        call_id="call-8",
    )
    trajectory = recorder.finish(
        terminal_state={"order_status": "cancelled"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.policy_violation is True
    assert result.first_error == "missing_confirmation"
    assert result.reward == -1.0


def test_exposes_verified_terminal_components():
    recorder = TrajectoryRecorder(task_id="retail-010", seed=43)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-5",
    )
    recorder.append_user("What is the status of order W794?")
    trajectory = recorder.finish(
        terminal_state={"order_status": "delivered"},
        evaluation={
            "task_success": True,
            "db_match": True,
            "communication_ok": True,
            "reward": 1.0,
        },
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.task_success is True
    assert result.db_match is True
    assert result.communication_ok is True
    assert result.reward_valid is True


def test_marks_invalid_infrastructure_result_without_model_penalty():
    recorder = TrajectoryRecorder(task_id="retail-012", seed=53)
    recorder.append_user("The simulator disconnected.")
    trajectory = recorder.finish(
        terminal_state={},
        evaluation={
            "task_success": False,
            "reward_valid": False,
            "reward": 0.0,
        },
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.reward_valid is False
    assert result.policy_violation is False
    assert result.reward == 0.0


def test_rejects_mutation_after_failed_authentication_result():
    recorder = TrajectoryRecorder(task_id="retail-014", seed=59)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "missing@example.com"},
        call_id="auth-6",
    )
    recorder.append_tool_result(
        call_id="auth-6",
        content="Error: User not found",
    )
    recorder.append_assistant("I will cancel order W795. Please confirm.")
    recorder.append_user("Yes, please.")
    recorder.append_tool_call(
        name="cancel_pending_order",
        arguments={"order_id": "W795", "reason": "no longer needed"},
        call_id="call-9",
    )
    trajectory = recorder.finish(
        terminal_state={"order_status": "cancelled"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.policy_violation is True
    assert result.first_error == "authentication_failure"
    assert result.reward == -1.0


def test_records_recoverable_tool_error_with_limited_reward_penalty():
    recorder = TrajectoryRecorder(task_id="retail-017", seed=61)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-7",
    )
    recorder.append_tool_result(
        call_id="auth-7",
        content={"user_id": "user-1"},
    )
    recorder.append_tool_call(
        name="get_order_details",
        arguments={"order_id": "W796"},
        call_id="call-10",
    )
    recorder.append_tool_result(
        call_id="call-10",
        content="Error: temporary lookup failure",
    )
    recorder.append_tool_call(
        name="get_order_details",
        arguments={"order_id": "W796"},
        call_id="call-11",
    )
    recorder.append_tool_result(
        call_id="call-11",
        content={"order_id": "W796", "status": "pending"},
    )
    trajectory = recorder.finish(
        terminal_state={"order_status": "pending"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.policy_violation is False
    assert result.first_error == "tool_error"
    assert result.tool_error_count == 1
    assert result.reward == 0.9


def test_allows_multiple_pending_read_only_tool_calls():
    recorder = TrajectoryRecorder(task_id="retail-018", seed=67)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-8",
    )
    recorder.append_tool_result(
        call_id="auth-8",
        content={"user_id": "user-1"},
    )
    recorder.append_tool_call(
        name="get_order_details",
        arguments={"order_id": "W797"},
        call_id="call-12",
    )
    recorder.append_tool_call(
        name="get_order_details",
        arguments={"order_id": "W798"},
        call_id="call-13",
    )
    trajectory = recorder.finish(
        terminal_state={},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = RetailPolicyVerifier().verify(trajectory)

    assert result.policy_violation is False
    assert result.first_error is None
    assert result.reward == 1.0
