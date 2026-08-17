from agent_for_business.trajectory import TrajectoryRecorder


def test_records_replayable_retail_events_in_order():
    recorder = TrajectoryRecorder(task_id="retail-001", seed=7)

    recorder.append_user("I want to check my order.")
    recorder.append_tool_call(
        name="get_order",
        arguments={"order_id": "W123"},
        call_id="call-1",
    )
    recorder.append_tool_result(
        call_id="call-1",
        content={"order_id": "W123", "status": "delivered"},
    )
    recorder.append_assistant("Your order has been delivered.")

    trajectory = recorder.finish(
        terminal_state={"order_status": "delivered"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    assert trajectory.task_id == "retail-001"
    assert trajectory.seed == 7
    assert [event.kind for event in trajectory.events] == [
        "user_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]
    assert trajectory.events[1].tool_name == "get_order"
    assert trajectory.events[1].arguments == {"order_id": "W123"}
    assert trajectory.events[2].tool_call_id == "call-1"
    assert trajectory.terminal_state == {"order_status": "delivered"}
    assert trajectory.evaluation["task_success"] is True

    serialized = trajectory.to_dict()
    assert serialized["task_id"] == "retail-001"
    assert serialized["events"][1]["tool_name"] == "get_order"
