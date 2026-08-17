from types import SimpleNamespace

from agent_for_business.tau_adapter import SimulationTrajectoryAdapter


class FakeSimulation:
    task_id = "retail-003"
    seed = 13

    def get_messages(self):
        return [
            SimpleNamespace(
                role="user",
                content="Please check my order.",
                tool_calls=None,
            ),
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call-2",
                        name="get_order",
                        arguments={"order_id": "W456"},
                    )
                ],
            ),
            SimpleNamespace(
                role="tool",
                id="call-2",
                content='{"status": "shipped"}',
            ),
            SimpleNamespace(
                role="assistant",
                content="Your order has shipped.",
                tool_calls=None,
            ),
        ]


def test_normalizes_tau_simulation_messages_into_trajectory():
    trajectory = SimulationTrajectoryAdapter().from_simulation(
        FakeSimulation(),
        terminal_state={"order_status": "shipped"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    assert trajectory.task_id == "retail-003"
    assert trajectory.seed == 13
    assert [event.kind for event in trajectory.events] == [
        "user_message",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]
    assert trajectory.events[1].tool_name == "get_order"
    assert trajectory.events[2].tool_call_id == "call-2"
