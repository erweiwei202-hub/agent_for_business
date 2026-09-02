from dataclasses import dataclass
from types import SimpleNamespace

from agent_for_business.grpo_agent import GenerationTrace
from agent_for_business.grpo_rollout import Tau2LocalRolloutRunner
from agent_for_business.policy_verifier import VerificationResult


@dataclass
class FakeSimulation:
    task_id: str = "retail-1"
    seed: int = 17
    info: dict = None

    def __post_init__(self):
        self.info = {
            "terminal_state": {"order_status": "cancelled"},
            "evaluation": {
                "task_success": True,
                "reward": 1.0,
                "reward_valid": True,
            },
        }

    def get_messages(self):
        return [
            SimpleNamespace(role="user", content="Cancel it", tool_calls=None),
            SimpleNamespace(
                role="assistant",
                content="Done",
                tool_calls=None,
            ),
        ]


def test_rollout_result_is_updateable_only_with_valid_reward_and_action_trace():
    trace = GenerationTrace(
        prompt_ids=(1, 2),
        response_ids=(3,),
        old_logprobs=(-0.2,),
        action_mask=(True,),
    )

    class FakeAdapter:
        def from_simulation(self, simulation, *, terminal_state, evaluation):
            return SimpleNamespace(
                task_id=simulation.task_id,
                seed=simulation.seed,
                events=[],
                evaluation=evaluation,
                terminal_state=terminal_state,
            )

    class FakeVerifier:
        def verify(self, trajectory):
            return VerificationResult(
                task_success=True,
                policy_violation=False,
                first_error=None,
                reward=1.0,
                reward_valid=True,
            )

    runner = Tau2LocalRolloutRunner(
        simulation_factory=lambda task_id, seed: (FakeSimulation(task_id, seed), (trace,)),
        adapter=FakeAdapter(),
        verifier=FakeVerifier(),
    )

    result = runner.run(task_id="retail-1", seed=17)

    assert result.verification.reward == 1.0
    assert result.trajectory.task_id == "retail-1"
    assert result.valid_for_update is True


def test_rollout_runner_retains_infrastructure_failure_as_invalid_result():
    runner = Tau2LocalRolloutRunner(
        simulation_factory=lambda task_id, seed: (_ for _ in ()).throw(
            RuntimeError("user simulator unavailable")
        )
    )

    result = runner.run(task_id="retail-1", seed=17)

    assert result.valid_for_update is False
    assert result.verification.reward_valid is False
    assert result.verification.first_error == "infrastructure_invalid"
    assert "user simulator unavailable" in result.error
