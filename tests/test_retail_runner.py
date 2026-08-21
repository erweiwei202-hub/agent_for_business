from types import SimpleNamespace

from agent_for_business.retail_runner import RetailTaskRunner
from agent_for_business.trajectory_store import TrajectoryStore


class FakeSimulation:
    task_id = "retail-004"
    seed = 19
    info = {
        "terminal_state": {"order_status": "cancelled"},
        "evaluation": {"task_success": True, "reward": 1.0},
    }

    def get_messages(self):
        return [
            SimpleNamespace(
                role="user",
                content="Cancel my pending order.",
                tool_calls=None,
            ),
            SimpleNamespace(
                role="assistant",
                content="The order has been cancelled.",
                tool_calls=None,
            ),
        ]


def test_runner_executes_one_task_and_persists_normalized_trajectory(tmp_path):
    calls = []

    def simulation_runner(task_id, seed):
        calls.append((task_id, seed))
        return FakeSimulation()

    store = TrajectoryStore(tmp_path / "trajectories.jsonl")
    runner = RetailTaskRunner(
        simulation_runner=simulation_runner,
        trajectory_store=store,
    )

    trajectory = runner.run(task_id="retail-004", seed=19)

    assert calls == [("retail-004", 19)]
    assert trajectory.task_id == "retail-004"
    assert trajectory.evaluation["task_success"] is True
    assert len(list(store.iter_trajectories())) == 1


def test_runner_calls_keyword_only_simulation_runner(tmp_path):
    calls = []

    def simulation_runner(*, task_id, seed):
        calls.append((task_id, seed))
        return FakeSimulation()

    runner = RetailTaskRunner(
        simulation_runner=simulation_runner,
        trajectory_store=TrajectoryStore(tmp_path / "trajectories.jsonl"),
    )

    runner.run(task_id="retail-004", seed=19)

    assert calls == [("retail-004", 19)]


def test_runner_maps_official_reward_info_components(tmp_path):
    class OfficialRewardInfoSimulation:
        task_id = "retail-013"
        seed = 23
        reward_info = SimpleNamespace(
            reward=1.0,
            db_check=SimpleNamespace(db_match=True),
            communicate_checks=[SimpleNamespace(met=True)],
        )

        def get_messages(self):
            return []

    store = TrajectoryStore(tmp_path / "trajectories.jsonl")
    runner = RetailTaskRunner(
        simulation_runner=lambda task_id, seed: OfficialRewardInfoSimulation(),
        trajectory_store=store,
    )

    trajectory = runner.run(task_id="retail-013", seed=23)

    assert trajectory.evaluation == {
        "reward": 1.0,
        "task_success": True,
        "db_match": True,
        "communication_ok": True,
        "reward_valid": True,
    }


def test_runner_completes_missing_evaluation_contract_fields(tmp_path):
    class PartiallyEvaluatedSimulation:
        task_id = "retail-014"
        seed = 25
        info = {
            "evaluation": {
                "reward": 1.0,
                "task_success": True,
                "db_match": True,
            }
        }
        reward_info = SimpleNamespace(
            reward=1.0,
            communicate_checks=[SimpleNamespace(met=True)],
        )

        def get_messages(self):
            return []

    runner = RetailTaskRunner(
        simulation_runner=lambda task_id, seed: PartiallyEvaluatedSimulation(),
        trajectory_store=TrajectoryStore(tmp_path / "trajectories.jsonl"),
    )

    trajectory = runner.run(task_id="retail-014", seed=25)

    assert trajectory.evaluation["reward_valid"] is True
    assert trajectory.evaluation["communication_ok"] is True


def test_runner_marks_official_success_reward_as_task_success(tmp_path):
    class SuccessfulOfficialSimulation:
        task_id = "retail-015"
        seed = 29
        reward_info = SimpleNamespace(reward=1.0)

        def get_messages(self):
            return []

    store = TrajectoryStore(tmp_path / "trajectories.jsonl")
    runner = RetailTaskRunner(
        simulation_runner=lambda task_id, seed: SuccessfulOfficialSimulation(),
        trajectory_store=store,
    )

    trajectory = runner.run(task_id="retail-015", seed=29)

    assert trajectory.evaluation["task_success"] is True


def test_runner_marks_missing_official_reward_as_invalid(tmp_path):
    class UnscoredSimulation:
        task_id = "retail-016"
        seed = 31

        def get_messages(self):
            return []

    store = TrajectoryStore(tmp_path / "trajectories.jsonl")
    runner = RetailTaskRunner(
        simulation_runner=lambda task_id, seed: UnscoredSimulation(),
        trajectory_store=store,
    )

    trajectory = runner.run(task_id="retail-016", seed=31)

    assert trajectory.evaluation["reward_valid"] is False
