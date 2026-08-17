from agent_for_business.trajectory import TrajectoryRecorder
from agent_for_business.trajectory_store import TrajectoryStore


def test_finished_trajectory_round_trips_through_jsonl(tmp_path):
    recorder = TrajectoryRecorder(task_id="retail-002", seed=11)
    recorder.append_user("Please check my order.")
    recorder.append_assistant("I will look up the order.")
    trajectory = recorder.finish(
        terminal_state={"order_status": "pending"},
        evaluation={"task_success": False, "reward": 0.0},
    )

    store = TrajectoryStore(tmp_path / "trajectories.jsonl")
    store.append(trajectory)

    loaded = list(store.iter_trajectories())

    assert len(loaded) == 1
    assert loaded[0].to_dict() == trajectory.to_dict()
