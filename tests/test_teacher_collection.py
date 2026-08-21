from agent_for_business.policy_verifier import RetailPolicyVerifier
from agent_for_business.teacher_collection import TeacherTrajectoryCollector
from agent_for_business.trajectory import TrajectoryRecorder
from agent_for_business.trajectory_store import TrajectoryStore


def make_trajectory(task_id, success, seed=1):
    recorder = TrajectoryRecorder(task_id=task_id, seed=seed)
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-1",
    )
    if success:
        recorder.append_user("Please check my order.")
        evaluation = {
            "task_success": True,
            "db_match": True,
            "communication_ok": True,
            "reward": 1.0,
        }
    else:
        recorder.append_user("The simulator failed.")
        evaluation = {
            "task_success": False,
            "reward": 0.0,
        }
    return recorder.finish(terminal_state={}, evaluation=evaluation)


def test_collector_routes_raw_accepted_and_failed_trajectories(tmp_path):
    queued = [make_trajectory("retail-013", True), make_trajectory("retail-013", False)]

    class FakeRunner:
        def run(self, *, task_id, seed):
            trajectory = queued.pop(0)
            trajectory.seed = seed
            return trajectory

    raw_store = TrajectoryStore(tmp_path / "raw.jsonl")
    accepted_store = TrajectoryStore(tmp_path / "accepted.jsonl")
    failed_store = TrajectoryStore(tmp_path / "failed.jsonl")
    collector = TeacherTrajectoryCollector(
        runner=FakeRunner(),
        verifier=RetailPolicyVerifier(),
        raw_store=raw_store,
        accepted_store=accepted_store,
        failed_store=failed_store,
    )

    summary = collector.collect(task_ids=["retail-013"], attempts_per_task=2)

    assert summary.raw_count == 2
    assert summary.accepted_count == 1
    assert summary.failed_count == 1
    assert len(list(raw_store.iter_trajectories())) == 2
    assert len(list(accepted_store.iter_trajectories())) == 1
    assert len(list(failed_store.iter_trajectories())) == 1


def test_collector_supports_parallel_workers(tmp_path):
    class ParallelRunner:
        def run(self, *, task_id, seed):
            return make_trajectory(task_id, True, seed=seed)

    raw_store = TrajectoryStore(tmp_path / "raw.jsonl")
    accepted_store = TrajectoryStore(tmp_path / "accepted.jsonl")
    failed_store = TrajectoryStore(tmp_path / "failed.jsonl")
    collector = TeacherTrajectoryCollector(
        runner=ParallelRunner(),
        verifier=RetailPolicyVerifier(),
        raw_store=raw_store,
        accepted_store=accepted_store,
        failed_store=failed_store,
    )

    summary = collector.collect(
        task_ids=["retail-021", "retail-022"],
        attempts_per_task=2,
        max_workers=2,
    )

    assert summary.raw_count == 4
    assert summary.accepted_count == 4
    assert summary.failed_count == 0


def test_collector_resumes_from_completed_runtime_json(tmp_path):
    previous = make_trajectory("retail-013", True, seed=1)
    runtime_store = TrajectoryStore(tmp_path / "runtime.jsonl")
    runtime_store.append(previous)

    class FailingRunner:
        def run(self, *, task_id, seed):
            raise AssertionError("completed runtime trajectory was rerun")

    raw_store = TrajectoryStore(tmp_path / "raw.jsonl")
    accepted_store = TrajectoryStore(tmp_path / "accepted.jsonl")
    failed_store = TrajectoryStore(tmp_path / "failed.jsonl")
    collector = TeacherTrajectoryCollector(
        runner=FailingRunner(),
        verifier=RetailPolicyVerifier(),
        raw_store=raw_store,
        accepted_store=accepted_store,
        failed_store=failed_store,
    )

    summary = collector.collect(
        task_ids=["retail-013"],
        attempts_per_task=1,
        base_seed=1,
        runtime_store=runtime_store,
    )

    assert summary.raw_count == 1
    assert summary.accepted_count == 1
    assert summary.failed_count == 0


def test_collector_retries_runtime_error_from_runtime_json(tmp_path):
    previous = make_trajectory("retail-013", False, seed=1)
    previous.evaluation["runtime_status"] = "error"
    runtime_store = TrajectoryStore(tmp_path / "runtime.jsonl")
    runtime_store.append(previous)
    calls = []

    class RetryRunner:
        def run(self, *, task_id, seed):
            calls.append((task_id, seed))
            return make_trajectory(task_id, True, seed=seed)

    raw_store = TrajectoryStore(tmp_path / "raw.jsonl")
    accepted_store = TrajectoryStore(tmp_path / "accepted.jsonl")
    failed_store = TrajectoryStore(tmp_path / "failed.jsonl")
    collector = TeacherTrajectoryCollector(
        runner=RetryRunner(),
        verifier=RetailPolicyVerifier(),
        raw_store=raw_store,
        accepted_store=accepted_store,
        failed_store=failed_store,
    )

    summary = collector.collect(
        task_ids=["retail-013"],
        attempts_per_task=1,
        base_seed=1,
        runtime_store=runtime_store,
    )

    assert calls == [("retail-013", 1)]
    assert summary.accepted_count == 1
