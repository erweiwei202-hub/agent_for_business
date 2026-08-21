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


def test_collector_creates_empty_accepted_file_when_all_attempts_fail(tmp_path):
    queued = [make_trajectory("retail-013", False)]

    class FakeRunner:
        def run(self, *, task_id, seed):
            return queued.pop(0)

    accepted_path = tmp_path / "accepted.jsonl"
    collector = TeacherTrajectoryCollector(
        runner=FakeRunner(),
        verifier=RetailPolicyVerifier(),
        raw_store=TrajectoryStore(tmp_path / "raw.jsonl"),
        accepted_store=TrajectoryStore(accepted_path),
        failed_store=TrajectoryStore(tmp_path / "failed.jsonl"),
    )

    collector.collect(task_ids=["retail-013"], attempts_per_task=1)

    assert accepted_path.exists()
    assert accepted_path.read_text(encoding="utf-8") == ""


def test_collector_resumes_without_rerunning_already_routed_trajectory(tmp_path):
    previous = make_trajectory("retail-013", True)
    runtime_store = TrajectoryStore(tmp_path / "runtime.jsonl")
    raw_store = TrajectoryStore(tmp_path / "raw.jsonl")
    accepted_store = TrajectoryStore(tmp_path / "accepted.jsonl")
    failed_store = TrajectoryStore(tmp_path / "failed.jsonl")
    runtime_store.append(previous)
    raw_store.append(previous)
    accepted_store.append(previous)

    class FailingRunner:
        def run(self, *, task_id, seed):
            raise AssertionError("an already routed trajectory was rerun")

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


def test_collector_routes_runtime_only_trajectory_without_rerunning(tmp_path):
    previous = make_trajectory("retail-013", False)
    runtime_store = TrajectoryStore(tmp_path / "runtime.jsonl")
    failed_store = TrajectoryStore(tmp_path / "failed.jsonl")
    runtime_store.append(previous)

    class FailingRunner:
        def run(self, *, task_id, seed):
            raise AssertionError("a runtime trajectory was rerun")

    raw_store = TrajectoryStore(tmp_path / "raw.jsonl")
    accepted_store = TrajectoryStore(tmp_path / "accepted.jsonl")
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
    assert summary.accepted_count == 0
    assert summary.failed_count == 1
    routed = list(failed_store.iter_trajectories())
    assert len(routed) == 1
    assert routed[0].evaluation["reward_valid"] is True
    assert routed[0].evaluation["communication_ok"] is None


def test_collector_keeps_collecting_after_one_worker_runtime_error(tmp_path):
    class PartiallyFailingRunner:
        def run(self, *, task_id, seed):
            if seed == 0:
                raise RuntimeError("temporary provider failure")
            return make_trajectory(task_id, True, seed=seed)

    runtime_store = TrajectoryStore(tmp_path / "runtime.jsonl")
    raw_store = TrajectoryStore(tmp_path / "raw.jsonl")
    accepted_store = TrajectoryStore(tmp_path / "accepted.jsonl")
    failed_store = TrajectoryStore(tmp_path / "failed.jsonl")
    collector = TeacherTrajectoryCollector(
        runner=PartiallyFailingRunner(),
        verifier=RetailPolicyVerifier(),
        raw_store=raw_store,
        accepted_store=accepted_store,
        failed_store=failed_store,
    )

    summary = collector.collect(
        task_ids=["retail-013"],
        attempts_per_task=2,
        base_seed=0,
        max_workers=2,
        runtime_store=runtime_store,
    )

    assert summary.raw_count == 1
    assert summary.accepted_count == 1
    assert summary.failed_count == 0
    assert summary.runtime_error_count == 1
    runtime_statuses = [
        trajectory.evaluation.get("runtime_status")
        for trajectory in runtime_store.iter_trajectories()
    ]
    assert "error" in runtime_statuses


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
