from agent_for_business.policy_verifier import RetailPolicyVerifier
from agent_for_business.pipeline import build_sft_dataset
from agent_for_business.pipeline import run_smoke
from agent_for_business.sft_dataset import SFTDatasetStore
from agent_for_business.trajectory import TrajectoryRecorder
from agent_for_business.trajectory_store import TrajectoryStore


def test_build_sft_dataset_writes_only_verified_examples(tmp_path):
    accepted_path = tmp_path / "accepted.jsonl"
    output_path = tmp_path / "sft.jsonl"
    recorder = TrajectoryRecorder(task_id="retail-pipeline-1", seed=107)
    recorder.append_user("What is the status of my order?")
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-pipeline-1",
    )
    recorder.append_assistant("Your order is pending.")
    trajectory = recorder.finish(
        terminal_state={"order_status": "pending"},
        evaluation={"task_success": True, "reward": 1.0},
    )
    TrajectoryStore(accepted_path).append(trajectory)

    summary = build_sft_dataset(
        input_path=accepted_path,
        output_path=output_path,
        verifier=RetailPolicyVerifier(),
    )

    assert summary == {
        "input_count": 1,
        "written_count": 1,
        "skipped_count": 0,
    }
    assert len(list(SFTDatasetStore(output_path).iter_examples())) == 1


def test_run_smoke_returns_verification_report_from_one_task():
    recorder = TrajectoryRecorder(task_id="retail-smoke-1", seed=109)
    recorder.append_user("Check my order.")
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-smoke-1",
    )
    recorder.append_assistant("Your order is pending.")
    trajectory = recorder.finish(
        terminal_state={"order_status": "pending"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    class FakeRunner:
        def run(self, *, task_id, seed):
            assert task_id == "retail-smoke-1"
            assert seed == 109
            return trajectory

    report = run_smoke(
        runner=FakeRunner(),
        verifier=RetailPolicyVerifier(),
        task_id="retail-smoke-1",
        seed=109,
    )

    assert report["task_id"] == "retail-smoke-1"
    assert report["verification"]["task_success"] is True
    assert report["verification"]["reward"] == 1.0
