import json
from types import SimpleNamespace

import pytest

from agent_for_business.policy_verifier import VerificationResult


def test_run_validation_benchmark_collects_seeded_results_in_json_report():
    from agent_for_business.validation_benchmark import run_validation_benchmark

    calls = []

    class FakeRunner:
        def run(self, *, task_id, seed):
            calls.append((task_id, seed))
            return SimpleNamespace(task_id=task_id)

    class FakeVerifier:
        def verify(self, trajectory):
            return VerificationResult(
                task_success=trajectory.task_id == "validation-1",
                policy_violation=False,
                first_error=None,
                reward=1.0 if trajectory.task_id == "validation-1" else 0.0,
            )

    report = run_validation_benchmark(
        task_ids=("validation-1", "validation-2"),
        runner=lambda: FakeRunner(),
        verifier=FakeVerifier(),
        model_label="raw-model",
        seed=41,
    )

    assert calls == [("validation-1", 41), ("validation-2", 41)]
    assert report["model"] == "raw-model"
    assert report["task_ids"] == ["validation-1", "validation-2"]
    assert report["summary"] == {
        "task_count": 2,
        "success_rate": 0.5,
        "policy_violation_rate": 0.0,
        "tool_error_rate": 0.0,
        "valid_rate": 1.0,
    }
    assert report["results"][0]["task_success"] is True
    json.dumps(report)


def test_run_validation_benchmark_rejects_empty_task_ids_before_runner_factory():
    from agent_for_business.validation_benchmark import run_validation_benchmark

    def unexpected_runner_factory():
        raise AssertionError("runner factory must not be called")

    with pytest.raises(ValueError, match="validation task_ids must not be empty"):
        run_validation_benchmark(
            task_ids=(),
            runner=unexpected_runner_factory,
            verifier=object(),
            model_label="raw-model",
            seed=41,
        )


def test_run_validation_benchmark_accepts_runner_instance_for_validation_only_ids():
    from agent_for_business.validation_benchmark import run_validation_benchmark

    calls = []

    class ValidationOnlyRunner:
        def run(self, *, task_id, seed):
            assert task_id.startswith("validation-")
            calls.append((task_id, seed))
            return SimpleNamespace(task_id=task_id)

    class FakeVerifier:
        def verify(self, trajectory):
            return VerificationResult(
                task_success=True,
                policy_violation=False,
                first_error=None,
                reward=1.0,
            )

    report = run_validation_benchmark(
        task_ids=["validation-3"],
        runner=ValidationOnlyRunner(),
        verifier=FakeVerifier(),
        model_label="sft-model",
        seed=73,
    )

    assert calls == [("validation-3", 73)]
    assert report["task_ids"] == ["validation-3"]


def test_compare_raw_sft_validation_uses_same_task_ids_seed_and_returns_gate():
    from agent_for_business.validation_benchmark import compare_raw_sft_validation

    calls = []

    class FakeRunner:
        def __init__(self, model, successful_tasks):
            self.model = model
            self.successful_tasks = successful_tasks

        def run(self, *, task_id, seed):
            calls.append((self.model, task_id, seed))
            return SimpleNamespace(
                model=self.model,
                task_id=task_id,
                task_success=task_id in self.successful_tasks,
            )

    class FakeVerifier:
        def verify(self, trajectory):
            return VerificationResult(
                task_success=trajectory.task_success,
                policy_violation=False,
                first_error=None,
                reward=1.0 if trajectory.task_success else 0.0,
            )

    task_ids = ("validation-1", "validation-2")
    report = compare_raw_sft_validation(
        task_ids=task_ids,
        raw_runner=lambda: FakeRunner("raw", {"validation-1"}),
        sft_runner=lambda: FakeRunner("sft", {"validation-1", "validation-2"}),
        verifier=FakeVerifier(),
        seed=101,
    )

    assert calls == [
        ("raw", "validation-1", 101),
        ("raw", "validation-2", 101),
        ("sft", "validation-1", 101),
        ("sft", "validation-2", 101),
    ]
    assert report["task_ids"] == list(task_ids)
    assert report["raw"]["model"] == "raw"
    assert report["sft"]["model"] == "sft"
    assert report["summary"]["raw"]["success_rate"] == 0.5
    assert report["summary"]["sft"]["success_rate"] == 1.0
    assert report["gate_decision"] == {
        "passed": True,
        "reason": "sft_ready_for_grpo",
    }
    json.dumps(report)


def test_compare_raw_sft_validation_blocks_invalid_sft_rewards():
    from agent_for_business.validation_benchmark import compare_raw_sft_validation

    class FakeRunner:
        def __init__(self, reward_valid):
            self.reward_valid = reward_valid

        def run(self, *, task_id, seed):
            return SimpleNamespace(task_id=task_id, reward_valid=self.reward_valid)

    class FakeVerifier:
        def verify(self, trajectory):
            return VerificationResult(
                task_success=True,
                policy_violation=False,
                first_error=None,
                reward=1.0,
                reward_valid=trajectory.reward_valid,
            )

    report = compare_raw_sft_validation(
        task_ids=["validation-1"],
        raw_runner=FakeRunner(True),
        sft_runner=FakeRunner(False),
        verifier=FakeVerifier(),
        seed=103,
    )

    assert report["gate_decision"] == {
        "passed": False,
        "reason": "benchmark_contains_invalid_rewards",
    }
