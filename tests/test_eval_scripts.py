import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from agent_for_business.policy_verifier import VerificationResult

ROOT = Path(__file__).parents[1]


def load_script_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_comprehensive_benchmark_verifies_serialized_simulations():
    module = load_script_module(
        "run_final_benchmark_comprehensive",
        "eval-scripts/run_final_benchmark.py",
    )

    class FakeSimulation:
        task_id = "5"
        trial = 0
        seed = 300
        info = {"terminal_state": {}, "evaluation": {"task_success": True}}

        def get_messages(self):
            return []

    seen = []

    class FakeAdapter:
        def from_simulation(self, simulation, *, terminal_state, evaluation):
            return SimpleNamespace(task_id=simulation.task_id, evaluation=evaluation)

    class FakeVerifier:
        def verify(self, trajectory):
            seen.append(trajectory)
            return VerificationResult(
                task_success=True,
                policy_violation=False,
                first_error=None,
                reward=0.75,
                reward_valid=True,
                db_match=True,
                communication_ok=False,
                tool_error_count=0,
            )

    report = module.build_comprehensive_benchmark(
        [{"task_id": "5", "trial": 0, "reward_info": {"reward": 1.0}}],
        expected_runs=1,
        deserializer=lambda payload: FakeSimulation(),
        adapter=FakeAdapter(),
        verifier=FakeVerifier(),
    )

    assert len(seen) == 1
    assert report["summary"]["tau_reward_mean"] == 1.0
    assert report["summary"]["verifier_reward_mean"] == 0.75
    assert report["simulations"][0]["verifier"]["db_match"] is True
    assert report["simulations"][0]["verifier"]["communication_ok"] is False


def test_build_comprehensive_benchmark_keeps_tau_reward_when_verifier_conversion_fails():
    module = load_script_module(
        "run_final_benchmark_conversion_error",
        "eval-scripts/run_final_benchmark.py",
    )

    report = module.build_comprehensive_benchmark(
        [{"task_id": "9", "trial": 1, "reward_info": {"reward": 0.5}}],
        expected_runs=1,
        deserializer=lambda payload: (_ for _ in ()).throw(
            ValueError("bad serialized simulation")
        ),
        adapter=object(),
        verifier=object(),
    )

    assert report["summary"]["tau_reward_mean"] == 0.5
    assert report["summary"]["verifier_invalid_count"] == 1
    assert "bad serialized simulation" in report["records"][0]["verifier_error"]


def test_markdown_summary_explains_tau_and_verifier_metrics(tmp_path):
    module = load_script_module(
        "run_final_benchmark_markdown",
        "eval-scripts/run_final_benchmark.py",
    )
    output_path = tmp_path / "benchmark.json"
    summary_path = tmp_path / "benchmark.md"
    payload = {
        "simulations": [
            {
                "id": "sim-1",
                "task_id": "5",
                "trial": 0,
                "termination_reason": "agent_end",
                "reward_info": {"reward": 1.0},
                "verifier": {
                    "task_id": "5",
                    "trial": 0,
                    "tau_reward": 1.0,
                    "tau_reward_valid": True,
                    "verifier_reward": 0.8,
                    "verifier_reward_valid": True,
                    "verifier_valid": True,
                    "task_success": True,
                    "policy_violation": False,
                    "first_error": None,
                    "tool_error_count": 0,
                    "db_match": True,
                    "communication_ok": True,
                    "termination_reason": "agent_end",
                    "verifier_error": None,
                },
            }
        ],
        "benchmark": {
            "summary": {
                "expected_runs": 1,
                "completed_runs": 1,
                "incomplete_runs": 0,
                "tau_reward_valid_count": 1,
                "tau_reward_invalid_count": 0,
                "tau_reward_valid_rate": 1.0,
                "tau_success_count": 1,
                "tau_success_rate": 1.0,
                "tau_reward_mean": 1.0,
                "db_match_true_count": 1,
                "db_match_present_count": 1,
                "db_match_missing_count": 0,
                "db_match_rate": 1.0,
                "communication_true_count": 1,
                "communication_present_count": 1,
                "communication_missing_count": 0,
                "communication_rate": 1.0,
                "termination_counts": {"agent_end": 1},
                "verifier_evaluated_count": 1,
                "verifier_invalid_count": 0,
                "verifier_reward_valid_count": 1,
                "verifier_reward_mean": 0.8,
                "policy_violation_count": 0,
                "policy_violation_rate": 0.0,
                "tool_error_run_count": 0,
                "tool_error_rate": 0.0,
                "tool_error_total": 0,
                "first_error_counts": {},
            },
            "records": [],
        },
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    module.write_benchmark_summary(
        output_path=output_path,
        summary_path=summary_path,
        agent_model="qwen-sft",
        vllm_api_base="http://127.0.0.1:8000/v1",
        user_model="user-model",
        num_trials=1,
        seed=300,
        max_concurrency=1,
    )

    markdown = summary_path.read_text(encoding="utf-8")
    assert "## τ² Metrics" in markdown
    assert "## Verifier / GRPO Metrics" in markdown
    assert "## Verifier First Errors" in markdown
    assert "## Metric Definitions" in markdown
    assert "tau_reward" in markdown
    assert "verifier_reward" in markdown
    assert "db_match_rate" in markdown
    assert "communication_rate" in markdown
    assert "DB reward" in markdown
    assert "COMMUNICATE" in markdown
    assert "-1.0" in markdown
    assert "gate_decision" not in markdown
    assert "SFTValidationGate" not in markdown


def test_enrich_benchmark_output_writes_shared_report_without_duplicate_simulations(
    tmp_path,
):
    module = load_script_module(
        "run_final_benchmark_enrich_output",
        "eval-scripts/run_final_benchmark.py",
    )

    class FakeSimulation:
        task_id = "5"
        seed = 300
        info = None
        reward_info = SimpleNamespace(reward=1.0)

        def get_messages(self):
            return []

    output_path = tmp_path / "benchmark.json"
    output_path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": "5",
                        "trial": 0,
                        "reward_info": {"reward": 1.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    module._deserialize_tau2_simulation = lambda payload: FakeSimulation()

    module.enrich_benchmark_output(output_path=output_path, expected_runs=1)
    enriched = json.loads(output_path.read_text(encoding="utf-8"))

    assert "summary" in enriched["benchmark"]
    assert "records" in enriched["benchmark"]
    assert "simulations" not in enriched["benchmark"]
    assert enriched["simulations"][0]["verifier"]["verifier_valid"] is True


def test_final_benchmark_command_uses_fixed_test_protocol(tmp_path):
    module = load_script_module(
        "run_final_benchmark",
        "eval-scripts/run_final_benchmark.py",
    )

    command = module.build_command(
        agent_model="qwen-base",
        vllm_api_base="http://127.0.0.1:8000/v1",
        vllm_api_key="EMPTY",
        user_model="user-model",
        user_api_base="https://user.example/v1",
        user_api_key="secret",
        output_path=tmp_path / "qwen-base-final.json",
        num_trials=3,
        seed=300,
        max_concurrency=1,
    )

    assert command[:4] == ["-m", "tau2.cli", "run", "--domain"]
    assert command[4] == "retail"
    assert command[command.index("--num-trials") + 1] == "3"
    assert command[command.index("--seed") + 1] == "300"
    task_ids = command[
        command.index("--task-ids") + 1 : command.index("--agent")
    ]
    assert len(task_ids) == 30
    assert task_ids[0] == "5"
    assert task_ids[-1] == "77"
    assert "--auto-resume" in command


def test_materialize_benchmark_output_copies_tau2_results_to_requested_json(tmp_path):
    module = load_script_module(
        "run_final_benchmark_materialize",
        "eval-scripts/run_final_benchmark.py",
    )
    checkpoint_results = tmp_path / ".qwen-sft-final.tau2" / "results.json"
    checkpoint_results.parent.mkdir()
    checkpoint_results.write_text('{"simulations": []}\n', encoding="utf-8")
    output_path = tmp_path / "qwen-sft-final.json"

    module.materialize_benchmark_output(
        checkpoint_results_path=checkpoint_results,
        output_path=output_path,
    )

    assert output_path.read_text(encoding="utf-8") == '{"simulations": []}\n'


def test_load_dotenv_does_not_overwrite_existing_values(tmp_path, monkeypatch):
    module = load_script_module(
        "run_final_benchmark_env",
        "eval-scripts/run_final_benchmark.py",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        'USER_API_KEY="from-file"\nUSER_API_BASE=https://example/v1\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("USER_API_KEY", "from-process")

    loaded = module.load_dotenv(env_file)

    assert loaded == 1
    assert module.os.environ["USER_API_KEY"] == "from-process"
    assert module.os.environ["USER_API_BASE"] == "https://example/v1"


def test_vllm_command_sets_requested_compatibility_environment():
    module = load_script_module("serve_qwen", "eval-scripts/serve_qwen.py")

    command = module.build_command(
        executable="vllm",
        model="Qwen/Qwen3.5-2B",
        served_model_name="qwen-base",
        host="0.0.0.0",
        port=8000,
        api_key="EMPTY",
        max_model_len=32768,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    )

    assert command == [
        "vllm",
        "serve",
        "Qwen/Qwen3.5-2B",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--served-model-name",
        "qwen-base",
        "--api-key",
        "EMPTY",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.85",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]


def test_vllm_command_supports_lora_tool_call_serving():
    module = load_script_module("serve_qwen_lora", "eval-scripts/serve_qwen.py")

    command = module.build_command(
        executable="vllm",
        model="Qwen/Qwen3.5-2B",
        served_model_name="qwen-sft",
        host="0.0.0.0",
        port=8000,
        api_key="EMPTY",
        max_model_len=32768,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
        enable_lora=True,
        lora_modules=["qwen-sft=outputs/sft/checkpoint-qwen/checkpoint-294"],
    )

    assert "--enable-lora" in command
    assert command[command.index("--lora-modules") + 1] == (
        "qwen-sft=outputs/sft/checkpoint-qwen/checkpoint-294"
    )
    assert "--enable-auto-tool-choice" in command
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
