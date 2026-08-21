import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_for_business.cli import (
    _runtime_llm_args,
    _project_env_path,
    build_parser,
    llm_args_from_env,
    load_project_env,
    main,
)
from agent_for_business.trajectory import TrajectoryRecorder
from agent_for_business.trajectory_store import TrajectoryStore


def test_cli_exposes_smoke_collection_and_sft_commands():
    parser = build_parser()

    smoke = parser.parse_args(["smoke", "--task-id", "0"])
    collect = parser.parse_args(["collect-teacher"])
    build = parser.parse_args(
        ["build-sft", "--input", "accepted.jsonl", "--output", "sft.jsonl"]
    )
    train = parser.parse_args(
        ["train-sft", "--dataset", "sft.jsonl", "--output-dir", "outputs/sft"]
    )

    assert smoke.command == "smoke"
    assert smoke.task_id == "0"
    assert collect.command == "collect-teacher"
    assert collect.max_workers == 4
    assert build.command == "build-sft"
    assert train.command == "train-sft"


def test_cli_reads_openai_compatible_llm_settings_from_environment():
    args = llm_args_from_env(
        "AGENT",
        {
            "AGENT_API_KEY": "secret",
            "AGENT_API_BASE": "http://127.0.0.1:8000/v1",
        },
    )

    assert args == {
        "api_key": "secret",
        "api_base": "http://127.0.0.1:8000/v1",
    }


def test_cli_defaults_to_anthropic_messages_for_deepseek():
    args = build_parser().parse_args(["smoke", "--task-id", "0"])

    assert args.agent_llm == "anthropic/deepseek-v4-flash"
    assert args.user_llm == "anthropic/deepseek-v4-flash"


def test_cli_reads_anthropic_credentials_from_shared_environment():
    args = llm_args_from_env(
        "AGENT",
        {
            "ANTHROPIC_API_KEY": "secret",
            "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
        },
    )

    assert args == {
        "api_key": "secret",
        "api_base": "https://api.deepseek.com/anthropic",
    }


def test_cli_reads_openai_compatible_credential_aliases():
    args = llm_args_from_env(
        "AGENT",
        {
            "OPENAI_API_KEY": "secret",
            "OPENAI_BASE_URL": "https://api.example.com/v1",
        },
    )

    assert args == {
        "api_key": "secret",
        "api_base": "https://api.example.com/v1",
    }


def test_cli_falls_back_to_project_env_when_started_elsewhere(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RETAIL_AGENT_ENV_FILE", raising=False)

    assert _project_env_path() == Path(__file__).resolve().parents[1] / ".env"


def test_load_project_env_reads_dotenv_values_without_overwriting_existing(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        'ANTHROPIC_API_KEY="from-file"\n'
        "ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic\n"
        "AGENT_LLM=anthropic/deepseek-v4-flash\n",
        encoding="utf-8",
    )
    environ = {"AGENT_LLM": "existing-model"}

    loaded = load_project_env(env_file, environ=environ)

    assert loaded == 2
    assert environ["ANTHROPIC_API_KEY"] == "from-file"
    assert environ["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert environ["AGENT_LLM"] == "existing-model"


def test_build_sft_cli_writes_output_and_prints_summary(tmp_path, capsys):
    input_path = tmp_path / "accepted.jsonl"
    output_path = tmp_path / "sft.jsonl"
    recorder = TrajectoryRecorder(task_id="retail-cli-1", seed=113)
    recorder.append_user("Check my order.")
    recorder.append_assistant("Your order is pending.")
    trajectory = recorder.finish(
        terminal_state={"order_status": "pending"},
        evaluation={"task_success": True, "reward": 1.0},
    )
    TrajectoryStore(input_path).append(trajectory)

    assert main(
        ["build-sft", "--input", str(input_path), "--output", str(output_path)]
    ) == 0

    assert output_path.exists()
    assert '"written_count": 1' in capsys.readouterr().out


def test_collect_teacher_creates_outputs_before_first_runner_attempt(
    tmp_path, monkeypatch
):
    import agent_for_business.cli as cli_module

    class FailingRunner:
        def run(self, *, task_id, seed):
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        cli_module,
        "create_tau2_retail_runner",
        lambda **kwargs: FailingRunner(),
    )
    monkeypatch.setattr(
        cli_module,
        "load_retail_task_partition",
        lambda path: SimpleNamespace(train=("retail-013",)),
    )

    output_dir = tmp_path / "teacher"
    assert main(
        [
            "collect-teacher",
            "--output-dir",
            str(output_dir),
            "--attempts-per-task",
            "1",
            "--max-workers",
            "1",
        ]
    ) == 0

    for name in (
        "runtime.jsonl",
        "raw.jsonl",
        "accepted.jsonl",
        "failed.jsonl",
        "collection_report.json",
    ):
        assert (output_dir / name).exists()

    report = json.loads(
        (output_dir / "collection_report.json").read_text(encoding="utf-8")
    )
    assert report["summary"] == {
        "raw_count": 0,
        "accepted_count": 0,
        "failed_count": 0,
        "runtime_error_count": 1,
    }


def test_cli_passes_request_timeout_to_both_llm_roles():
    parsed = build_parser().parse_args(
        ["smoke", "--task-id", "0", "--request-timeout", "12.5"]
    )

    assert _runtime_llm_args(parsed, "agent")["timeout"] == 12.5
    assert _runtime_llm_args(parsed, "user")["timeout"] == 12.5
