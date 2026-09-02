import json

import pytest

import agent_for_business.cli as cli_module
from agent_for_business.cli import (
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


def test_cli_exposes_grpo_command_with_confirmed_training_defaults():
    parser = build_parser()

    grpo = parser.parse_args(["grpo", "--model", "sft-checkpoint"])

    assert grpo.command == "grpo"
    assert grpo.model == "sft-checkpoint"
    assert grpo.groups_per_batch == 50
    assert grpo.group_size == 4
    assert grpo.batch_epochs == 2
    assert grpo.max_workers == 4
    assert grpo.inference_microbatch == 2
    assert grpo.clip_ratio == 0.2
    assert grpo.kl_beta == 0.001
    assert grpo.max_rollout_batches == 2


def test_grpo_cli_reads_user_llm_from_project_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "USER_LLM=gpt-5.6-luna\n"
        "USER_API_BASE=https://user.example/v1\n"
        "USER_API_KEY=user-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("USER_LLM", raising=False)
    monkeypatch.delenv("USER_API_BASE", raising=False)
    monkeypatch.delenv("USER_API_KEY", raising=False)
    monkeypatch.setenv("RETAIL_AGENT_ENV_FILE", str(env_file))
    captured = {}

    def fake_train_grpo(config):
        captured["config"] = config
        return {"optimizer_steps": 0}

    monkeypatch.setattr(cli_module, "train_grpo", fake_train_grpo)

    assert main(
        [
            "grpo",
            "--model",
            "sft-checkpoint",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    ) == 0

    config = captured["config"]
    assert config.user_llm == "gpt-5.6-luna"
    assert config.user_llm_args == {
        "api_key": "user-secret",
        "api_base": "https://user.example/v1",
    }


def test_cli_exposes_local_grpo_rollout_and_optimizer_options():
    parser = build_parser()

    grpo = parser.parse_args(
        [
            "grpo",
            "--model",
            "sft-checkpoint",
            "--user-llm",
            "anthropic/user-sim",
            "--learning-rate",
            "0.00001",
            "--temperature",
            "0.2",
            "--device",
            "cuda",
            "--parallel-generation",
            "--resume-from",
            "outputs/grpo/checkpoint-1",
        ]
    )

    assert grpo.user_llm == "anthropic/user-sim"
    assert grpo.learning_rate == 0.00001
    assert grpo.temperature == 0.2
    assert grpo.device == "cuda"
    assert grpo.parallel_generation is True
    assert grpo.resume_from == "outputs/grpo/checkpoint-1"


def test_cli_rejects_non_positive_grpo_batch_parameters():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["grpo", "--model", "sft-checkpoint", "--groups-per-batch", "0"]
        )


def test_cli_rejects_negative_grpo_loss_coefficients():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["grpo", "--model", "sft-checkpoint", "--kl-beta", "-0.1"]
        )


def test_grpo_cli_passes_config_to_trainer_and_writes_manifest(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("AGENT_LLM", raising=False)
    monkeypatch.delenv("USER_LLM", raising=False)
    monkeypatch.setenv(
        "RETAIL_AGENT_ENV_FILE", str(tmp_path / "missing-test.env")
    )
    captured = {}

    def fake_train_grpo(config):
        captured["config"] = config
        return {"optimizer_steps": 4}

    monkeypatch.setattr(cli_module, "train_grpo", fake_train_grpo)

    assert main(
        [
            "grpo",
            "--model",
            "sft-checkpoint",
            "--output-dir",
            str(tmp_path),
            "--max-rollout-batches",
            "3",
        ]
    ) == 0

    config = captured["config"]
    assert config.model_name == "sft-checkpoint"
    assert config.groups_per_batch == 50
    assert config.group_size == 4
    assert config.batch_epochs == 2
    assert config.max_rollout_batches == 3

    manifest = json.loads(
        (tmp_path / "grpo_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_name"] == "sft-checkpoint"
    assert manifest["groups_per_batch"] == 50
    assert manifest["group_size"] == 4
    assert manifest["rollout_plan"] == {
        "batches": 3,
        "groups_per_batch": 50,
        "rollouts_per_group": 4,
        "rollouts_per_batch": 200,
        "total_rollouts": 600,
    }
    assert manifest["runtime"]["python_version"]
    assert "torch" in manifest["runtime"]["dependencies"]
    assert '"status": "trained"' in capsys.readouterr().out


def test_grpo_cli_records_lora_model_source_in_manifest(
    tmp_path, monkeypatch
):
    adapter_dir = tmp_path / "sft-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-2B",
                "peft_type": "LORA",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "RETAIL_AGENT_ENV_FILE", str(tmp_path / "missing-test.env")
    )
    monkeypatch.setattr(
        cli_module,
        "train_grpo",
        lambda config: {"optimizer_steps": 1},
    )
    output_dir = tmp_path / "grpo"

    assert main(
        [
            "grpo",
            "--model",
            str(adapter_dir),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    manifest = json.loads(
        (output_dir / "grpo_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_source"] == {
        "kind": "lora",
        "base_model_name_or_path": "Qwen/Qwen3.5-2B",
        "adapter_path": str(adapter_dir),
    }


def test_grpo_cli_records_plain_base_model_as_lora_backed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "RETAIL_AGENT_ENV_FILE", str(tmp_path / "missing-test.env")
    )
    monkeypatch.setattr(
        cli_module,
        "train_grpo",
        lambda config: {"optimizer_steps": 1},
    )
    output_dir = tmp_path / "grpo"

    assert main(
        [
            "grpo",
            "--model",
            "Qwen/Qwen3.5-2B",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    manifest = json.loads(
        (output_dir / "grpo_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_source"] == {
        "kind": "lora",
        "base_model_name_or_path": "Qwen/Qwen3.5-2B",
        "adapter_path": None,
    }


def test_grpo_cli_preserves_failure_diagnostics_without_success_report(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("AGENT_LLM", raising=False)
    monkeypatch.delenv("USER_LLM", raising=False)
    monkeypatch.setenv(
        "RETAIL_AGENT_ENV_FILE", str(tmp_path / "missing-test.env")
    )
    def failing_train_grpo(config):
        raise RuntimeError("rollout backend unavailable")

    monkeypatch.setattr(cli_module, "train_grpo", failing_train_grpo)

    with pytest.raises(RuntimeError, match="rollout backend unavailable"):
        main(
            [
                "grpo",
                "--model",
                "sft-checkpoint",
                "--output-dir",
                str(tmp_path),
            ]
        )

    failure = json.loads(
        (tmp_path / "grpo_failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert "rollout backend unavailable" in failure["error"]
    assert not (tmp_path / "grpo_result.json").exists()


def test_cli_rejects_non_finite_grpo_loss_coefficients():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["grpo", "--model", "sft-checkpoint", "--clip-ratio", "nan"]
        )


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
