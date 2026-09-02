import importlib
import json

import pytest

from agent_for_business.sft_dataset import SFTDatasetStore, SFTExample


def test_default_sft_config_is_action_only_lora_for_qwen():
    module = importlib.import_module("agent_for_business.sft_training")

    config = module.SFTTrainingConfig()

    assert config.model_name == "Qwen/Qwen3.5-2B"
    assert config.use_lora is True
    assert config.action_only is True
    assert config.num_train_epochs == 2


def test_default_sft_config_targets_qwen35_text_projection_modules():
    module = importlib.import_module("agent_for_business.sft_training")

    config = module.SFTTrainingConfig()

    assert config.lora_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )


def test_config_rejects_full_parameter_finetuning():
    module = importlib.import_module("agent_for_business.sft_training")

    with pytest.raises(ValueError, match="use_lora"):
        module.SFTTrainingConfig(use_lora=False)


def test_config_rejects_non_action_only_training():
    module = importlib.import_module("agent_for_business.sft_training")

    with pytest.raises(ValueError, match="action_only"):
        module.SFTTrainingConfig(action_only=False)


def test_config_rejects_more_than_two_epochs():
    module = importlib.import_module("agent_for_business.sft_training")

    with pytest.raises(ValueError, match="num_train_epochs"):
        module.SFTTrainingConfig(num_train_epochs=3)


def test_load_sft_dataset_reads_store_and_builds_action_only_labels(tmp_path):
    module = importlib.import_module("agent_for_business.sft_training")
    path = tmp_path / "accepted-sft.jsonl"
    SFTDatasetStore(path).append(
        SFTExample(
            task_id="retail-001",
            messages=[
                {"role": "user", "content": "Check order W1."},
                {"role": "assistant", "content": "The order is pending."},
            ],
            trainable_message_indices=(1,),
        )
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["role"] == "user"
            return {
                "input_ids": list(range(101, 101 + len(messages))),
                "assistant_masks": [
                    1 if message["role"] == "assistant" else 0
                    for message in messages
                ],
            }

    records = module.load_sft_dataset(path, tokenizer=FakeTokenizer())

    assert records == [
        {
            "input_ids": [101, 102],
            "assistant_masks": [0, 1],
            "labels": [-100, 102],
        }
    ]


def test_config_carries_dataset_and_output_paths(tmp_path):
    module = importlib.import_module("agent_for_business.sft_training")
    dataset_path = tmp_path / "accepted-sft.jsonl"
    output_dir = tmp_path / "sft-run"

    config = module.SFTTrainingConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    assert config.dataset_path == dataset_path
    assert config.output_dir == output_dir


def test_load_sft_dataset_rejects_examples_without_action_tokens(tmp_path):
    module = importlib.import_module("agent_for_business.sft_training")
    path = tmp_path / "accepted-sft.jsonl"
    SFTDatasetStore(path).append(
        SFTExample(
            task_id="retail-no-action",
            messages=[{"role": "user", "content": "Check order W2."}],
            trainable_message_indices=(),
        )
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {
                "input_ids": [101, 102],
                "assistant_masks": [0, 0],
            }

    with pytest.raises(ValueError, match="action-only"):
        module.load_sft_dataset(path, tokenizer=FakeTokenizer())


def test_load_sft_dataset_rejects_empty_store(tmp_path):
    module = importlib.import_module("agent_for_business.sft_training")

    with pytest.raises(ValueError, match="action-only"):
        module.load_sft_dataset(tmp_path / "empty.jsonl", tokenizer=object())


def test_train_sft_uses_injected_trainer_and_persists_config(tmp_path):
    module = importlib.import_module("agent_for_business.sft_training")
    dataset_path = tmp_path / "accepted-sft.jsonl"
    output_dir = tmp_path / "sft-run"
    SFTDatasetStore(dataset_path).append(
        SFTExample(
            task_id="retail-train",
            messages=[
                {"role": "user", "content": "Check order W3."},
                {"role": "assistant", "content": "The order is pending."},
            ],
            trainable_message_indices=(1,),
        )
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {
                "input_ids": list(range(101, 101 + len(messages))),
                "assistant_masks": [
                    1 if message["role"] == "assistant" else 0
                    for message in messages
                ],
            }

    events = []

    class FakeTrainer:
        def train(self):
            events.append("train")
            return {"status": "trained"}

        def save_model(self, output_dir):
            events.append(("save_model", output_dir))

    def trainer_factory(**kwargs):
        events.append(("factory", kwargs))
        return FakeTrainer()

    config = module.SFTTrainingConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    result = module.train_sft(
        config,
        tokenizer=FakeTokenizer(),
        trainer_factory=trainer_factory,
    )

    assert result == {"status": "trained"}
    assert events[0][0] == "factory"
    assert events[0][1]["config"] == config
    assert events[0][1]["train_dataset"][0]["labels"] == [-100, 102]
    assert events[1:] == ["train", ("save_model", str(output_dir))]
    saved_config = json.loads(
        (output_dir / "sft_training_config.json").read_text(encoding="utf-8")
    )
    assert saved_config["model_name"] == "Qwen/Qwen3.5-2B"
    assert saved_config["use_lora"] is True
    assert saved_config["action_only"] is True
    assert saved_config["num_train_epochs"] == 2
