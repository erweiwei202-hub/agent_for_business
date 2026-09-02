import json
import sys
import types

import pytest

from agent_for_business.grpo_training import (
    GRPOTrainingConfig,
    load_grpo_model,
    resolve_grpo_model_source,
    train_grpo,
)


def test_resolves_sft_lora_adapter_to_its_base_model(tmp_path):
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

    source = resolve_grpo_model_source(adapter_dir)

    assert source.adapter_path == str(adapter_dir)
    assert source.base_model_name_or_path == "Qwen/Qwen3.5-2B"
    assert source.is_lora is True


def test_loads_full_model_without_requiring_peft(monkeypatch, tmp_path):
    loaded = []

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name):
            loaded.append(model_name)
            return "full-policy"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM),
    )
    monkeypatch.setitem(sys.modules, "peft", None)

    model_dir = tmp_path / "full-model"
    model_dir.mkdir()

    assert load_grpo_model(model_dir, use_lora=False) == "full-policy"
    assert loaded == [str(model_dir)]


def test_loads_base_model_in_bfloat16_on_supported_cuda(monkeypatch, tmp_path):
    import torch

    loaded = {}

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            loaded["model_name"] = model_name
            loaded["kwargs"] = kwargs
            return "base-policy"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM),
    )
    monkeypatch.setitem(sys.modules, "peft", None)

    model_dir = tmp_path / "full-model"
    model_dir.mkdir()

    assert load_grpo_model(model_dir, use_lora=False) == "base-policy"
    assert loaded == {
        "model_name": str(model_dir),
        "kwargs": {"torch_dtype": torch.bfloat16},
    }


def test_enables_gradient_checkpointing_on_trainable_base_lora(
    monkeypatch, tmp_path
):
    calls = []

    class FakeBaseModel:
        def gradient_checkpointing_enable(self):
            calls.append("gradient_checkpointing_enable")

        def enable_input_require_grads(self):
            calls.append("enable_input_require_grads")

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return FakeBaseModel()

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_get_peft_model(base_model, peft_config):
        calls.append(("wrap", base_model, peft_config))
        return base_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(
            LoraConfig=FakeLoraConfig,
            get_peft_model=fake_get_peft_model,
        ),
    )

    model_dir = tmp_path / "full-model"
    model_dir.mkdir()

    load_grpo_model(model_dir)

    assert [item for item in calls if isinstance(item, str)] == [
        "gradient_checkpointing_enable",
        "enable_input_require_grads",
    ]


def test_loads_plain_base_model_with_trainable_lora(monkeypatch, tmp_path):
    calls = []

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name):
            calls.append(("base", model_name))
            return "base-policy"

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            calls.append(("config", kwargs))
            self.kwargs = kwargs

    def fake_get_peft_model(base_model, peft_config):
        calls.append(("wrap", base_model, peft_config.kwargs))
        return "trainable-lora-policy"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(
            LoraConfig=FakeLoraConfig,
            get_peft_model=fake_get_peft_model,
        ),
    )

    model_dir = tmp_path / "full-model"
    model_dir.mkdir()

    assert load_grpo_model(model_dir) == "trainable-lora-policy"
    assert calls == [
        ("base", str(model_dir)),
        (
            "config",
            {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": [
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
                ],
            },
        ),
        (
            "wrap",
            "base-policy",
            {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": [
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
                ],
            },
        ),
    ]


def test_exports_grpo_model_loading_api_from_package():
    import agent_for_business

    assert agent_for_business.GRPOModelSource is not None
    assert agent_for_business.load_grpo_model is not None
    assert agent_for_business.resolve_grpo_model_source is not None


def test_loads_lora_policy_as_trainable_adapter_without_merging(monkeypatch, tmp_path):
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
    calls = []

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name):
            calls.append(("base", model_name))
            return "base-policy"

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, base_model, adapter_path, **kwargs):
            calls.append(("adapter", base_model, adapter_path, kwargs))
            return "trainable-lora-policy"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(PeftModel=FakePeftModel),
    )

    assert load_grpo_model(adapter_dir) == "trainable-lora-policy"
    assert calls == [
        ("base", "Qwen/Qwen3.5-2B"),
        (
            "adapter",
            "base-policy",
            str(adapter_dir),
            {"is_trainable": True},
        ),
    ]


def test_train_grpo_passes_loaded_policy_to_online_trainer_factory():
    config = GRPOTrainingConfig(model_name="sft-adapter")
    policy = object()
    seen = {}

    def model_loader(source):
        seen["source"] = source
        return policy

    class FakeTrainer:
        def train(self):
            return {"status": "trained"}

    def trainer_factory(received_config, received_policy):
        seen["config"] = received_config
        seen["policy"] = received_policy
        return FakeTrainer()

    result = train_grpo(
        config,
        trainer_factory=trainer_factory,
        model_loader=model_loader,
    )

    assert result == {"status": "trained"}
    assert seen["config"] is config
    assert seen["policy"] is policy
    assert seen["source"].base_model_name_or_path == "sft-adapter"


def test_train_grpo_uses_online_trainer_by_default(monkeypatch):
    import agent_for_business.grpo_training as module

    seen = {}
    policy = object()
    tokenizer = object()

    class FakeTrainer:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def train(self):
            return {"optimizer_steps": 1}

    monkeypatch.setattr(module, "OnlineGRPOTrainer", FakeTrainer)
    monkeypatch.setattr(module, "_load_grpo_tokenizer", lambda config: tokenizer)

    result = module.train_grpo(
        GRPOTrainingConfig(model_name="sft-adapter"),
        model_loader=lambda source: policy,
    )

    assert result == {"optimizer_steps": 1}
    assert seen["policy_model"] is policy
    assert seen["tokenizer"] is tokenizer


def test_grpo_training_config_keeps_confirmed_defaults():
    config = GRPOTrainingConfig(model_name="sft-checkpoint")

    assert config.groups_per_batch == 50
    assert config.group_size == 4
    assert config.batch_epochs == 2
    assert config.max_workers == 4
    assert config.inference_microbatch == 2
    assert config.clip_ratio == 0.2
    assert config.kl_beta == 0.001
    assert config.max_rollout_batches == 2
    assert config.use_lora is True


def test_grpo_training_config_rejects_invalid_batch_and_loss_values():
    with pytest.raises(ValueError, match="groups_per_batch"):
        GRPOTrainingConfig(model_name="sft-checkpoint", groups_per_batch=0)

    with pytest.raises(ValueError, match="kl_beta"):
        GRPOTrainingConfig(model_name="sft-checkpoint", kl_beta=float("nan"))
