"""GRPO 训练配置、模型装载和在线 trainer 的稳定入口。"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .sft_training import SFTTrainingConfig

# Populated lazily so importing the configuration remains safe without torch.
OnlineGRPOTrainer: Any = None


@dataclass(frozen=True)
class GRPOTrainingConfig:
    """描述一个可复现的 Retail 在线 GRPO 运行。"""

    model_name: str
    output_dir: Union[str, Path] = "outputs/grpo"
    split_tasks: Union[str, Path] = (
        "vendor/tau2-bench/data/tau2/domains/retail/split_tasks.json"
    )
    groups_per_batch: int = 50
    group_size: int = 4
    batch_epochs: int = 2
    max_workers: int = 4
    # Number of rollout traces processed per gradient-accumulation microbatch.
    inference_microbatch: int = 2
    parallel_generation: bool = False
    clip_ratio: float = 0.2
    kl_beta: float = 0.001
    seed: int = 42
    # Total rollout batches for the complete run; resume continues from the
    # saved batch cursor until this target is reached.
    max_rollout_batches: int = 2
    user_llm: str = "anthropic/deepseek-v4-flash"
    user_llm_args: Optional[Dict[str, Any]] = None
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    device: str = "auto"
    # Periodic checkpoint interval in optimizer steps; the trainer writes at
    # completed batch boundaries and also saves the final step.
    checkpoint_every: int = 1
    resume_from: Optional[Union[str, Path]] = None
    # Plain base-model paths are wrapped with a fresh LoRA adapter by default.
    use_lora: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        for name in (
            "groups_per_batch",
            "group_size",
            "batch_epochs",
            "max_workers",
            "inference_microbatch",
            "max_rollout_batches",
        ):
            if getattr(self, name) <= 0:
                raise ValueError("{} must be positive".format(name))
        for name in ("clip_ratio", "kl_beta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("{} must be finite and non-negative".format(name))
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.temperature:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")

    @property
    def rollout_plan(self) -> Dict[str, int]:
        """Return the exact number of groups and simulations this run requests."""

        rollouts_per_batch = self.groups_per_batch * self.group_size
        return {
            "batches": self.max_rollout_batches,
            "groups_per_batch": self.groups_per_batch,
            "rollouts_per_group": self.group_size,
            "rollouts_per_batch": rollouts_per_batch,
            "total_rollouts": self.max_rollout_batches * rollouts_per_batch,
        }


@dataclass(frozen=True)
class GRPOModelSource:
    """Resolved policy source used by the GRPO trainer.

    A LoRA SFT checkpoint contains only adapter weights.  In that case the
    source keeps both the adapter directory and the base model identifier so a
    trainer can load the policy without first materialising a merged model.
    """

    requested_model_name_or_path: str
    base_model_name_or_path: str
    adapter_path: Optional[str] = None

    @property
    def is_lora(self) -> bool:
        """Whether the source is a LoRA adapter rather than a full model."""
        return self.adapter_path is not None


def resolve_grpo_model_source(
    model_name_or_path: Union[str, Path],
) -> GRPOModelSource:
    """Resolve a full model path or a local PEFT LoRA adapter directory.

    Hugging Face repository IDs are left untouched.  A local directory is
    treated as a LoRA adapter only when it contains ``adapter_config.json``;
    this keeps ordinary base-model paths compatible with the same interface.
    """

    requested = str(model_name_or_path)
    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    if not adapter_config_path.is_file():
        return GRPOModelSource(
            requested_model_name_or_path=requested,
            base_model_name_or_path=requested,
        )

    try:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "invalid LoRA adapter_config.json: {}".format(adapter_config_path)
        ) from error

    if adapter_config.get("peft_type") != "LORA":
        raise ValueError(
            "GRPO supports only LoRA adapters; got peft_type={!r}".format(
                adapter_config.get("peft_type")
            )
        )

    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("LoRA adapter_config.json must define base_model_name_or_path")

    return GRPOModelSource(
        requested_model_name_or_path=requested,
        base_model_name_or_path=base_model,
        adapter_path=requested,
    )


def load_grpo_model(
    model_name_or_path: Union[str, Path],
    *,
    model_loader: Optional[Any] = None,
    use_lora: bool = True,
) -> Any:
    """Load a GRPO policy from a base model or a trainable LoRA adapter.

    ``model_loader`` is a test and embedding seam.  When omitted, the
    training dependencies are imported lazily. Existing adapter directories
    are loaded with ``is_trainable=True``; plain base-model paths receive a
    fresh adapter using the same LoRA defaults as SFT. ``use_lora=False`` is
    an explicit escape hatch for callers that need an unwrapped full model.
    """

    source = resolve_grpo_model_source(model_name_or_path)
    if model_loader is not None:
        return model_loader(source)

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "GRPO model loading requires transformers in the training environment"
        ) from error

    base_model = AutoModelForCausalLM.from_pretrained(
        source.base_model_name_or_path,
        **_base_model_load_kwargs(),
    )
    if not source.is_lora and not use_lora:
        return base_model

    try:
        import peft
    except ImportError as error:
        raise RuntimeError(
            "GRPO LoRA loading requires peft in the training environment"
        ) from error

    if source.is_lora:
        model = peft.PeftModel.from_pretrained(
            base_model,
            source.adapter_path,
            is_trainable=True,
        )
        return _prepare_trainable_model(model)

    sft_defaults = SFTTrainingConfig()
    peft_config = peft.LoraConfig(
        r=sft_defaults.lora_r,
        lora_alpha=sft_defaults.lora_alpha,
        lora_dropout=sft_defaults.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(sft_defaults.lora_target_modules),
    )
    return _prepare_trainable_model(peft.get_peft_model(base_model, peft_config))


def _prepare_trainable_model(model: Any) -> Any:
    """Enable activation checkpointing on the trainable GRPO policy."""

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def _base_model_load_kwargs() -> Dict[str, Any]:
    """Prefer BF16 for CUDA model loading while keeping CPU imports usable."""

    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "GRPO requires a CUDA device with BF16 support for base-model loading"
        )
    return {"torch_dtype": torch.bfloat16}


def _load_grpo_tokenizer(config: GRPOTrainingConfig) -> Any:
    """Load the tokenizer from the adapter directory or its base model."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "GRPO tokenizer loading requires transformers in the training environment"
        ) from error
    source = resolve_grpo_model_source(config.resume_from or config.model_name)
    tokenizer_source = (
        source.requested_model_name_or_path
        if source.is_lora
        else source.base_model_name_or_path
    )
    return AutoTokenizer.from_pretrained(tokenizer_source)


def train_grpo(
    config: GRPOTrainingConfig,
    *,
    trainer_factory: Optional[Any] = None,
    model_loader: Optional[Any] = None,
) -> Any:
    """Run the online trainer through a stable seam.

    The default path constructs the real online tau2/torch trainer. A factory
    is still supported for integration tests and embedding without moving
    rollout or loss logic into argparse.
    The factory receives ``(config, policy_model)``; the policy may be a
    trainable PEFT model when ``config.model_name`` points to the SFT adapter.
    """
    policy_source = config.resume_from or config.model_name
    policy_model = load_grpo_model(
        policy_source,
        model_loader=model_loader,
        use_lora=config.use_lora,
    )
    if trainer_factory is not None:
        trainer = trainer_factory(config, policy_model)
        return trainer.train()

    global OnlineGRPOTrainer
    if OnlineGRPOTrainer is None:
        from .grpo_online import OnlineGRPOTrainer as trainer_class
    else:
        trainer_class = OnlineGRPOTrainer
    reference_model = None
    if config.resume_from:
        reference_model = load_grpo_model(
            config.model_name,
            model_loader=model_loader,
            use_lora=config.use_lora,
        )
    trainer = trainer_class(
        config=config,
        policy_model=policy_model,
        tokenizer=_load_grpo_tokenizer(config),
        reference_model=reference_model,
    )
    return trainer.train()
