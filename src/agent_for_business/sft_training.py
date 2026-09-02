"""Qwen Action-only LoRA SFT 的配置、数据入口和训练组装。"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .sft_dataset import QwenActionOnlyTokenFormatter, SFTDatasetStore


@dataclass(frozen=True)
class SFTTrainingConfig:
    """限制训练实验规模并集中保存模型、LoRA 和输出配置。"""

    model_name: str = "Qwen/Qwen3.5-2B"
    dataset_path: Optional[Union[str, Path]] = None
    output_dir: Union[str, Path] = "outputs/sft"
    use_lora: bool = True
    action_only: bool = True
    num_train_epochs: int = 2
    max_length: Optional[int] = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    seed: int = 42
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        # Full-attention projections.
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # MLP projections.
        "gate_proj",
        "up_proj",
        "down_proj",
        # Qwen3.5 linear-attention projections.
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )

    def __post_init__(self) -> None:
        # 项目主线只允许 Action-only LoRA，避免误启动全参数或全消息训练。
        if not self.use_lora:
            raise ValueError(
                "use_lora must be True; full-parameter fine-tuning is disabled"
            )
        if not self.action_only:
            raise ValueError("action_only must be True")
        if self.num_train_epochs < 1 or self.num_train_epochs > 2:
            raise ValueError("num_train_epochs must be between 1 and 2")


def load_sft_dataset(
    path: Union[str, Path],
    *,
    tokenizer: Any,
    max_length: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """读取 SFT JSONL，生成 token labels，并拒绝空 action target 数据集。"""
    formatter = QwenActionOnlyTokenFormatter()
    records: List[Dict[str, Any]] = []
    for example in SFTDatasetStore(path).iter_examples():
        formatted = formatter.format(
            tokenizer=tokenizer,
            example=example,
            max_length=max_length,
        )
        # 一个样本至少要有一个非 -100 label，否则训练只会消费上下文。
        if not any(label != -100 for label in formatted["labels"]):
            raise ValueError("SFT example has no action-only tokens")
        records.append(formatted)
    if not records:
        raise ValueError("SFT dataset has no action-only examples")
    return records


def train_sft(
    config: SFTTrainingConfig,
    *,
    tokenizer: Any = None,
    trainer_factory: Any = None,
) -> Any:
    """训练配置的 Action-only SFT，并保存配置文件与 LoRA checkpoint。"""
    if config.dataset_path is None:
        raise ValueError("dataset_path is required for SFT training")

    if tokenizer is None:
        # transformers/peft/trl 都延迟导入，使数据构建和单元测试不依赖训练环境。
        tokenizer = _load_tokenizer(config.model_name)

    train_dataset = load_sft_dataset(
        config.dataset_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sft_training_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    if trainer_factory is None:
        trainer = _build_default_trainer(
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
        )
    else:
        trainer = trainer_factory(
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
        )

    result = trainer.train()
    trainer.save_model(str(output_dir))
    return result


def _load_tokenizer(model_name: str) -> Any:
    """按需加载 tokenizer，并将缺少训练依赖转换为清晰错误。"""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "SFT training requires transformers in the training environment"
        ) from error
    return AutoTokenizer.from_pretrained(model_name)


def _build_default_trainer(
    *,
    config: SFTTrainingConfig,
    tokenizer: Any,
    train_dataset: List[Dict[str, Any]],
) -> Any:
    """构造默认 Transformers/PEFT/TRL trainer；测试可通过 factory 注入替代品。"""
    try:
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, TrainingArguments
        from trl import SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "SFT training requires transformers, peft, and trl in the training environment"
        ) from error

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=config.seed,
    )
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.lora_target_modules),
    )
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=_to_training_dataset(train_dataset),
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def _to_training_dataset(records: List[Dict[str, Any]]) -> Any:
    """有 datasets 时转成 Dataset，否则保留 list 以支持轻量测试 trainer。"""
    try:
        from datasets import Dataset
    except ImportError:
        return records
    return Dataset.from_list(records)
