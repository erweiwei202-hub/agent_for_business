"""把合规 Retail 轨迹转换为 Action-only SFT 数据。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .policy_verifier import RetailPolicyVerifier
from .trajectory import Trajectory


@dataclass(frozen=True)
class SFTExample:
    """一条聊天格式 SFT 样本及其中可计算 loss 的 message 下标。"""

    task_id: str
    messages: List[Dict[str, Any]]
    trainable_message_indices: Tuple[int, ...]


@dataclass(frozen=True)
class SFTBuildResult:
    """保存成功构建的样本和被跳过的 task id，便于审计数据损耗。"""

    examples: Tuple[SFTExample, ...]
    skipped_task_ids: Tuple[str, ...]


class ActionOnlySFTRenderer:
    """保留完整对话上下文，但显式标记 assistant action 为训练目标。"""

    def render(self, trajectory: Trajectory) -> SFTExample:
        """将统一轨迹渲染为消息列表，并记录可训练 message 的位置。"""
        messages: List[Dict[str, Any]] = []
        trainable_indices: List[int] = []

        for event in trajectory.events:
            if event.kind == "user_message":
                messages.append({"role": "user", "content": str(event.content or "")})
            elif event.kind == "tool_call":
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": event.tool_call_id or "",
                                "type": "function",
                                "function": {
                                    "name": event.tool_name or "",
                                    # Qwen3.5's chat template iterates over
                                    # arguments with ``items``; keep this as a
                                    # mapping instead of a JSON string.
                                    "arguments": event.arguments or {},
                                },
                            }
                        ],
                    }
                )
                trainable_indices.append(len(messages) - 1)
            elif event.kind == "tool_result":
                # tool observation 是决策上下文，保留它但不加入 trainable_indices。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": event.tool_call_id or "",
                        "content": self._content_to_text(event.content),
                    }
                )
            elif event.kind == "assistant_message":
                messages.append(
                    {"role": "assistant", "content": str(event.content or "")}
                )
                trainable_indices.append(len(messages) - 1)
            else:
                raise ValueError("Unsupported trajectory event kind: " + event.kind)

        if not trainable_indices:
            # 没有 assistant action 的样本无法产生有效 loss，必须显式拒绝。
            raise ValueError("trajectory has no assistant training target")

        return SFTExample(
            task_id=trajectory.task_id,
            messages=messages,
            trainable_message_indices=tuple(trainable_indices),
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """把结构化工具结果稳定地编码为聊天消息中的文本。"""
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        return str(content or "")


class ActionOnlySFTDatasetBuilder:
    """只接收 reward 有效、任务成功且无策略违规的轨迹作为 SFT 正例。"""

    def __init__(
        self,
        *,
        verifier: RetailPolicyVerifier,
        renderer: Optional[ActionOnlySFTRenderer] = None,
    ) -> None:
        self._verifier = verifier
        self._renderer = renderer or ActionOnlySFTRenderer()

    def build(self, trajectories: Iterable[Trajectory]) -> SFTBuildResult:
        """验证并渲染轨迹，返回 accepted 样本和跳过原因对应的 task id。"""
        examples: List[SFTExample] = []
        skipped_task_ids: List[str] = []

        for trajectory in trajectories:
            verification = self._verifier.verify(trajectory)
            accepted = (
                # accepted 是 SFT 数据边界，不等同于“轨迹文件中存在”。
                verification.reward_valid
                and verification.task_success
                and not verification.policy_violation
            )
            if not accepted:
                skipped_task_ids.append(trajectory.task_id)
                continue
            try:
                examples.append(self._renderer.render(trajectory))
            except ValueError:
                skipped_task_ids.append(trajectory.task_id)

        return SFTBuildResult(
            examples=tuple(examples),
            skipped_task_ids=tuple(skipped_task_ids),
        )


class QwenActionOnlyTokenFormatter:
    """只为 SFTExample 指定的消息构造 Action-only labels。"""

    def format(
        self,
        *,
        tokenizer: Any,
        example: SFTExample,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """应用 chat template，并只保留指定消息的 token 作为训练目标。"""
        self._validate_trainable_indices(example)
        base_kwargs: Dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "add_generation_prompt": False,
        }
        kwargs = dict(base_kwargs)
        if max_length is not None:
            kwargs.update({"max_length": max_length, "truncation": True})

        encoded = tokenizer.apply_chat_template(example.messages, **kwargs)
        input_ids = list(encoded["input_ids"])
        message_end_offsets = self._message_end_offsets(
            tokenizer=tokenizer,
            messages=example.messages,
            base_kwargs=base_kwargs,
        )

        formatted = dict(encoded)
        selected_indices = set(example.trainable_message_indices)
        labels = [-100] * len(input_ids)
        message_start = 0
        for message_index, message_end in enumerate(message_end_offsets):
            if message_index in selected_indices:
                # 截断只保留仍位于 input_ids 中的目标 token。
                for token_index in range(
                    message_start, min(message_end, len(input_ids))
                ):
                    labels[token_index] = input_ids[token_index]
            message_start = message_end

        # -100 是 PyTorch/Transformers 交叉熵常用的 ignore index。
        formatted["labels"] = labels
        return formatted

    @staticmethod
    def _validate_trainable_indices(example: SFTExample) -> None:
        """拒绝越界或指向非 assistant 消息的训练目标。"""
        message_count = len(example.messages)
        for index in example.trainable_message_indices:
            if index < 0 or index >= message_count:
                raise ValueError("trainable message index is out of range")
            if example.messages[index].get("role") != "assistant":
                raise ValueError("trainable message must have assistant role")

    @staticmethod
    def _message_end_offsets(
        *,
        tokenizer: Any,
        messages: List[Dict[str, Any]],
        base_kwargs: Dict[str, Any],
    ) -> List[int]:
        """Find cumulative token lengths so each message has a token interval."""
        offsets: List[int] = []
        for end in range(1, len(messages) + 1):
            prefix = tokenizer.apply_chat_template(
                messages[:end],
                **base_kwargs,
            )
            offsets.append(len(prefix["input_ids"]))
        return offsets


class SFTDatasetStore:
    """以 JSONL 保存渲染后的 SFTExample，便于中断后继续读取。"""

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)

    def append(self, example: SFTExample) -> None:
        """追加一条 SFT 样本，并将 tuple 下标转换为 JSON 数组。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": example.task_id,
            "messages": example.messages,
            "trainable_message_indices": list(example.trainable_message_indices),
        }
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def iter_examples(self) -> Iterator[SFTExample]:
        """按文件顺序恢复 SFT 样本；不存在文件时产生空迭代。"""
        if not self._path.exists():
            return

        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                yield SFTExample(
                    task_id=payload["task_id"],
                    messages=payload["messages"],
                    trainable_message_indices=tuple(
                        payload["trainable_message_indices"]
                    ),
                )
