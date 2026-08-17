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
                                    "arguments": json.dumps(
                                        event.arguments or {},
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    ),
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
    """根据 tokenizer 的 assistant token mask 构造 Action-only labels。"""

    def format(
        self,
        *,
        tokenizer: Any,
        example: SFTExample,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """应用 Qwen chat template，并将非 assistant token 的 label 设为 -100。"""
        kwargs: Dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_assistant_tokens_mask": True,
            "add_generation_prompt": False,
        }
        if max_length is not None:
            kwargs.update({"max_length": max_length, "truncation": True})

        encoded = tokenizer.apply_chat_template(example.messages, **kwargs)
        input_ids = list(encoded["input_ids"])
        assistant_mask = encoded.get("assistant_masks")
        if assistant_mask is None:
            assistant_mask = encoded.get("assistant_tokens_mask")
        if assistant_mask is None:
            raise ValueError("tokenizer did not return assistant token mask")
        # mask 必须与 input_ids 一一对应，否则 labels 会把 loss 错位到上下文。
        if len(input_ids) != len(assistant_mask):
            raise ValueError("assistant token mask length does not match input_ids")

        formatted = dict(encoded)
        # -100 是 PyTorch/Transformers 交叉熵常用的 ignore index。
        formatted["labels"] = [
            token_id if bool(is_assistant) else -100
            for token_id, is_assistant in zip(input_ids, assistant_mask)
        ]
        return formatted


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
