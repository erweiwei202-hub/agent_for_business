"""把合规 Retail 轨迹转换为 Action-only SFT 数据。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .policy_verifier import RetailPolicyVerifier
from .trajectory import Trajectory


def _unwrap_teacher_message(content: Any) -> Any:
    """Convert tau2's JSON message envelope to visible assistant text."""
    if not isinstance(content, str):
        return content
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(decoded, dict) and isinstance(decoded.get("message"), str):
        return decoded["message"]
    return content


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

        # Do not persist the simulator's terminal user message as model input.
        # It occurs after the final assistant action and has no next policy
        # response to learn.
        leading_assistant_count = 0
        while (
            leading_assistant_count < len(messages)
            and messages[leading_assistant_count].get("role") == "assistant"
            and not messages[leading_assistant_count].get("tool_calls")
        ):
            leading_assistant_count += 1
        if leading_assistant_count:
            messages = messages[leading_assistant_count:]
            trainable_indices = [
                index - leading_assistant_count
                for index in trainable_indices
                if index >= leading_assistant_count
            ]

        last_assistant_index = max(
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        )
        messages = messages[: last_assistant_index + 1]
        messages = [
            {
                **message,
                "content": (
                    _unwrap_teacher_message(message.get("content"))
                    if message.get("role") == "assistant"
                    and not message.get("tool_calls")
                    else message.get("content")
                ),
            }
            for message in messages
        ]

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
        messages, trainable_message_indices = self._normalize_messages(example)
        base_kwargs: Dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "add_generation_prompt": False,
        }
        kwargs = dict(base_kwargs)
        if max_length is not None:
            kwargs.update({"max_length": max_length, "truncation": True})

        encoded = tokenizer.apply_chat_template(messages, **kwargs)
        input_ids = list(encoded["input_ids"])
        assistant_spans = self._assistant_message_spans(
            tokenizer=tokenizer,
            input_ids=input_ids,
            messages=messages,
            base_kwargs=base_kwargs,
        )

        formatted = dict(encoded)
        selected_indices = set(trainable_message_indices)
        labels = [-100] * len(input_ids)
        for message_index in selected_indices:
            span = assistant_spans.get(message_index)
            if span is None:
                # The selected assistant turn was truncated before its EOS.
                # It cannot provide a complete causal-LM target.
                continue
            message_start, message_end = span
            for token_index in range(message_start, message_end):
                labels[token_index] = input_ids[token_index]

        # -100 是 PyTorch/Transformers 交叉熵常用的 ignore index。
        formatted["labels"] = labels
        return formatted

    @classmethod
    def _normalize_messages(
        cls,
        example: SFTExample,
    ) -> Tuple[List[Dict[str, Any]], Tuple[int, ...]]:
        """Normalize teacher messages to a final assistant training turn."""
        drop_count = 0
        while (
            drop_count < len(example.messages)
            and example.messages[drop_count].get("role") == "assistant"
            and not example.messages[drop_count].get("tool_calls")
        ):
            drop_count += 1

        messages = example.messages[drop_count:]
        trainable_indices = tuple(
            index - drop_count
            for index in example.trainable_message_indices
            if index >= drop_count
        )
        if not trainable_indices:
            raise ValueError(
                "SFT example has no action-only training target after normalization"
            )

        # A terminal UserSimulator stop message is an observation after the
        # final agent action, not a target the policy should learn to continue.
        last_assistant_index = max(
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        )
        messages = messages[: last_assistant_index + 1]
        normalized_messages: List[Dict[str, Any]] = []
        for message in messages:
            normalized = dict(message)
            if normalized.get("role") == "assistant" and not normalized.get(
                "tool_calls"
            ):
                normalized["content"] = cls._unwrap_teacher_message(
                    normalized.get("content")
                )
            normalized_messages.append(normalized)

        for index in trainable_indices:
            if normalized_messages[index].get("role") != "assistant":
                raise ValueError("normalized trainable message must have assistant role")

        return normalized_messages, trainable_indices

    @staticmethod
    def _unwrap_teacher_message(content: Any) -> Any:
        """Convert tau2's JSON message envelope to visible assistant text."""
        return _unwrap_teacher_message(content)

    @classmethod
    def _assistant_message_spans(
        cls,
        *,
        tokenizer: Any,
        input_ids: List[int],
        messages: List[Dict[str, Any]],
        base_kwargs: Dict[str, Any],
    ) -> Dict[int, Tuple[int, int]]:
        """Find assistant spans in the final rendered sequence.

        Prefix rendering is unsafe for Qwen templates because the template's
        reasoning scaffold depends on which user message is last. The final
        sequence is authoritative: each assistant header is paired with the
        next EOS token in that same sequence.
        """
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ]
        if not assistant_indices:
            raise ValueError("SFT example has no assistant messages")

        encode = getattr(tokenizer, "encode", None)
        if encode is None:
            # Small dependency-free test tokenizers may only implement the
            # formatter's original prefix API; keep that seam compatible.
            offsets = cls._message_end_offsets(
                tokenizer=tokenizer,
                messages=messages,
                base_kwargs=base_kwargs,
            )
            spans: Dict[int, Tuple[int, int]] = {}
            start = 0
            for index, end in enumerate(offsets):
                if index in assistant_indices:
                    spans[index] = (start, min(end, len(input_ids)))
                start = end
            return spans

        header_ids = list(encode("<|im_start|>assistant\n", add_special_tokens=False))
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if not header_ids or eos_token_id is None:
            raise ValueError("tokenizer must expose assistant header and EOS token")

        spans = {}
        search_start = 0
        for message_index in assistant_indices:
            assistant_start = cls._find_subsequence(
                input_ids,
                header_ids,
                start=search_start,
            )
            if assistant_start is None:
                # The remainder of the conversation was truncated away.
                break
            eos_position = next(
                (
                    position
                    for position in range(
                        assistant_start + len(header_ids), len(input_ids)
                    )
                    if input_ids[position] == eos_token_id
                ),
                None,
            )
            if eos_position is None:
                # Do not train a partial assistant response without EOS.
                break
            spans[message_index] = (assistant_start, eos_position + 1)
            search_start = eos_position + 1
        return spans

    @staticmethod
    def _find_subsequence(
        values: List[int], needle: List[int], *, start: int
    ) -> Optional[int]:
        """Return the first index of ``needle`` in ``values`` after ``start``."""
        last_start = len(values) - len(needle)
        for index in range(start, last_start + 1):
            if values[index : index + len(needle)] == needle:
                return index
        return None

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
