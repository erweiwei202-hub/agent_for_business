"""Local Qwen policy adapter for tau2 half-duplex rollouts."""

import json
import re
from contextlib import nullcontext
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from tau2.agent.base_agent import HalfDuplexAgent
except ImportError:  # pragma: no cover - exercised only without optional tau2
    class HalfDuplexAgent:  # type: ignore[no-redef]
        """Import-safe fallback for core tests without tau2 installed."""

        def __init__(self, tools: list[Any], domain_policy: str) -> None:
            self.tools = tools
            self.domain_policy = domain_policy


@dataclass(frozen=True)
class ParsedToolCall:
    """Dependency-free representation of one model-emitted tool call."""

    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ParsedAssistant:
    """Parsed assistant response before conversion to tau2 message types."""

    content: Optional[str]
    tool_calls: Tuple[ParsedToolCall, ...] = ()


@dataclass(frozen=True)
class GenerationTrace:
    """Token-level data required to replay one assistant action sequence."""

    prompt_ids: Tuple[int, ...]
    response_ids: Tuple[int, ...]
    old_logprobs: Tuple[float, ...]
    action_mask: Tuple[bool, ...]


@dataclass
class LocalQwenAgentState:
    """Conversation state accepted by tau2's half-duplex orchestrator."""

    system_messages: list[Any]
    messages: list[Any]


def parse_qwen_response(text: str) -> ParsedAssistant:
    """Parse Qwen3.5 text or its XML function-call format.

    Qwen3.5's chat template emits function calls as ``<tool_call>`` blocks
    containing ``<function=name>`` and named ``<parameter=...>`` elements.
    JSON objects/arrays and JSON literals are decoded; ordinary scalar values
    remain strings because tool schemas commonly distinguish IDs from numbers.
    """

    normalized = str(text or "").strip()
    if not normalized:
        raise ValueError("model response is empty")

    blocks = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>", normalized, flags=re.DOTALL
    )
    if not blocks:
        if "<tool_call>" in normalized or "</tool_call>" in normalized:
            raise ValueError("malformed Qwen tool_call XML")
        return ParsedAssistant(
            content=_unwrap_message_envelope(_strip_reasoning(normalized))
        )

    calls = []
    seen_calls = set()
    for block in blocks:
        function_match = re.search(
            r"<function=([^>]+)>(.*?)</function>", block, flags=re.DOTALL
        )
        if function_match is None:
            raise ValueError("Qwen tool_call is missing a function block")
        name = function_match.group(1).strip()
        if not name:
            raise ValueError("Qwen tool_call function name is empty")
        function_body = function_match.group(2)
        arguments: Dict[str, Any] = {}
        for parameter_match in re.finditer(
            r"<parameter=([^>]+)>(.*?)</parameter>",
            function_body,
            flags=re.DOTALL,
        ):
            parameter_name = parameter_match.group(1).strip()
            if not parameter_name:
                raise ValueError("Qwen tool_call parameter name is empty")
            arguments[parameter_name] = _parse_parameter_value(
                parameter_match.group(2)
            )
        call_key = (
            name,
            json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if call_key in seen_calls:
            continue
        seen_calls.add(call_key)
        calls.append(ParsedToolCall(name=name, arguments=arguments))

    return ParsedAssistant(content=None, tool_calls=tuple(calls))


def sample_qwen_response(
    *,
    model: Any,
    tokenizer: Any,
    messages: Iterable[Dict[str, Any]],
    tools: Iterable[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    generation_lock: Optional[Any] = None,
) -> Tuple[ParsedAssistant, GenerationTrace]:
    """Generate one Qwen response and retain old action-token log-probabilities."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "local GRPO rollout requires torch in the training environment"
        ) from error

    encoded = tokenizer.apply_chat_template(
        list(messages),
        tools=list(tools),
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = input_ids.device
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    generation_config = getattr(model, "generation_config", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(generation_config, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(generation_config, "pad_token_id", None)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if temperature > 0.0:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    lock_context = generation_lock or nullcontext()
    with lock_context, torch.inference_mode():
        output = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = int(input_ids.shape[-1])
    response_tensor = output.sequences[0, prompt_length:]
    response_ids = tuple(int(token_id) for token_id in response_tensor.tolist())
    old_logprobs = []
    for score, token_id in zip(output.scores, response_ids):
        old_logprobs.append(
            float(torch.log_softmax(score[0], dim=-1)[token_id].item())
        )
    if len(old_logprobs) != len(response_ids):
        raise ValueError("generation scores are not aligned with response tokens")
    response_text = tokenizer.decode(
        list(response_ids),
        skip_special_tokens=True,
    )
    return parse_qwen_response(response_text), GenerationTrace(
        prompt_ids=tuple(int(token_id) for token_id in input_ids[0].tolist()),
        response_ids=response_ids,
        old_logprobs=tuple(old_logprobs),
        action_mask=tuple(True for _ in response_ids),
    )


class LocalQwenAgent(HalfDuplexAgent):
    """A trainable local Qwen policy compatible with tau2 half-duplex runs."""

    STOP_TOKEN = "###STOP###"

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        tools: list[Any],
        domain_policy: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generation_lock: Optional[Any] = None,
        system_factory: Optional[Any] = None,
        tool_call_factory: Optional[Any] = None,
        assistant_factory: Optional[Any] = None,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generation_lock = generation_lock or RLock()
        self._traces: list[GenerationTrace] = []

        if any(
            factory is None
            for factory in (system_factory, tool_call_factory, assistant_factory)
        ):
            try:
                from tau2.data_model.message import (
                    AssistantMessage,
                    SystemMessage,
                    ToolCall,
                )
            except ImportError as error:
                raise RuntimeError(
                    "LocalQwenAgent requires tau2 message types at runtime"
                ) from error
            system_factory = system_factory or SystemMessage
            tool_call_factory = tool_call_factory or ToolCall
            assistant_factory = assistant_factory or AssistantMessage
        self._system_factory = system_factory
        self._tool_call_factory = tool_call_factory
        self._assistant_factory = assistant_factory

    @property
    def system_prompt(self) -> str:
        """Keep the tau2 policy contract in the local model's context."""

        return (
            "<instructions>\n"
            "You are a customer service agent that helps the user according "
            "to the <policy> provided below.\n"
            "In each turn, either send a message or make necessary tool calls, "
            "never both. Follow the policy and use valid tool arguments.\n"
            "ID rules: product_id identifies a product and item_id identifies "
            "a concrete variant. Use get_product_details for product_id and "
            "get_item_details for item_id. Do not repeat identical calls within "
            "one assistant message. Transient failures may be retried with the "
            "same arguments; argument errors require corrected arguments.\n"
            "Never invent a user_id. If the user_id is unknown, use "
            "find_user_id_by_email or find_user_id_by_name_zip. Ordinary replies "
            "must be plain user-facing text, not a JSON message envelope.\n"
            "</instructions>\n<policy>\n"
            + self.domain_policy
            + "\n</policy>"
        )

    def get_init_state(
        self, message_history: Optional[list[Any]] = None
    ) -> LocalQwenAgentState:
        """Create a state with a system prompt and optional tau2 history."""

        return LocalQwenAgentState(
            system_messages=[
                self._system_factory(role="system", content=self.system_prompt)
            ],
            messages=list(message_history or []),
        )

    def generate_next_message(
        self, message: Any, state: LocalQwenAgentState
    ) -> tuple[Any, LocalQwenAgentState]:
        """Generate one assistant message and append it to the conversation."""

        if message is not None:
            if hasattr(message, "tool_messages"):
                state.messages.extend(list(message.tool_messages))
            else:
                state.messages.append(message)

        parsed, trace = sample_qwen_response(
            model=self.model,
            tokenizer=self.tokenizer,
            messages=self._render_messages(state),
            tools=self._tool_schemas(),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            generation_lock=self.generation_lock,
        )
        self._traces.append(trace)
        assistant = self._to_tau2_assistant(parsed)
        state.messages.append(assistant)
        return assistant, state

    def drain_generation_traces(self) -> Tuple[GenerationTrace, ...]:
        """Return and clear traces collected since the previous drain."""

        traces = tuple(self._traces)
        self._traces.clear()
        return traces

    def set_seed(self, seed: int) -> None:
        """Seed local sampling when torch is available."""

        try:
            import torch
        except ImportError:
            return
        torch.manual_seed(seed)

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        """Recognize tau2-compatible explicit stop text."""

        return cls.STOP_TOKEN in str(getattr(message, "content", None) or "")

    def _tool_schemas(self) -> list[Dict[str, Any]]:
        return [tool.openai_schema for tool in self.tools]

    def _render_messages(self, state: LocalQwenAgentState) -> list[Dict[str, Any]]:
        messages = [self._message_to_dict(message) for message in state.system_messages]
        messages.extend(self._message_to_dict(message) for message in state.messages)
        return messages

    @staticmethod
    def _message_to_dict(message: Any) -> Dict[str, Any]:
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        if hasattr(message, "dict"):
            return message.dict(exclude_none=True)
        return {
            key: value
            for key, value in vars(message).items()
            if value is not None
        }

    def _to_tau2_assistant(self, parsed: ParsedAssistant) -> Any:
        if not parsed.tool_calls:
            return self._assistant_factory(role="assistant", content=parsed.content)

        tool_calls = [
            self._tool_call_factory(
                id="local-call-{}".format(index),
                name=call.name,
                arguments=call.arguments,
                requestor="assistant",
            )
            for index, call in enumerate(parsed.tool_calls)
        ]
        return self._assistant_factory(
            role="assistant",
            content=None,
            tool_calls=tool_calls,
        )


def _parse_parameter_value(value: str) -> Any:
    """Decode structured parameter values while preserving scalar IDs."""

    stripped = value.strip()
    if stripped[:1] in {"{", "[", '"'} or stripped in {
        "true",
        "false",
        "null",
    }:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError("invalid JSON tool parameter") from error
    return stripped


def _strip_reasoning(text: str) -> str:
    """Keep only the visible assistant response after an optional think block."""

    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def _unwrap_message_envelope(text: str) -> str:
    """Convert the teacher/API JSON message envelope to visible text."""

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(decoded, dict) and isinstance(decoded.get("message"), str):
        return decoded["message"]
    return text
