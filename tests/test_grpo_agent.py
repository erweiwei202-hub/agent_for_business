from types import SimpleNamespace

import torch

from agent_for_business.grpo_agent import (
    LocalQwenAgent,
    parse_qwen_response,
    sample_qwen_response,
)


def test_parses_qwen_xml_tool_call_with_scalar_and_json_parameters():
    parsed = parse_qwen_response(
        """<tool_call>
<function=get_order_details>
<parameter=order_id>
12345
</parameter>
<parameter=fields>
[\"status\", \"items\"]
</parameter>
</function>
</tool_call>"""
    )

    assert parsed.content is None
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_order_details"
    assert parsed.tool_calls[0].arguments == {
        "order_id": "12345",
        "fields": ["status", "items"],
    }


def test_parses_plain_qwen_response_after_reasoning_block():
    parsed = parse_qwen_response(
        "<think>internal reasoning</think>\nThe order is now cancelled."
    )

    assert parsed.content == "The order is now cancelled."
    assert parsed.tool_calls == ()


def test_parses_teacher_message_envelope_as_visible_assistant_text():
    parsed = parse_qwen_response('{"message":"The order is now cancelled."}')

    assert parsed.content == "The order is now cancelled."
    assert parsed.tool_calls == ()


def test_deduplicates_identical_tool_calls_but_keeps_different_arguments():
    parsed = parse_qwen_response(
        """<tool_call><function=get_user_details>
<parameter=user_id>yusuf_rossi_9620</parameter>
</function></tool_call>
<tool_call><function=get_user_details>
<parameter=user_id>yusuf_rossi_9620</parameter>
</function></tool_call>
<tool_call><function=get_user_details>
<parameter=user_id>sara_doe_496</parameter>
</function></tool_call>"""
    )

    assert [(call.name, call.arguments) for call in parsed.tool_calls] == [
        ("get_user_details", {"user_id": "yusuf_rossi_9620"}),
        ("get_user_details", {"user_id": "sara_doe_496"}),
    ]


def test_samples_response_and_marks_only_generated_tokens_as_actions():
    class FakeTokenizer:
        eos_token_id = 7
        pad_token_id = 8

        def apply_chat_template(self, messages, *, tools, add_generation_prompt,
                                 tokenize, return_tensors, return_dict):
            assert add_generation_prompt is True
            assert tokenize is True
            assert return_dict is True
            assert tools == [{"type": "function", "function": {"name": "lookup"}}]
            return {"input_ids": torch.tensor([[10, 11]])}

        def decode(self, token_ids, skip_special_tokens=False):
            assert token_ids == [20, 21]
            assert skip_special_tokens is True
            return "The order is ready."

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, *, input_ids, return_dict_in_generate, output_scores,
                     **kwargs):
            assert input_ids.tolist() == [[10, 11]]
            assert return_dict_in_generate is True
            assert output_scores is True
            assert kwargs["eos_token_id"] == 7
            assert kwargs["pad_token_id"] == 8
            return type(
                "GenerationOutput",
                (),
                {
                "sequences": torch.tensor([[10, 11, 20, 21]]),
                "scores": [
                    torch.zeros((1, 30)),
                    torch.zeros((1, 30)),
                ],
                },
            )()

    parsed, trace = sample_qwen_response(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        messages=[{"role": "user", "content": "Where is my order?"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        max_new_tokens=2,
        temperature=0.0,
    )

    assert parsed.content == "The order is ready."
    assert trace.prompt_ids == (10, 11)
    assert trace.response_ids == (20, 21)
    assert trace.action_mask == (True, True)
    assert len(trace.old_logprobs) == 2


def test_local_agent_converts_parsed_tool_call_to_tau2_message():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is True
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, token_ids, skip_special_tokens=False):
            return (
                "<tool_call><function=lookup>"
                "<parameter=order_id>123</parameter>"
                "</function></tool_call>"
            )

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return SimpleNamespace(
                sequences=torch.tensor([[1, 2, 3]]),
                scores=[torch.zeros((1, 10))],
            )

    def system_factory(**kwargs):
        return SimpleNamespace(**kwargs)

    def tool_call_factory(**kwargs):
        return SimpleNamespace(**kwargs)

    def assistant_factory(**kwargs):
        return SimpleNamespace(**kwargs)

    agent = LocalQwenAgent(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        tools=[SimpleNamespace(openai_schema={"name": "lookup"})],
        domain_policy="be safe",
        system_factory=system_factory,
        tool_call_factory=tool_call_factory,
        assistant_factory=assistant_factory,
    )
    state = agent.get_init_state()

    message, state = agent.generate_next_message(
        SimpleNamespace(role="user", content="Find my order"), state
    )

    assert message.content is None
    assert message.tool_calls[0].name == "lookup"
    assert message.tool_calls[0].arguments == {"order_id": "123"}
    assert len(agent.drain_generation_traces()) == 1


def test_local_agent_system_prompt_distinguishes_product_and_item_ids():
    agent = LocalQwenAgent(
        model=object(),
        tokenizer=object(),
        tools=[],
        domain_policy="be safe",
        system_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        tool_call_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        assistant_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    prompt = agent.system_prompt

    assert "You are a customer service agent that helps the user" in prompt
    assert "ID rules:" in prompt
    assert "product_id" in prompt
    assert "item_id" in prompt
    assert "get_product_details" in prompt
    assert "get_item_details" in prompt
    assert "Do not repeat identical calls within one assistant message" in prompt
    assert "Transient failures may be retried with the same arguments" in prompt
    assert "argument errors require corrected arguments" in prompt
    assert "Never invent a user_id." in prompt
    assert (
        "If the user_id is unknown, use find_user_id_by_email or "
        "find_user_id_by_name_zip."
    ) in prompt
