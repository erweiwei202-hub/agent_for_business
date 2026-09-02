from agent_for_business.policy_verifier import RetailPolicyVerifier
from agent_for_business.sft_dataset import (
    ActionOnlySFTDatasetBuilder,
    ActionOnlySFTRenderer,
    QwenActionOnlyTokenFormatter,
    SFTDatasetStore,
    SFTExample,
)
from agent_for_business.trajectory import TrajectoryRecorder


def test_renders_observations_but_trains_only_assistant_messages():
    recorder = TrajectoryRecorder(task_id="retail-019", seed=71)
    recorder.append_user("What is the status of order W799?")
    recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-9",
    )
    recorder.append_tool_result(
        call_id="auth-9",
        content={"user_id": "user-1"},
    )
    recorder.append_assistant("I found the order. It is pending.")
    trajectory = recorder.finish(
        terminal_state={"order_status": "pending"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    example = ActionOnlySFTRenderer().render(trajectory)

    assert example.task_id == "retail-019"
    assert [message["role"] for message in example.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert example.messages[2]["content"] == '{"user_id": "user-1"}'
    assert example.trainable_message_indices == (1, 3)
    assert example.messages[1]["tool_calls"][0]["function"]["name"] == (
        "find_user_id_by_email"
    )
    assert example.messages[1]["tool_calls"][0]["function"]["arguments"] == {
        "email": "user@example.com"
    }


def test_rejects_trajectory_without_assistant_training_target():
    recorder = TrajectoryRecorder(task_id="retail-020", seed=73)
    recorder.append_user("What is the status of order W800?")
    trajectory = recorder.finish(
        terminal_state={},
        evaluation={"task_success": False, "reward": 0.0},
    )

    try:
        ActionOnlySFTRenderer().render(trajectory)
    except ValueError as error:
        assert str(error) == "trajectory has no assistant training target"
    else:
        raise AssertionError("expected an empty SFT target to be rejected")


def test_renderer_writes_clean_terminal_assistant_example():
    recorder = TrajectoryRecorder(task_id="retail-clean-render", seed=75)
    recorder.append_user("Please finish the order.")
    recorder.append_assistant('{"message":"The order is complete."}')
    recorder.append_user("Thanks for your help. ###STOP###")
    trajectory = recorder.finish(
        terminal_state={"order_status": "complete"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    example = ActionOnlySFTRenderer().render(trajectory)

    assert example.messages == [
        {"role": "user", "content": "Please finish the order."},
        {"role": "assistant", "content": "The order is complete."},
    ]
    assert example.trainable_message_indices == (1,)


def test_dataset_builder_excludes_policy_badcase_from_sft():
    good_recorder = TrajectoryRecorder(task_id="retail-good", seed=79)
    good_recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-good",
    )
    good_recorder.append_assistant("Your order is pending.")
    good = good_recorder.finish(
        terminal_state={"order_status": "pending"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    bad_recorder = TrajectoryRecorder(task_id="retail-bad", seed=83)
    bad_recorder.append_tool_call(
        name="find_user_id_by_email",
        arguments={"email": "user@example.com"},
        call_id="auth-bad",
    )
    bad_recorder.append_tool_call(
        name="cancel_pending_order",
        arguments={"order_id": "W801", "reason": "no longer needed"},
        call_id="cancel-bad",
    )
    bad = bad_recorder.finish(
        terminal_state={"order_status": "cancelled"},
        evaluation={"task_success": True, "reward": 1.0},
    )

    result = ActionOnlySFTDatasetBuilder(
        verifier=RetailPolicyVerifier(),
    ).build([good, bad])

    assert [example.task_id for example in result.examples] == ["retail-good"]
    assert result.skipped_task_ids == ("retail-bad",)


def test_token_formatter_masks_non_selected_tokens():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {
                "input_ids": list(range(101, 101 + len(messages))),
                "attention_mask": [1] * len(messages),
                "assistant_masks": [
                    1 if message["role"] == "assistant" else 0
                    for message in messages
                ],
            }

    example = SFTExample(
        task_id="retail-mask",
        messages=[
            {"role": "user", "content": "Check order W802."},
            {"role": "assistant", "content": "First action."},
            {"role": "tool", "content": "Observation."},
            {"role": "assistant", "content": "Final answer."},
        ],
        trainable_message_indices=(1, 3),
    )

    encoded = QwenActionOnlyTokenFormatter().format(
        tokenizer=FakeTokenizer(),
        example=example,
    )

    assert encoded["labels"] == [-100, 102, -100, 104]


def test_token_formatter_trains_only_selected_message_indices():
    class PrefixAwareTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            # Each serialized message occupies one token, so prefix lengths
            # identify the token span belonging to each message.
            return {
                "input_ids": list(range(100, 100 + len(messages))),
                "attention_mask": [1] * len(messages),
                # Deliberately mark every assistant token as trainable. The
                # formatter must narrow this to trainable_message_indices.
                "assistant_masks": [
                    1 if message["role"] == "assistant" else 0
                    for message in messages
                ],
            }

    example = SFTExample(
        task_id="retail-selected-targets",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "action one"},
            {"role": "tool", "content": "observation"},
            {"role": "assistant", "content": "action two"},
        ],
        trainable_message_indices=(1,),
    )

    encoded = QwenActionOnlyTokenFormatter().format(
        tokenizer=PrefixAwareTokenizer(),
        example=example,
    )

    assert encoded["labels"] == [-100, 101, -100, -100]


def test_token_formatter_does_not_request_unsupported_assistant_mask():
    class QwenTemplateTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert "return_assistant_tokens_mask" not in kwargs
            return {
                "input_ids": list(range(100, 100 + len(messages))),
                "attention_mask": [1] * len(messages),
            }

    example = SFTExample(
        task_id="retail-qwen-template",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "action"},
        ],
        trainable_message_indices=(1,),
    )

    QwenActionOnlyTokenFormatter().format(
        tokenizer=QwenTemplateTokenizer(),
        example=example,
    )


def test_token_formatter_drops_leading_assistant_greeting_for_qwen_template():
    class UserFirstTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if messages[0]["role"] != "user":
                raise ValueError("No user query found in messages")
            return {
                "input_ids": list(range(100, 100 + len(messages))),
                "attention_mask": [1] * len(messages),
            }

    example = SFTExample(
        task_id="retail-leading-greeting",
        messages=[
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "Check my order."},
            {"role": "assistant", "content": "I found your order."},
        ],
        trainable_message_indices=(0, 2),
    )

    encoded = QwenActionOnlyTokenFormatter().format(
        tokenizer=UserFirstTokenizer(),
        example=example,
    )

    assert encoded["labels"] == [-100, 101]


def test_token_formatter_normalizes_terminal_user_and_teacher_message_wrapper():
    rendered_messages = []

    class NormalizingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            rendered_messages.append(messages)
            return {
                "input_ids": list(range(100, 100 + len(messages))),
                "attention_mask": [1] * len(messages),
            }

    example = SFTExample(
        task_id="retail-terminal-user",
        messages=[
            {"role": "user", "content": "Please finish the order."},
            {"role": "assistant", "content": '{"message":"The order is complete."}'},
            {"role": "user", "content": "Thanks. ###STOP###"},
        ],
        trainable_message_indices=(1,),
    )

    encoded = QwenActionOnlyTokenFormatter().format(
        tokenizer=NormalizingTokenizer(),
        example=example,
    )

    assert rendered_messages[0][-1] == {
        "role": "assistant",
        "content": "The order is complete.",
    }
    assert encoded["labels"] == [-100, 101]


def test_token_formatter_keeps_assistant_labels_inside_each_eos_boundary():
    class BoundaryAwareTokenizer:
        eos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            assert text == "<|im_start|>assistant\n"
            assert add_special_tokens is False
            return [20]

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is False
            # The real Qwen template adds a think scaffold when a prefix ends
            # on an assistant message. This makes prefix lengths unsuitable
            # as boundaries for the complete conversation.
            ids = []
            for index, message in enumerate(messages):
                role = message["role"]
                if role == "user":
                    ids.extend([10 + index, 99, 2])
                elif role == "assistant":
                    header = 20
                    ids.extend([header, header + 1, 2])
                    if index == len(messages) - 1:
                        ids.extend([77, 78])
                else:
                    raise AssertionError("unexpected role in test fixture")
            return {"input_ids": ids}

    example = SFTExample(
        task_id="retail-eos-boundary",
        messages=[
            {"role": "user", "content": "First request."},
            {"role": "assistant", "content": "First action."},
            {"role": "user", "content": "Second request."},
            {"role": "assistant", "content": "Final answer."},
        ],
        trainable_message_indices=(1, 3),
    )

    encoded = QwenActionOnlyTokenFormatter().format(
        tokenizer=BoundaryAwareTokenizer(),
        example=example,
    )

    assert encoded["labels"] == [
        -100,
        -100,
        -100,
        20,
        21,
        2,
        -100,
        -100,
        -100,
        20,
        21,
        2,
        -100,
        -100,
    ]


def test_token_formatter_skips_assistant_turns_truncated_before_eos():
    class TruncatingTokenizer:
        eos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            return [20]

        def apply_chat_template(self, messages, **kwargs):
            ids = []
            for index, message in enumerate(messages):
                if message["role"] == "user":
                    ids.extend([10 + index, 99, 2])
                else:
                    ids.extend([20, 21, 2])
            if "max_length" in kwargs:
                ids = ids[: kwargs["max_length"]]
            return {"input_ids": ids}

    example = SFTExample(
        task_id="retail-truncated-target",
        messages=[
            {"role": "user", "content": "First request."},
            {"role": "assistant", "content": "First action."},
            {"role": "user", "content": "Second request."},
            {"role": "assistant", "content": "Second action."},
        ],
        trainable_message_indices=(1, 3),
    )

    encoded = QwenActionOnlyTokenFormatter().format(
        tokenizer=TruncatingTokenizer(),
        example=example,
        max_length=6,
    )

    assert encoded["labels"] == [-100, -100, -100, 20, 21, 2]


def test_sft_dataset_store_round_trips_examples(tmp_path):
    recorder = TrajectoryRecorder(task_id="retail-store", seed=97)
    recorder.append_user("Check order W803.")
    recorder.append_assistant("The order is pending.")
    example = ActionOnlySFTRenderer().render(
        recorder.finish(
            terminal_state={"order_status": "pending"},
            evaluation={"task_success": True, "reward": 1.0},
        )
    )
    store = SFTDatasetStore(tmp_path / "sft.jsonl")

    store.append(example)

    assert list(store.iter_examples()) == [example]
