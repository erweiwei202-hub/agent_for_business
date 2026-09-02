import pytest
import torch

from agent_for_business.grpo_objective import grpo_loss, sequence_logprobs


def test_grpo_loss_uses_only_action_tokens_and_preserves_gradient_flow():
    old = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    current = torch.tensor([[0.0, 0.1]], dtype=torch.float32, requires_grad=True)
    reference = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    result = grpo_loss(
        old_logprobs=old,
        current_logprobs=current,
        reference_logprobs=reference,
        advantages=torch.tensor([1.0]),
        action_mask=torch.tensor([[False, True]]),
        clip_ratio=0.2,
        kl_beta=0.001,
    )
    result.loss.backward()

    assert result.action_token_count == 1
    assert result.policy_objective.item() == pytest.approx(torch.exp(torch.tensor(0.1)).item())
    assert current.grad[0, 0].item() == pytest.approx(0.0)
    assert current.grad[0, 1].item() < 0.0


def test_sequence_logprobs_aligns_logits_with_response_tokens():
    class FakeModel:
        def __call__(self, *, input_ids, use_cache):
            assert use_cache is False
            logits = torch.tensor(
                [
                    [
                        [0.0, 1.0, 2.0],
                        [0.0, 2.0, 1.0],
                        [1.0, 0.0, 2.0],
                    ]
                ],
                requires_grad=True,
            )
            return type("Output", (), {"logits": logits})()

    values = sequence_logprobs(
        FakeModel(),
        torch.tensor([[1, 2, 0]]),
        response_start=1,
    )

    assert values.shape == (1, 2)
    assert values[0, 0].item() == pytest.approx(
        torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=-1)[2].item()
    )


def test_sequence_logprobs_moves_cpu_ids_to_model_device():
    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, *, input_ids, use_cache):
            assert use_cache is False
            assert input_ids.device == self.device
            return type(
                "Output",
                (),
                {"logits": torch.zeros((1, input_ids.shape[1], 4))},
            )()

    values = sequence_logprobs(FakeModel(), torch.tensor([[1, 2]]), 1)

    assert values.shape == (1, 1)


def test_sequence_logprobs_requests_only_response_logits():
    captured = {}

    class FakeModel:
        def __call__(self, *, input_ids, use_cache, logits_to_keep):
            captured["use_cache"] = use_cache
            captured["logits_to_keep"] = logits_to_keep
            assert input_ids.shape == (1, 4)
            return type(
                "Output",
                (),
                {"logits": torch.zeros((1, 2, 4))},
            )()

    values = sequence_logprobs(
        FakeModel(),
        torch.tensor([[1, 2, 3, 0]]),
        response_start=2,
    )

    assert values.shape == (1, 2)
    assert captured["use_cache"] is False
    assert torch.equal(captured["logits_to_keep"], torch.tensor([1, 2]))
