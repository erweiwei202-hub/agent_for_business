"""Differentiable torch objectives for the online Retail GRPO loop."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectiveResult:
    """Scalar GRPO terms and the number of tokens used by the objective."""

    loss: Any
    policy_objective: Any
    reference_kl: Any
    action_token_count: int


def grpo_loss(
    *,
    old_logprobs: Any,
    current_logprobs: Any,
    reference_logprobs: Any,
    advantages: Any,
    action_mask: Any,
    clip_ratio: float = 0.2,
    kl_beta: float = 0.001,
) -> ObjectiveResult:
    """Compute action-masked clipped GRPO loss with sampled reference KL."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("GRPO objective requires torch") from error

    if old_logprobs.shape != current_logprobs.shape:
        raise ValueError("old and current logprobs must have the same shape")
    if reference_logprobs.shape != current_logprobs.shape:
        raise ValueError("reference and current logprobs must have the same shape")
    if action_mask.shape != current_logprobs.shape:
        raise ValueError("action mask and logprobs must have the same shape")
    if advantages.shape[0] != current_logprobs.shape[0]:
        raise ValueError("advantages must have one value per rollout")
    if clip_ratio < 0.0 or kl_beta < 0.0:
        raise ValueError("clip_ratio and kl_beta must be non-negative")

    old_logprobs = old_logprobs.to(current_logprobs)
    reference_logprobs = reference_logprobs.to(current_logprobs)
    advantages = advantages.to(current_logprobs)
    mask = action_mask.to(device=current_logprobs.device, dtype=torch.bool)
    action_count = int(mask.sum().item())
    if action_count == 0:
        raise ValueError("at least one action token is required")

    expanded_advantages = advantages.to(current_logprobs).unsqueeze(-1)
    ratio = torch.exp(current_logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    unclipped = ratio * expanded_advantages
    clipped = clipped_ratio * expanded_advantages
    surrogate = torch.minimum(unclipped, clipped)
    policy_objective = surrogate.masked_select(mask).mean()

    delta = reference_logprobs - current_logprobs
    sampled_kl = torch.expm1(delta) - delta
    reference_kl = sampled_kl.masked_select(mask).mean()
    loss = -policy_objective + kl_beta * reference_kl
    return ObjectiveResult(
        loss=loss,
        policy_objective=policy_objective,
        reference_kl=reference_kl,
        action_token_count=action_count,
    )


def sequence_logprobs(model: Any, input_ids: Any, response_start: int) -> Any:
    """Return differentiable log-probabilities for generated response tokens."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("GRPO objective requires torch") from error

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("sequence_logprobs expects one [1, sequence] input")
    if response_start <= 0 or response_start >= input_ids.shape[1]:
        raise ValueError("response_start must identify tokens after a prompt")

    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = input_ids.device
    input_ids = input_ids.to(device)
    targets = input_ids[:, response_start:]
    response_positions = torch.arange(
        response_start - 1,
        input_ids.shape[1] - 1,
        device=input_ids.device,
    )
    try:
        output = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=response_positions,
        )
    except TypeError as error:
        # Older causal-LM implementations do not expose logits_to_keep.
        # Keep the compatibility fallback, while using the optimized path for
        # models such as Qwen3.5 that support selecting output positions.
        if "logits_to_keep" not in str(error):
            raise
        output = model(input_ids=input_ids, use_cache=False)
    logits = output.logits
    if logits.shape[1] == input_ids.shape[1]:
        logits = logits[:, response_start - 1 : -1, :]
    if logits.shape[1] != targets.shape[1]:
        raise ValueError("model logits and response tokens are not aligned")
    return torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
