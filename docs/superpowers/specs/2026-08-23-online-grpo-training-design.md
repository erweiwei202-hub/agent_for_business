# Online GRPO Training Design

## Goal

Turn the existing GRPO configuration and mathematical helpers into a real
single-GPU online GRPO loop that starts from the SFT LoRA adapter, runs the
local Qwen policy inside the existing tau2 Retail orchestrator, scores each
trajectory with the project Verifier, updates only action tokens, and saves
resumable checkpoints.

## Decisions

- Use in-process Transformers + PEFT for rollout and training so generated
  token ids, old log-probabilities, current log-probabilities, and gradients
  share one tokenizer/model contract.
- Reuse tau2's `Environment`, `Orchestrator`, `UserSimulator`, tool execution,
  and official evaluation. Do not duplicate Retail environment transitions.
- Add a local half-duplex tau2 agent that renders the Qwen chat template,
  parses Qwen3.5 `<tool_call>` XML, and records per-assistant-turn token traces.
- Keep user simulation remote/configurable through the existing tau2 user
  simulator; only the agent policy is replaced by the local trainable model.
- Compute the differentiable clipped objective and reference KL in torch. The
  existing pure-Python `grpo_core` remains the framework-independent contract
  test surface.
- Use a generation lock around the shared local model. Rollout workers may
  overlap tau2/user/environment work, while GPU generation remains serialized
  and deterministic.
- Invalid/infrastructure rollouts are retained in the batch report but are
  excluded from the optimizer update. A batch with no valid action traces
  fails explicitly.

## Public interfaces

- `LocalQwenAgent(...).drain_generation_traces()` returns the token traces
  collected during its latest simulation.
- `OnlineGRPOTrainer(config, policy_model, tokenizer, ...)` exposes `train()`
  and returns a JSON-safe result containing optimizer steps, rollout counts,
  reward/loss history, and checkpoint paths.
- `train_grpo()` constructs the tokenizer and `OnlineGRPOTrainer` by default;
  `trainer_factory` remains available for tests and embedding.

## Data flow

```text
SFT LoRA adapter
      -> trainable PEFT policy + frozen reference policy
      -> tau2 Orchestrator with LocalQwenAgent
      -> Qwen token traces + SimulationRun
      -> RetailTaskRunner normalization + RetailPolicyVerifier
      -> group rewards / advantages
      -> current/reference logprob replay
      -> clipped GRPO loss + reference KL
      -> LoRA optimizer step and resumable checkpoint
```

## Error handling

- Malformed model output becomes a valid assistant text response only when it
  contains usable text; malformed tool XML is recorded as an agent error and
  the simulation is scored invalid rather than silently inventing a call.
- Missing `transformers`, `peft`, or tau2 dependencies produce actionable
  runtime errors at execution time; importing the core package remains safe.
- Checkpoint writes use a temporary directory and a manifest containing model
  source, seed, optimizer step, batch configuration, and metric history.

## Testing

- Pure parser tests cover text responses, one/multiple tool calls, JSON scalar
  parameters, malformed XML, and action-token trace boundaries.
- Agent tests use a fake tokenizer/model and verify tau2 message conversion and
  captured traces without importing GPU dependencies.
- Objective tests verify gradient flow, action-only masking, clipping, KL, and
  rejection of empty/invalid groups.
- Trainer tests use fake rollout/model/optimizer seams to verify group sampling,
  invalid-rollout filtering, checkpoint/resume, and JSON-safe results.
- A real AutoDL smoke command remains the required hardware validation for the
  full tau2 + Qwen path.
