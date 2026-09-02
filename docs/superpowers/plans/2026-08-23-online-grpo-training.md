# Online GRPO Training Implementation Plan

> **For agentic workers:** Execute this plan inline task-by-task with a RED → GREEN checkpoint after every behavior. Existing uncommitted user changes must be preserved.

**Goal:** Implement a real single-GPU online GRPO trainer that directly starts from the SFT LoRA adapter, runs the local Qwen policy in tau2 Retail, performs action-only clipped updates with reference KL, and saves resumable checkpoints.

**Architecture:** A local half-duplex tau2 agent owns prompt rendering, Qwen XML tool-call parsing, generation, and per-turn token traces. A rollout coordinator reuses tau2's environment/orchestrator and the project Verifier. A torch trainer replays traces against current and frozen-reference policies, computes the differentiable objective, updates trainable PEFT parameters, and persists checkpoints.

**Tech Stack:** Python 3.12, PyTorch, Transformers, PEFT, tau2-bench, pytest, existing project trajectory/verifier contracts.

## Global Constraints

- SFT output `outputs/sft/checkpoint-qwen/checkpoint-294` is a PEFT LoRA adapter and must not require merge.
- The policy objective trains assistant action tokens only; user/tool observation tokens are context.
- The existing tau2 Retail environment and user simulator remain the source of live task execution.
- Validation and final-test tasks are never used for online GRPO rollouts.
- Current Windows tests must remain import-safe without torch/transformers/peft/tau2 installed.

### Task 1: Qwen action parsing and trace contract

**Files:**
- Create: `src/agent_for_business/grpo_agent.py`
- Test: `tests/test_grpo_agent.py`

**Interfaces:**
- `GenerationTrace(prompt_ids, response_ids, old_logprobs, action_mask)` stores one assistant generation.
- `parse_qwen_response(text) -> ParsedAssistant` parses text or Qwen3.5 XML tool calls.
- `LocalQwenAgent.drain_generation_traces() -> tuple[GenerationTrace, ...]` exposes traces after a rollout.

- [x] Write one failing parser test for a `<tool_call>` with scalar and JSON parameters.
- [x] Run the focused test and verify the missing parser fails.
- [x] Implement the minimal parser and tau2 `ToolCall` conversion behind lazy tau2 imports.
- [x] Run the focused test and verify it passes.
- [x] Add one failing trace test proving response ids are action-masked while prompt ids are context.
- [x] Implement fake-model/tokenizer generation and trace capture.
- [x] Run parser and trace tests.

### Task 2: Local agent inside tau2 half-duplex orchestration

**Files:**
- Modify: `src/agent_for_business/grpo_agent.py`
- Create: `src/agent_for_business/grpo_rollout.py`
- Test: `tests/test_grpo_rollout.py`

**Interfaces:**
- `Tau2LocalRolloutRunner.run(task_id, seed) -> RolloutResult` builds a tau2 environment/user/orchestrator using injected builders and returns the `SimulationRun`, `Trajectory`, `VerificationResult`, and traces.
- `RolloutResult.valid_for_update` is true only when the Verifier reward is valid and at least one action trace exists.

- [x] Write a failing fake tau2 integration test for user message → local tool call → environment result → final assistant message.
- [x] Run it and verify the runner path is absent.
- [x] Implement the runner using tau2 `build_environment`, `build_user`, `Orchestrator`, and `run_simulation` through lazy imports.
- [x] Convert the simulation with `RetailTaskRunner.evaluation_from_simulation` and score it with `RetailPolicyVerifier`.
- [x] Run the focused rollout tests.

### Task 3: Differentiable GRPO objective and logprob replay

**Files:**
- Create: `src/agent_for_business/grpo_objective.py`
- Test: `tests/test_grpo_objective.py`

**Interfaces:**
- `sequence_logprobs(model, input_ids, response_start) -> Tensor` returns differentiable log-probabilities for response tokens.
- `grpo_loss(old_logprobs, current_logprobs, reference_logprobs, advantages, action_mask, clip_ratio, kl_beta) -> ObjectiveResult` returns loss, policy objective, KL, and action-token count.

- [x] Write a failing test that checks gradients reach only selected response/action positions.
- [x] Implement shifted causal logprob gathering and masked torch reductions.
- [x] Add clipping, negative-advantage behavior, reference KL, and empty-mask validation tests.
- [x] Run all objective tests.

### Task 4: Online trainer, batching, optimizer, checkpoint/resume

**Files:**
- Modify: `src/agent_for_business/grpo_training.py`
- Create: `src/agent_for_business/grpo_online.py`
- Test: `tests/test_grpo_online.py`

**Interfaces:**
- `OnlineGRPOTrainer.train() -> dict` performs rollout groups, computes advantages, replays traces, updates the optimizer, and saves checkpoints.
- `OnlineGRPOTrainer.save_checkpoint(step) -> Path` saves PEFT adapter/tokenizer/optimizer/manifest.
- `GRPOTrainingConfig` gains user-simulator, rollout, optimizer, device, checkpoint, and resume fields.

- [x] Write a failing trainer test for one fake group with rewards `[0, 1, 1, 0]` and one optimizer update.
- [x] Implement the minimal sequential trainer seam with injected rollout and objective functions.
- [x] Add failing tests for invalid rollout filtering and zero-action batch rejection.
- [x] Add checkpoint and resume tests using a fake model/optimizer.
- [x] Implement real Transformers/PEFT model replay and tau2 rollout factory behind lazy imports.
- [x] Run focused online trainer tests.

### Task 5: CLI/default wiring, exports, documentation, verification

**Files:**
- Modify: `src/agent_for_business/cli.py`
- Modify: `src/agent_for_business/__init__.py`
- Modify: `README.md`
- Modify: `docs/autodl-runbook.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- `train_grpo(config)` creates the tokenizer and real `OnlineGRPOTrainer` when no factory is injected.
- CLI writes training metrics and checkpoint paths to `grpo_result.json`.

- [x] Write a failing CLI test proving default GRPO no longer raises the old “trainer unavailable” error when dependencies/factory seams are available.
- [x] Wire config, model, tokenizer, trainer, and result manifest.
- [x] Update docs with the real command and explicit AutoDL-only requirement.
- [x] Run targeted tests, compileall, Ruff, and the full suite; report tau2 dependency failures separately if the local environment lacks tau2.
