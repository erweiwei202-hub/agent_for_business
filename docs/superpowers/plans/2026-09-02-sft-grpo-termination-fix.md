# SFT-to-GRPO Conversation Termination Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SFT examples match the local GRPO protocol and prevent GRPO updates when a rollout group has no relative reward signal.

**Architecture:** Keep the existing trajectory and JSONL contracts, but normalize examples at rendering/formatting boundaries. SFT formatting will identify assistant spans from the actual rendered token sequence instead of independently rendered prefixes, and the local Qwen parser will unwrap the teacher's `{"message": ...}` envelope. GRPO keeps Qwen `<|im_end|>` as the primary generation stop; `###STOP###` remains only a simulator/defensive control string.

**Tech Stack:** Python, pytest, Transformers tokenizer/chat template, existing trajectory/SFT/GRPO modules.

## Global Constraints

- Do not add `###STOP###` to assistant SFT targets.
- Preserve tool-call XML/structured arguments and train only assistant spans.
- The assistant target span must include its generated `<|im_end|>` EOS and no following user/tool tokens.
- Keep imports usable without optional Transformers/PEFT/tau2 dependencies.
- Preserve unrelated user worktree changes.
- Count valid and invalid rollouts per rollout, not per group.
- Skip groups whose valid rewards are all equal; fail batches with no valid action-token traces.

---

### Task 1: Normalize terminal messages and teacher text

**Files:**
- Modify: `src/agent_for_business/sft_dataset.py`
- Test: `tests/test_sft_dataset.py`

**Interfaces:**
- `QwenActionOnlyTokenFormatter.format(...)` continues accepting `SFTExample` and returns tokenized records.
- Normalization removes leading assistant-only greetings, trims messages after the final trainable assistant, and unwraps assistant text shaped as `{"message": "..."}`.

- [x] **Step 1: Write the failing test** for trailing terminal user messages being excluded and JSON message envelopes being normalized before tokenization.
- [x] **Step 2: Run only those tests** and confirm the old implementation includes the terminal user or preserves the JSON envelope.
- [x] **Step 3: Implement the smallest normalization change** while preserving trainable indices and tool calls.
- [x] **Step 4: Run the focused tests** and confirm they pass.

### Task 2: Replace prefix-offset labels with rendered assistant spans

**Files:**
- Modify: `src/agent_for_business/sft_dataset.py`
- Test: `tests/test_sft_dataset.py`

**Interfaces:**
- `QwenActionOnlyTokenFormatter.format` labels only assistant message spans found in the final rendered sequence.
- The formatter raises a clear error if the tokenizer cannot expose the expected assistant header spans.

- [x] **Step 1: Write the failing regression test** using a tokenizer seam that reproduces the Qwen checkpoint's template behavior: a selected assistant label segment must end at EOS and must not decode into the next user/tool message.
- [x] **Step 2: Run the regression test** and confirm it fails with the current prefix-based span calculation.
- [x] **Step 3: Implement assistant-header/EOS span discovery** against the final `input_ids`, including the selected message indices.
- [x] **Step 4: Run the focused formatter tests** and confirm they pass.

### Task 3: Align local Qwen parsing with normalized SFT text

**Files:**
- Modify: `src/agent_for_business/grpo_agent.py`
- Test: `tests/test_grpo_agent.py`

**Interfaces:**
- `parse_qwen_response('{"message":"..."}')` returns visible content `...`.
- Plain text, tool-call XML, and reasoning stripping remain unchanged.

- [x] **Step 1: Write the failing parser test** for the teacher JSON message envelope.
- [x] **Step 2: Run the focused parser test** and confirm it fails because the old parser returns the raw JSON string.
- [x] **Step 3: Implement the minimal JSON-envelope unwrapping**.
- [x] **Step 4: Run the focused parser tests** and confirm they pass.

### Task 4: Make the launcher use the intended clean dataset

**Files:**
- Modify: `scripts/sft_train.sh`
- Test: `tests/test_pipeline_entrypoints.py` or an existing launcher test if applicable.

**Interfaces:**
- The shell launcher uses `outputs/sft/accepted-qwen-clean.jsonl` as its default dataset and keeps the existing output/checkpoint behavior.

- [x] **Step 1: Add or update the launcher assertion** for the clean dataset path.
- [x] **Step 2: Run the focused launcher test** and confirm it fails against the current `accepted.jsonl` default.
- [x] **Step 3: Change only the dataset default**.
- [x] **Step 4: Run the focused launcher test** and confirm it passes.

### Task 5: Full verification and data audit

**Files:**
- No new production files.
- Test: `tests/test_sft_dataset.py`, `tests/test_sft_training.py`, `tests/test_grpo_agent.py`, full test suite.

- [x] **Step 1: Run all changed-area tests.**
- [x] **Step 2: Re-audit `outputs/sft/accepted-qwen-clean.jsonl` through the formatter** and verify no label segment includes a later role and every selected assistant EOS is labeled.
- [x] **Step 3: Run the complete test suite; record unrelated worktree/environment failures separately.**
- [x] **Step 4: Inspect the final diff and report any environment-only limitations** such as unavailable GPU/tau2 runtime.

### Task 6: Handle all-equal GRPO rewards safely

**Files:**
- Modify: `src/agent_for_business/grpo_online.py`
- Test: `tests/test_grpo_online.py`

**Interfaces:**
- `OnlineGRPOTrainer.train()` reports `skipped_no_signal_groups` and advances the batch cursor without an optimizer update when every valid group has equal rewards.
- `OnlineGRPOTrainer._update_batch()` returns `(None, stats, 0)` for an all-equal valid batch and retains the existing error for batches without valid action-token traces.

- [x] **Step 1: Write the failing test** for two valid rollouts with equal zero reward and assert no optimizer step.
- [x] **Step 2: Run the test** and verify the old implementation does not report the skip behavior.
- [x] **Step 3: Implement the skip path and correct per-rollout invalid counting.**
- [x] **Step 4: Run `tests/test_grpo_online.py` and verify all 13 tests pass.**
