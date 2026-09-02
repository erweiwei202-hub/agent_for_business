# GRPO Base Model LoRA Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with a test-first checkpoint after each behavior. Preserve unrelated working-tree changes.

**Goal:** Allow `MODEL="Qwen/Qwen3.5-2B"` to load the base model with a trainable LoRA adapter so GRPO updates only adapter parameters instead of full-model weights.

**Architecture:** Existing adapter directories continue to load through `PeftModel.from_pretrained(..., is_trainable=True)`. A plain base-model path will load the base model and wrap it with `get_peft_model()` using the same LoRA hyperparameters and target modules already used by SFT. The training configuration records that the effective policy is LoRA-backed while checkpoint/resume paths remain compatible with adapter checkpoints.

**Tech Stack:** Python 3.12, Transformers, PEFT, pytest, existing `GRPOTrainingConfig` and CLI.

## Global Constraints

- Keep training action-only and LoRA-based; do not add full-parameter GRPO as the default.
- Keep dependency imports lazy so core tests remain runnable without training dependencies.
- Reuse the SFT LoRA settings: rank 8, alpha 16, dropout 0.05, and the existing Qwen attention/MLP/linear-attention target modules.
- Preserve existing SFT-adapter loading and `--resume-from` behavior.
- Do not change rollout count, objective math, or checkpoint format in this change.

### Task 1: Lock the base-model LoRA loading contract

**Files:**
- Modify: `tests/test_grpo_training.py`

**Interfaces:**
- `load_grpo_model(base_model_path)` must call PEFT `get_peft_model()` for a plain model path.
- The PEFT configuration passed to `get_peft_model()` must match the SFT defaults.
- Existing adapter-path behavior must continue calling `PeftModel.from_pretrained()`.

- [ ] **Step 1: Add a test for base-model LoRA wrapping**

  Mock Transformers and PEFT, load a plain model path, and assert that `LoraConfig` receives `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, `bias="none"`, `task_type="CAUSAL_LM"`, and the SFT target modules; assert that `get_peft_model()` receives the loaded base model and this config.

- [ ] **Step 2: Run the focused test and verify it fails**

  Run: `python -m pytest -q tests/test_grpo_training.py -k base_model_lora`

  Expected: FAIL because the current plain-model branch returns the unwrapped base model and does not import or call PEFT.

### Task 2: Implement LoRA injection for plain base models

**Files:**
- Modify: `src/agent_for_business/grpo_training.py`
- Modify: `tests/test_grpo_training.py`

**Interfaces:**
- `load_grpo_model()` keeps lazy imports and returns a trainable PEFT model for a plain base-model path.
- A local `use_lora` option remains available for callers that explicitly need raw full-model loading, while GRPO configuration defaults to LoRA.

- [ ] **Step 1: Implement the smallest loader change**

  Add a default-enabled LoRA choice to `GRPOTrainingConfig` and `load_grpo_model()`. For a plain model path, construct the SFT-equivalent `LoraConfig`, call `get_peft_model(base_model, peft_config)`, and leave the existing adapter branch unchanged. Pass the configuration through `train_grpo()` for both new runs and policy loading during resume.

- [ ] **Step 2: Run focused tests and verify they pass**

  Run: `python -m pytest -q tests/test_grpo_training.py`

  Expected: PASS, including the new base-model LoRA test and existing adapter/resume tests.

### Task 3: Make run metadata and documentation explicit

**Files:**
- Modify: `src/agent_for_business/cli.py`
- Modify: `README.md`
- Modify: `docs/autodl-runbook.md`

**Interfaces:**
- A base model configured for default GRPO loading is reported as LoRA-backed rather than full-parameter training.
- The documented command keeps `MODEL="Qwen/Qwen3.5-2B"` and explains that only LoRA parameters are trainable.

- [ ] **Step 1: Update metadata and usage text**

  Record the effective model kind as LoRA for a plain base path under the default configuration, document the PEFT requirement, and state that the GRPO checkpoint contains adapter weights.

- [ ] **Step 2: Run the relevant test suite and static checks**

  Run: `python -m pytest -q tests/test_grpo_training.py tests/test_cli.py tests/test_grpo_online.py`

  Run: `git diff --check`

  Expected: all selected tests pass and `git diff --check` is clean.
