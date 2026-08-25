# GRPO Training Shell Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a root-level Bash launcher whose editable variables invoke the existing GRPO CLI with all supported training parameters, including checkpoint save frequency and resume path.

**Architecture:** `GRPO_train.sh` will change to the repository root, define all GRPO values in one configuration block, build optional arguments only when configured, and call `python -u -m agent_for_business.cli grpo`. The existing CLI and `OnlineGRPOTrainer` remain unchanged; checkpoint behavior is controlled through `--checkpoint-every` and `--resume-from`.

**Tech Stack:** Bash, Python 3.12+, existing `agent_for_business.cli` argparse entrypoint, existing `OnlineGRPOTrainer` checkpoint implementation.

## Global Constraints

- Use plain `python`; do not use `uv run`.
- Keep the launcher at the repository root as `GRPO_train.sh`.
- Assume the current Python environment already contains the project and training dependencies.
- Preserve existing CLI option names and existing checkpoint semantics.
- Do not modify existing source files or unrelated dirty-worktree changes.

---

### Task 1: Add and validate the GRPO launcher

**Files:**
- Create: `GRPO_train.sh`
- Test: shell syntax check and CLI help parsing; no GPU training run

**Interfaces:**
- Consumes: `agent_for_business.cli` subcommand `grpo` and its current options.
- Produces: a runnable launcher with editable variables for model, output,
  rollout, optimization, generation, device, checkpoint, and resume settings.

- [ ] **Step 1: Create the launcher configuration block**

  Define these editable variables near the top of `GRPO_train.sh`:

  ```bash
  MODEL="outputs/sft/checkpoint-qwen/checkpoint-294"
  OUTPUT_DIR="outputs/grpo"
  SPLIT_TASKS="vendor/tau2-bench/data/tau2/domains/retail/split_tasks.json"
  GROUPS_PER_BATCH="50"
  GROUP_SIZE="4"
  BATCH_EPOCHS="2"
  MAX_WORKERS="4"
  INFERENCE_MICROBATCH="2"
  CLIP_RATIO="0.2"
  KL_BETA="0.001"
  SEED="42"
  MAX_ROLLOUT_BATCHES="2"
  USER_LLM="gpt-5.6-luna"
  USER_API_BASE=""
  LEARNING_RATE="1e-5"
  WEIGHT_DECAY="0.0"
  TEMPERATURE="0.7"
  TOP_P="0.95"
  MAX_NEW_TOKENS="512"
  DEVICE="auto"
  CHECKPOINT_EVERY="1"
  RESUME_FROM=""
  ```

- [ ] **Step 2: Add safe shell setup and repository-root handling**

  Start the script with:

  ```bash
  #!/usr/bin/env bash
  set -euo pipefail

  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ```

  This makes relative paths resolve from the repository root and ensures a
  failed Python process stops the launcher.

- [ ] **Step 3: Assemble optional arguments without empty flag values**

  Use a Bash array so `--user-api-base` and `--resume-from` are omitted when
  their variables are empty:

  ```bash
  OPTIONAL_ARGS=()
  if [[ -n "$USER_API_BASE" ]]; then
    OPTIONAL_ARGS+=(--user-api-base "$USER_API_BASE")
  fi
  if [[ -n "$RESUME_FROM" ]]; then
    OPTIONAL_ARGS+=(--resume-from "$RESUME_FROM")
  fi
  ```

- [ ] **Step 4: Invoke the existing GRPO CLI with every configured option**

  Use the current subcommand form, not a nonexistent `--grpo` flag:

  ```bash
  python -u -m agent_for_business.cli grpo \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --split-tasks "$SPLIT_TASKS" \
    --groups-per-batch "$GROUPS_PER_BATCH" \
    --group-size "$GROUP_SIZE" \
    --batch-epochs "$BATCH_EPOCHS" \
    --max-workers "$MAX_WORKERS" \
    --inference-microbatch "$INFERENCE_MICROBATCH" \
    --clip-ratio "$CLIP_RATIO" \
    --kl-beta "$KL_BETA" \
    --seed "$SEED" \
    --max-rollout-batches "$MAX_ROLLOUT_BATCHES" \
    --user-llm "$USER_LLM" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    "${OPTIONAL_ARGS[@]}"
  ```

- [ ] **Step 5: Validate shell syntax**

  Run:

  ```bash
  bash -n GRPO_train.sh
  ```

  Expected: exit code `0` and no output.

- [ ] **Step 6: Validate the Python CLI entrypoint without starting training**

  Run:

  ```bash
  python -m agent_for_business.cli grpo --help
  ```

  Expected: exit code `0` and help output containing `--checkpoint-every` and
  `--resume-from`.

- [ ] **Step 7: Review the final diff and report the launcher usage**

  Confirm only `GRPO_train.sh` is added by this implementation, then report:

  ```bash
  bash GRPO_train.sh
  ```

  and explain that setting `RESUME_FROM="outputs/grpo/checkpoint-10"` resumes
  from that checkpoint while `CHECKPOINT_EVERY` controls future saves.
