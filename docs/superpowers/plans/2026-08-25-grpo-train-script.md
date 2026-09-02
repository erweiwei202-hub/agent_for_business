# GRPO Training Shell Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a root-level Bash launcher whose editable variables invoke the existing GRPO CLI with all supported training parameters, including checkpoint save frequency and resume path.

**Architecture:** `GRPO_train.sh` will change to the repository root, add the local `src` directory to `PYTHONPATH`, define editable GRPO values in one configuration block, automatically select the latest complete numeric checkpoint, and call `python -u -m agent_for_business.cli grpo`. The script will not override user simulator settings; the existing CLI will read them from `.env` and the process environment. `OnlineGRPOTrainer` will save periodic checkpoints at completed batch boundaries and save a final checkpoint when the periodic interval does not land on the final step.

**Tech Stack:** Bash, Python 3.12+, existing `agent_for_business.cli` argparse entrypoint, existing `OnlineGRPOTrainer` checkpoint implementation.

## Global Constraints

- Use plain `python`; do not use `uv run`.
- Keep the launcher at the repository root as `GRPO_train.sh`.
- Assume the current Python environment already contains the training dependencies.
- Preserve existing CLI option names and load `USER_LLM`/`USER_API_BASE` from `.env` or the process environment.
- Do not modify unrelated dirty-worktree changes.

---

### Task 1: Add and validate the GRPO launcher

**Files:**
- Create: `GRPO_train.sh`
- Modify: `src/agent_for_business/grpo_online.py:train`
- Test: `tests/test_grpo_online.py`, shell syntax check, and CLI help parsing; no GPU training run

**Interfaces:**
- Consumes: `agent_for_business.cli` subcommand `grpo` and its current options.
- Produces: a runnable launcher with editable variables for model, output,
  rollout, optimization, generation, device, and checkpoint settings; it
  automatically resumes the latest checkpoint.

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
  LEARNING_RATE="1e-5"
  WEIGHT_DECAY="0.0"
  TEMPERATURE="0.7"
  TOP_P="0.95"
  MAX_NEW_TOKENS="512"
  DEVICE="auto"
  CHECKPOINT_EVERY="10"
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

- [ ] **Step 3: Add automatic checkpoint discovery**

  Scan only numeric `checkpoint-N` directories and select the largest step:

  ```bash
  LATEST_CHECKPOINT=""
  LATEST_STEP=-1
  for checkpoint in "$OUTPUT_DIR"/checkpoint-*; do
    [[ -d "$checkpoint" ]] || continue
    [[ -f "$checkpoint/grpo_manifest.json" ]] || continue
    [[ -f "$checkpoint/optimizer.pt" ]] || continue
    step="${checkpoint##*-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    if (( 10#$step > LATEST_STEP )); then
      LATEST_STEP=$((10#$step))
      LATEST_CHECKPOINT="$checkpoint"
    fi
  done

  RESUME_ARGS=()
  if [[ -n "$LATEST_CHECKPOINT" ]]; then
    RESUME_ARGS+=(--resume-from "$LATEST_CHECKPOINT")
  fi
  ```

- [ ] **Step 4: Add final-checkpoint regression coverage first**

  Add a trainer test with `checkpoint_every=10` and one optimizer step. It
  must assert that `checkpoint-1/optimizer.pt` and
  `checkpoint-1/grpo_manifest.json` exist even though step 1 is not periodic.

- [ ] **Step 5: Implement batch-boundary checkpoint persistence**

  Mark periodic checkpoints as due during the epoch loop, but save them only
  after all epochs for the current batch finish. After the rollout-batch loop
  and before returning the result, save the final step only when it is nonzero
  and was not already saved periodically:

  ```python
  if (
      self.optimizer_steps
      and self.optimizer_steps % self.config.checkpoint_every != 0
  ):
      self.save_checkpoint(self.optimizer_steps)
  ```

- [ ] **Step 6: Invoke the existing GRPO CLI with every configured option**

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
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    "${RESUME_ARGS[@]}"
  ```

- [ ] **Step 7: Run the regression test**

  Run:

  ```bash
  pytest -q tests/test_grpo_online.py::test_online_trainer_saves_final_checkpoint_when_interval_is_larger
  ```

  Expected: PASS.

- [ ] **Step 8: Validate shell syntax**

  Run:

  ```bash
  bash -n GRPO_train.sh
  ```

  Expected: exit code `0` and no output.

- [ ] **Step 9: Validate the Python CLI entrypoint without starting training**

  Run:

  ```bash
  python -m agent_for_business.cli grpo --help
  ```

  Expected: exit code `0` and help output containing `--checkpoint-every` and
  `--resume-from`.

- [ ] **Step 10: Review the final diff and report the launcher usage**

  Confirm only `GRPO_train.sh` is added by this implementation, then report:

  ```bash
  bash GRPO_train.sh
  ```

  and explain that an existing `outputs/grpo/checkpoint-N` is selected
  automatically while `CHECKPOINT_EVERY` controls future periodic saves.
