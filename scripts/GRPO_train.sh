#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# ===== Edit GRPO parameters here =====
# MODEL="outputs/sft/checkpoint-qwen/checkpoint-294"
MODEL="Qwen/Qwen3.5-2B"
OUTPUT_DIR="outputs/grpo-base-lora"
SPLIT_TASKS="vendor/tau2-bench/data/tau2/domains/retail/split_tasks.json"

# Keep a complete console transcript beside the structured progress log.
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/console.log") 2>&1

GROUPS_PER_BATCH="10"
GROUP_SIZE="4"
BATCH_EPOCHS="2"
# Maximum number of concurrent rollout worker threads.
MAX_WORKERS="2"
INFERENCE_MICROBATCH="1"
PARALLEL_GENERATION="0"

CLIP_RATIO="0.2"
KL_BETA="0.001"
SEED="42"
# Number of rollout batches for the whole run. Each batch collects
# GROUPS_PER_BATCH * GROUP_SIZE rollouts.
MAX_ROLLOUT_BATCHES="24"

LEARNING_RATE="1e-5"
# L2-style parameter penalty; 0 disables weight decay and matches the
# project's current GRPO default.
WEIGHT_DECAY="0.0"
TEMPERATURE="0.7"
TOP_P="0.95"
MAX_NEW_TOKENS="256"
DEVICE="auto"

# Save at completed batch boundaries once N optimizer steps have accumulated,
# plus the final step. This avoids resuming in the middle of a batch.
CHECKPOINT_EVERY="1"

# Automatically resume from the numerically latest checkpoint, if one exists.
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
  echo "Resuming GRPO from $LATEST_CHECKPOINT"
  RESUME_ARGS+=(--resume-from "$LATEST_CHECKPOINT")
else
  echo "No GRPO checkpoint found under $OUTPUT_DIR; starting a new run"
fi

GENERATION_ARGS=()
if [[ "$PARALLEL_GENERATION" == "1" ]]; then
  GENERATION_ARGS+=(--parallel-generation)
fi

python -u -m agent_for_business.cli grpo \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --split-tasks "$SPLIT_TASKS" \
  --groups-per-batch "$GROUPS_PER_BATCH" \
  --group-size "$GROUP_SIZE" \
  --batch-epochs "$BATCH_EPOCHS" \
  --max-workers "$MAX_WORKERS" \
  --inference-microbatch "$INFERENCE_MICROBATCH" \
  "${GENERATION_ARGS[@]}" \
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
