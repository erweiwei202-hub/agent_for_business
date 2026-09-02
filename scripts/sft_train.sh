#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# ===== Edit SFT parameters here =====
MODEL="Qwen/Qwen3.5-2B"
DATASET="outputs/sft/accepted-qwen-clean.jsonl"
OUTPUT_DIR="outputs/sft/checkpoint"
EPOCHS="2"
# Leave empty to use the trainer default sequence length.
MAX_LENGTH=""

ARGS=(
  train-sft
  --model "$MODEL"
  --dataset "$DATASET"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
)

if [[ -n "$MAX_LENGTH" ]]; then
  ARGS+=(--max-length "$MAX_LENGTH")
fi

python -u -m agent_for_business.cli "${ARGS[@]}"
