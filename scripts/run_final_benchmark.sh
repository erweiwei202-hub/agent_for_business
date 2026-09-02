#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ===== Edit parameters here =====
ENV_FILE=".env"
AGENT_LLM="openai/qwen-sft"
VLLM_API_BASE="http://127.0.0.1:8000/v1"
VLLM_API_KEY="EMPTY"
USER_LLM="gpt-5.6-luna"
OUTPUT="outputs/benchmarks/qwen-sft-final.json"
SUMMARY_OUTPUT="outputs/benchmarks/qwen-sft-final.md"
NUM_TRIALS="3"
SEED="300"
MAX_CONCURRENCY="1"

python -u eval-scripts/run_final_benchmark.py \
  --env-file "$ENV_FILE" \
  --agent-llm "$AGENT_LLM" \
  --vllm-api-base "$VLLM_API_BASE" \
  --vllm-api-key "$VLLM_API_KEY" \
  --user-llm "$USER_LLM" \
  --output "$OUTPUT" \
  --summary-output "$SUMMARY_OUTPUT" \
  --num-trials "$NUM_TRIALS" \
  --seed "$SEED" \
  --max-concurrency "$MAX_CONCURRENCY"
