export HF_ENDPOINT=https://hf-mirror.com
#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ===== Edit parameters here =====
MODEL="Qwen/Qwen3.5-2B"
SERVED_MODEL_NAME="qwen-base"
HOST="0.0.0.0"
PORT="8000"
API_KEY="EMPTY"

MAX_MODEL_LEN="32768"
MAX_NUM_SEQS="4"
GPU_MEMORY_UTILIZATION="0.7"

ENABLE_AUTO_TOOL_CHOICE="1"
TOOL_CALL_PARSER="qwen3_coder"

ENABLE_LORA="1"
LORA_MODULES="qwen-sft=outputs/sft/checkpoint-qwen/checkpoint-294"


TOOL_ARGS=()
if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
  TOOL_ARGS+=(--enable-auto-tool-choice)
fi
TOOL_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")

LORA_ARGS=()
if [[ "$ENABLE_LORA" == "1" ]]; then
  LORA_ARGS+=(--enable-lora --lora-modules "$LORA_MODULES")
fi

python -u eval-scripts/serve_qwen.py \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${TOOL_ARGS[@]}" \
  "${LORA_ARGS[@]}"
