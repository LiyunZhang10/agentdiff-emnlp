#!/usr/bin/env bash
# gpu_kit/serve_qwen14b.sh
# Boot a vllm OpenAI-compatible server for Qwen-2.5-14B-Instruct on 1 H20.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.venv/bin/activate"

# Use the model already cached on this cluster's shared storage if present;
# otherwise fall back to HuggingFace cache.
MODEL_PATH="${MODEL_PATH:-/jizhicfs/stephnialuo/models/Qwen2.5-14B-Instruct}"
if [ ! -d "$MODEL_PATH" ]; then
  MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
fi

PORT="${PORT:-8000}"
echo "[serve] model = $MODEL_PATH"
echo "[serve] port  = $PORT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name qwen2.5-14b \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16
