#!/usr/bin/env bash
# gpu_kit/serve_llama70b.sh
# Boot a vllm OpenAI-compatible server for Llama-3.3-70B-Instruct on 2 H20.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.venv/bin/activate"

MODEL_PATH="${MODEL_PATH:-/jizhicfs/beipingpan/taiji_pipline/Llama-3.3-70B-Instruct-hz}"
if [ ! -d "$MODEL_PATH" ]; then
  MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct"
fi

PORT="${PORT:-8001}"
echo "[serve] model = $MODEL_PATH"
echo "[serve] port  = $PORT"

# 70B on 2× H20 (96 GB each) → tensor_parallel=2, bf16 fits.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name llama-3.3-70b \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --dtype bfloat16
