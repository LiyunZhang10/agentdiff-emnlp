#!/usr/bin/env bash
# gpu_kit/serve_qwen14b_gpu1.sh
# 第二实例：跑在 GPU1，端口 8001
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.venv/bin/activate"

MODEL_PATH="${MODEL_PATH:-/jizhicfs/stephnialuo/models/Qwen2.5-14B-Instruct}"
if [ ! -d "$MODEL_PATH" ]; then
  MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"
fi

PORT="${PORT:-8001}"
echo "[serve-gpu1] model = $MODEL_PATH"
echo "[serve-gpu1] port  = $PORT"
echo "[serve-gpu1] CUDA_VISIBLE_DEVICES = 1"

CUDA_VISIBLE_DEVICES=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name qwen2.5-14b \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16
