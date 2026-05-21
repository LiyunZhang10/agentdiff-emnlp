#!/usr/bin/env bash
# gpu_kit/install.sh
# One-shot install for the GPU node (TencentOS / RHEL-ish, Python 3.9, no internet
# access except via star-proxy.oa.com which is already in env).
#
# Result: a fresh venv at gpu_kit/.venv with vllm + transformers + requests.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

echo "[install] creating venv at $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"

echo "[install] upgrading pip"
pip install --upgrade pip wheel setuptools

echo "[install] installing vllm + companions (CUDA 12.2 wheels)"
# vllm 0.6.x supports CUDA 12.x; the H20 is a Hopper card, fully supported.
pip install \
    "vllm>=0.6.0,<0.8" \
    "transformers>=4.43" \
    "accelerate>=0.30" \
    "requests>=2.31" \
    "numpy<2"

echo "[install] sanity"
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"

echo "[install] DONE. activate with: source $VENV/bin/activate"
