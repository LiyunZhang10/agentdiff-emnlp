# GPU Kit — running heavy AgentDiff experiments on the H20 node

This kit lets you run perturbation experiments for **larger models that the
free-tier APIs can't host** (Qwen-2.5-14B, Llama-3.3-70B, Gemma-2-9B, etc.) on
a GPU node that has **2× NVIDIA H20 (96 GB each)** and a working `vllm`
install.

The kit reuses [`code/agentdiff_v2.py`](../code/agentdiff_v2.py) verbatim —
only the LLM backend is replaced (vllm OpenAI-compatible server instead of
ollama / groq / mimo).

---

## 0. What you need on the GPU node

* Linux + Python 3.9 (already present on TencentOS H20 nodes)
* Drivers + CUDA 12.x (already present: nvidia-smi shows CUDA 12.2)
* Network access through `star-proxy.oa.com:3128` for HuggingFace + PyPI
  (already exported in `~/.bashrc` on these nodes — `pip` works out of the box)
* Models on shared storage (already present at `/jizhicfs/...`):
  * `Qwen2.5-14B-Instruct` → `/jizhicfs/stephnialuo/models/Qwen2.5-14B-Instruct`
  * `Llama-3.3-70B-Instruct-hz` → `/jizhicfs/beipingpan/taiji_pipline/Llama-3.3-70B-Instruct-hz`

---

## 1. The 5-step recipe

```bash
# === ON THE GPU NODE ===

# (1) clone this repo and enter it
cd /data
git clone https://<USER>:<TOKEN>@github.com/<USER>/agentdiff-emnlp.git
cd agentdiff-emnlp

# (2) install vllm into a local venv (one-time, ~10 min)
bash gpu_kit/install.sh

# (3a) start a vllm server in tmux/screen — pick ONE of:
tmux new -s serve
bash gpu_kit/serve_qwen14b.sh         # 1× H20 → port 8000
# OR
bash gpu_kit/serve_llama70b.sh        # 2× H20 → port 8001
# wait until you see "Uvicorn running on http://0.0.0.0:8000"
# detach: Ctrl-B then D

# (3b) (optional) sanity probe in another shell
source gpu_kit/.venv/bin/activate
python3 gpu_kit/vllm_llm_fn.py qwen2.5-14b
# should print:  PONG

# (4) launch the actual experiment (resumable)
tmux new -s run
source gpu_kit/.venv/bin/activate

# Qwen-2.5-14B fix benchmark (200 questions × 3 agents × 3 benchmarks)
python3 gpu_kit/client_run.py \
    --model qwen2.5-14b \
    --slug qwen25_14b_vllm \
    --base-url http://localhost:8000/v1 \
    --benchmarks gsm8k math hotpotqa \
    --agents react cot direct \
    --n-samples 200

# Llama-3.3-70B (one benchmark to keep wall-clock manageable)
python3 gpu_kit/client_run.py \
    --model llama-3.3-70b \
    --slug llama33_70b_vllm \
    --base-url http://localhost:8001/v1 \
    --benchmarks gsm8k math \
    --agents react cot \
    --n-samples 100

# (5) push results back to GitHub when done
cd /data/agentdiff-emnlp
git add results/runs_real_qwen25_14b_vllm \
        results/runs_real_llama33_70b_vllm \
        results/results_real_qwen25_14b_vllm \
        results/results_real_llama33_70b_vllm
git commit -m "gpu: add Qwen2.5-14B + Llama-3.3-70B runs"
git push
```

---

## 2. Wall-clock estimates (on 2× H20 96 GB)

| Model | Per-question | Cell (200 q × 19 perturbations × 3 agents) | Notes |
|---|---|---|---|
| Qwen-2.5-14B (1× H20) | ~3 s | **~9.5 h / benchmark** | fits comfortably |
| Llama-3.3-70B (2× H20, TP=2) | ~6 s | **~19 h / benchmark** | use `--n-samples 100` |
| Gemma-2-9B (1× H20) | ~2 s | **~6 h / benchmark** | optional, low priority |

**Recommended use of one overnight window (until 10 AM tomorrow):**

1. **Phase A (0–2 h)**: install + serve + sanity-probe.
2. **Phase B (2–11 h)**: Qwen-2.5-14B on **gsm8k + math** (≈ 8 h).
3. **Phase C (11–19 h)**: Llama-3.3-70B on **gsm8k** only, n=100 (≈ 8 h).
4. **Phase D (last 1 h)**: aggregate + git push.

If anything stalls, the client is **fully resumable** (it skips already-written
`<sid>.json` files), just re-run `client_run.py` with the same `--slug`.

---

## 3. Where outputs land

```
results/
└── runs_real_qwen25_14b_vllm/
    ├── gsm8k/
    │   ├── react/
    │   │   ├── gsm8k_test_0.json
    │   │   ├── gsm8k_test_1.json
    │   │   └── ...
    │   ├── cot/
    │   └── direct/
    ├── math/
    └── hotpotqa/

results/results_real_qwen25_14b_vllm/
├── gsm8k_react.json
├── gsm8k_cot.json
└── ...   ← per-cell aggregates (Δ, semantic vs surface, propagation patterns)
```

These plug straight into `code/aggregate_conditional.py` and
`code/make_paper_figures.py` once pulled back.

---

## 4. After the run — back on the control PC

```bash
git pull
python3 code/aggregate_conditional.py     # rebuild 26-cell aggregate
python3 code/make_paper_figures.py        # regenerate figures
python3 code/md_to_docx.py paper/paper.md -o paper/Paper_EN.docx
```

The paper text is auto-updated by re-running the analysis pipeline. Any
figures, tables, and statistics in `paper.md` that depend on aggregated
results will be refreshed.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `vllm: command not found` | `source gpu_kit/.venv/bin/activate` |
| `OOM` on 70B | drop to `--max-model-len 4096` or `--gpu-memory-utilization 0.85` |
| `CUDA out of memory` mid-run | another process is on the GPU; `nvidia-smi`, kill it |
| pip can't reach pypi | the proxy is in `~/.bashrc`; `source ~/.bashrc` and retry |
| HuggingFace download blocked | use the local `/jizhicfs` path (already wired in the serve scripts) |
| Server starts but `client_run.py` 404s | check `--served-model-name` matches `--model` flag |
