# Mechanism Pilot — Decoding Instability via First-Token Logprob KL

## What this is

A focused, statistically clean pilot test for **one candidate mechanism**:
*decoding-time divergence*. We measure whether semantic-preserving rewrites
push the model's first-token next-token distribution further from the
original than surface-only edits do.

This is the **only mechanism candidate** out of the 6 considered that we
chose to test, because:

| candidate | reason rejected / accepted |
|---|---|
| **decoding instability** (logprob KL) | ✅ chosen — clean, vllm-native, no white-box needed |
| representation drift (hidden cosine) | deferred — needs HF transformers, slower; do only if pilot passes |
| attention redistribution | rejected — well-known unreliable signal (Jain & Wallace 2019) |
| token saliency (gradient × input) | rejected — gradient noise (Adebayo 2018) |
| retrieval instability | N/A — our ReAct uses simulated tools |
| latent routing (MoE expert switch) | N/A — Qwen-2.5-14B / Llama-3.3-70B are dense |

## Pre-registered pass criteria

| condition | threshold |
|---|---|
| paired t-test (KL_sem − KL_sur, n questions) | p < 0.01 |
| effect size: ratio of means | KL_sem / KL_sur > 1.3 |
| Normalized Edit Distance (NED) balance | |NED_sem − NED_sur| < 0.15 |

If all three pass → **extend to all 6 models, then write into paper**.
If any fails → **stop, do not write into paper**, log results in `docs/`.

## Execution recipe (on the GPU node)

```bash
# 0) make sure vllm is up on Qwen-2.5-14B (port 8000)
tmux ls   # should show 'serve' session
# if not:
tmux new -s serve
bash gpu_kit/serve_qwen14b.sh
# wait for 'Uvicorn running on http://0.0.0.0:8000'
# Ctrl-B D to detach

# 1) sanity-check vllm logprobs support
source gpu_kit/.venv/bin/activate
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5-14b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":1,"logprobs":true,"top_logprobs":5}' \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['logprobs']['content'][0]['top_logprobs'])"
# Expected: a list of 5 {'token':..., 'logprob':...} dicts

# 2) run the pilot (≈3-5 minutes, 20 questions × 6 prompts = 120 calls)
mkdir -p results/probe_decoding_kl
python3 gpu_kit/probe_decoding_kl.py \
    --variants results/runs_real_qwen25_7b_fix/gsm8k_cot_real_qwen25_7b_fix.jsonl \
    --model qwen2.5-14b \
    --base-url http://localhost:8000/v1 \
    --out results/probe_decoding_kl/qwen25_14b_gsm8k_pilot.json

# 3) push results back
git add results/probe_decoding_kl/
git commit -m "pilot: decoding KL probe on qwen25_14b / gsm8k (n=20)"
git push
```

## Then back on the control PC

```bash
git pull
python3 code/analyze_probe_kl.py results/probe_decoding_kl/qwen25_14b_gsm8k_pilot.json
```

The analyzer will print a clear PASS / MARGINAL / UNDERPOWERED / FAIL verdict.

## If pilot PASSES

Extend the same probe to all 6 models × 3 benchmarks × 2 agents (only need
first-token logprobs, no agent rollout, so this is FAST). Add a new
"Mechanism" subsection to `paper.md` with the new figure.

## If pilot FAILS (any condition)

Do NOT add a mechanism section. The paper's current "robust empirical
regularity, existing theories fail, no mechanism claim" framing is
preserved. We log the negative result in `docs/decoding_kl_negative.md`.
