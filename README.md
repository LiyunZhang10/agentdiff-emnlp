# AgentDiff: Semantic vs Surface Inconsistency in LLM Agents

> **Anonymized supplementary material accompanying an EMNLP 2026 Findings submission.**
> All identifying information (authors, affiliations, repository URLs, API keys,
> internal paths) has been removed from this snapshot.

This repository accompanies the paper **"When Do LLM Agents Treat Surface Noise
Differently from Semantic Noise? A 44-Cell Measurement Study with a Held-Out
Trace-Level Validation."** It contains all source code, raw experimental
trajectories, analysis scripts, and the camera-ready PDF.

---

## TL;DR

We run **44 (model × benchmark × scaffold) cells** across 6 model families on
3 benchmarks (GSM8K, MATH, HotpotQA), measuring how often the agent's final
answer flips when the question is rewritten in a meaning-preserving way
(`paraphrase`, `synonym`) versus a presentation-only way (`reorder`, `format`,
`distractor`). We document a **capability-gated, task-tractable dichotomy**:
once both gates open, semantic-preserving rewrites destabilize agents
substantially more than surface edits (mean **+14.32 pp** on the original
6-model panel; replicated in direction on a held-out 7th model run of 1,800
trajectories).

The full paper is in [`paper/acl/acl_paper.pdf`](paper/acl/acl_paper.pdf).

---

## Repository contents

| Path | What |
|---|---|
| `paper/acl/acl_paper.tex` + `acl_paper.pdf` | LaTeX source + camera-ready PDF (ACL Rolling Review format, anonymized) |
| `paper/acl/figs/` | All 6 main figures (vector PDF) |
| `paper/acl/references.bib` | Bibliography |
| `paper/paper.md` | Markdown source the LaTeX was generated from |
| `code/` | 25 Python scripts: experiment driver, perturbation generators, aggregation, statistical tests, figure generation |
| `data/` | The 5 task files (GSM8K test, MATH test + ablation, HotpotQA test + ablation) |
| `results/` | Raw `.jsonl` trajectories for every cell + aggregated CSV/JSON tables |
| `gpu_kit/` | Self-contained workflow for the held-out Qwen-2.5-14B run on a GPU node (vLLM serving + client) |

---

## Quick reproduction (CPU-only)

All headline numbers in the paper can be recomputed from `results/` without
running any LLM:

```bash
# 1. Aggregate the 26-cell main panel + 16-cell severity-matched extension
python3 code/aggregate_conditional.py

# 2. Re-run the merged 42-cell analysis (held-out + main panel)
python3 code/merged_analysis_42cells.py

# 3. Trace-level mechanism probes on the held-out 1,800 trajectories
python3 code/trace_mechanism_probes.py

# 4. Regenerate the 6 main figures
python3 code/make_paper_figures.py
python3 code/make_fig_mechanism.py
```

Aggregated outputs land in `results/results_conditional/` and
`results/conditional_v2/`. Figure PDFs land in `paper/acl/figs/`.

---

## Reproducing the experiments from scratch

The experiments in `results/` were collected with a free-tier API mix:

* `ollama` (local CPU) for ≤ 7 B open-weight models
* `groq` for ≤ 70 B open-weight models
* `gemini` 2.0 flash (Google AI Studio) for closed-source baseline calls
* a frontier proprietary OpenAI-compatible API for the closed-source agent

Set environment variables before running:

```bash
export GROQ_API_KEY=...        # https://console.groq.com
export GOOGLE_API_KEY=...      # https://aistudio.google.com
export MIMO_API_KEY=...        # any frontier OpenAI-compatible API
export MIMO_BASE_URL=...       # base URL for the above (optional)
```

Then either:

```bash
# Smoke-test one cell (Llama-3.2-3B / GSM8K / CoT)
python3 code/agentdiff_v2.py --model llama3.2:3b --bench gsm8k --scaffold cot

# Full 26-cell sweep (24-48 h on a 32-core CPU node, free-tier rate-limited)
python3 code/run_cross_model.py
```

The held-out Qwen-2.5-14B / 1,800-trajectory run was executed on a GPU node
running vLLM. See `gpu_kit/README_GPU.md` for the exact 5-command workflow.

---

## What is in `results/`

| Directory | Cells | Trajectories | Notes |
|---|---|---|---|
| `runs_real_llama32_1b_*` | 4 (GSM8K + HotpotQA × CoT/ReAct) | ~1 600 | Llama-3.2-1B |
| `runs_real_llama32_3b_*` | 6 + 2 gen-swap | ~3 200 | Llama-3.2-3B |
| `runs_real_llama31_8b_*` | 6 + 2 gen-swap | ~3 200 | Llama-3.1-8B |
| `runs_real_qwen25_3b_*` | 6 + 2 gen-swap | ~3 200 | Qwen-2.5-3B |
| `runs_real_qwen25_7b_*` | 6 + 2 gen-swap | ~3 200 | Qwen-2.5-7B |
| `runs_real_mistral_7b_fix` | 2 | ~800 | Mistral-7B (fix-set only) |
| `runs_real_mimo_v25_pro_*` | 6 + 2 gen-swap | ~3 200 | Frontier closed-source proprietary |
| `runs_real_qwen25_14b_vllm` | **9** (3 benchmarks × CoT/ReAct/Direct) | **1 800** | **Held-out validation, GPU vLLM** |
| `results_conditional/` | — | — | Aggregated CSV/JSON, OLS results, mechanism stats |
| `conditional_v2/` | — | — | Held-out 9-cell aggregate + merged 42-cell analysis |

---

## Anonymization notes

* `code/track_f_second_judge.py` and `code/api_router.py` previously hard-coded
  a frontier-API key; both now read from `MIMO_API_KEY` / `MIMO_BASE_URL`
  environment variables (any OpenAI-compatible endpoint works).
* All absolute paths (`/data/workspace/...`) have been replaced with relative
  paths so the scripts run from the repo root.
* No author names, affiliations, internal hostnames, or version-control
  identifiers (commit hashes from non-anonymized branches) appear in this
  snapshot.
* The `paper/acl/acl_paper.tex` already declares `\author{Anonymous ACL
  Submission}` and uses `\usepackage[review]{acl}`.

---

## License

Code: MIT (see `LICENSE`).
Paper text and figures: CC-BY 4.0 once de-anonymized after acceptance.
