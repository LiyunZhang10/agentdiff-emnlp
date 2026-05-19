# AgentDiff: Semantic vs Surface Inconsistency in LLM Agents

> **EMNLP 2026 Findings submission (in preparation).**
> An empirical study of how meaning-preserving lexical rewrites destabilize
> multi-step LLM agents more than presentation-level surface edits — across
> benchmarks, scaffolds, and model families.

---

## 🎯 What this repo contains

| Path | What | Size |
|---|---|---|
| [`paper/`](paper/) | The paper itself: `paper.md` (source), `Paper_EN.docx`, `Paper_EN.pdf`, `Paper_ACL.tex`, `Paper_ACL.pdf`, all figures | ~6 MB |
| [`code/`](code/) | All experiment + analysis scripts (22 files) | ~320 KB |
| [`data/`](data/) | The 5 task files (GSM8K, MATH, HotpotQA test/ablation) | ~500 KB |
| [`results/`](results/) | All raw run outputs already produced (27 model × benchmark cells) | ~63 MB |
| [`docs/`](docs/) | Design docs: `EMNLP_FINDINGS_ROADMAP.md`, `PAPER_OUTLINE_v3.md`, `intake_report.md`, `experiment_plan.md` | ~40 KB |
| [`gpu_kit/`](gpu_kit/) | **Self-contained kit to run the missing 70B / heavy experiments on a GPU node** | small |

Total checked in: ~70 MB (well below GitHub's 1 GB recommended limit, no LFS needed).

---

## 🧪 Headline finding (one paragraph)

Across **3 benchmarks** (GSM8K, MATH, HotpotQA) × **8 model families**
(Llama-3.2 1B/3B, Llama-3.1 8B, Qwen-2.5 3B/7B, Mistral 7B, Gemma-2 9B,
MiMo-v2.5-pro), we ran 19 controlled perturbations per cell and measured
*semantic-preserving lexical rewrites* (`syn_swap`, `paraphrase`,
`negation_double`, …) versus *surface-only edits* (`whitespace`, `case`,
`punct`, …). We find a **robust empirical regularity**: meaning-preserving
lexical rewrites consistently produce larger answer-instability deltas than
presentation edits, and this gap **scales positively with task tractability
and model capability** — a *capability-gated, task-tractable dichotomy*.
Existing tractability- and topology-based explanations fail to predict it.

> **Capability-Gated AND Task-Tractable Dichotomy** (26 cells aggregated):
> Pearson r = +0.37, p = 0.050; OLS slope = +17.7 pp / acc unit.
> At acc ≥ 0.65, **8/8 capable cells** show Δ > 0 (mean +14.6 pp,
> Fisher exact p = 0.0016); below threshold, only 5/18 (mean −1.2 pp).
> Mechanism: χ² = 9.93, p = 0.042 over 5-pattern distribution; semantic
> perturbations are **3× less likely to self-correct** (0.8 % vs 2.6 %,
> p = 0.005) and diverge **0.11 step earlier** (paired *t* = −2.30, p = 0.021).

---

## 📁 Repository structure

```
agentdiff-emnlp/
├── README.md                       ← you are here
├── LICENSE                         (MIT)
├── .gitignore
│
├── paper/                          ← the manuscript & camera-ready figures
│   ├── paper.md                    ← canonical source (markdown, ~48 KB)
│   ├── Paper_EN.docx               ← Word export
│   ├── Paper_EN.pdf                ← PDF export
│   ├── Paper_ACL.tex               ← ACL/EMNLP LaTeX source
│   ├── Paper_ACL.pdf
│   ├── figures/                    ← 5 main figures (PDF + PNG)
│   ├── paper_figs_v2/              ← alternative figure set
│   └── figs_v3/                    ← supplementary heatmaps / bars
│
├── code/                           ← reproducible pipeline
│   ├── agentdiff_v2.py             ← main perturbation+agent driver
│   ├── agentdiff_probe.py
│   ├── run_cross_model.py          ← cross-family evaluation orchestrator
│   ├── api_router.py               ← unified ollama/groq/openai-compat router
│   ├── make_paper_figures.py       ← regenerates figs from results/
│   ├── make_paper_figs_n50.py
│   ├── md_to_docx.py               ← paper.md → docx with embedded images
│   ├── plot_dichotomy_heatmap.py
│   ├── analyze_propagation_dichotomy.py
│   ├── aggregate_conditional.py    ← builds 26-cell aggregate
│   ├── aggregate_fix_models.py
│   ├── checklist_vs_agentdiff.py
│   ├── sanity_judge.py
│   ├── track_a_severity_audit.py        ← Track A: severity-controlled
│   ├── track_a2_severity_matched.py
│   ├── track_b_robust_inference.py      ← Track B: scaffold robustness
│   ├── track_c_family_cluster.py        ← Track C: family clustering
│   ├── track_c_genrank.py
│   ├── track_d_embedding_severity.py    ← Track D: embedding severity
│   ├── track_d_within_benchmark.py
│   ├── track_e_embedding_cascade.py     ← Track E: cascade analysis
│   └── track_f_second_judge.py          ← Track F: second-judge robustness
│
├── data/                           ← task sets
│   ├── gsm8k_test.jsonl
│   ├── math_test.jsonl             ← deep-math
│   ├── math_ablation.jsonl
│   ├── hotpotqa_test.jsonl
│   └── hotpotqa_ablation.jsonl
│
├── results/                        ← 63 MB of already-produced runs
│   ├── results_conditional/        ← aggregate analysis outputs
│   ├── runs_real_llama32_1b_*/     ← Llama-3.2 1B (fix + hpqa)
│   ├── runs_real_llama32_3b_*/     ← Llama-3.2 3B
│   ├── runs_real_llama31_8b_*/     ← Llama-3.1 8B
│   ├── runs_real_qwen25_3b_*/      ← Qwen-2.5 3B
│   ├── runs_real_qwen25_7b_*/      ← Qwen-2.5 7B
│   ├── runs_real_mistral_7b_*/     ← Mistral 7B (partial)
│   └── runs_real_mimo_v25_pro_*/   ← MiMo-v2.5-pro (frontier API)
│
├── docs/                           ← internal planning notes
│   ├── EMNLP_FINDINGS_ROADMAP.md
│   ├── PAPER_OUTLINE_v3.md
│   ├── intake_report.md
│   ├── experiment_plan.md
│   └── OVERNIGHT_PLAN_2026-05-13.md
│
└── gpu_kit/                        ← run heavy experiments on a GPU node
    └── README_GPU.md               ← copy-paste workflow (see below)
```

---

## 🚀 Quick reproduction (light, no GPU)

```bash
git clone <THIS_REPO> agentdiff-emnlp
cd agentdiff-emnlp

# 1. regenerate all figures from existing results/
python3 code/make_paper_figures.py

# 2. regenerate the 26-cell aggregate analysis
python3 code/aggregate_conditional.py

# 3. rebuild Word + PDF from paper.md
python3 code/md_to_docx.py paper/paper.md -o paper/Paper_EN.docx
```

Everything in `paper/` and `results/` is already produced; the scripts above
just verify reproducibility.

---

## 🖥 Running the heavy experiments on a GPU node

The experiments **already in `results/`** were run via free-tier APIs
(Ollama local + Groq + Gemini + MiMo). To extend to **larger open-weight
models that the API tier can't host** (Qwen-2.5-14B / Llama-3.3-70B / etc.),
use [`gpu_kit/`](gpu_kit/).

The kit is designed for the following workflow:

```text
  ┌──────────────┐  git clone   ┌───────────────┐  vllm serve   ┌────────┐
  │  this repo   │ ───────────► │   GPU node    │ ────────────► │ models │
  │ (control PC) │              │ (2× H20 96GB) │               └────────┘
  └──────────────┘              └───────┬───────┘
         ▲                              │ python client_run.py
         │   git push results/_new      ▼
         │                       results/_gpu_new/
         └─── git pull ─────────────────┘
```

See [`gpu_kit/README_GPU.md`](gpu_kit/README_GPU.md) for the exact 5-command
recipe to clone, serve, run, and push results back.

---

## 📌 What still needs to be added (open work)

These are the gaps the GPU node will close:

1. **Qwen-2.5-14B** — 14 B reference model, fix benchmark (3 perturbation × 200 q)
2. **Llama-3.3-70B** — frontier open model, fix benchmark
3. **Gemma-2 9B** — additional family beyond Llama / Qwen / Mistral
4. **Mistral 7B full sweep** — current run is smoke only
5. **HotpotQA cross-generator transfer** — `genqwen14b` / `genmimo` columns

---

## 🔖 Citation

```bibtex
@inproceedings{agentdiff2026,
  title  = {When Do LLM Agents Treat Surface Noise Differently from Semantic Noise?},
  author = {Anonymous},
  booktitle = {Findings of EMNLP},
  year   = {2026},
  note   = {Under review}
}
```
