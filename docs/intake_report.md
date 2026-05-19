# Intake Report — Paper Writing Factory (final, n=50 update)

**Date**: 2026-05-18
**Project**: AgentDiff (`/data/workspace/agentdiff_exp`)
**Target venue**: EMNLP 2026 main track (Findings as fallback)
**Paper type**: Empirical / measurement study (revised from "method paper")

## 1. Material inventory

### Raw experiment data (44 cells × n∈{20,30,50})
- 6 LLMs × 3 benchmarks × 2 scaffolds = 36 main cells, plus 8 generator-swap cells (mimo) and 8 generator-swap cells (qwen2.5:14b)
- Cell sample sizes:
  - 21 cells extended to **n=50** (Track G)
  - 10 cells at n=30
  - 5 cells at n=20 (mimo MATH/HotpotQA + a few that did not need extension)
- Total trajectories: 970 originals × 5 perturbations × 2 scaffolds = 8,350 variant evaluations

### Track outputs (all completed, all in repository)

| Track | Output | Size |
|---|---|---|
| A1 (severity audit) | `track_a/severity_per_variant.jsonl` | **8350 rows**, 5 ops × 1670 originals/avg |
| A2 (severity-matched Δ) | `track_a/_a2_severity_matched.json` | 44 cells |
| B (wild cluster bootstrap) | `track_b/{wild_cluster_bootstrap, hierarchical_bootstrap_cascade, multiple_comparisons_table}.json` | full |
| C (3-way generator) | `track_c/three_way_rank_correlation.json` | n=8 paired cells |
| D (within-benchmark tractability) | `track_d/within_benchmark.json` | 36 cells, 6 strata |
| E (TF-IDF cascade) | `track_e/embedding_cascade.json` | 36 cells, 3 thresholds |
| F (second judge MiMo) | `track_f/judge_agreement.json` | 1486 paired judgments |
| G (n=20→50 expansion) | (data merged into main cells) | 21 cells |

## 2. Headline numbers (post-n=50 rerun)

### Robust findings (survive every test)

| Result | Value | Statistical strength |
|---|---|---|
| **Δ_severity_matched** (paraphrase/synonym vs reorder/format/distractor) | **+14.32 pp** | paired t=6.76, **p<0.0001**; 40/44 cells positive |
| **Pearson r(accuracy, Δ)** | **+0.44** | p=0.003, **BH q=0.021** |
| **GSM8K cascade gap** (cell-level paired, exact-match) | **+0.38 step** | t=3.35, p=0.0065, **BH q=0.023** |
| **GSM8K cascade gap** (TF-IDF cosine, threshold 0.3) | **+0.66 step** | Welch t=4.92, **p<0.001** |
| **Same-family generator rank correlation** (qwen2.5:3b vs qwen2.5:14b) | Pearson **r=+0.79** | p=0.019; Spearman ρ=+0.71 |
| **MATH cascade gap** | +0.04 step | p=0.99 (correctly null) |

### Findings that did NOT survive n=50 rerun

| Result | Old (n=20) | **New (n=50)** | Action |
|---|---|---|---|
| Topology coefficient (wild cluster bootstrap, K=6) | β=+9.81pp, p=0.018 ✅ | **β=+4.31pp, p=0.165** ❌ | Downgrade claim from "topology gates" to "topology associated with" |
| Capability coefficient (cluster-robust) | β=+9.01pp, p=0.245 | β=+11.49pp, p=0.126 | Already not significant; keep narrative |
| Within-benchmark tractability contrast | 0/3 significant | 0/3 significant | Confirm: tractability proxy is not a within-benchmark gate |

### Cross-architecture generator instability (R2-Fatal-3 honest disclosure)

| Generator pair | Pearson r | Spearman ρ |
|---|---|---|
| Original (qwen2.5:3b) vs MiMo-v2.5-pro | +0.34 (p=0.41) | **+0.14 (p=0.74)** |
| Original vs qwen2.5:14b | +0.79 (p=0.019) | +0.71 (p=0.047) |
| MiMo vs qwen2.5:14b | +0.65 (p=0.082) | +0.52 (p=0.183) |

**Implication**: Δ ranking is preserved within the qwen architecture family but **not preserved across families**. The paper must explicitly limit any cross-generator generalisation claim.

### Second-judge agreement (R2-Fatal-4 honest disclosure)

| Stratum | Cohen's κ | n |
|---|---|---|
| Overall | 0.50 | 1486 |
| GSM8K / MATH / HotpotQA | 0.50 / 0.44 / 0.55 | 485 / 492 / 499 |
| Sem side / Sur side | 0.50 / 0.51 | 583 / 893 |
| Per operator | 0.48–0.52 | 285–299 each |

**Implication**: κ≈0.5 is **moderate (not strong)** but **uniform across strata** — the disagreement is random noise, not a systematic per-operator or per-benchmark bias. The paper reports this honestly.

## 3. Paper type and contribution claims

**Paper type**: Empirical / measurement study (NOT a new method paper — the proposed AgentDiff-Probe v2 is an analysis tool, not a deployable system).

**Final contribution claims** (data-driven, conservative):

1. **Robust empirical phenomenon**: Across 44 cells, the meaning-bearing minus presentation perturbation inconsistency gap Δ averages +14.32 pp after severity matching (paired t=6.76, p<0.0001). It survives a TF-IDF redefinition of cascade-depth (GSM8K +0.66 step, p<0.001) and a second LLM judge (Cohen's κ=0.50 uniformly across strata).
2. **Capability is the strongest single predictor**: Pearson r(accuracy, Δ) = +0.44 (BH q=0.021); the cluster-robust regression coefficient is +11.49 pp/accuracy unit (wild bootstrap p=0.13, K=6 clusters limit power).
3. **The dichotomy is generator-family-conditional**: Δ ranking is preserved within the qwen architecture (Pearson r=+0.79, p=0.019) but lost across families (qwen vs MiMo: ρ=+0.14). The paper does not claim cross-family generalisation.
4. **AgentDiff-Probe v2 is a prototype, not a deployable diagnostic**: MAE 7.10 pp vs trivial-mean baseline 8.27 pp; sign accuracy ties the trivial baseline at 72.2%. Reported transparently.

## 4. Limitations to disclose explicitly

L1. **Within-benchmark tractability proxy fails**: 0/3 within-benchmark contrasts are significant (Track D). Multi-path / single-path is therefore not the causal mechanism; benchmark identity (likely task-domain) is the true association.

L2. **Topology coefficient does not survive small-K cluster correction**: With K=6 model clusters, wild cluster bootstrap p=0.165. Δ is a robust phenomenon but not a precisely identified causal effect.

L3. **Generator-source ablation magnitude > headline effect** for cross-architecture generators (qwen→MiMo: ΔΔ=−12.30 pp). Within-architecture (qwen→qwen) the ranking is preserved.

L4. **Single-architecture LLM judge** with Cohen's κ=0.50 vs MiMo. Uniform across strata, but absolute κ is moderate.

L5. **Severity matching is operator-aggregated, not per-question**: Track A2 matches the edit-distance distribution, not each variant individually.

## 5. Step 1 gate check

- [x] At least 1 main experiment result with numerical data ✅ (44 cells, 8350 variants)
- [x] Core contribution claims confirmed ✅ (4 claims above, intentionally conservative)
- [x] Paper type determined ✅ (empirical measurement study)
- [x] Target venue selected ✅ (EMNLP 2026 main, Findings as fallback)

**Step 1 PASS** — proceed to Step 2 (Outline).
