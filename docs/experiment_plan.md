# AgentDiff Experiment Plan

## 1. Resource Assessment

### Machine Profile
| Resource | Available | Required | Status |
|----------|-----------|----------|--------|
| CPU | 48 cores | 32 cores | ✅ Exceeds |
| RAM | 123 GB | 60 GB | ✅ Exceeds |
| GPU | None | None | ✅ CPU-only design |
| Disk | 500 GB | 50 GB | ✅ Exceeds |
| API Budget | $0 (free tier) | $0 | ✅ Free APIs only |

### API Budget Planning
| Provider | Daily Limit | Est. Calls Needed | Days |
|----------|-------------|-------------------|------|
| Ollama (local Qwen2.5-7B) | Unlimited | ~15,000 | 2-3 days |
| Groq (Llama 3.3 70B) | ~28,800/day | ~5,000 (cross-model) | 1 day |
| Google AI Studio (Gemini 2.0 Flash) | 1,500/day | ~3,000 (cross-model) | 2 days |

**Strategy**: Use Ollama (local) as primary for all main experiments. Use Groq + Gemini for cross-model experiments only.

### RAM Estimation
| Component | RAM | Notes |
|-----------|-----|-------|
| OS + Python | 4 GB | Baseline |
| Ollama Qwen2.5-7B (Q4) | 6 GB | Primary LLM |
| Experiment scripts | 2 GB | Data + processing |
| **Total** | **12 GB** | Well within 123 GB |

## 2. Research Hypothesis

**H1 (Main)**: ≥25% of LLM agent benchmark successes are inconsistent across semantically equivalent input variants.

**H2 (Patching)**: AgentDiff's failure-mode-specific prompt patches reduce inconsistency by ≥30% without degrading baseline accuracy.

**H3 (Generalization)**: Failure modes and patches discovered on one agent/model transfer to other agents/models.

## 3. Experiment Matrix

| ID | Purpose | Dataset | Baselines | Metrics | Est. Samples | Est. Time |
|----|---------|---------|-----------|---------|---------------|-----------|
| E1 | Main: inconsistency discovery | HotpotQA (500 samples) | single-run, self-consistency, generic-robust | consistency_rate, failure_rate, patch_effectiveness | 500 × 5 variants × 3 seeds | ~8h |
| E2 | Main: inconsistency discovery | MATH L3-5 (300 samples) | single-run, self-consistency, generic-robust | consistency_rate, failure_rate, patch_effectiveness | 300 × 5 variants × 3 seeds | ~5h |
| E3 | Main: inconsistency discovery | GSM8K (200 samples) | single-run, self-consistency, generic-robust | consistency_rate, failure_rate, patch_effectiveness | 200 × 5 variants × 3 seeds | ~3h |
| E4 | Ablation: perturbation types | HotpotQA (200 samples) | remove each perturbation type | per-type failure_rate | 200 × 5 types × 3 seeds | ~4h |
| E5 | Ablation: propagation graphs | MATH (200 samples) | with/without graphs | attribution_accuracy | 200 × 2 × 3 seeds | ~2h |
| E6 | Ablation: patch granularity | HotpotQA (200 samples) | system-level vs step-level | patch_effectiveness | 200 × 2 × 3 seeds | ~2h |
| E7 | Cross-agent transfer | HotpotQA (200 samples) | ReAct vs CoT agent | transfer_success_rate | 200 × 2 agents × 3 seeds | ~3h |
| E8 | Cross-model generalization | HotpotQA (200 samples) | Qwen-7B, Llama-3.3-70B, Gemini | model-specific patterns | 200 × 3 models × 3 seeds | ~6h |

**Total estimated time**: ~33 hours (parallelizable to ~12h with multi-provider)

## 4. Baselines

| # | Baseline | Description | Implementation |
|---|----------|-------------|----------------|
| B1 | Single-run | Standard agent evaluation on original task only | Trivial |
| B2 | Self-consistency | Run agent K times on same input, majority vote | Wang et al., 2023 |
| B3 | Generic-robust | Add generic robustness instruction to system prompt | Manual prompt |
| B4 | AgentDiff (ours) | Full pipeline: variant gen → consistency analysis → targeted patches | Our method |

## 5. Evaluation Metrics

| Metric | Definition | Range |
|--------|-----------|-------|
| **Consistency Rate** | % of tasks where agent gives same answer across all variants | [0, 100] |
| **Failure Rate** | % of tasks with ≥1 inconsistent variant | [0, 100] |
| **Patch Effectiveness** | (failure_rate_before - failure_rate_after) / failure_rate_before | [0, 1] |
| **Accuracy** | % of tasks with correct final answer | [0, 100] |
| **Accuracy Preservation** | accuracy_after_patch / accuracy_before_patch | Should be ≥0.95 |

## 6. Statistical Testing

- **Primary**: Paired t-test (our method vs each baseline)
- **Secondary**: Wilcoxon signed-rank test (non-parametric)
- **Confidence**: 95% CI via bootstrap (10,000 resamples)
- **Significance threshold**: p < 0.05

## 7. Perturbation Types (for Variant Generation)

| Type | Description | Example |
|------|-------------|---------|
| **Paraphrase** | Rephrase question preserving meaning | "What is X?" → "Can you tell me X?" |
| **Reorder** | Reorder context sentences | Swap paragraph order in context |
| **Synonym** | Replace key terms with synonyms | "calculate" → "compute" |
| **Format** | Change formatting (bullets, numbering) | Numbered list → prose |
| **Distractor** | Add irrelevant but plausible context | Add unrelated fact |

## Step 1-2 Gate Check
- [x] RAM peak < 60 GB: YES (est. 12 GB)
- [x] No GPU dependency: YES
- [x] API calls within free tier: YES (primarily local Ollama)
- [x] Total runtime < 7 days: YES (est. 33h)
- [x] ≥3 baselines: YES (4 baselines)
- [x] Each experiment has dataset + metrics: YES
- [x] ≥3 ablation variants: YES (5 ablations)
- [x] Statistical testing plan: YES
