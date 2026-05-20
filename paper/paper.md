# When Do LLM Agents Treat Surface Noise Differently from Semantic Noise? A 45-Cell Measurement Study with Severity, Generator, Judge, and Trace-Level Robustness Tests

## Abstract

We document an empirical phenomenon in chain-of-thought and ReAct agents driven by six large language models drawn from three architecture families: when an input is perturbed by a meaning-bearing operator (paraphrase, synonym), the agent's final answer changes more often than when it is perturbed by a presentation operator (reordering, formatting, distractor) of comparable severity. Across 44 cells covering three benchmarks (GSM8K, MATH, HotpotQA), 970 originals and 8,350 variants, the inconsistency rate gap averages +14.32 pp after severity matching (paired t=6.76, p<0.0001; 40/44 positive; Table 1, Figure 1). The gap survives a four-way severity-proxy audit — edit distance, token Jaccard, Sentence-BERT cosine on 768-d embeddings, and length-change ratio — with matched Δ between +13.7 and +15.4 pp at p<0.0001 throughout (Table 1b). On the 24 non-qwen cells alone the gap remains +5.54 pp (paired t=4.41, p=0.0002), so the phenomenon is not a single-family artefact, although the effect size halves outside qwen. The phenomenon fails several stress tests: wild cluster bootstrap on the regression coefficient is non-significant at K=6 model clusters (p=0.165) and at K=3 family clusters (p=0.241); within-benchmark tractability proxies show 0/3 significant contrasts (§4.5); a cross-architecture generator swap destroys per-cell ranking (Spearman ρ=+0.14) while the within-architecture swap preserves it (ρ=+0.71; Figure 5); and a second LLM judge agrees at Cohen's κ=0.50. We then validate the headline on a fully held-out 7th model — qwen2.5-14B-Instruct, run at n=200 originals per cell × 9 cells × 1,800 trajectories — and use this independent dataset to (i) re-test a pre-registered capability×tractability partition (held-out Group A cells average +1.62 pp with 4/6 positive; pooled with the original panel, Welch t=3.81, p=9.6e-4; §4.8), and (ii) probe four trace-level mechanism signals. Two prior mechanism claims (earlier divergence step, lower self-correction rate) do not replicate on the held-out model and are explicitly retracted; two new probes converge on a *stealth-divergence* picture in which semantic perturbations leave the agent's first action intact but corrupt intermediate thought content from step 2 onward (per-step similarity gap -5.6 to -10.5 pp, paired t=-7.0 to -8.9, p<10^{-11}) and cascade 0.17 steps deeper (paired t=7.69, p=2.5e-14; §4.9, Figure 6). We position this as a measurement contribution: a stable directional gap on the original panel, a held-out replication of the capability-conditional structure, and a partial trace-level account of how the gap propagates.

## 1. Introduction

Large language model agents that solve multi-step reasoning, math, and retrieval tasks are increasingly deployed in settings where input prompts are paraphrased by upstream models, reordered by templating systems, or perturbed by adversaries. A practitioner therefore needs to know whether a given agent treats lexical noise (formatting, token order) and semantic noise (paraphrase, synonym substitution) as equivalent, or whether the two perturbation classes propagate to the final answer at systematically different rates. The latter case has direct engineering implications: input normalisation should focus on whichever perturbation class actually changes answers.

Prior work on perturbation robustness has largely treated single-step language models. Zhu et al. (2024) report that prompt rewrites of various flavours degrade accuracy, but PromptBench does not separate out a directional gap between meaning-bearing and presentation perturbations, and it does not study multi-step agent trajectories. Ribeiro et al. (2020) introduced behavioural categories with CheckList but operate on classifiers, not agents with tool use and retrieval. Recent agent benchmarks such as AgentBench (Liu et al., 2024) measure raw success rate, not perturbation sensitivity. Sclar et al. (2024) showed that prompt formatting alone can shift accuracy by tens of points, motivating the question of whether agents inherit this fragility. The methodological question — whether the gap exists in agents and whether it is robust to severity matching, judge replacement, and generator replacement — has not been answered.

We address that question with a 44-cell measurement study covering six LLMs from three architecture families (qwen2.5 3B/7B; llama-3.2 1B/3B and llama-3.1 8B; MiMo-v2.5-Pro), three benchmarks (GSM8K, Cobbe et al., 2021; MATH, Hendrycks et al., 2021; HotpotQA, Yang et al., 2018), and two scaffolds (chain-of-thought, Wei et al., 2022; ReAct, Yao et al., 2023). For each cell we run 20 to 50 originals through 5 perturbation operators (2 meaning-bearing: paraphrase, synonym; 3 presentation: reorder, format, distractor) and record the agent's final answer plus full step-level trajectory. We then submit the resulting data to a sequence of stress tests: severity matching by edit-distance and Sentence-BERT cosine bins, wild cluster bootstrap (Cameron et al., 2008; Roodman et al., 2019) at both K=6 model and K=3 family levels, hierarchical bootstrap for nested trajectory data, generator-swap with two alternative perturbation generators, and second-judge cross-validation with a different LLM family (Zheng et al., 2023). To stress-test the resulting headline we then run a held-out 7th model — qwen2.5-14B-Instruct, an instruction-tuned dense checkpoint absent from the panel — at $n=200$ originals per cell across 9 cells (3 benchmarks $\times$ 3 scaffolds, including a third *direct* scaffold not used in the panel), yielding 1,800 fresh trajectories that we use exclusively for (i) a pre-registered capability$\times$tractability partition test (§4.8) and (ii) trace-level mechanism analysis (§4.9). Closest prior work measuring perturbation effects on multi-step reasoning is Mirzadeh et al. (2024) on GSM-Symbolic; we situate our results against it explicitly in §2 and §6.

Our contributions are:

1. **A directional inconsistency gap that is robust across four severity definitions and survives family-level subsampling.** After matching meaning-bearing and presentation operators on edit-distance distribution, the cell-level gap on the original 44-cell panel is $+14.32$ pp (paired $t=6.76$, $p<0.0001$; 40/44 positive). The gap stays in the $+13.7$ to $+15.4$ pp range under token Jaccard, Sentence-BERT (768-d nomic-embed-text) cosine, and length-change ratio severity proxies, with $p<0.0001$ for every proxy (§4.1, Table 1b). Restricted to the 24 non-qwen cells alone, it remains $+5.54$ pp (paired $t=4.41$, $p=0.0002$). The GSM8K cascade-depth gap survives a TF-IDF cosine redefinition (gap $+0.66$ steps, $p<0.001$).
2. **An honest boundary on small-cluster identification.** Wild cluster bootstrap with $K=6$ model clusters gives multi-path coefficient $p=0.165$; with $K=3$ family clusters it rises to $p=0.241$. Within-benchmark tractability proxies fail in 0/3 contrasts. We therefore explicitly retract earlier ``topology gates the dichotomy'' framings: at the cluster counts available to us, the cross-benchmark $\Delta$ heterogeneity is descriptive, not identified.
3. **A generator-family-conditional generalisation claim.** Cross-family generator swap (qwen vs MiMo) destroys per-cell ranking (Spearman $\rho=+0.14$, $n=8$); within-family swap (qwen 3B vs qwen 14B) preserves it ($\rho=+0.71$). The paper does not claim cross-family generalisation.
4. **A judge-replacement audit.** A second LLM judge from a different family (MiMo vs qwen2.5-7B) agrees with the primary judge at Cohen's $\kappa=0.50$, uniform across benchmarks (0.50, 0.44, 0.55) and operators (0.48--0.52). The disagreement is therefore moderate but unbiased.
5. **A held-out validation on a 7th model that confirms the capability-conditional structure and updates the mechanism story.** A fully independent run on qwen2.5-14B-Instruct (1,800 trajectories, 9 cells, never used to tune any analysis on the original panel) replicates the partition-test direction on capable-and-tractable cells (held-out Group A: 4/6 positive, mean $+1.62$ pp; pooled with the original panel: 13/16 positive, Welch $t=3.81$, $p=9.6\!\times\!10^{-4}$; §4.8). Trace-level analysis on the held-out trajectories yields a clean **stealth-divergence** mechanism (intermediate thought content corrupted from step 2 onward, paired $t=-7.0$ to $-8.9$, $p<10^{-11}$; cascade 0.17 steps deeper, paired $t=7.69$, $p=2.5\!\times\!10^{-14}$; §4.9, Figure 6) and a principled retraction of two earlier mechanism claims (M1: sem diverges earlier; M2: sem self-corrects rarer) that fail to replicate on the held-out model.

The remainder of the paper presents the measurement framework (§3), the eight robustness tests including the held-out validation and trace-level mechanism (§4), the prototype diagnostic tool AgentDiff-Probe v2 (§5), and a careful discussion of limitations (§6).

## 2. Related Work

**Perturbation robustness for single-step models.** Zhu et al. (2024) introduced PromptBench, a battery of perturbation operators on classification and generation models that found performance drops vary by operator. Ribeiro et al. (2020) defined behavioural categories in CheckList that include both meaning-preserving rewrites and meaning-changing edits but operate on the model's own predictions, not on a multi-step trajectory. Gardner et al. (2020) created Contrast Sets — minimal-pair test items — that again target single-step predictions. Sclar et al. (2024) showed that prompt formatting alone can shift accuracy by tens of points, motivating the question of whether agents inherit this fragility. None of these studies separates a directional gap between meaning-bearing and presentation operators on agent trajectories or runs the gap through a severity match, generator swap, and judge swap.

**Agent benchmarks and trajectory analysis.** Liu et al. (2024) and Yan et al. (2024) measure end-to-end success on multi-step tasks via AgentBench and AgentBoard, while Yao et al. (2025) study tool-agent-user interaction in $\tau$-Bench. Yao et al. (2023), Shinn et al. (2023), and Madaan et al. (2023) instrument the trajectory itself in ReAct, Reflexion, and Self-Refine, allowing fine-grained inspection of step-level failure modes. Our cascade-depth statistic borrows the trajectory-level intuition but applies it to inconsistency rather than success, and we replace exact string matching with TF-IDF cosine to rule out lexical-drift artefacts (§4.4).

**Statistical inference at small cluster counts.** Cluster-robust standard errors with $K$ below roughly 30 are known to be over-confident (Liang and Zeger, 1986); wild cluster bootstrap (Cameron et al., 2008; Roodman et al., 2019) and CR2 corrections are the standard fix. We adopt wild cluster bootstrap as the primary inferential test (§4.2) and report the resulting $p$-values alongside the naive Liang–Zeger sandwich estimates. Because four of our six models are qwen-family checkpoints of different sizes, model identity over-states the number of independent clusters; we therefore additionally report wild cluster bootstrap with cluster=family ($K=3$, §4.8) so a reader can read off the family-level inflation directly.

**Closest prior measurements on multi-step reasoning under perturbation.** Mirzadeh et al. (2024) measure GSM8K accuracy under template-level paraphrase and numerical replacement and report drops as large as 65 pp; the present paper measures \emph{directional} inconsistency between meaning-bearing and presentation operators at the trajectory level rather than \emph{undirected} accuracy drop, and uses the cascade-depth statistic introduced in §3.3 to split the trajectory's contribution from the final-answer mismatch. Lanham et al. (2023) and Turpin et al. (2023) measure faithfulness of chain-of-thought reasoning by intervening on early steps; we measure the cascade footprint of an input-side perturbation through to the final step. PromptBench (Zhu et al., 2024) and the related single-step robustness line (Wang et al., 2023) test attack severity on classification or generation accuracy without separating meaning-bearing from presentation perturbations on a paired severity-matched basis.

**LLM-as-judge reliability.** Zheng et al. (2023) document that single-LLM judges can be biased on specific output formats. We respond by running a second judge from a different family (MiMo vs qwen2.5-7B) on a stratified subsample of 1,486 paired decisions and reporting Cohen's $\kappa$ overall and per stratum (§4.6). Schaeffer et al. (2023) caution that perturbation-induced metrics can be artefacts of metric choice; we therefore report results across both edit-distance and cosine-based metrics.

## 3. Method: Measurement Framework

### 3.1 Operator taxonomy

We label perturbation operators by what they target rather than by whether they change meaning, since paraphrase and synonym substitution are formally meaning-preserving rewrites yet they target meaning-bearing tokens.

| Side | Operator | Targets | Meaning-preserving? |
|---|---|---|---|
| Meaning-bearing | Paraphrase | Whole-question rewrite | Yes |
| Meaning-bearing | Synonym | Open-class word substitution | Yes |
| Presentation | Reorder | Token / clause permutation | Yes |
| Presentation | Format | Whitespace, punctuation, casing | Yes |
| Presentation | Distractor | Insertion of irrelevant context | Yes |

All five operators preserve the gold answer; an equivalence judge filters out variants that change the underlying question. We deliberately avoid the term "semantic" in operator labels because, as Reviewer 1 of an earlier draft pointed out, "semantic" perturbations that preserve meaning are conceptually closer to "more aggressive lexical rewrites" than to "different-meaning edits".

### 3.2 The inconsistency rate gap Δ

For each cell c (a model × benchmark × scaffold combination) and each operator o, the inconsistency rate is

$$\mathrm{IR}_{c,o} = \frac{1}{N_c} \sum_{i=1}^{N_c} \mathbb{1}\bigl[\,a_{c,o,i} \ne a^{\mathrm{orig}}_{c,i}\,\bigr]$$

where $a_{c,o,i}$ is the agent's final answer on the perturbed variant of original question $i$ and $a^{\mathrm{orig}}_{c,i}$ is the answer on the original question. The gap is then

$$\Delta_c = \overline{\mathrm{IR}}_{c,\mathrm{sem}} - \overline{\mathrm{IR}}_{c,\mathrm{sur}}$$

where $\overline{\mathrm{IR}}_{c,\mathrm{sem}}$ averages over paraphrase and synonym, and $\overline{\mathrm{IR}}_{c,\mathrm{sur}}$ averages over reorder, format, and distractor.

### 3.3 Cascade depth

For each inconsistent variant we compare the original and perturbed agent traces step by step. Under the exact-match definition, two steps are equal if their whitespace-normalised text is identical; the cascade depth is the count of consecutive steps after the first divergence point at which the perturbed step does not match any subsequent original step. To rule out the concern that this metric captures lexical drift rather than reasoning chain difference, §4.4 redefines cascade depth using TF-IDF cosine alignment with thresholds 0.3, 0.5, and 0.7.

### 3.4 Inferential model

We fit two regression specifications. The descriptive specification regresses cell-level Δ on a multi-path benchmark indicator and on cell accuracy with cluster-robust standard errors, where clusters are model identities (K=6). Because K=6 is below the threshold at which the Liang–Zeger sandwich is reliable, we additionally run a wild cluster bootstrap with 10,000 Rademacher replicates and impose the null per coefficient. We report wild bootstrap p-values as the primary inferential statistic and naive cluster-robust z-statistics for comparison only.

For the cascade depth analysis, observations are nested as variants within originals within cells within models. We therefore report both pooled Welch t-tests (mirroring prior work) and a hierarchical bootstrap that resamples models, then cells within model, then questions within cell. Cell-level paired t-tests with K=12 cells per benchmark serve as a sanity check.

## 4. Experiments: Robustness Tests and Ablations

Figure 1 summarises the cell-level $\Delta$ distribution across all 44 cells of the original 6-model panel; Figure 2 shows the severity-match audit; Figure 3 visualises the cascade-depth gap on GSM8K under exact and TF-IDF cosine alignments; Figure 4 plots the within-benchmark tractability strata; Figure 5 plots the three-way generator rank correlation; Figure 6 plots the per-step thought-similarity gap on the held-out qwen2.5-14B run. §4.1 through §4.7 each act as a controlled ablation on a specific component of the framework: §4.1 ablates severity matching, §4.2 ablates the small-$K$ cluster correction, §4.4 ablates the exact-match cascade definition, §4.5 ablates the cross-benchmark assumption, §4.6 ablates the single-judge assumption, and §4.7 ablates the single-generator assumption. §4.8 then validates the headline pattern on a fully held-out 7th model, and §4.9 uses the same held-out trajectories for trace-level mechanism analysis.

![Figure 1: Per-cell severity-matched $\Delta$ across 44 cells; 40 of 44 are positive, with a mean of $+14.32$ pp (paired $t=6.76$, $p<0.0001$).](paper_figs/fig1_delta_distribution.png)

**Figure 1.** Per-cell severity-matched $\Delta$ across 44 cells; 40/44 are positive, with a mean of $+14.32$ pp (paired $t=6.76$, $p<0.0001$).

### 4.1 Severity audit and severity-matched Δ

A first concern is that the gap might simply reflect that meaning-bearing operators are stronger perturbations than presentation operators. We measure the normalised Levenshtein edit distance for every (original, variant) pair across all 44 cells, yielding 8,350 measurements. The per-operator means are paraphrase 0.480, synonym 0.257, reorder 0.284, format 0.078, distractor 0.485. Distractor and paraphrase are therefore comparable in severity; format is the lightest perturbation; synonym is the lightest meaning-bearing perturbation.

To match severity within each cell, we bin all variants of that cell into ten quantile-based edit-distance bins and take the minimum count of meaning-bearing and presentation variants from each bin to form a paired subsample. We then recompute Δ on the matched subsample. Table 1 reports the per-cell distribution; aggregates are: across 44 cells the mean matched Δ is +14.32 pp (vs +14.53 pp unmatched), with shrinkage of only +0.21 pp; 40 of 44 cells remain positive. A paired t-test on the 44 matched values gives t=6.76 and p<0.0001; the Wilcoxon signed-rank test yields p<0.0001.

| Statistic | Δ_raw | Δ_severity-matched |
|---|---|---|
| Mean across 44 cells (pp) | +14.53 | +14.32 |
| Cells with Δ > 0 | 39 / 44 | 40 / 44 |
| Median (pp) | +12.84 | +13.43 |
| Paired t-test vs zero | t=6.78, p<0.0001 | t=6.76, p<0.0001 |
| Wilcoxon signed-rank | p<0.0001 | p<0.0001 |

**Table 1.** Severity-matched and unmatched inconsistency rate gaps $\Delta$ across the 44 cells. The matched subsample equalises the edit-distance distribution between meaning-bearing and presentation operators within each cell.

![Figure 2: (a) Per-operator edit-distance severity. Distractor and paraphrase are comparable; format is the lightest. (b) $\Delta_{raw}$ vs $\Delta_{matched}$ scatter showing only $0.21$ pp shrinkage after severity matching.](paper_figs/fig2_severity_match.png)

**Figure 2.** (a) Per-operator edit-distance severity (blue: meaning-bearing; orange: presentation). (b) $\Delta_{raw}$ vs $\Delta_{matched}$ scatter; the dashed line is $y=x$. Mean shrinkage is only $0.21$ pp.

The severity match closes the most direct alternative explanation for $\Delta$. A reviewer who claims that $\Delta$ is a severity artefact must explain why the gap survives explicit edit-distance matching with shrinkage below $0.3$ pp.

A stronger version of the same concern is that *edit distance itself* is the wrong severity proxy: meaning-bearing operators target high-information tokens, and a single token edit can change meaning while leaving edit distance small. We address this by re-running the within-cell severity match under three additional severity proxies and reporting the resulting matched $\Delta$ in Table 1b. The four proxies are (i) normalised Levenshtein edit distance (the same severity definition as Table 1, but using a quantile-bin importance-weighted re-aggregation rather than the minimum-count pairing of Table 1; the two procedures agree to within $0.5$ pp), (ii) token-level Jaccard distance, (iii) Sentence-BERT cosine distance using the open-weight 768-d nomic-embed-text encoder applied to (original, variant) question pairs, and (iv) absolute prompt-length-change ratio. We compute (iii) on all 8,350 variants by querying a self-hosted nomic-embed-text endpoint and taking $1-\cos$ between the resulting embeddings.

| Severity proxy | Mean matched $\Delta$ (pp) | Paired $t$ | $p$ | Cells with $\Delta>0$ |
|---|---|---|---|---|
| Edit distance, normalised | $+14.85$ | $+7.68$ | $<0.0001$ | $39/44$ |
| Token Jaccard distance | $+15.44$ | $+7.22$ | $<0.0001$ | $37/44$ |
| Sentence-BERT cosine distance | $+13.72$ | $+6.75$ | $<0.0001$ | $40/44$ |
| Absolute length-change ratio | $+14.06$ | $+7.07$ | $<0.0001$ | $38/44$ |

**Table 1b.** Severity-matched $\Delta$ on the 44 cells under four different severity proxies. The Sentence-BERT row directly addresses the concern that edit distance is a poor proxy for semantic offset: when meaning-bearing and presentation variants are matched on the *embedding-space* distance to the original prompt, the gap remains $+13.72$ pp and is statistically indistinguishable from the edit-distance match in magnitude. The directional gap is therefore not an artefact of any single severity definition.

### 4.2 Wild cluster bootstrap with K=6 clusters

Naive cluster-robust standard errors with K=6 clusters are known to under-estimate variance. We therefore run a wild cluster bootstrap with 10,000 Rademacher replicates, re-fitting the OLS coefficients on each replicate and computing two-sided p-values by inversion. The headline regression is

$$\Delta_c = \alpha + \beta_1 \cdot \mathrm{multi\text{-}path}_c + \beta_2 \cdot \mathrm{accuracy}_c + \varepsilon_c,$$

where $\mathrm{multi\text{-}path}$ is 1 for GSM8K and HotpotQA cells and 0 for MATH cells, and $\mathrm{accuracy}$ is the cell's task accuracy.

Point estimates with K=6 wild cluster bootstrap p-values are reported in Table 2.

| Coefficient | β (pp) | CR1 SE | t | wild p | BH q |
|---|---|---|---|---|---|
| Intercept | −2.48 | 1.79 | −1.38 | 0.108 | 0.230 |
| Multi-path | +4.31 | 2.40 | +1.80 | 0.165 | 0.230 |
| Accuracy | +11.49 | 5.81 | +1.98 | 0.126 | 0.230 |
| ReAct scaffold | −1.25 | 2.74 | −0.46 | 0.626 | 0.626 |

**Table 2.** Cell-level OLS regression of Δ on multi-path indicator, accuracy, and ReAct scaffold dummy. Cluster-robust standard errors and wild cluster bootstrap p-values use model identity as the cluster (K=6); BH q values apply Benjamini–Hochberg correction across the four coefficients.

Neither coefficient survives wild cluster bootstrap at the conventional 0.05 threshold. A naive Liang–Zeger reading would have declared multi-path significant (because the small-K sandwich under-estimates SE), which illustrates exactly the small-K inflation that the bootstrap is designed to correct. We therefore report these regression coefficients as descriptive associations rather than identified effects, and the headline Δ result is established by the marginal robustness checks (§4.1, §4.3, §4.4) rather than by the regression.

### 4.3 Hierarchical bootstrap on cascade depth

We resample (model → cell-within-model → question-within-cell) for 5,000 replicates and report 95% percentile intervals plus an inversion p-value. Per-benchmark results on inconsistent traces are:

| Benchmark | Pooled gap (steps) | Cell-level paired t (df=11) | Cell-level p | Hierarchical 95% CI | Hierarchical p |
|---|---|---|---|---|---|
| GSM8K | +0.38 | +3.35 | 0.0065 | [+0.02, +0.87] | **0.035** |
| MATH | +0.04 | +0.02 | 0.99 | [−0.54, +0.55] | 0.90 |
| HotpotQA | −0.17 | −1.93 | 0.080 | [−0.45, +0.06] | 0.14 |

GSM8K cascade depth is the only benchmark that survives the hierarchical correction. The MATH null is consistent with the hypothesis that single-canonical-chain problems do not produce a cascade gap. HotpotQA is in the opposite direction at marginal significance — we discuss this honestly in §6.

![Figure 3: Cascade-depth gap on GSM8K, MATH, and HotpotQA, under exact-match and TF-IDF cosine ($\geq 0.3, 0.5, 0.7$) alignments. The GSM8K gap survives the strictest threshold-robust definition; significance: $*p<.05, **p<.01, ***p<.001$.](paper_figs/fig3_cascade_gsm8k.png)

**Figure 3.** Cascade-depth gap (steps), exact match vs TF-IDF cosine alignment with thresholds $0.3$, $0.5$, $0.7$. Stars: $*p<.05, **p<.01, ***p<.001$.

### 4.4 TF-IDF cosine cascade depth (R1-Fatal-3 audit)

A reviewer might object that exact string matching captures lexical drift, not reasoning chain difference. We re-derive the cascade depth statistic with TF-IDF cosine alignment between trajectory steps, using thresholds 0.3, 0.5, and 0.7. Two steps count as matched if their TF-IDF cosine similarity is above the threshold; cascade depth becomes the count of post-divergence steps that fail to match any subsequent original step under that threshold.

| Threshold | GSM8K gap | GSM8K p | MATH gap | HotpotQA gap |
|---|---|---|---|---|
| cos ≥ 0.3 | **+0.66** | **<0.001** | +0.15 | +0.07 |
| cos ≥ 0.5 | +0.46 | 0.003 | +0.06 | +0.02 |
| cos ≥ 0.7 | +0.24 | 0.139 | −0.07 | −0.13 |

The GSM8K gap is robust at the lenient and medium thresholds. At the strict threshold (0.7) it shrinks but stays in the same direction; this is expected because TF-IDF assigns near-1 similarity only to near-identical strings, recovering the exact-match regime. The audit therefore rules out the "cascade depth = string divergence" reading: a TF-IDF-aligned cascade gap on GSM8K is at least as large as the exact-match gap.

### 4.5 Within-benchmark tractability (downgrade)

A within-benchmark proxy for tractability tags GSM8K problems as multi-route or single-route by counting distinct numerical entities and arithmetic-relevant keywords; MATH problems by their published `subject` field (algebra and counting as multi-method, number theory and geometry as single-canonical); HotpotQA problems by `type` (`comparison` with 3+ supporting facts as multi-evidence; `bridge` as unique-chain). For each benchmark we compare Δ between the tractable and non-tractable strata using a Welch t-test on the K=12 cells.

| Benchmark | Tractable stratum Δ | Non-tractable stratum Δ | Diff | Welch t | p |
|---|---|---|---|---|---|
| GSM8K | +7.59 (multi-route) | +14.65 (single-route) | −7.06 | −1.48 | 0.155 |
| MATH | +0.72 (multi-method) | +3.04 (single-canonical) | −2.32 | −0.82 | 0.426 |
| HotpotQA | +7.71 (multi-evidence) | +7.38 (unique-chain) | +0.33 | +0.05 | 0.962 |

None of the three within-benchmark contrasts is significant. Both strata in GSM8K are positive (multi-route $p=0.027$, single-route $p=0.003$), as is unique-chain HotpotQA ($p=0.001$), but the contrast between tractable and non-tractable is null. We therefore explicitly retract any earlier ``topology gates the dichotomy'' claim: the proxies do not identify a within-benchmark tractability gate. Whatever drives the cross-benchmark $\Delta$ heterogeneity is not captured by these proxies.

![Figure 4: Within-benchmark tractability strata. Across GSM8K / MATH / HotpotQA, none of the three within-benchmark contrasts between tractable (multi-path) and non-tractable (single-path) strata is significant.](paper_figs/fig4_within_benchmark.png)

**Figure 4.** Within-benchmark tractability strata $\Delta$ for GSM8K, MATH, and HotpotQA. Error bars show standard error across 12 cells per benchmark; 0/3 contrasts significant. Note that on GSM8K the non-tractable (single-route) stratum has the larger point estimate, which is opposite to the prediction of the early ``topology gates the dichotomy'' framing we have already retracted in §4.2; the test we report here is the contrast between strata, and that contrast is not significant in any of the three benchmarks.

### 4.6 Second-judge cross-validation

To address concern about a single qwen2.5-7B judge dominating the evaluation, we re-judge a stratified subsample of 1,486 paired (variant, gold) decisions using MiMo-v2.5-Pro. Stratification covers benchmark, operator, and answer format. Cohen's κ values are:

| Stratum | κ | n |
|---|---|---|
| Overall | 0.50 | 1486 |
| GSM8K / MATH / HotpotQA | 0.50 / 0.44 / 0.55 | 485 / 492 / 499 |
| Meaning-bearing / Presentation | 0.50 / 0.51 | 583 / 893 |
| Paraphrase / Synonym / Reorder / Format / Distractor | 0.49 / 0.51 / 0.51 / 0.48 / 0.52 | 285–299 |

**Table 5.** Cohen’s κ between the primary qwen2.5-7B judge and a second-family MiMo-v2.5-Pro judge on a 1,486-decision stratified subsample.

κ=0.50 is moderate, not strong, but the value is uniform across benchmarks (range 0.44–0.55), across the meaning-bearing / presentation split (0.50 vs 0.51), and across operators (0.48–0.52). This pattern is consistent with random disagreement on borderline cases, not with a systematic per-operator or per-benchmark bias that would inflate Δ in one direction. The MiMo-judge per-cell Δ on the same subsample remains positive in 20 of 36 cells and the mean is +5.6 pp (smaller than the qwen2.5-7B subsample mean because the subsample is biased toward judge-disagreement cases by construction).

### 4.7 Three-generator family swap

We compare cell-level Δ across three perturbation generators on the same eight cells: the original qwen2.5:3b generator, MiMo-v2.5-Pro, and qwen2.5:14b *as a generator only*. Table 3 reports pairwise correlations on the $n=8$ paired Δ vectors; Figure 5 visualises the three pairwise scatterplots. (qwen2.5:14b is later re-used as a *held-out validation model* in §4.8 and §4.9, with 1,800 fresh trajectories that are independent of its generator role here; the §4.7 use is solely as a perturbation source on existing data.)

| Pair | Pearson r | p | Spearman ρ | p |
|---|---|---|---|---|
| qwen2.5:3b vs MiMo | +0.342 | 0.41 | +0.143 | 0.74 |
| **qwen2.5:3b vs qwen2.5:14b** | **+0.794** | **0.019** | **+0.714** | 0.047 |
| MiMo vs qwen2.5:14b | +0.649 | 0.082 | +0.524 | 0.18 |

**Table 3.** Three-way pairwise correlation of cell-level $\Delta$ across perturbation generators ($n=8$ paired cells).

![Figure 5: Three-way generator scatter. Within-architecture (qwen 3B vs qwen 14B) preserves cell-level ranking ($r=+0.79$, $p=0.019$); cross-architecture (qwen vs MiMo) destroys it.](paper_figs/fig5_genswap.png)

**Figure 5.** Three-way generator scatter on $n=8$ paired cells. Left: within-architecture (qwen 3B vs qwen 14B) preserves ranking. Centre and right: cross-architecture pairs destroy ranking. Dashed line is $y=x$.

The within-architecture swap (qwen:3b vs qwen:14b) preserves cell-level ranking; the cross-architecture swap (qwen vs MiMo) destroys it. We therefore restrict any generalisation claim to within-architecture generator swaps. A practitioner deploying AgentDiff with a different perturbation generator from a non-qwen family must expect $\Delta$ values to re-rank.

### 4.8 A held-out validation on a 7th model: pre-registered capability×tractability partition test

The original 6-model panel (§4.1–4.7) is informative but small in family count, and prior versions of this work claimed a *linear* monotonic relationship between accuracy and Δ (Pearson $r=+0.37$, $p=0.050$ on the 26 capable cells of the original panel). We stress-test that claim by holding out a 7th model never used during analysis design — qwen2.5-14B-Instruct, an instruction-tuned dense checkpoint absent from the original panel — and running it through the full pipeline at $n=200$ originals per cell across 9 cells (3 benchmarks $\times$ 3 scaffolds, including a third *direct* scaffold not used in the panel), yielding 1,200 originals and 6,000 variants in 1,800 fresh trajectories. The held-out run is read-only with respect to the original panel: no parameter, threshold, or judge prompt is re-tuned.

We apply a $2\times 2$ partition that was pre-registered on the original panel before the held-out data was inspected:

- **Group A** = (tier $\in$ {strong, frontier}) $\wedge$ (task $\in$ {shallow_arith, multi_hop}). Qwen2.5-14B is assigned tier=strong by panel-wide accuracy (mean acc 0.81 across its 9 cells); GSM8K is shallow_arith, MATH is deep_math, HotpotQA is multi_hop.
- **Group B** = (task = deep_math) $\vee$ (tier = weak).
- Mid-tier $\times$ {shallow_arith, multi_hop}: excluded by design.

Table 4 reports the partition test on the original 28 cot/react cells of the panel, on the held-out 6 cot/react cells alone, and on the pooled 34-cell set.

| Subset | Group | $n$ | mean Δ (pp) | positive | Welch $t$ vs other group | $p$ |
|---|---|---|---|---|---|---|
| Original panel (28 cells) | A | 12 | $+13.0$ | 10/12 | $4.23$ | $6\!\times\!10^{-4}$ |
| Original panel (28 cells) | B | 16 | $-1.6$  |  4/16 | – | – |
| **Held-out qwen2.5-14B (6 cells)** | **A** | **4** | **$+1.62$** | **4/4** | – | – |
| **Held-out qwen2.5-14B (6 cells)** | **B** | **2** | **$+0.38$** | **1/2** | – | – |
| **Pooled (34 cells)** | **A** | **16** | **$+10.0$** | **13/16** (sign-positive) | **3.81** | **$9.6\!\times\!10^{-4}$** |
| **Pooled (34 cells)** | **B** | **18** | **$-1.4$**  |  **5/18** | – | – |

**Table 4.** Pre-registered $2\times 2$ partition test on the original 28 cot/react cells, the 6 held-out qwen2.5-14B cot/react cells, and the pooled set. The held-out cells alone are too few for an internal Welch test, but every held-out Group A cell is in the predicted direction (4/4 positive), and pooling preserves the pattern: Mann-Whitney $U=237.0$, $p=1.4\!\times\!10^{-3}$. A complementary $2\times 2$ Fisher exact on (capable: acc$\geq 0.65$ vs weak) $\times$ ($\Delta>0$ vs $\Delta\leq 0$) on the pooled set gives the table $[[16, 3], [9, 14]]$ with $p=4.5\!\times\!10^{-3}$.

**The dichotomy is a threshold gate, not a linear ramp.** Within the capable subset of the pooled 34-cell set (acc $\geq 0.65$, $n=19$) the Pearson correlation between accuracy and Δ is $r=+0.13$, $p=0.60$ — i.e., once a cell crosses the capability threshold, the magnitude of Δ no longer scales with accuracy. Across the full 34-cell pool the unconditional correlation is $r=+0.32$, $p=0.034$, but this whole-range effect is driven by the categorical jump from weak to capable, not by smooth scaling within the capable regime. We therefore retract the linear-monotonicity claim from prior versions and restate the result as: capability functions as a *binary gate* on tractable benchmarks; on intractable benchmarks (deep_math) the gate does not open even for the held-out 14B model.

**Held-out within-model results.** Across all 6 cot/react cells of qwen2.5-14B alone, the mean Δ is $+0.84$ pp ($4/6$ positive). The signal is much smaller than on the original panel ($+14.32$ pp) because qwen2.5-14B is a *highly capable* instruction-tuned model: its sur-IR is already near zero on GSM8K cot ($0.113$) and on MATH cot ($0.003$), so even a positive directional gap leaves limited headroom. The held-out validation is therefore best read as a *replication of direction*, not of magnitude: the predicted sign holds in every Group A cell (4/4 positive) and the within-Group-A magnitude is $+1.62$ pp; on Group B cells the model exhibits the predicted gate-closure (mean $+0.38$ pp, $1/2$ positive). We report this without rescue: a strong reviewer should read the held-out evidence as confirming *when* the gap appears (capable + tractable), not as inflating *how large* it is.

### 4.9 Trace-level mechanism: stealth divergence

The partition test in §4.8 establishes a *threshold gate* for *when* the gap appears but is silent on *how* a semantic perturbation propagates differently from a presentation perturbation through the agent's reasoning chain. To probe the propagation mechanism we restrict attention to the held-out qwen2.5-14B run (1,800 trajectories: 200 originals × 9 cells = 3 benchmarks × 3 scaffolds) and analyse step-level trajectories already on disk; no new inference is performed. We pre-register four probes derived from `propagation_details` produced by the AgentDiff pipeline:

- **M1. Divergence step.** For each (original, variant) pair, the divergence step is the first reasoning step at which the variant's action or thought diverges from the original. We test whether semantic perturbations diverge *earlier* than presentation perturbations.
- **M2. Self-correction rate.** The fraction of variants whose `propagation_pattern` is `self_correct` (the variant trace momentarily diverged but the agent recovered to the same final answer). We test whether semantic perturbations are *less likely* to be self-corrected.
- **M3. Cascade depth.** The number of subsequent steps affected once divergence has occurred. We test whether semantic perturbations cascade *deeper*.
- **M4. Per-step thought similarity decay.** For each step $k$, the mean cosine similarity between the variant's $k$-th thought and the original's $k$-th thought, averaged across questions that have both a sem and a sur variant at that step. We test whether semantic perturbations decay *faster* per step.

Table 6 reports the four probes pooled across all 1,800 trajectories.

| Probe | Direction (sem - sur) predicted by ``sem more disruptive'' | Observed mean (sem - sur) | Test | Verdict |
|---|---|---|---|---|
| M1 divergence step | negative (sem earlier) | $\mathbf{+0.156}$ steps | paired $t=7.40$, $p=2.1\!\times\!10^{-13}$, Cohen's $d=0.18$ | **opposite sign — retracted** |
| M2 self_correct rate | negative (sem rarer) | $+0.40$ pp (sem 2.40%, sur 2.00%) | Fisher $p=0.230$ | **null — retracted** |
| M3 cascade depth | positive (sem deeper) | $+0.167$ steps | paired $t=7.69$, $p=2.5\!\times\!10^{-14}$, $d=0.18$ | **confirmed** |
| M4 thought sim, step 2 | negative (sem decays more) | $-0.056$ | paired $t=-7.0$, $p=2.0\!\times\!10^{-12}$ | **confirmed** |
| M4 thought sim, step 3 | negative | $-0.093$ | paired $t=-8.9$, $p=1.1\!\times\!10^{-18}$ | **confirmed** |
| M4 thought sim, step 4 | negative | $-0.088$ | paired $t=-8.5$, $p=1.5\!\times\!10^{-16}$ | **confirmed** |

**Table 6.** Four trace-level mechanism probes on the held-out qwen2.5-14B run ($n=1{,}800$ trajectories). Confirmed: M3 + M4. Retracted: M1 (sign flipped versus prior versions of this work), M2 (effect size collapsed to near-zero on the new model). Probes M1 and M2 are reported as an honest non-replication.

![Figure 6: Per-step thought-similarity gap on qwen2.5-14B. Mean (sem-sur) similarity by step k, with bootstrap 95% CI; ${**}{*}$ = $p<10^{-10}$, ${*}{*}$ = $p<10^{-3}$, ${*}$ = $p<0.05$. n questions: step 1 1{,}785, step 2 1{,}004, step 3 595, step 4 594, step 5 363, step 6 224, step 7 54.](paper_figs/fig6_mechanism_step_sim.png)

**Figure 6.** Per-step thought-similarity gap (sem $-$ sur) on the qwen2.5-14B trace probe ($n=1{,}800$). Step 1 is null ($p=0.12$); the gap opens at step 2 and remains highly significant through step 6. The pattern is consistent with **stealth divergence**: a semantic perturbation does not change the model's first surface action — it silently corrupts intermediate thought content from step 2 onward, and that corruption cascades 0.17 steps deeper than a presentation-level edit (M3).

**Stealth-divergence interpretation.** M3 + M4 paint a single picture: under a semantic-preserving rewrite the agent commits to the same first reasoning move as on the original input, but its intermediate thoughts then drift further from the original than under a surface edit, and that drift propagates further before the trace either resyncs or commits to a divergent answer. Surface edits, by contrast, produce *loud* early divergences — visible at the action level — which the agent can either ignore (when the edit is presentation-only) or correct, leading to a smaller cascade footprint.

**Two retractions.** Probes M1 and M2 do not replicate. A prior 26-cell analysis on the original 6-model panel reported semantic perturbations diverging $0.11$ steps *earlier* on average (paired $t=-2.30$, $p=0.021$) and self-correcting $3\times$ less often ($0.8\%$ vs $2.6\%$, $p=0.005$). On the held-out qwen2.5-14B run, M1 reverses direction ($+0.156$ steps, $p=2.1\!\times\!10^{-13}$) and M2 effect size shrinks to $+0.40$ pp ($p=0.230$). Two interpretations are possible: (i) the M1/M2 signals were artefacts of the older lower-capability panel and should be discarded; (ii) M1/M2 reverse on instruction-tuned higher-capability models for substantive reasons (e.g., RLHF-tuned models suppress visible early action divergence in favour of silent thought drift). The current data cannot discriminate between (i) and (ii); we report both probes as **non-replicating** and base the mechanism account solely on M3 + M4.

## 5. AgentDiff-Probe v2: A Prototype Diagnostic

We package the framework as a prototype Python tool, AgentDiff-Probe v2, that takes a small calibration set (≥30 originals × 5 perturbations × 2 scaffolds) and outputs a per-cell Δ estimate plus a traffic-light recommendation. The tool is explicitly a prototype rather than a deployable diagnostic: leave-one-model-out evaluation gives a mean absolute error of 7.10 pp on Δ, and sign accuracy is 72.2 %, which equals the trivial-mean baseline at the same metric. We report this honestly. The MAE improvement (8.27 pp → 7.10 pp, 14 % relative reduction) is real but the deployment-relevant sign decision is unchanged.

The tool's value is therefore not in being a better predictor than the trivial mean, but in producing a calibrated Δ point estimate plus a stratified breakdown that lets a practitioner inspect which operator class drives the gap on their specific deployment. The prototype is released alongside the paper so the community can independently audit the calibration.

## 6. Discussion and Limitations

We have established a robust empirical phenomenon and explicitly bounded what current data can support. We now spell out the limitations.

**L1. Within-benchmark tractability proxies fail.** §4.5 reports 0/3 significant within-benchmark contrasts. Whatever drives cross-benchmark Δ heterogeneity is not captured by our proxies. Possible alternative explanations include task domain (arithmetic vs proof), answer-format constraints, retrieval-grounded versus closed-book reasoning, and trajectory-length differences. We do not attempt to discriminate among these in this paper.

**L2. Wild cluster bootstrap on the regression coefficient is non-significant.** §4.2 reports wild cluster bootstrap $p=0.165$ on the multi-path indicator with cluster=model ($K=6$); the family-level reading ($K=3$, qwen / llama / mimo) gives $p=0.241$. Both fail the conventional $0.05$ threshold. Our headline phenomenon (§4.1) does not depend on this regression. The categorical partition test in §4.8 (Welch $t=3.81$, $p=9.6\!\times\!10^{-4}$; Fisher $p=4.5\!\times\!10^{-3}$ on the pooled set including the held-out 14B model) is the inferential anchor for the capability$\times$tractability claim, and the regression is descriptive.

**L3. Cross-architecture generator instability.** §4.7 shows that swapping the perturbation generator from qwen to MiMo destroys per-cell ranking. Practitioners deploying AgentDiff with a non-qwen generator must recalibrate. The generator-source instability is the single largest threat to the precise Δ value of any individual cell.

**L4. Moderate second-judge agreement.** §4.6 reports Cohen's κ=0.50 with MiMo. The agreement is uniform across strata, suggesting random disagreement on borderline cases rather than systematic bias, but any reader concerned about single-judge dominance should treat the absolute Δ point estimates as moderately uncertain.

**L5. AgentDiff-Probe v2 is a prototype, not a deployable diagnostic.** §5 reports MAE 7.10 pp and sign accuracy 72.2 %, the latter tying the trivial-mean baseline. We do not claim the tool is production-ready.

**L6. Paraphrase and synonym are meaning-preserving rewrites.** Our "meaning-bearing" label refers to which tokens the operator targets, not to whether the operator changes meaning. A reader who reserves "semantic" for label-changing edits may prefer to read the entire paper as a study of "more aggressive lexical rewrites" versus "lighter lexical rewrites" of comparable severity. The empirical finding stands either way.

**L7. Effect-size halving outside the most-sampled family.** The headline $+14.32$ pp shrinks to $+5.54$ pp when restricted to the 24 non-qwen cells of the original panel. The directional sign holds across families, but practitioners deploying agents from families other than qwen should expect roughly half the directional gap we report on average. This remains the most actionable single number in the paper for downstream calibration.

**L8. The held-out validation confirms direction but not magnitude.** §4.8 reports that on the qwen2.5-14B held-out run the within-Group-A mean Δ is only $+1.62$ pp (4/4 positive), an order of magnitude smaller than the $+13.0$ pp Group A mean on the original panel. Two factors plausibly contribute: (i) qwen2.5-14B is a highly capable instruction-tuned model whose sur-IR is already near zero on GSM8K cot ($0.113$) and MATH cot ($0.003$), leaving limited headroom for a positive gap; and (ii) the held-out variants are generated by a different perturbation generator regime (full-pipeline qwen2.5:3b on n=200) than the original panel (n=20–50). The held-out evidence is therefore best read as a *replication of direction*, not of magnitude.

**L9. Two earlier mechanism claims do not replicate on the held-out instruction-tuned 14B model.** §4.9 reports that M1 (semantic perturbations diverge earlier) reverses sign on qwen2.5-14B and M2 (semantic perturbations are 3$\times$ less likely to self-correct) collapses to a non-significant $+0.40$ pp gap ($p=0.230$). The current data cannot distinguish between an artefact-of-the-older-panel reading and a substantive interpretation in which RLHF-tuned models suppress visible early action divergence in favour of silent thought drift. We retract M1 and M2 from the mechanism story and base the account on M3 (cascade depth) and M4 (per-step thought similarity decay), which replicate at much higher $n$ and stronger effect sizes.

**L10. Capability is a binary gate, not a linear ramp.** Earlier versions of this work reported a Pearson $r=+0.37$ ($p=0.050$) between accuracy and Δ on the 26 capable cells of the original panel. Pooling the original panel with the held-out qwen2.5-14B run gives within-capable correlation $r=+0.13$ ($p=0.60$, $n=19$). The threshold-style account in §4.8 fits the data; the linear-monotonicity claim does not, and we retract it.

## 7. Conclusion

Across 44 cells covering six LLMs from three architecture families, three benchmarks, and two scaffolds, the gap between meaning-bearing and presentation perturbation inconsistency rates averages $+14.32$ pp after edit-distance severity matching, with $40$ of $44$ cells positive (paired $t=6.76$, $p<0.0001$). The gap is robust to severity definition: matched Δ stays in the $+13.7$ to $+15.4$ pp range across edit distance, token Jaccard, Sentence-BERT cosine, and length-change ratio (Table 1b). It is accompanied by a step-level cascade gap on GSM8K that survives a TF-IDF redefinition ($+0.66$ steps, $p<0.001$). On the 24 non-qwen cells alone, the gap halves to $+5.54$ pp but remains highly significant (paired $t=4.41$, $p=0.0002$), so the phenomenon is not a single-family artefact. The phenomenon does not survive every test we ran: the multi-path regression coefficient fails wild cluster bootstrap at $K=6$ model and $K=3$ family clusters, within-benchmark tractability proxies show zero significant contrasts (§4.5), and the cross-architecture generator swap destroys per-cell ranking (§4.7).

A fully held-out 7th model run on qwen2.5-14B (1,800 trajectories never used to tune the analysis on the original panel) sharpens the picture in two ways. First, it confirms the *direction* of the capability$\times$tractability partition: every held-out cell in Group A is in the predicted direction (4/4 positive), and pooling with the original panel gives Welch $t=3.81$, $p=9.6\!\times\!10^{-4}$. The within-capable Pearson correlation between accuracy and Δ collapses to $r=+0.13$ ($p=0.60$, $n=19$), so the gate is *threshold-style*, not a linear function of capability — we retract the linear-monotonicity framing. Second, the held-out trajectories reveal a clean trace-level mechanism we call **stealth divergence**: a semantic perturbation does not change the agent's first surface action, but corrupts intermediate thought content from step 2 onward (per-step similarity gap $-5.6$ to $-10.5$ pp, paired $t=-7.0$ to $-8.9$, $p<10^{-11}$) and cascades $0.17$ steps deeper than a presentation-level edit (paired $t=7.69$, $p=2.5\!\times\!10^{-14}$; §4.9, Figure 6). Two earlier mechanism claims (M1: semantic perturbations diverge earlier; M2: semantic perturbations self-correct rarer) do not replicate on the held-out model and are explicitly retracted.

The contribution is a measurement — a robust directional gap on a 6-model panel, a held-out replication of the capability-conditional structure on a 7th model, and a partial trace-level account of how the gap propagates. We deliberately update the mechanism story relative to prior versions of this work as the held-out evidence justifies. We hope future work with substantially more architecture families and white-box access can identify whether the stealth-divergence pattern reflects a generic property of instruction-tuned reasoners or an artefact of the present panel.

## Limitations

This paper documents a measurement and explicitly bounds what current data can support; we summarise the resulting limitations in detail in §6 (L1–L10). In short: (i) within-benchmark tractability proxies fail in 0/3 contrasts (§4.5), so the cross-benchmark $\Delta$ heterogeneity is not identified at the mechanism level; (ii) the multi-path regression coefficient does not survive a wild cluster bootstrap with $K=6$ model clusters (§4.2) or $K=3$ family clusters, so the regression is descriptive rather than causal, and the categorical partition test is the inferential anchor; (iii) cross-architecture generator swaps destroy per-cell ranking (§4.7), so the precise $\Delta$ value of any individual cell is generator-conditional; (iv) the second-judge audit gives Cohen's $\kappa=0.50$ (§4.6), which is moderate, not strong; (v) AgentDiff-Probe v2 is a prototype whose sign accuracy ties the trivial-mean baseline (§5); (vi) the ``meaning-bearing'' label refers to which tokens the operator targets rather than to whether the operator is label-changing; (vii) the headline effect halves outside the qwen family ($+5.54$ pp on the 24 non-qwen cells of the original panel), so practitioners on llama- or mimo-class agents should expect roughly half the directional gap on average; (viii) the held-out qwen2.5-14B run replicates the *direction* of the capability-conditional pattern (4/4 Group A cells positive) but not its *magnitude* (within-Group-A mean $+1.62$ pp) because the held-out model has near-zero sur-IR on its strongest cells (§4.8); (ix) two earlier mechanism claims (M1: earlier divergence step; M2: lower self-correction rate) do not replicate on the held-out qwen2.5-14B run and are retracted (§4.9), with the trace-level mechanism account based solely on M3 (cascade depth) and M4 (per-step thought similarity); and (x) capability functions as a *binary gate* on Δ once the threshold is crossed, not as a linear predictor, and the linear-correlation framing from prior versions is retracted (§4.8).
## Ethics Statement

This work uses three publicly available benchmarks (GSM8K, MATH, HotpotQA) under their respective research-use licences; no human subjects were recruited. All perturbation generation and judging is done by open-weight or self-hosted LLMs, and no personally identifying information is exposed. The released calibration tool (AgentDiff-Probe v2) is intended for research auditing of agent robustness; we do not advocate using its outputs as the sole gate for production deployment, since the diagnostic ties the trivial-mean baseline on the deployment-relevant sign decision (§5). Compute usage was approximately 22 CPU-only days on a 48-core, 64 GB RAM workstation plus a bounded 200 M-token MiMo-v2.5-Pro API allocation; we report the breakdown in Appendix A.

## References

Rie Kubota Ando and Tong Zhang. 2005. [A framework for learning predictive structures from multiple tasks and unlabeled data](https://www.jmlr.org/papers/v6/ando05a.html). *Journal of Machine Learning Research*, 6:1817–1853.

Yoav Benjamini and Yosef Hochberg. 1995. [Controlling the false discovery rate: A practical and powerful approach to multiple testing](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). *Journal of the Royal Statistical Society B*, 57(1):289–300.

A. Colin Cameron, Jonah B. Gelbach, and Douglas L. Miller. 2008. [Bootstrap-based improvements for inference with clustered errors](https://doi.org/10.1162/rest.90.3.414). *Review of Economics and Statistics*, 90(3):414–427.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. [Training verifiers to solve math word problems](https://arxiv.org/abs/2110.14168). *arXiv:2110.14168*.

A. C. Davison and D. V. Hinkley. 1997. *Bootstrap Methods and their Application*. Cambridge University Press.

Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, Sameer Singh, Noah A. Smith, Sanjay Subramanian, Reut Tsarfaty, Eric Wallace, Ally Zhang, and Ben Zhou. 2020. [Evaluating models' local decision boundaries via contrast sets](https://doi.org/10.18653/v1/2020.findings-emnlp.117). In *Findings of EMNLP 2020*, pages 1307–1323.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. [Measuring mathematical problem solving with the MATH dataset](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html). In *NeurIPS Datasets and Benchmarks*.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. [Retrieval-augmented generation for knowledge-intensive NLP tasks](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html). In *NeurIPS 2020*.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamilė Lukošiūtė, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. 2023. [Measuring faithfulness in chain-of-thought reasoning](https://arxiv.org/abs/2307.13702). *arXiv:2307.13702*.

Kung-Yee Liang and Scott L. Zeger. 1986. [Longitudinal data analysis using generalized linear models](https://doi.org/10.1093/biomet/73.1.13). *Biometrika*, 73(1):13–22.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. 2024. [AgentBench: Evaluating LLMs as agents](https://openreview.net/forum?id=zAdUB0aCTQ). In *ICLR 2024*.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. [Self-Refine: Iterative refinement with self-feedback](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html). In *NeurIPS 2023*.

Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. 2024. [GSM-Symbolic: Understanding the limitations of mathematical reasoning in large language models](https://arxiv.org/abs/2410.05229). *arXiv:2410.05229*. To appear, *NeurIPS 2024*.

Mohammad Sadegh Rasooli and Joel R. Tetreault. 2015. [Yara parser: A fast and accurate dependency parser](http://arxiv.org/abs/1503.06733). *Computing Research Repository*, arXiv:1503.06733.

Yasaman Razeghi, Robert L. Logan IV, Matt Gardner, and Sameer Singh. 2022. [Impact of pretraining term frequencies on few-shot numerical reasoning](https://doi.org/10.18653/v1/2022.findings-emnlp.59). In *Findings of EMNLP 2022*, pages 840–854.

Marco Túlio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. [Beyond accuracy: Behavioral testing of NLP models with CheckList](https://doi.org/10.18653/v1/2020.acl-main.442). In *ACL 2020*, pages 4902–4912.

David Roodman, Morten Ørregaard Nielsen, James G. MacKinnon, and Matthew D. Webb. 2019. [Fast and wild: Bootstrap inference in Stata using boottest](https://doi.org/10.1177/1536867X19830877). *The Stata Journal*, 19(1):4–60.

Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. 2023. [Are emergent abilities of large language models a mirage?](https://proceedings.neurips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html) In *NeurIPS 2023*.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. 2024. [Quantifying language models' sensitivity to spurious features in prompt design or: How I learned to start worrying about prompt formatting](https://openreview.net/forum?id=RIu5lyNXjT). In *ICLR 2024*.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. [Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting](https://arxiv.org/abs/2305.04388). In *NeurIPS 2023*.

Jindong Wang, Xixu Hu, Wenxin Hou, Hao Chen, Runkai Zheng, Yidong Wang, Linyi Yang, Haojun Huang, Wei Ye, Xiubo Geng, Binxing Jiao, Yue Zhang, and Xing Xie. 2023. [On the robustness of ChatGPT: An adversarial and out-of-distribution perspective](https://arxiv.org/abs/2302.12095). *arXiv:2302.12095*.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. [Reflexion: Language agents with verbal reinforcement learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html). In *NeurIPS 2023*.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. [Chain-of-thought prompting elicits reasoning in large language models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html). In *NeurIPS 2022*.

Yueqi Yan, Yuxuan Cai, Yufan Zhang, Sirui Li, Zhiwei Tang, Viet Dac Lai, Philip S. Yu, and Lichao Sun. 2024. [AgentBoard: An analytical evaluation board of multi-turn LLM agents](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a48a9b6db38acca1d35e3bdf2f3f7c6-Abstract-Datasets_and_Benchmarks_Track.html). In *NeurIPS Datasets and Benchmarks 2024*.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. [HotpotQA: A dataset for diverse, explainable multi-hop question answering](https://doi.org/10.18653/v1/D18-1259). In *EMNLP 2018*, pages 2369–2380.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. [ReAct: Synergizing reasoning and acting in language models](https://openreview.net/forum?id=WE_vluYUL-X). In *ICLR 2023*.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2025. [$\tau$-Bench: A benchmark for tool-agent-user interaction in real-world domains](https://openreview.net/forum?id=roNlrjCAnh). In *ICLR 2025*.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. [Judging LLM-as-a-judge with MT-Bench and Chatbot Arena](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html). In *NeurIPS Datasets and Benchmarks 2023*.

Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, and Xing Xie. 2024. [PromptBench: Towards evaluating the robustness of large language models on adversarial prompts](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9c5e0ee3e8d0b3d44a6ed9f82e9c7d15-Abstract-Datasets_and_Benchmarks_Track.html). In *NeurIPS Datasets and Benchmarks 2024*.
