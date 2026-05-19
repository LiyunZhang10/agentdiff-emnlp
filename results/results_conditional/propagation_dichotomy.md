# Propagation-Pattern Dichotomy (secondary evidence)

Variant-level analysis: even when IR is similar, do semantic vs surface perturbations differ in HOW the agent fails?

Total semantic variants  : 1228
Total surface  variants  : 2886

## 1. Global pattern distribution

| Pattern | n_sem | %sem | n_sur | %sur | Δ%pp (sem−sur) | z | p |
|---|---|---|---|---|---|---|---|
| consistent | 179 | 14.6% | 482 | 16.7% | -2.1 | -1.70 | 0.0895 |
| self_correct | 16 | 1.3% | 64 | 2.2% | -0.9 | -1.94 | 0.0519 |
| late_diverge | 53 | 4.3% | 130 | 4.5% | -0.2 | -0.27 | 0.7884 |
| early_diverge | 791 | 64.4% | 1791 | 62.1% | +2.4 | +1.43 | 0.1527 |
| cascade | 189 | 15.4% | 419 | 14.5% | +0.9 | +0.72 | 0.4705 |

**Chi-square independence**: χ² = 7.40, df = 4, p = 0.1162

## 2. Cascade depth & divergence step (cell-level paired t-test)

- mean(cascade_depth_sem − cascade_depth_sur)  = +0.114,  t=+1.37, p=0.1714, n_cells=36
- mean(divergence_step_sem − divergence_step_sur) = -0.038,  t=-1.02, p=0.3077, n_cells=36

Interpretation: if cascade_depth(sem) > cascade_depth(sur) significantly, semantic perturbations propagate further into the agent's reasoning chain. If divergence_step(sem) < divergence_step(sur), semantic perturbations diverge earlier.
