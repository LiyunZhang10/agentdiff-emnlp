# Trace-level mechanism probes on Qwen-2.5-14B (n_traj=1800)

All signals are derived from `propagation_details` produced by the AgentDiff pipeline. No hidden representations or token-level logprobs are queried — every probe is a pure secondary analysis of step-level traces already on disk.

## M1. Divergence step (does sem diverge earlier?)

- n_paired (have both sem and sur): 1785
- mean(sem - sur) = +0.156 steps  (negative => sem earlier)
- paired t = 7.398, p = 2.122e-13, Cohen's d = 0.175
- Wilcoxon W = 333436, p = 5.411e-12
- bootstrap 95% CI: [+0.113, +0.198]

### M1 by cell

| cell | n | mean(sem-sur) | t | p |
|---|---|---|---|---|
| gsm8k/cot | 200 | +0.542 | 5.436 | 1.587e-07 |
| gsm8k/direct | 200 | +0.060 | 2.289 | 0.0231 |
| gsm8k/react | 200 | +0.008 | 0.278 | 0.7813 |
| hotpotqa/cot | 195 | +0.366 | 3.741 | 0.0002416 |
| hotpotqa/direct | 195 | -0.117 | -7.097 | 2.34e-11 |
| hotpotqa/react | 195 | +0.032 | 0.840 | 0.4017 |
| math/cot | 200 | +0.452 | 4.610 | 7.177e-06 |
| math/direct | 200 | -0.182 | -12.423 | 1.318e-26 |
| math/react | 200 | +0.235 | 8.797 | 6.71e-16 |

## M2. Self-correct rate (sem vs sur, pooled)

- sem self_correct: 83 / 3459 = 0.0240
- sur self_correct: 108 / 5400 = 0.0200
- diff: +0.40 pp (negative => sem less likely to recover)
- Fisher exact two-sided p = 0.2302, OR = 1.205

### M2 by cell

| cell | sem rate | sur rate | diff(pp) | Fisher p |
|---|---|---|---|---|
| gsm8k/cot | 0.0550 | 0.0300 | +2.50 | 0.06864 |
| gsm8k/direct | 0.0000 | 0.0000 | +0.00 | 1 |
| gsm8k/react | 0.0000 | 0.0000 | +0.00 | 1 |
| hotpotqa/cot | 0.0840 | 0.0950 | -1.10 | 0.6454 |
| hotpotqa/direct | 0.0000 | 0.0000 | +0.00 | 1 |
| hotpotqa/react | 0.0000 | 0.0000 | +0.00 | 1 |
| math/cot | 0.0781 | 0.0550 | +2.31 | 0.1815 |
| math/direct | 0.0000 | 0.0000 | +0.00 | 1 |
| math/react | 0.0000 | 0.0000 | +0.00 | 1 |

## M3. Cascade depth

- n_paired: 1785
- mean(sem - sur) = +0.167  (positive => sem cascades deeper)
- paired t = 7.687, p = 2.474e-14, Cohen's d = 0.182
- Wilcoxon W = 148415, p = 7.49e-19
- bootstrap 95% CI: [+0.125, +0.213]

## M4. Thought similarity decay per step

- For each step k, compute mean(sem_thought_sim_k - sur_thought_sim_k) across questions that have both kinds of variant at that step.

| step k | n | mean(sem-sur) | t | p | 95% CI |
|---|---|---|---|---|---|
| 1 | 1785 | -0.0102 | -1.563 | 0.1183 | [-0.0231, +0.0022] |
| 2 | 1004 | -0.0559 | -7.121 | 2.043e-12 | [-0.0709, -0.0406] |
| 3 | 595 | -0.0926 | -9.122 | 1.133e-18 | [-0.1127, -0.0735] |
| 4 | 594 | -0.0884 | -8.504 | 1.487e-16 | [-0.1081, -0.0690] |
| 5 | 363 | -0.1050 | -6.981 | 1.406e-11 | [-0.1343, -0.0751] |
| 6 | 224 | -0.0948 | -4.369 | 1.91e-05 | [-0.1381, -0.0543] |
| 7 | 54 | -0.0573 | -2.099 | 0.04058 | [-0.1149, -0.0063] |
| 8 | 11 | -0.0425 | -0.576 | 0.5775 | [-0.1719, +0.0992] |
| 9 | 3 | +0.0055 | 0.048 | 0.9662 | [-0.1905, +0.2069] |
| 10 | 1 | +0.0000 | n/a | n/a | [+0.0000, +0.0000] |
