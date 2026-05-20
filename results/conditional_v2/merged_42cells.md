# Merged 42-cell analysis (36 old + 6 new Qwen-2.5-14B cot/react)

**Question:** does the original Capability×Tractability dichotomy still hold when we add 6 new high-n (n=200/cell) cells from Qwen-2.5-14B (a strong-tier, dense, instruction-tuned model absent from the original 6-model panel)?

## 1. Pre-registered partition (cot/react agents only)

Same rules as original paper:
- Group A = (tier ∈ {strong, frontier}) AND (task ∈ {shallow_arith, multi_hop})
- Group B = (task = deep_math) OR (tier = weak)
- Mid × {shallow, multi_hop}: excluded by design

### 1a. Old-only baseline reproduction (sanity)

- Group A: n=12, mean Δ = +13.01pp, positive 10/12
- Group B: n=16, mean Δ = -1.59pp, positive 4/16
- Welch t = 4.225, df = 16.0, p = 0.0006411

### 1b. NEW: 6 Qwen-2.5-14B cot/react cells alone

| bench | task | agent | acc | sem_ir | sur_ir | Δ (pp) | group |
|---|---|---|---|---|---|---|---|
| gsm8k | shallow_arith | cot | 0.930 | 0.092 | 0.113 | -2.08 | A |
| gsm8k | shallow_arith | react | 0.915 | 0.107 | 0.102 | +0.58 | A |
| math | deep_math | cot | 0.770 | 0.003 | 0.003 | -0.08 | B |
| math | deep_math | react | 0.945 | 0.095 | 0.087 | +0.83 | B |
| hotpotqa | multi_hop | cot | 0.785 | 0.028 | 0.002 | +2.65 | A |
| hotpotqa | multi_hop | react | 0.780 | 0.049 | 0.017 | +3.16 | A |

- Group A subset (Qwen-14B): n=4, mean Δ = +1.08pp, positive 3/4
- Group B subset (Qwen-14B): n=2, mean Δ = +0.37pp, positive 1/2

### 1c. MERGED 42-cell partition test (the headline)

- Group A: n=16, mean Δ = +10.02pp, positive 13/16
- Group B: n=18, mean Δ = -1.38pp, positive 5/18
- Welch t = 3.807, df = 22.1, p = 0.0009551
- Mann-Whitney U = 237.0, p = 0.001415

## 2. Capability-gating regression (acc vs Δ)

- Capable cells (acc>=0.65) in 42-pool: n=19, Pearson r = +0.128, p = 0.6018
- All 42 cells: n=42, Pearson r = +0.317, p = 0.04061

## 3. 2x2 Fisher exact (capable vs weak × Δ>0 vs Δ≤0)

|         | Δ>0 | Δ≤0 |
|---|---|---|
| capable (acc≥0.65) | 16 | 3 |
| weak    (acc<0.65) | 9 | 14 |

- Fisher exact two-sided p = 0.004486, OR = 8.296

## 4. Robustness: Qwen-14B 'direct' agent (3 cells, exploration only)

Direct agents are NOT in the original 36-cell panel. We report them separately rather than fold into the partition test.

| bench | task | acc | sem_ir | sur_ir | Δ (pp) | group |
|---|---|---|---|---|---|---|
| gsm8k | shallow_arith | 0.660 | 0.228 | 0.243 | -1.58 | A |
| math | deep_math | 0.780 | 0.045 | 0.210 | -16.50 | B |
| hotpotqa | multi_hop | 0.770 | 0.013 | 0.019 | -0.60 | A |
