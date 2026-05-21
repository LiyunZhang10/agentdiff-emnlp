# Qwen-2.5-14B per-cell summary (9 cells × 200 questions)

| benchmark | agent | acc | sem rate | sur rate | Δ (pp) | t | p | 95% CI (pp) |
|---|---|---|---|---|---|---|---|---|
| gsm8k | cot | 0.930 | 0.092 | 0.113 | -2.08 | -1.85 | 0.06444 | [-4.17, +0.17] |
| gsm8k | react | 0.915 | 0.107 | 0.102 | +0.58 | +0.38 | 0.7052 | [-2.33, +3.75] |
| gsm8k | direct | 0.660 | 0.228 | 0.243 | -1.58 | -0.72 | 0.471 | [-6.08, +2.67] |
| math | cot | 0.770 | 0.003 | 0.003 | -0.08 | -0.24 | 0.8088 | [-0.67, +0.58] |
| math | react | 0.945 | 0.095 | 0.087 | +0.83 | +0.68 | 0.4948 | [-1.75, +3.33] |
| math | direct | 0.780 | 0.045 | 0.210 | -16.50 | -10.53 | 0 | [-19.67, -13.33] |
| hotpotqa | cot | 0.785 | 0.028 | 0.002 | +2.65 | +3.22 | 0.001271 | [+1.28, +4.44] |
| hotpotqa | react | 0.780 | 0.049 | 0.017 | +3.16 | +3.19 | 0.001431 | [+1.45, +5.30] |
| hotpotqa | direct | 0.770 | 0.013 | 0.019 | -0.60 | -1.09 | 0.2741 | [-1.71, +0.51] |

## Capability-gating split (acc threshold = 0.65)

- Capable cells (acc>=0.65): n=9, mean Δ=-1.51pp, positive Δ in 4/9
- Weak cells (acc<0.65):    n=0, mean Δ=+0.00pp, positive Δ in 0/0

## Overall (9 cells, mean across cells)

- mean Δ across 9 cells: -1.51 pp
- positive Δ cells: 4/9
