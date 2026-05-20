#!/usr/bin/env python3
"""
code/make_fig_mechanism.py

Figure for the new §4.9 mechanism section: per-step thought-similarity
gap (sem - sur) on Qwen-2.5-14B, n=1800.

Saves:
  paper/paper_figs/fig6_mechanism_step_sim.png  (300 DPI)
  paper/paper_figs/fig6_mechanism_step_sim.pdf  (vector)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/workspace/agentdiff-emnlp"
SRC  = os.path.join(ROOT, "results/conditional_v2/trace_mech_qwen25_14b.json")
OUT_DIR = os.path.join(ROOT, "paper/paper_figs")
os.makedirs(OUT_DIR, exist_ok=True)

with open(SRC) as f:
    d = json.load(f)

per_step = d["M4_thought_similarity_per_step"]["per_step"]
steps  = sorted((int(k) for k in per_step), key=int)
means  = [per_step[str(k)]["mean_sem_minus_sur"] for k in steps]
ci_low = [per_step[str(k)]["ci95"][0] for k in steps]
ci_hi  = [per_step[str(k)]["ci95"][1] for k in steps]
ns     = [per_step[str(k)]["n"] for k in steps]
ps     = [(per_step[str(k)]["paired_t"] or {}).get("p", float("nan")) for k in steps]

# Drop steps with n < 30 (too few paired questions for stable CI)
keep = [i for i, n in enumerate(ns) if n >= 30]
steps   = [steps[i]   for i in keep]
means   = [means[i]   for i in keep]
ci_low  = [ci_low[i]  for i in keep]
ci_hi   = [ci_hi[i]   for i in keep]
ns      = [ns[i]      for i in keep]
ps      = [ps[i]      for i in keep]

fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=300)

ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.6)
ax.fill_between(steps, ci_low, ci_hi, color="#5b9bd5", alpha=0.25, label="95% CI")
ax.plot(steps, means, marker="o", color="#1f4e79", lw=1.6, markersize=5,
        label="Qwen-2.5-14B (n=1800)")

# annotate p-values at each step
for k, m, p_v in zip(steps, means, ps):
    if p_v < 1e-10:
        sig = "***"
    elif p_v < 1e-3:
        sig = "**"
    elif p_v < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    ax.annotate(sig, (k, m), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=8, color="black")

ax.set_xlabel("Reasoning step $k$")
ax.set_ylabel("Mean thought-similarity gap\n"
              r"$\overline{\mathrm{sem}_k - \mathrm{sur}_k}$")
ax.set_xticks(steps)
ax.set_title("Stealth divergence: semantic perturbations corrupt\n"
             "intermediate thoughts more from step 2 onward",
             fontsize=10)
ax.legend(loc="lower left", fontsize=8, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "fig6_mechanism_step_sim.png")
out_pdf = os.path.join(OUT_DIR, "fig6_mechanism_step_sim.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print("wrote", out_png)
print("wrote", out_pdf)
