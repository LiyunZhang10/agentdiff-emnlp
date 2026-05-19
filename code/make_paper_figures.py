#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figures.py — Publication-quality figures for the
"Capability-Gated Dichotomy" framing.

Outputs:
    results_conditional/figures/
        fig1_capability_gated.png   (paper Figure 1)
        fig1_capability_gated.pdf
        fig2_per_cell_bars.png      (paper Figure 2)
        fig2_per_cell_bars.pdf
        fig3_perturbation_breakdown.png
        fig3_perturbation_breakdown.pdf
        fig4_threshold_curve.png    (Δ vs sweeping accuracy threshold)

Stat outputs (text):
    results_conditional/capability_gated_stats.txt
"""
import os, sys, json, math, csv
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "results_conditional")
FIG  = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# ----------------------- load -----------------------
J = json.load(open(os.path.join(OUT, "dichotomy_summary.json")))
cells = [c for c in J["cells"]
         if c.get("accuracy") is not None and c.get("delta") is not None]

# ordering reference
TIER_ORDER = ["weak", "mid", "strong", "frontier"]
TIER_COLOR = {"weak":"#888888","mid":"#1f77b4","strong":"#2ca02c","frontier":"#d62728"}
TASK_MARK  = {"shallow_arith":"o","deep_math":"s","multi_hop":"^"}
TASK_LABEL = {"shallow_arith":"GSM8K","deep_math":"MATH","multi_hop":"HotpotQA"}

# ----------------------- stats -----------------------
def pearson(xs, ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sxx=sum((x-mx)**2 for x in xs); syy=sum((y-my)**2 for y in ys)
    r=sxy/math.sqrt(sxx*syy) if sxx*syy>0 else 0
    if abs(r) >= 1.0: return r, 0.0
    t=r*math.sqrt((n-2)/max(1-r*r,1e-9))
    p=math.erfc(abs(t)/math.sqrt(2))
    return r, p

def spearman(xs, ys):
    def rk(v):
        o=sorted(range(len(v)), key=lambda i: v[i])
        rk=[0.0]*len(v); i=0
        while i<len(v):
            j=i
            while j+1<len(v) and v[o[j+1]]==v[o[i]]: j+=1
            avg=(i+j)/2.0+1.0
            for k in range(i,j+1): rk[o[k]]=avg
            i=j+1
        return rk
    rx=rk(xs); ry=rk(ys); n=len(xs)
    return pearson(rx, ry)

def fishers_exact_2x2(a,b,c,d):
    """Fisher exact two-sided p (small samples).
       a,b,c,d are cell counts of 2x2 table."""
    from math import lgamma
    def lcomb(n,k):
        return lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1)
    n=a+b+c+d
    r1=a+b; r2=c+d; c1=a+c; c2=b+d
    def lp(x):
        return lcomb(c1,x)+lcomb(c2,r1-x)-lcomb(n,r1)
    obs=lp(a)
    p=0.0
    for x in range(max(0,r1-c2), min(c1,r1)+1):
        if lp(x) <= obs+1e-12:
            p += math.exp(lp(x))
    return p

xs = [c["accuracy"] for c in cells]
ys = [c["delta"] for c in cells]

r_p, p_p = pearson(xs, ys)
r_s, p_s = spearman(xs, ys)

# Linear regression
n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
sxx=sum((x-mx)**2 for x in xs)
slope = sxy/sxx if sxx>0 else 0
intercept = my - slope*mx
# residuals -> SE
preds = [intercept+slope*x for x in xs]
resid = [y-p for y,p in zip(ys,preds)]
sse = sum(r*r for r in resid)
sigma2 = sse/(n-2)
se_slope = math.sqrt(sigma2/sxx) if sxx>0 else 0
ci_low  = slope - 1.96*se_slope
ci_high = slope + 1.96*se_slope

# Threshold sweep — for each acc threshold T, compute mean Δ above and below
ths = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
sweep = []
for T in ths:
    above=[(c["accuracy"], c["delta"]) for c in cells if c["accuracy"]>=T]
    below=[(c["accuracy"], c["delta"]) for c in cells if c["accuracy"]< T]
    if not above or not below: continue
    a_mean = sum(d for _,d in above)/len(above)
    b_mean = sum(d for _,d in below)/len(below)
    a_pos  = sum(1 for _,d in above if d>0)
    b_pos  = sum(1 for _,d in below if d>0)
    # Fisher exact on 2x2: above_pos, above_neg | below_pos, below_neg
    a_neg = len(above)-a_pos
    b_neg = len(below)-b_pos
    fp = fishers_exact_2x2(a_pos, a_neg, b_pos, b_neg)
    sweep.append({
        "T":T, "n_above":len(above), "n_below":len(below),
        "a_mean":a_mean, "b_mean":b_mean, "gap":a_mean-b_mean,
        "a_pos":a_pos, "b_pos":b_pos, "fisher_p":fp,
    })

# Best threshold = max gap (or smallest fisher p)
best = min(sweep, key=lambda s: s["fisher_p"]) if sweep else None

# Save stats text
with open(os.path.join(OUT, "capability_gated_stats.txt"), "w") as f:
    p=lambda s: f.write(s+"\n")
    p("# Capability-Gated Dichotomy — Statistical Evidence")
    p("")
    p(f"Total cells analysed : {n}")
    p("")
    p("## 1. Continuous correlation (accuracy ↔ Δ)")
    p(f"  Pearson  r = {r_p:+.3f}  (p ≈ {p_p:.4f})")
    p(f"  Spearman ρ = {r_s:+.3f}  (p ≈ {p_s:.4f})")
    p("")
    p("## 2. Linear regression Δ = α + β·Accuracy")
    p(f"  intercept α = {100*intercept:+.2f}pp")
    p(f"  slope     β = {100*slope:+.2f}pp / accuracy-unit")
    p(f"  95% CI on β = [{100*ci_low:+.2f}, {100*ci_high:+.2f}]pp")
    p("")
    p("## 3. Threshold sweep (Fisher exact on Δ>0 above-vs-below)")
    p("|  T  | n_above | a_pos/a_neg |  mean_Δ_above | n_below | b_pos/b_neg |  mean_Δ_below | gap | Fisher p |")
    p("|-----|---------|-------------|---------------|---------|-------------|---------------|-----|----------|")
    for s in sweep:
        p("| %.2f | %3d | %d/%d | %+5.1fpp | %3d | %d/%d | %+5.1fpp | %+5.1fpp | %.4f |" % (
            s["T"], s["n_above"], s["a_pos"], s["n_above"]-s["a_pos"], 100*s["a_mean"],
            s["n_below"], s["b_pos"], s["n_below"]-s["b_pos"], 100*s["b_mean"],
            100*s["gap"], s["fisher_p"]))
    if best:
        p("")
        p(f"## Best threshold: T = {best['T']:.2f}  (Fisher p = {best['fisher_p']:.4f})")

# ----------------------- Fig 1 (paper key figure) -----------------------
# accuracy × delta scatter + regression band + threshold band
fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(0, color="grey", linewidth=0.7, linestyle="--", zorder=0)
T = best["T"] if best else 0.6
ax.axvspan(T, 1.0, color="#d62728", alpha=0.06, zorder=0,
           label=f"acc ≥ {T:.2f} (capable region)")

# regression line + 95% band (use plain lists for compat)
xx = [i/199.0 for i in range(200)]
yy = [intercept + slope*x for x in xx]
yy_lo = [intercept + ci_low*x for x in xx]
yy_hi = [intercept + ci_high*x for x in xx]
yy_band_low  = [min(a,b)*100 for a,b in zip(yy_lo, yy_hi)]
yy_band_high = [max(a,b)*100 for a,b in zip(yy_lo, yy_hi)]
ax.fill_between(xx, yy_band_low, yy_band_high,
                color="black", alpha=0.10, zorder=1, label="95% CI")
ax.plot(xx, [v*100 for v in yy], color="black", linewidth=1.6, zorder=2,
        label=f"OLS slope = {100*slope:+.1f}pp/acc")

# scatter
for c in cells:
    ax.scatter(c["accuracy"], 100*c["delta"],
               color=TIER_COLOR.get(c["tier"],"k"),
               marker=TASK_MARK.get(c["task"],"o"),
               s=85, alpha=0.9, edgecolor="black", linewidth=0.6, zorder=3)

ax.set_xlabel("Original-task accuracy", fontsize=11)
ax.set_ylabel(r"$\Delta$ = sem_IR $-$ sur_IR  (percentage points)", fontsize=11)
ax.set_title("Capability-gated dichotomy: $\\Delta$ emerges with task accuracy", fontsize=12)
ax.set_xlim(-0.02, 1.02)
ax.grid(True, alpha=0.25)

# annotate stats on plot
note = (f"Pearson r = {r_p:+.2f} (p = {p_p:.3f})\n"
        f"Spearman ρ = {r_s:+.2f} (p = {p_s:.3f})\n"
        f"OLS slope = {100*slope:+.1f}pp / unit acc\n"
        f"Above acc≥{T:.2f}: mean Δ = {100*best['a_mean']:+.1f}pp ({best['a_pos']}/{best['n_above']})\n"
        f"Below acc<{T:.2f}: mean Δ = {100*best['b_mean']:+.1f}pp ({best['b_pos']}/{best['n_below']})\n"
        f"Fisher exact p = {best['fisher_p']:.4f}")
ax.text(0.02, -16, note, fontsize=8, family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))

# legends
legend_tier = [Line2D([0],[0],marker="o",color="w",markerfacecolor=TIER_COLOR[t],
                       markeredgecolor="black",label=t,markersize=9)
               for t in TIER_ORDER]
legend_task = [Line2D([0],[0],marker=TASK_MARK[t],color="w",markerfacecolor="grey",
                       markeredgecolor="black",label=TASK_LABEL[t],markersize=9)
               for t in TASK_MARK]
leg1 = ax.legend(handles=legend_tier, title="Tier", loc="upper left",
                 fontsize=8, framealpha=0.85)
ax.add_artist(leg1)
ax.legend(handles=legend_task, title="Task", loc="upper right",
          fontsize=8, framealpha=0.85)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_capability_gated.png"), dpi=200)
plt.savefig(os.path.join(FIG, "fig1_capability_gated.pdf"))
plt.close()

# ----------------------- Fig 2: per-cell Δ bar chart, sorted by accuracy -----------------------
sorted_cells = sorted(cells, key=lambda c: c["accuracy"])
fig, ax = plt.subplots(figsize=(8, 5.5))
xs2 = list(range(len(sorted_cells)))
ys2 = [100*c["delta"] for c in sorted_cells]
colors = [TIER_COLOR[c["tier"]] for c in sorted_cells]
ax.bar(xs2, ys2, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
labels = [f"{c['display'][:6]}/{c['bench'][:5]}/{c['agent'][:3]}\n(acc={c['accuracy']:.2f})"
          for c in sorted_cells]
ax.set_xticks(xs2)
ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
ax.set_ylabel(r"$\Delta$ (pp)", fontsize=11)
ax.set_title("Per-cell $\\Delta$, sorted by original-task accuracy", fontsize=12)
ax.grid(True, axis="y", alpha=0.25)
ax.legend(handles=legend_tier, title="Tier", loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_per_cell_bars.png"), dpi=200)
plt.savefig(os.path.join(FIG, "fig2_per_cell_bars.pdf"))
plt.close()

# ----------------------- Fig 3: per-perturbation breakdown -----------------------
SEM_LIST = ["paraphrase","synonym"]
SUR_LIST = ["reorder","format","distractor"]
ALL_LIST = SEM_LIST + SUR_LIST
type_vals = defaultdict(list)
for c in cells:
    for t in ALL_LIST:
        v = c["per_type"].get(t)
        if v is not None: type_vals[t].append(v)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
labels=ALL_LIST
means=[100*sum(type_vals[t])/len(type_vals[t]) for t in labels]
stds =[100*math.sqrt(sum((v-sum(type_vals[t])/len(type_vals[t]))**2 for v in type_vals[t])/(len(type_vals[t])-1)) if len(type_vals[t])>1 else 0
       for t in labels]
clr = ["#d62728" if t in SEM_LIST else "#1f77b4" for t in labels]
ax.bar(range(len(labels)), means, yerr=stds, color=clr, edgecolor="black",
       capsize=5, alpha=0.85)
for i,(m,s) in enumerate(zip(means,stds)):
    ax.text(i, m+s+1.5, f"{m:.1f}%", ha="center", fontsize=9)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20)
ax.set_ylabel("Mean IR across cells (%)")
ax.set_title("(a) Per-type IR (mean ± SD across cells)")
ax.legend(handles=[Patch(facecolor="#d62728",label="Semantic"),
                    Patch(facecolor="#1f77b4",label="Surface")], loc="lower right")
ax.grid(True, axis="y", alpha=0.25)

# (b) split by capable/incapable
ax = axes[1]
cap_cells = [c for c in cells if c["accuracy"] >= T]
inc_cells = [c for c in cells if c["accuracy"] <  T]
def avg_pt(grp, t):
    vs = [c["per_type"].get(t) for c in grp if c["per_type"].get(t) is not None]
    return (sum(vs)/len(vs)) if vs else None
cap_means = [100*avg_pt(cap_cells, t) for t in labels if avg_pt(cap_cells,t) is not None]
inc_means = [100*avg_pt(inc_cells, t) for t in labels if avg_pt(inc_cells,t) is not None]
x = np.arange(len(labels))
w = 0.35
ax.bar(x-w/2, cap_means, w, color=clr, edgecolor="black", label="acc ≥ %.2f" % T, alpha=0.95, hatch="//")
ax.bar(x+w/2, inc_means, w, color=clr, edgecolor="black", label="acc < %.2f" % T, alpha=0.55)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
ax.set_ylabel("Mean IR (%)")
ax.set_title(f"(b) Per-type IR split by capable / incapable cells (T={T:.2f})")
ax.legend(loc="lower right"); ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_perturbation_breakdown.png"), dpi=200)
plt.savefig(os.path.join(FIG, "fig3_perturbation_breakdown.pdf"))
plt.close()

# ----------------------- Fig 4: threshold sweep -----------------------
fig, ax = plt.subplots(figsize=(7, 4))
Ts = [s["T"] for s in sweep]
gaps = [100*s["gap"] for s in sweep]
fps = [s["fisher_p"] for s in sweep]
ax.plot(Ts, gaps, "o-", color="black", linewidth=1.5, label="mean Δ above − below")
ax.set_xlabel("Accuracy threshold T")
ax.set_ylabel("Δ gap (pp)", color="black")
ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
ax2 = ax.twinx()
ax2.plot(Ts, fps, "s--", color="#d62728", linewidth=1.2, label="Fisher exact p")
ax2.axhline(0.05, color="#d62728", linewidth=0.6, linestyle=":")
ax2.set_ylabel("Fisher exact p-value", color="#d62728")
ax2.set_yscale("log")
ax.set_title("Threshold sweep: gap and significance vs accuracy threshold")
fig.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_threshold_curve.png"), dpi=200)
plt.savefig(os.path.join(FIG, "fig4_threshold_curve.pdf"))
plt.close()

print("done.")
print(" stats -> %s" % os.path.join(OUT, "capability_gated_stats.txt"))
print(" figs  -> %s/" % FIG)
