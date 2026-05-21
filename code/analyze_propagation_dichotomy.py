#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_propagation_dichotomy.py

Secondary validation of the conditional dichotomy: even when Δ_IR ≈ 0,
do semantic vs surface perturbations differ in HOW they fail
(propagation pattern: early_diverge / late_diverge / cascade / self_correct
/ consistent)?

Reads runs_real_*_fix/  and  runs_real_*_hpqa/  jsonl files,
inspects the per-variant `propagation_details` list, and produces:

  - results_conditional/propagation_dichotomy.md
  - results_conditional/propagation_dichotomy.json
  - results_conditional/figures/fig5_propagation_pattern.png/pdf
"""
import os, sys, json, glob, re, math
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "results_conditional")
FIG  = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

SEM = ["paraphrase", "synonym"]
SUR = ["reorder", "format", "distractor"]

PATTERNS = ["consistent","self_correct","late_diverge","early_diverge","cascade"]

# ----------------- discover & load -----------------
def discover_jsonls():
    paths=[]
    for d in sorted(glob.glob(os.path.join(ROOT, "runs_real_*_fix"))):
        for jp in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            paths.append(jp)
    for d in sorted(glob.glob(os.path.join(ROOT, "runs_real_*_hpqa"))):
        for jp in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            paths.append(jp)
    for jp in sorted(glob.glob(os.path.join(ROOT, "runs_real_mimo_v25_pro/*.jsonl"))):
        paths.append(jp)
    return paths

def load_records(jp):
    rows=[]
    with open(jp) as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: r=json.loads(ln)
            except: continue
            if "error" in r: continue
            rows.append(r)
    by={r.get("sample_id",""):r for r in rows}
    return list(by.values())

# ----------------- aggregate variant-level patterns -----------------
sem_pattern_count = defaultdict(int)   # global
sur_pattern_count = defaultdict(int)
total_sem = 0
total_sur = 0

# also: per-cell, for paired analysis
per_cell = defaultdict(lambda: {"sem":defaultdict(int), "sur":defaultdict(int),
                                 "n_sem":0, "n_sur":0,
                                 "sem_cascade_depth":[], "sur_cascade_depth":[],
                                 "sem_diverge_step":[], "sur_diverge_step":[]})

# also: identify cells split into capable / incapable
cells_acc = {}  # cell_key -> accuracy

for jp in discover_jsonls():
    fname = os.path.basename(jp)
    # cell key = file path basename without .jsonl
    cell_key = fname.replace(".jsonl","")
    rows = load_records(jp)
    if not rows: continue
    acc = sum(1 for r in rows if r.get("original_result",{}).get("is_correct",False)) / len(rows)
    cells_acc[cell_key] = acc

    for r in rows:
        pd = r.get("propagation_details", [])
        for v in pd:
            ptype  = v.get("perturbation_type","unknown")
            pat    = v.get("propagation_pattern","unknown")
            cdepth = v.get("cascade_depth", 0) or 0
            dstep  = v.get("divergence_step", 0) or 0
            if ptype in SEM:
                sem_pattern_count[pat] += 1
                total_sem += 1
                per_cell[cell_key]["sem"][pat] += 1
                per_cell[cell_key]["n_sem"]   += 1
                per_cell[cell_key]["sem_cascade_depth"].append(cdepth)
                per_cell[cell_key]["sem_diverge_step"].append(dstep)
            elif ptype in SUR:
                sur_pattern_count[pat] += 1
                total_sur += 1
                per_cell[cell_key]["sur"][pat] += 1
                per_cell[cell_key]["n_sur"]   += 1
                per_cell[cell_key]["sur_cascade_depth"].append(cdepth)
                per_cell[cell_key]["sur_diverge_step"].append(dstep)

# ----------------- chi-square test on global pattern distribution -----------------
def chi_square_2way(obs_a, obs_b, categories):
    """obs_a, obs_b are dicts cat->count.  Returns (chi2, df, p)."""
    rows = [[obs_a.get(c,0), obs_b.get(c,0)] for c in categories]
    # row totals, col totals, grand total
    R = [sum(r) for r in rows]
    C = [sum(rows[i][0] for i in range(len(rows))),
         sum(rows[i][1] for i in range(len(rows)))]
    N = sum(C)
    chi2 = 0.0
    df = (len(categories)-1)*1
    for i,c in enumerate(categories):
        for j in (0,1):
            exp = R[i]*C[j]/N if N>0 else 0
            if exp==0: continue
            obs = rows[i][j]
            chi2 += (obs-exp)**2/exp
    # crude chi2->p via gamma incomplete (df, x) — approx for small df
    # use survival function: P(chi2 > x | df) ~ Q(df/2, x/2)
    # use math.gamma based incomplete gamma series.
    def regularized_upper_gamma(s, x):
        # series for lower then 1-…  for accuracy we use scipy-free recursion
        if x <= 0: return 1.0
        # Q(s, x) via continued fraction (Numerical Recipes, simplified)
        # Use series for x < s+1 else CF
        if x < s + 1:
            term = 1.0/s; total=term; n=1
            while n < 200:
                term *= x/(s+n)
                total += term
                if abs(term) < 1e-12: break
                n += 1
            P = total * math.exp(-x + s*math.log(x) - math.lgamma(s))
            return 1.0 - P
        else:
            b = x + 1.0 - s; c = 1e30; d = 1.0/b; h = d
            for n in range(1, 200):
                an = -n*(n-s); b += 2.0
                d = an*d + b
                if abs(d) < 1e-30: d = 1e-30
                c = b + an/c
                if abs(c) < 1e-30: c = 1e-30
                d = 1.0/d; delta = d*c
                h *= delta
                if abs(delta-1.0) < 1e-12: break
            return h * math.exp(-x + s*math.log(x) - math.lgamma(s))
    p = regularized_upper_gamma(df/2.0, chi2/2.0)
    return chi2, df, p

chi2, df, p_chi = chi_square_2way(sem_pattern_count, sur_pattern_count, PATTERNS)

# ----------------- per-pattern proportions and z-test -----------------
def two_prop_z(p1, n1, p2, n2):
    if n1==0 or n2==0: return None
    p_pool = (p1*n1 + p2*n2)/(n1+n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    if se==0: return None
    z = (p1-p2)/se
    p = math.erfc(abs(z)/math.sqrt(2))
    return z,p

per_pat_test = {}
for pat in PATTERNS:
    p_sem = sem_pattern_count.get(pat,0)/total_sem if total_sem else 0
    p_sur = sur_pattern_count.get(pat,0)/total_sur if total_sur else 0
    z,pval = two_prop_z(p_sem, total_sem, p_sur, total_sur) or (None,None)
    per_pat_test[pat] = {
        "n_sem": sem_pattern_count.get(pat,0), "n_sur": sur_pattern_count.get(pat,0),
        "p_sem": p_sem, "p_sur": p_sur,
        "diff_pp": 100*(p_sem-p_sur),
        "z": z, "p": pval,
    }

# ----------------- cascade depth & divergence step (paired t on cell-level means) -----------------
def cell_means():
    pairs_cd = []  # (cell_key, mean_sem_cd, mean_sur_cd)
    pairs_ds = []
    for k, d in per_cell.items():
        if d["sem_cascade_depth"] and d["sur_cascade_depth"]:
            pairs_cd.append((k,
                             sum(d["sem_cascade_depth"])/len(d["sem_cascade_depth"]),
                             sum(d["sur_cascade_depth"])/len(d["sur_cascade_depth"])))
        if d["sem_diverge_step"] and d["sur_diverge_step"]:
            pairs_ds.append((k,
                             sum(d["sem_diverge_step"])/len(d["sem_diverge_step"]),
                             sum(d["sur_diverge_step"])/len(d["sur_diverge_step"])))
    return pairs_cd, pairs_ds

pairs_cd, pairs_ds = cell_means()

def paired_t(diffs):
    n=len(diffs)
    if n<3: return None
    m=sum(diffs)/n
    sd=math.sqrt(sum((d-m)**2 for d in diffs)/(n-1))
    if sd==0: return None
    se=sd/math.sqrt(n)
    t=m/se
    p=math.erfc(abs(t)/math.sqrt(2))
    return {"n":n,"mean":m,"sd":sd,"t":t,"p":p}

cd_diffs = [s-u for _,s,u in pairs_cd]
ds_diffs = [s-u for _,s,u in pairs_ds]
cd_t = paired_t(cd_diffs)
ds_t = paired_t(ds_diffs)

# ----------------- write report -----------------
md=[]
md.append("# Propagation-Pattern Dichotomy (secondary evidence)")
md.append("")
md.append("Variant-level analysis: even when IR is similar, do semantic vs surface perturbations differ in HOW the agent fails?")
md.append("")
md.append(f"Total semantic variants  : {total_sem}")
md.append(f"Total surface  variants  : {total_sur}")
md.append("")
md.append("## 1. Global pattern distribution")
md.append("")
md.append("| Pattern | n_sem | %sem | n_sur | %sur | Δ%pp (sem−sur) | z | p |")
md.append("|---|---|---|---|---|---|---|---|")
for pat in PATTERNS:
    t = per_pat_test[pat]
    z = "—" if t["z"] is None else f"{t['z']:+.2f}"
    pv = "—" if t["p"] is None else (f"<0.0001" if t["p"]<1e-4 else f"{t['p']:.4f}")
    md.append("| %s | %d | %.1f%% | %d | %.1f%% | %+.1f | %s | %s |" % (
        pat, t["n_sem"], 100*t["p_sem"], t["n_sur"], 100*t["p_sur"], t["diff_pp"], z, pv))
md.append("")
md.append(f"**Chi-square independence**: χ² = {chi2:.2f}, df = {df}, p = {p_chi:.4f}")
md.append("")
md.append("## 2. Cascade depth & divergence step (cell-level paired t-test)")
md.append("")
if cd_t:
    md.append(f"- mean(cascade_depth_sem − cascade_depth_sur)  = {cd_t['mean']:+.3f},  t={cd_t['t']:+.2f}, p={cd_t['p']:.4f}, n_cells={cd_t['n']}")
if ds_t:
    md.append(f"- mean(divergence_step_sem − divergence_step_sur) = {ds_t['mean']:+.3f},  t={ds_t['t']:+.2f}, p={ds_t['p']:.4f}, n_cells={ds_t['n']}")
md.append("")
md.append("Interpretation: if cascade_depth(sem) > cascade_depth(sur) significantly, "
          "semantic perturbations propagate further into the agent's reasoning chain. "
          "If divergence_step(sem) < divergence_step(sur), semantic perturbations diverge earlier.")
md.append("")

with open(os.path.join(OUT, "propagation_dichotomy.md"),"w") as f:
    f.write("\n".join(md))

# JSON
with open(os.path.join(OUT, "propagation_dichotomy.json"),"w") as f:
    json.dump({
        "totals":{"sem":total_sem,"sur":total_sur},
        "per_pattern": per_pat_test,
        "chi2":chi2,"df":df,"p":p_chi,
        "cascade_depth_paired_t": cd_t,
        "divergence_step_paired_t": ds_t,
    }, f, indent=2, default=str)

# Figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8.5,4.5))
xs = np.arange(len(PATTERNS))
w = 0.38
sem_pcts = [100*sem_pattern_count.get(p,0)/total_sem if total_sem else 0 for p in PATTERNS]
sur_pcts = [100*sur_pattern_count.get(p,0)/total_sur if total_sur else 0 for p in PATTERNS]
ax.bar(xs-w/2, sem_pcts, w, color="#d62728", edgecolor="black",
        label=f"Semantic (n={total_sem})", alpha=0.9)
ax.bar(xs+w/2, sur_pcts, w, color="#1f77b4", edgecolor="black",
        label=f"Surface (n={total_sur})", alpha=0.9)
for i,(s,u) in enumerate(zip(sem_pcts,sur_pcts)):
    ax.text(i-w/2, s+0.4, f"{s:.1f}", ha="center", fontsize=8)
    ax.text(i+w/2, u+0.4, f"{u:.1f}", ha="center", fontsize=8)
    # significance star
    pv = per_pat_test[PATTERNS[i]]["p"]
    if pv is not None and pv < 0.001:
        ax.text(i, max(s,u)+3, "***", ha="center", fontsize=12, fontweight="bold")
    elif pv is not None and pv < 0.01:
        ax.text(i, max(s,u)+3, "**", ha="center", fontsize=12, fontweight="bold")
    elif pv is not None and pv < 0.05:
        ax.text(i, max(s,u)+3, "*", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(PATTERNS, rotation=15)
ax.set_ylabel("Percentage of variants (%)")
ax.set_title(f"How agents fail: semantic vs surface propagation patterns "
             f"(χ²={chi2:.1f}, p={p_chi:.4f})")
ax.legend(loc="upper right")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIG,"fig5_propagation_pattern.png"), dpi=200)
plt.savefig(os.path.join(FIG,"fig5_propagation_pattern.pdf"))
plt.close()

print("DONE")
print(" md  -> %s" % os.path.join(OUT,"propagation_dichotomy.md"))
print(" json-> %s" % os.path.join(OUT,"propagation_dichotomy.json"))
print(" fig -> %s/fig5_propagation_pattern.{png,pdf}" % FIG)
