#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_conditional.py — Authoritative aggregator for the
"Conditional Dichotomy" claim.

Reads (only fix-judged sources):
    runs_real_*_fix/{gsm8k,math}_{cot,react}_real_*_fix.jsonl     (n=20 each)
    runs_real_*_hpqa/hotpotqa_{cot,react}_real_*_hpqa.jsonl       (n=30 each)
    runs_real_mimo_v25_pro/{gsm8k,math}_{cot,react}_real_mimo_v25_pro.jsonl
                                                                  (n=20, may be partial)

Outputs:
    results_conditional/dichotomy_summary.md       — main paper-ready table
    results_conditional/dichotomy_summary.json     — machine readable
    results_conditional/per_cell_long.csv          — flat per-cell rows
    results_conditional/figures/fig1_acc_vs_delta.png
    results_conditional/figures/fig2_per_type_grouped.png
    results_conditional/figures/fig3_heatmap_dichotomy.png
"""
import os, sys, json, glob, re, math, csv
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "results_conditional")
FIG  = os.path.join(OUT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEM = ["paraphrase", "synonym"]
SUR = ["reorder", "format", "distractor"]
ALL = SEM + SUR

# capability tier (independent of dichotomy direction; based on params + accuracy)
TIER = {
    "llama32_1b":   ("weak",     "Llama-3.2-1B"),
    "llama32_3b":   ("mid",      "Llama-3.2-3B"),
    "qwen25_3b":    ("mid",      "Qwen2.5-3B"),
    "qwen25_7b":    ("strong",   "Qwen2.5-7B"),
    "llama31_8b":   ("strong",   "Llama-3.1-8B"),
    "mimo_v25_pro": ("frontier", "MiMo-v2.5-pro"),
}
TASK = {
    "gsm8k":    "shallow_arith",
    "math":     "deep_math",
    "hotpotqa": "multi_hop",
}

# ----------------- I/O -----------------
def load_jsonl(jp):
    if not os.path.isfile(jp): return []
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

def discover_cells():
    cells=[]
    for d in sorted(glob.glob(os.path.join(ROOT, "runs_real_*_fix"))):
        slug=os.path.basename(d)[len("runs_real_"):-len("_fix")]
        if slug not in TIER: continue
        for jp in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            m=re.match(r"(gsm8k|math)_(cot|react)_real_.+\.jsonl$", os.path.basename(jp))
            if not m: continue
            cells.append((slug, m.group(1), m.group(2), jp))
    for d in sorted(glob.glob(os.path.join(ROOT, "runs_real_*_hpqa"))):
        slug=os.path.basename(d)[len("runs_real_"):-len("_hpqa")]
        if slug not in TIER: continue
        for jp in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            m=re.match(r"hotpotqa_(cot|react)_real_.+\.jsonl$", os.path.basename(jp))
            if not m: continue
            cells.append((slug, "hotpotqa", m.group(1), jp))
    for jp in sorted(glob.glob(os.path.join(ROOT, "runs_real_mimo_v25_pro/*.jsonl"))):
        m=re.match(r"(gsm8k|math)_(cot|react)_real_.+\.jsonl$", os.path.basename(jp))
        if not m: continue
        cells.append(("mimo_v25_pro", m.group(1), m.group(2), jp))
    return cells

def cell_stats(rows):
    if not rows: return None
    n=len(rows)
    acc = sum(1 for r in rows if r.get("original_result", {}).get("is_correct", False)) / n
    ir  = sum(1 for r in rows
              if not r.get("consistency_analysis", {}).get("is_consistent", True)) / n
    sem_per=[]; sur_per=[]; pti_sum=defaultdict(float); pti_cnt=defaultdict(int)
    for r in rows:
        pti=r.get("consistency_analysis",{}).get("per_type_inconsistency",{})
        if not pti: continue
        sv=[pti[t] for t in SEM if t in pti]
        uv=[pti[t] for t in SUR if t in pti]
        if sv: sem_per.append(sum(sv)/len(sv))
        if uv: sur_per.append(sum(uv)/len(uv))
        for t,v in pti.items():
            pti_cnt[t]+=1; pti_sum[t]+=v
    return {
        "n": n, "accuracy": acc, "ir": ir,
        "sem_ir": sum(sem_per)/len(sem_per) if sem_per else None,
        "sur_ir": sum(sur_per)/len(sur_per) if sur_per else None,
        "delta":  (sum(sem_per)/len(sem_per) - sum(sur_per)/len(sur_per))
                  if sem_per and sur_per else None,
        "per_type": {t: pti_sum[t]/pti_cnt[t] if pti_cnt[t] else None for t in ALL},
        "sem_per_sample": sem_per,
        "sur_per_sample": sur_per,
    }

# ----------------- stats -----------------
def wilcoxon(diffs):
    nz=[d for d in diffs if d!=0]; n=len(nz)
    if n<6: return None,None,n
    ad=[abs(d) for d in nz]
    order=sorted(range(n), key=lambda i: ad[i])
    ranks=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and ad[order[j+1]]==ad[order[i]]: j+=1
        avg=(i+j)/2.0+1.0
        for k in range(i,j+1): ranks[order[k]]=avg
        i=j+1
    Wp=sum(ranks[k] for k in range(n) if nz[k]>0)
    Wn=sum(ranks[k] for k in range(n) if nz[k]<0)
    W=min(Wp,Wn)
    mu=n*(n+1)/4.0; var=n*(n+1)*(2*n+1)/24.0
    cnt=defaultdict(int)
    for v in ad: cnt[v]+=1
    tie=sum(c**3-c for c in cnt.values() if c>1)
    var-=tie/48.0
    if var<=0: return W,1.0,n
    z=(W-mu)/math.sqrt(var)
    p=math.erfc(abs(z)/math.sqrt(2))
    return W,p,n

def welch(a,b):
    if len(a)<2 or len(b)<2: return None
    ma=sum(a)/len(a); mb=sum(b)/len(b)
    va=sum((x-ma)**2 for x in a)/(len(a)-1)
    vb=sum((x-mb)**2 for x in b)/(len(b)-1)
    se=math.sqrt(va/len(a)+vb/len(b))
    if se==0: return None
    t=(ma-mb)/se
    df=(va/len(a)+vb/len(b))**2 / ((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
    p=math.erfc(abs(t)/math.sqrt(2))
    return {"t":t,"p":p,"df":df,"mean_a":ma,"mean_b":mb}

def mannwhitney(a,b):
    n1,n2=len(a),len(b)
    if n1<3 or n2<3: return None
    comb=[(v,0) for v in a]+[(v,1) for v in b]; comb.sort()
    ranks=[0]*len(comb); i=0
    while i<len(comb):
        j=i
        while j+1<len(comb) and comb[j+1][0]==comb[i][0]: j+=1
        avg=(i+j)/2.0+1.0
        for k in range(i,j+1): ranks[k]=avg
        i=j+1
    R1=sum(ranks[k] for k in range(len(comb)) if comb[k][1]==0)
    U1=R1-n1*(n1+1)/2.0; U2=n1*n2-U1; U=min(U1,U2)
    mu=n1*n2/2.0; sig=math.sqrt(n1*n2*(n1+n2+1)/12.0)
    z=(U-mu)/sig if sig>0 else 0
    p=math.erfc(abs(z)/math.sqrt(2))
    return {"U":U,"z":z,"p":p}

# ----------------- aggregate -----------------
def main():
    cells = discover_cells()
    per_cell = []
    for slug,bench,agent,jp in cells:
        rows = load_jsonl(jp)
        st = cell_stats(rows)
        if st is None: continue
        tier, display = TIER.get(slug, ("?", slug))
        task = TASK.get(bench, "?")
        in_A = (tier in ("strong","frontier")) and (task in ("shallow_arith","multi_hop"))
        in_B = (task=="deep_math") or (tier=="weak")
        per_cell.append({
            "slug":slug, "display":display, "tier":tier,
            "bench":bench, "task":task, "agent":agent, "jp":jp,
            "in_A":in_A, "in_B":in_B,
            **st
        })

    # ---- group statistics ----
    A_deltas = [c["delta"] for c in per_cell if c["in_A"] and c["delta"] is not None]
    B_deltas = [c["delta"] for c in per_cell if c["in_B"] and c["delta"] is not None]
    grp = {
        "A_n": len(A_deltas), "A_mean": sum(A_deltas)/len(A_deltas) if A_deltas else None,
        "A_pos": sum(1 for d in A_deltas if d>0),
        "B_n": len(B_deltas), "B_mean": sum(B_deltas)/len(B_deltas) if B_deltas else None,
        "B_pos": sum(1 for d in B_deltas if d>0),
        "welch": welch(A_deltas, B_deltas),
        "mannwhitney": mannwhitney(A_deltas, B_deltas),
    }

    # ---- markdown ----
    def fpct(v): return ("%.1f%%" % (100*v)) if v is not None else "—"
    def fpp(v):  return ("%+.1fpp" % (100*v)) if v is not None else "—"

    md=[]
    md.append("# AgentDiff — Conditional Dichotomy Summary")
    md.append("")
    md.append("**Claim**: Semantic-vs-surface inconsistency dichotomy in LLM agents is *conditional*: it emerges only under (i) sufficiently capable models, and (ii) shallow-reasoning or multi-hop tasks; on weak models or deep-math tasks it disappears or reverses.")
    md.append("")
    md.append(f"_{len(per_cell)} cells aggregated. Generated by `aggregate_conditional.py`._")
    md.append("")
    md.append("## 1. Per-cell dichotomy table")
    md.append("")
    md.append("| Model (tier) | Task | Bench | Agent | n | Acc | IR | sem_IR | sur_IR | Δ |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    order=sorted(per_cell, key=lambda c: (c["tier"], c["slug"], c["bench"], c["agent"]))
    for c in order:
        md.append("| %s (%s) | %s | %s | %s | %d | %s | %s | %s | %s | %s |" % (
            c["display"], c["tier"], c["task"], c["bench"], c["agent"], c["n"],
            fpct(c["accuracy"]), fpct(c["ir"]),
            fpct(c["sem_ir"]), fpct(c["sur_ir"]), fpp(c["delta"]),
        ))
    md.append("")

    md.append("## 2. Group analysis (the headline test)")
    md.append("")
    md.append("Pre-registered partition (independent of any per-cell Δ value):")
    md.append("- **Group A** = (tier ∈ {strong, frontier}) AND (task ∈ {shallow_arith, multi_hop})")
    md.append("- **Group B** = (task = deep_math) OR (tier = weak)")
    md.append("")
    md.append(f"- Group A: n_cells = {grp['A_n']}, mean Δ = {fpp(grp['A_mean'])}, Δ>0 in {grp['A_pos']}/{grp['A_n']}")
    md.append(f"- Group B: n_cells = {grp['B_n']}, mean Δ = {fpp(grp['B_mean'])}, Δ>0 in {grp['B_pos']}/{grp['B_n']}")
    if grp["welch"]:
        w=grp["welch"]
        md.append(f"- **Welch t-test**: t = {w['t']:.3f}, p ≈ {w['p']:.4f}, df ≈ {w['df']:.1f}")
    if grp["mannwhitney"]:
        m=grp["mannwhitney"]
        md.append(f"- **Mann-Whitney U**: U = {m['U']:.1f}, z = {m['z']:.3f}, p ≈ {m['p']:.4f}")
    md.append("")

    md.append("## 3. Per-perturbation-type IR (averaged across all cells)")
    md.append("")
    md.append("| Type | Class | Cells | Mean IR |")
    md.append("|---|---|---|---|")
    type_vals=defaultdict(list)
    for c in per_cell:
        for t in ALL:
            v=c["per_type"].get(t)
            if v is not None: type_vals[t].append(v)
    rows=[]
    for t in ALL:
        vs=type_vals[t]
        if not vs: continue
        rows.append((t, "SEM" if t in SEM else "SUR", len(vs), sum(vs)/len(vs)))
    rows.sort(key=lambda r: -r[3])
    for t,cls,n,m in rows:
        md.append("| %s | %s | %d | %s |" % (t, cls, n, fpct(m)))
    md.append("")

    md.append("## 4. Per-cell Wilcoxon signed-rank test")
    md.append("")
    md.append("| Model | Bench/Agent | n | Δ | n+ | n- | n0 | Wilcoxon p |")
    md.append("|---|---|---|---|---|---|---|---|")
    for c in order:
        diffs=[s-u for s,u in zip(c["sem_per_sample"], c["sur_per_sample"])]
        W,p,nz=wilcoxon(diffs)
        np_=sum(1 for d in diffs if d>0)
        nn=sum(1 for d in diffs if d<0)
        n0=sum(1 for d in diffs if d==0)
        ps = ("<0.0001" if (p is not None and p<1e-4) else
              ("%.4f" % p) if p is not None else "n/a")
        md.append("| %s | %s/%s | %d | %s | %d | %d | %d | %s |" % (
            c["display"], c["bench"], c["agent"], c["n"],
            fpp(c["delta"]), np_, nn, n0, ps))
    md.append("")

    md.append("## 5. Group composition (transparency)")
    md.append("")
    md.append("**Group A cells (predicted positive):**")
    for c in order:
        if c["in_A"]:
            md.append("- %s | %s | %s | Δ=%s" % (c["display"], c["bench"], c["agent"], fpp(c["delta"])))
    md.append("")
    md.append("**Group B cells (predicted negative):**")
    for c in order:
        if c["in_B"]:
            md.append("- %s | %s | %s | Δ=%s" % (c["display"], c["bench"], c["agent"], fpp(c["delta"])))
    md.append("")

    with open(os.path.join(OUT, "dichotomy_summary.md"), "w") as f:
        f.write("\n".join(md))

    # ---- json ----
    with open(os.path.join(OUT, "dichotomy_summary.json"), "w") as f:
        # strip per-sample arrays to keep size small
        payload=[]
        for c in per_cell:
            d=dict(c); d.pop("sem_per_sample", None); d.pop("sur_per_sample", None); d.pop("jp", None)
            payload.append(d)
        json.dump({"cells": payload, "groups": grp}, f, indent=2, default=str)

    # ---- csv (flat) ----
    with open(os.path.join(OUT, "per_cell_long.csv"), "w") as f:
        w=csv.writer(f)
        w.writerow(["slug","display","tier","bench","task","agent","n","accuracy",
                    "ir","sem_ir","sur_ir","delta",
                    "para","syn","reord","fmt","dist","in_A","in_B"])
        for c in per_cell:
            pt=c["per_type"]
            w.writerow([c["slug"], c["display"], c["tier"], c["bench"], c["task"], c["agent"],
                        c["n"], c["accuracy"], c["ir"], c["sem_ir"], c["sur_ir"], c["delta"],
                        pt.get("paraphrase"), pt.get("synonym"), pt.get("reorder"),
                        pt.get("format"), pt.get("distractor"),
                        int(c["in_A"]), int(c["in_B"])])

    # ---- figures ----
    try: import matplotlib
    except ImportError: matplotlib=None
    if matplotlib:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Fig 1: accuracy vs delta
        fig,ax=plt.subplots(figsize=(7,5))
        TIER_COLOR={"weak":"#888888","mid":"#1f77b4","strong":"#2ca02c","frontier":"#d62728"}
        TASK_MARK={"shallow_arith":"o","deep_math":"s","multi_hop":"^"}
        for c in per_cell:
            if c["accuracy"] is None or c["delta"] is None: continue
            ax.scatter(c["accuracy"], 100*c["delta"],
                       color=TIER_COLOR.get(c["tier"],"k"),
                       marker=TASK_MARK.get(c["task"],"o"),
                       s=70, alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Original-task accuracy")
        ax.set_ylabel("Δ = sem_IR − sur_IR  (percentage points)")
        ax.set_title("Conditional Dichotomy: Δ emerges with capability")
        # legends
        from matplotlib.lines import Line2D
        legend_tier=[Line2D([0],[0],marker="o",color="w",markerfacecolor=v,
                            markeredgecolor="k",label=k,markersize=9)
                     for k,v in TIER_COLOR.items()]
        legend_task=[Line2D([0],[0],marker=v,color="w",markerfacecolor="grey",
                            markeredgecolor="k",label=k,markersize=9)
                     for k,v in TASK_MARK.items()]
        leg1=ax.legend(handles=legend_tier, title="Tier", loc="upper left", fontsize=8)
        ax.add_artist(leg1)
        ax.legend(handles=legend_task, title="Task", loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG,"fig1_acc_vs_delta.png"), dpi=200)
        plt.close()

        # Fig 2: per-type bar grouped by SEM/SUR
        fig,ax=plt.subplots(figsize=(7,4))
        labels=[r[0] for r in rows]; vals=[100*r[3] for r in rows]
        colors=["#d62728" if r[1]=="SEM" else "#1f77b4" for r in rows]
        ax.bar(range(len(labels)), vals, color=colors, edgecolor="black")
        for i,v in enumerate(vals):
            ax.text(i, v+1, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_ylabel("Mean IR across cells (%)")
        ax.set_title("Per-perturbation-type inconsistency rate")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#d62728",label="Semantic"),
                            Patch(facecolor="#1f77b4",label="Surface")], loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG,"fig2_per_type_grouped.png"), dpi=200)
        plt.close()

        # Fig 3: Δ heatmap (model × bench/agent)
        models = sorted({c["display"] for c in per_cell},
                        key=lambda d: ["Llama-3.2-1B","Llama-3.2-3B","Qwen2.5-3B",
                                       "Qwen2.5-7B","Llama-3.1-8B","MiMo-v2.5-pro"].index(d)
                                      if d in ["Llama-3.2-1B","Llama-3.2-3B","Qwen2.5-3B",
                                       "Qwen2.5-7B","Llama-3.1-8B","MiMo-v2.5-pro"] else 999)
        cols=[(b,a) for b in ["gsm8k","math","hotpotqa"] for a in ["cot","react"]]
        Z=[[None]*len(cols) for _ in models]
        for c in per_cell:
            i=models.index(c["display"])
            j=cols.index((c["bench"],c["agent"]))
            Z[i][j]=100*c["delta"] if c["delta"] is not None else None
        import numpy as np
        Zarr=np.array([[v if v is not None else np.nan for v in row] for row in Z])
        fig,ax=plt.subplots(figsize=(7, 0.6*len(models)+1.5))
        im=ax.imshow(Zarr, cmap="RdBu_r", aspect="auto", vmin=-30, vmax=30)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{b}/{a}" for b,a in cols], rotation=30, ha="right")
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
        for i in range(len(models)):
            for j in range(len(cols)):
                v=Z[i][j]
                if v is None: ax.text(j,i,"—",ha="center",va="center",fontsize=8,color="grey")
                else: ax.text(j,i,f"{v:+.0f}",ha="center",va="center",
                              fontsize=8, color="white" if abs(v)>15 else "black")
        plt.colorbar(im, ax=ax, label="Δ (pp)")
        ax.set_title("Δ = sem_IR − sur_IR across (model × task × agent)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG,"fig3_heatmap_dichotomy.png"), dpi=200)
        plt.close()

    print("DONE.")
    print(" md  -> %s" % os.path.join(OUT, "dichotomy_summary.md"))
    print(" json-> %s" % os.path.join(OUT, "dichotomy_summary.json"))
    print(" csv -> %s" % os.path.join(OUT, "per_cell_long.csv"))
    print(" fig -> %s/" % FIG)

if __name__ == "__main__":
    main()
