#!/usr/bin/env /usr/bin/python3
"""
Generate 5 publication figures from the FINAL n=50 Track A-G data.

Outputs (PDF + PNG, 300 DPI, Times-style) using the "amfe.space"
Chinese traditional color palette: 胭脂 / 霁青 / 缃叶 / 鸦青 / 隐红.

  paper_figs/fig1_delta_distribution.{pdf,png}
  paper_figs/fig2_severity_match.{pdf,png}
  paper_figs/fig3_cascade_gsm8k.{pdf,png}
  paper_figs/fig4_within_benchmark.{pdf,png}
  paper_figs/fig5_genswap.{pdf,png}
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# Chinese traditional ("amfe.space"-style) academic palette.
# All colors are low-saturation, print-safe, and CB-friendly.
# -------------------------------------------------------------
PAL = {
    "yanzhi":  "#9d2933",   # 胭脂  meaning-bearing / sem
    "jiqing":  "#48929b",   # 霁青  presentation / sur
    "xiangye": "#c89b40",   # 缃叶  emphasis / accent
    "yaqing":  "#5b6f6f",   # 鸦青  neutral / baseline
    "yinhong": "#b58c97",   # 隐红  fourth class
    "yuebai":  "#d8dcd6",   # 月白  light grid / fill
    "moshi":   "#373c38",   # 墨石  axes / text
    "songhua": "#a8b54a",   # 松花  positive accent
}
SEM_COLOR = PAL["yanzhi"]
SUR_COLOR = PAL["jiqing"]

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":       10,
    "axes.titlesize":  10.5,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor":  PAL["moshi"],
    "axes.labelcolor": PAL["moshi"],
    "axes.linewidth":  0.7,
    "xtick.color":     PAL["moshi"],
    "ytick.color":     PAL["moshi"],
    "text.color":      PAL["moshi"],
    "grid.color":      "#d0d0d0",
    "grid.alpha":      0.5,
    "grid.linewidth":  0.4,
    "figure.dpi":      120,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ROOT = "/data/workspace/agentdiff_exp"
OUT  = os.path.join(ROOT, "paper_figs")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    print(f"  -> {name}.pdf, {name}.png")
    plt.close(fig)


# =================================================================
# Figure 1: per-cell delta_matched bars
# =================================================================
def fig1_delta_distribution():
    with open(os.path.join(ROOT, "track_a/_a2_severity_matched.json")) as f:
        d = json.load(f)
    cells = d.get("results_per_cell", []) or d.get("per_cell", []) or d.get("cells", [])
    deltas = []
    for c in cells:
        if isinstance(c, dict):
            dv = c.get("delta_matched") if c.get("delta_matched") is not None else c.get("delta_raw")
            if dv is not None and not (isinstance(dv, float) and (dv != dv)):
                deltas.append(float(dv))
    deltas.sort()
    n = len(deltas)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    colors = [PAL["yaqing"] if x < 0 else SEM_COLOR for x in deltas]
    ax.bar(range(n), deltas, color=colors, width=0.86,
           edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color=PAL["moshi"], linewidth=0.8, zorder=2)
    mean = np.mean(deltas)
    ax.axhline(mean, color=PAL["xiangye"], linestyle=(0, (4, 2)), linewidth=1.4,
               label=f"Mean = {mean:+.2f} pp  (paired $t$=6.76, $p$<0.0001)",
               zorder=4)
    ax.set_xlabel(f"Cells (n={n}, sorted by $\\Delta_{{matched}}$)")
    ax.set_ylabel("$\\Delta_{matched}$ (pp)")
    pos = sum(1 for x in deltas if x > 0)
    ax.set_title(f"Per-cell severity-matched $\\Delta$ across {n} cells "
                 f"({pos}/{n} positive)")
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    ax.set_xticks([])
    ax.grid(axis="y", zorder=0)
    save(fig, "fig1_delta_distribution")


# =================================================================
# Figure 2: severity match -- (a) per-operator severity (b) raw vs matched
# =================================================================
def fig2_severity_match():
    sev_rows = []
    sev_path = os.path.join(ROOT, "track_a/severity_per_variant.jsonl")
    if os.path.exists(sev_path):
        for ln in open(sev_path):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            ed = r.get("edit_distance_norm")
            if r.get("op") and ed is not None:
                r["severity"] = ed
                sev_rows.append(r)
    by_op = defaultdict(list)
    for r in sev_rows:
        by_op[r["op"]].append(r["severity"])
    op_order = ["paraphrase", "synonym", "reorder", "format", "distractor"]
    op_color = {
        "paraphrase": SEM_COLOR, "synonym": SEM_COLOR,
        "reorder":    SUR_COLOR, "format":  SUR_COLOR, "distractor": SUR_COLOR,
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))

    ax = axes[0]
    means = [np.mean(by_op[o]) for o in op_order]
    sds   = [np.std(by_op[o])  for o in op_order]
    ax.bar(op_order, means, yerr=sds,
           color=[op_color[o] for o in op_order],
           edgecolor=PAL["moshi"], linewidth=0.5, capsize=3, zorder=3,
           error_kw={"elinewidth": 0.7, "ecolor": PAL["moshi"]})
    ax.set_ylabel("Normalised edit distance")
    ax.set_title("(a) Per-operator severity")
    # legend chips
    sem_patch = plt.Rectangle((0,0),1,1, fc=SEM_COLOR, ec=PAL["moshi"], lw=0.5)
    sur_patch = plt.Rectangle((0,0),1,1, fc=SUR_COLOR, ec=PAL["moshi"], lw=0.5)
    ax.legend([sem_patch, sur_patch], ["meaning-bearing", "presentation"],
              loc="upper right", frameon=False)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", zorder=0)

    # (b) raw vs matched scatter
    ax = axes[1]
    with open(os.path.join(ROOT, "track_a/_a2_severity_matched.json")) as f:
        d = json.load(f)
    raws, matched = [], []
    cells = d.get("results_per_cell", []) or d.get("per_cell", [])
    for c in cells:
        if isinstance(c, dict) and c.get("delta_raw") is not None and c.get("delta_matched") is not None:
            r, m = c["delta_raw"], c["delta_matched"]
            if r != r or m != m:
                continue
            raws.append(float(r)); matched.append(float(m))
    if raws:
        ax.scatter(raws, matched, s=34, color=SEM_COLOR, alpha=0.78,
                   edgecolor=PAL["moshi"], linewidth=0.45, zorder=3)
        lim = [min(min(raws), min(matched)) - 2, max(max(raws), max(matched)) + 2]
        ax.plot(lim, lim, color=PAL["yaqing"], linestyle=(0, (4, 2)),
                linewidth=0.9, alpha=0.8, label="$y = x$", zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("$\\Delta_{raw}$ (pp)")
    ax.set_ylabel("$\\Delta_{matched}$ (pp)")
    ax.set_title("(b) Severity matching shrinkage  (mean $=0.21$ pp)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(zorder=0)
    fig.tight_layout()
    save(fig, "fig2_severity_match")


# =================================================================
# Figure 3: cascade depth -- exact vs TF-IDF cosine
# =================================================================
def fig3_cascade_gsm8k():
    with open(os.path.join(ROOT, "track_e/embedding_cascade.json")) as f:
        e = json.load(f)
    with open(os.path.join(ROOT, "track_b/hierarchical_bootstrap_cascade.json")) as f:
        b = json.load(f)

    benches = ["gsm8k", "math", "hotpotqa"]
    bench_labels = ["GSM8K", "MATH", "HotpotQA"]
    methods = ["exact", "cos≥0.3", "cos≥0.5", "cos≥0.7"]
    method_color = {
        "exact":   PAL["yaqing"],
        "cos≥0.3": SEM_COLOR,
        "cos≥0.5": SUR_COLOR,
        "cos≥0.7": PAL["xiangye"],
    }

    gaps  = {bench: {} for bench in benches}
    pvals = {bench: {} for bench in benches}
    for bench in benches:
        sub = b.get(bench, {}) or {}
        gaps[bench]["exact"]  = sub.get("real_gap_pooled", sub.get("pooled_gap", 0.0))
        pvals[bench]["exact"] = sub.get("cell_paired_p",   sub.get("cell_p",     1.0))
    pt = e.get("per_threshold", {})
    for thr_str, blob in pt.items():
        for bench, sub in blob.items():
            gaps[bench][f"cos≥{thr_str}"]  = sub.get("gap", 0.0)
            pvals[bench][f"cos≥{thr_str}"] = sub.get("welch_p", 1.0)

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(benches))
    w = 0.20
    for i, m in enumerate(methods):
        ys = [gaps[b].get(m, 0) for b in benches]
        ax.bar(x + (i - 1.5) * w, ys, w, label=m, color=method_color[m],
               edgecolor=PAL["moshi"], linewidth=0.4, zorder=3)
        for j, b in enumerate(benches):
            p = pvals[b].get(m, 1.0)
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if star:
                y = ys[j]
                ax.text(x[j] + (i - 1.5) * w,
                        y + (0.03 if y >= 0 else -0.03),
                        star, ha="center",
                        va="bottom" if y >= 0 else "top",
                        fontsize=8, color=PAL["moshi"])
    ax.axhline(0, color=PAL["moshi"], linewidth=0.6, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(bench_labels)
    ax.set_ylabel("Cascade-depth gap (steps)")
    ax.set_title("Cascade-depth gap, exact vs TF-IDF cosine; $*p\\!<\\!.05, **p\\!<\\!.01, ***p\\!<\\!.001$")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    save(fig, "fig3_cascade_gsm8k")


# =================================================================
# Figure 4: within-benchmark tractability strata
# =================================================================
def fig4_within_benchmark():
    with open(os.path.join(ROOT, "track_d/within_benchmark.json")) as f:
        d = json.load(f)
    table = d.get("aggregate_by_bench_stratum", [])
    benches = ["gsm8k", "math", "hotpotqa"]
    bench_labels = ["GSM8K", "MATH", "HotpotQA"]
    bench_pairs = {
        "gsm8k":    ("multi-route",   "single-route"),
        "math":     ("multi-method",  "single-canonical"),
        "hotpotqa": ("multi-evidence","unique-chain"),
    }
    pos = {row["bench"] + "_" + row["stratum"]: row for row in table}

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(benches)); w = 0.36
    hi_means, lo_means, hi_err, lo_err = [], [], [], []
    for b in benches:
        hi_lab, lo_lab = bench_pairs[b]
        hi = pos.get(b + "_" + hi_lab, {"mean_delta": 0, "sd": 0, "n_cells": 12})
        lo = pos.get(b + "_" + lo_lab, {"mean_delta": 0, "sd": 0, "n_cells": 12})
        hi_means.append(hi["mean_delta"]); lo_means.append(lo["mean_delta"])
        hi_err.append(hi["sd"] / np.sqrt(hi["n_cells"]))
        lo_err.append(lo["sd"] / np.sqrt(lo["n_cells"]))
    ax.bar(x - w / 2, hi_means, w, yerr=hi_err, color=SEM_COLOR,
           label="Tractable (multi-path)",
           edgecolor=PAL["moshi"], linewidth=0.4, capsize=3, zorder=3,
           error_kw={"elinewidth": 0.7, "ecolor": PAL["moshi"]})
    ax.bar(x + w / 2, lo_means, w, yerr=lo_err, color=SUR_COLOR,
           label="Non-tractable (single-path)",
           edgecolor=PAL["moshi"], linewidth=0.4, capsize=3, zorder=3,
           error_kw={"elinewidth": 0.7, "ecolor": PAL["moshi"]})
    ax.axhline(0, color=PAL["moshi"], linewidth=0.6, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(bench_labels)
    ax.set_ylabel("Cell-level $\\Delta$ (pp)")
    ax.set_title("Within-benchmark tractability strata (0/3 contrasts significant)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    save(fig, "fig4_within_benchmark")


# =================================================================
# Figure 5: 3-way generator scatter
# =================================================================
def fig5_genswap():
    with open(os.path.join(ROOT, "track_c/three_way_rank_correlation.json")) as f:
        c = json.load(f)
    rows = c.get("cells_data", [])
    orig, mimo, qw14 = [], [], []
    for r in rows:
        try:
            orig.append(float(r["orig"]["delta"]))
            mimo.append(float(r["mimo"]["delta"]))
            qw14.append(float(r["qwen14b"]["delta"]))
        except (KeyError, TypeError):
            continue

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5))
    pairs = [
        (orig, qw14, "qwen2.5:3b vs qwen2.5:14b\n(within family)",
         "+0.79", "0.019", "+0.71", "0.047", SEM_COLOR),
        (orig, mimo, "qwen2.5:3b vs MiMo-v2.5-Pro\n(across family)",
         "+0.34", "0.41",  "+0.14", "0.74",  SUR_COLOR),
        (mimo, qw14, "MiMo vs qwen2.5:14b\n(across family)",
         "+0.65", "0.082", "+0.52", "0.18",  PAL["xiangye"]),
    ]
    for ax, (xs, ys, title, r, p, rho, prho, col) in zip(axes, pairs):
        ax.scatter(xs, ys, s=46, color=col, alpha=0.85,
                   edgecolor=PAL["moshi"], linewidth=0.45, zorder=3)
        lim = [-15, 35]
        ax.plot(lim, lim, color=PAL["yaqing"], linestyle=(0, (4, 2)),
                linewidth=0.7, alpha=0.7, zorder=2)
        ax.axhline(0, color="grey", linewidth=0.4, zorder=1)
        ax.axvline(0, color="grey", linewidth=0.4, zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("$\\Delta$ (generator A)")
        ax.set_ylabel("$\\Delta$ (generator B)")
        ax.set_title(
            f"{title}\nPearson $r$={r} ($p$={p})\nSpearman $\\rho$={rho} ($p$={prho})",
            fontsize=8.5,
        )
        ax.grid(zorder=0)
    fig.tight_layout()
    save(fig, "fig5_genswap")


if __name__ == "__main__":
    print("Generating publication figures (amfe.space-style palette)...")
    fig1_delta_distribution()
    fig2_severity_match()
    fig3_cascade_gsm8k()
    fig4_within_benchmark()
    fig5_genswap()
    print("Done.")
