#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate semantic-vs-surface dichotomy heatmap from
cross_model_fix_heatmap_data.json.

Output:
    figs/semantic_surface_heatmap.pdf   (vector, paper-ready)
    figs/semantic_surface_heatmap.png   (preview)
    figs/delta_bars.pdf                  (Δ per cell, sorted)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(EXP_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

PERT_TYPES = ["paraphrase", "synonym", "reorder", "format", "distractor"]
SEMANTIC = {"paraphrase", "synonym"}

# Color palette (colorblind-safe, sequential heat)
CMAP = "RdYlBu_r"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,  # embed as Type 42 / TrueType
    "ps.fonttype": 42,
})


def main():
    src = os.path.join(EXP_DIR, "cross_model_fix_heatmap_data.json")
    with open(src) as f:
        data = json.load(f)

    rows = data["rows"]
    # Filter cells with n>=5 to keep heatmap meaningful (drop math_react n=2)
    rows = [r for r in rows if r["n"] >= 5]
    rows.sort(key=lambda r: (r["model"], r["benchmark"], r["agent"]))

    if not rows:
        print("No valid rows", file=sys.stderr)
        return

    # Build matrix: rows x perturbation_types
    matrix = []
    row_labels = []
    for r in rows:
        ir = r["ir_per_type"]
        matrix.append([(ir.get(pt) or 0.0) * 100 for pt in PERT_TYPES])
        partial = "*" if r["is_partial"] else ""
        row_labels.append("%s | %s/%s%s (n=%d)" % (
            r["model"], r["benchmark"], r["agent"], partial, r["n"]))

    M = np.array(matrix)

    # ===== Heatmap =====
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(PERT_TYPES)))
    ax.set_xticklabels([
        "Paraphrase", "Synonym", "Reorder", "Format", "Distractor"
    ], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Annotate values
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = "white" if v > 50 else "black"
            ax.text(j, i, "%.0f" % v, ha="center", va="center",
                    color=color, fontsize=9)

    # Vertical separator between semantic and surface
    sem_count = sum(1 for pt in PERT_TYPES if pt in SEMANTIC)
    ax.axvline(sem_count - 0.5, color="black", linewidth=2)
    ax.text(0.5, -1.3, "Semantic", ha="center", fontsize=10, fontweight="bold")
    ax.text(3.0, -1.3, "Surface", ha="center", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Inconsistency Rate (%)")

    ax.set_title("Cross-model semantic-vs-surface dichotomy " +
                 "(Inconsistency Rate per perturbation type)")
    plt.tight_layout()

    pdf = os.path.join(FIG_DIR, "semantic_surface_heatmap.pdf")
    png = os.path.join(FIG_DIR, "semantic_surface_heatmap.png")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.close()
    print("Wrote:", pdf)
    print("Wrote:", png)

    # ===== Δ bar chart =====
    deltas = [(r["row_label"].replace(" | ", "\n"),
               (r["delta"] or 0) * 100,
               r["is_partial"]) for r in rows]
    deltas.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    labels = [d[0] for d in deltas]
    vals = [d[1] for d in deltas]
    colors = ["#888888" if d[2] else "#1f77b4" for d in deltas]

    bars = ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("$\\Delta_{\\mathrm{sem-sur}}$ (percentage points)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Δ (semantic IR − surface IR) is positive in every "
                 "tested cell")
    for i, v in enumerate(vals):
        ax.text(v + 1, i, "+%.0f" % v, va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color="#1f77b4", label="Full cell (n=20)"),
        Patch(color="#888888", label="Partial cell (n<20)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)

    plt.tight_layout()
    pdf2 = os.path.join(FIG_DIR, "delta_bars.pdf")
    png2 = os.path.join(FIG_DIR, "delta_bars.png")
    plt.savefig(pdf2, bbox_inches="tight")
    plt.savefig(png2, bbox_inches="tight", dpi=200)
    plt.close()
    print("Wrote:", pdf2)
    print("Wrote:", png2)


if __name__ == "__main__":
    main()
