#!/usr/bin/env /usr/bin/python3
"""
Track C analysis — three-generator rank correlation (Reviewer 2 / E4).

Input cells:
  * runs_real_<slug>/                  → original generator (qwen2.5:3b/llama3.2:3b)
  * runs_real_<slug>_genmimo/          → second generator (mimo-v2.5-pro)
  * runs_real_<slug>_genqwen14b/       → third generator (qwen2.5:14b)

For each of the 8 paired cells, compute Δ = IR_sem − IR_sur with each generator.
Then compute pairwise Pearson and Spearman rank correlations across the three
generators on the 8-cell paired Δ vectors.

Decision rule:
  * If all 3 pairwise rho ≥ 0.7  →  ordering claim rescued
  * If any pair rho ≤ 0.3        →  downgrade: declare framework single-generator-only
  * Else                          →  intermediate; report honestly

Output: track_c/three_way_rank_correlation.json + track_c/_c_summary.txt
"""
import json
import os
import math
import statistics as st
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = "."
SEM = {"paraphrase", "synonym"}
SUR = {"reorder", "format", "distractor"}

# 8 paired cells (matches genswap design)
PAIRED_CELLS = [
    # (slug_orig,        bench,    scaf,   needs_hpqa_suffix_for_orig)
    ("mimo_v25_pro",       "gsm8k",    "cot",   False),
    ("mimo_v25_pro",       "gsm8k",    "react", False),
    ("mimo_v25_pro_hpqa",  "hotpotqa", "cot",   False),
    ("mimo_v25_pro_hpqa",  "hotpotqa", "react", False),
    ("llama31_8b_fix",     "gsm8k",    "cot",   False),
    ("qwen25_7b_fix",      "gsm8k",    "cot",   False),
    ("llama32_3b_fix",     "gsm8k",    "cot",   False),
    ("qwen25_3b_fix",      "gsm8k",    "cot",   False),
]


def cell_delta(jsonl_path):
    """Return (n_originals, IR_sem, IR_sur, Δ) from a cell jsonl, or None if missing."""
    if not os.path.exists(jsonl_path):
        return None
    sem_inc = []
    sur_inc = []
    n = 0
    for ln in open(jsonl_path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        n += 1
        oa = r.get("original_result", {}).get("final_answer")
        for det in r.get("propagation_details", []):
            op = det.get("perturbation_type")
            if op not in SEM and op not in SUR:
                continue
            v = det.get("variant_answer")
            inc = (v is not None and oa is not None and v != oa)
            if op in SEM:
                sem_inc.append(inc)
            else:
                sur_inc.append(inc)
    if n == 0 or not sem_inc or not sur_inc:
        return None
    sem_ir = sum(sem_inc) / len(sem_inc) * 100
    sur_ir = sum(sur_inc) / len(sur_inc) * 100
    return {"n": n, "IR_sem": sem_ir, "IR_sur": sur_ir, "delta": sem_ir - sur_ir}


def main():
    cells_data = []  # one row per (slug, bench, scaf): orig, mimo, qwen14b
    for slug, bench, scaf, _ in PAIRED_CELLS:
        # Original generator
        orig_path = os.path.join(ROOT, f"runs_real_{slug}", f"{bench}_{scaf}_real_{slug}.jsonl")
        # Mimo generator
        mimo_path = os.path.join(
            ROOT, f"runs_real_{slug}_genmimo",
            f"{bench}_{scaf}_real_{slug}_genmimo.jsonl",
        )
        # Qwen14b generator (third)
        q14_path = os.path.join(
            ROOT, f"runs_real_{slug}_genqwen14b",
            f"{bench}_{scaf}_real_{slug}_genqwen14b.jsonl",
        )
        orig = cell_delta(orig_path)
        mimo = cell_delta(mimo_path)
        q14 = cell_delta(q14_path)
        cells_data.append({
            "slug": slug, "bench": bench, "scaf": scaf,
            "orig": orig, "mimo": mimo, "qwen14b": q14,
        })

    # Print per-cell table
    print(f"{'Cell':<35} {'orig Δ':>8} {'mimo Δ':>8} {'qwen14b Δ':>10}")
    print("-" * 65)
    valid_orig, valid_mimo, valid_q14 = [], [], []
    for c in cells_data:
        name = f"{c['slug']}/{c['bench']}/{c['scaf']}"
        o = c["orig"]["delta"] if c["orig"] else None
        m = c["mimo"]["delta"] if c["mimo"] else None
        q = c["qwen14b"]["delta"] if c["qwen14b"] else None
        os_ = f"{o:+.2f}" if o is not None else "  --"
        ms = f"{m:+.2f}" if m is not None else "  --"
        qs = f"{q:+.2f}" if q is not None else "    --"
        print(f"  {name:<33} {os_:>8} {ms:>8} {qs:>10}")
        if o is not None and m is not None and q is not None:
            valid_orig.append(o); valid_mimo.append(m); valid_q14.append(q)

    if len(valid_orig) < 3:
        print(f"\nNot enough complete triplets ({len(valid_orig)}) for correlations; need ≥3.")
        return

    # Pairwise correlations
    pairs = [
        ("orig vs mimo", valid_orig, valid_mimo),
        ("orig vs qwen14b", valid_orig, valid_q14),
        ("mimo vs qwen14b", valid_mimo, valid_q14),
    ]
    out = {"n_paired_cells": len(valid_orig), "cells_data": cells_data, "correlations": {}}
    print(f"\n=== Three-way pairwise correlations (n={len(valid_orig)}) ===")
    print(f"{'Pair':<25}  {'Pearson r':>10}  {'p':>7}  {'Spearman ρ':>11}  {'p':>7}")
    for label, x, y in pairs:
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        out["correlations"][label] = {
            "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp,
        }
        print(f"  {label:<23}  {pr:>+10.3f}  {pp:>.4f}  {sr:>+11.3f}  {sp:>.4f}")

    # Decision rule: rank correlation thresholds
    rhos = [out["correlations"][p[0]]["spearman_rho"] for p in pairs]
    out["min_rho"] = min(rhos) if rhos else None
    out["max_rho"] = max(rhos) if rhos else None
    if all(r >= 0.7 for r in rhos):
        verdict = "STRONG_RESCUE: ordering preserved across all three generators (all ρ ≥ 0.7)"
    elif any(r <= 0.3 for r in rhos):
        verdict = "WEAK: at least one pair ρ ≤ 0.3 → framework should be re-scoped"
    else:
        verdict = "MIXED: 0.3 < min ρ < 0.7 → ordering partially preserved; report cautiously"
    out["verdict"] = verdict
    print(f"\nDECISION: {verdict}")

    # Write artifacts
    os.makedirs(os.path.join(ROOT, "track_c"), exist_ok=True)
    out_path = os.path.join(ROOT, "track_c", "three_way_rank_correlation.json")
    with open(out_path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(out_path + ".tmp", out_path)

    summary_path = os.path.join(ROOT, "track_c", "_c_summary.txt")
    with open(summary_path, "w") as f:
        f.write(
            f"Track C complete. n_paired_cells={len(valid_orig)}.\n"
            f"Pearson r and Spearman ρ for each pair:\n"
        )
        for label, _, _ in pairs:
            c = out["correlations"][label]
            f.write(
                f"  {label:<22} Pearson r={c['pearson_r']:+.3f} (p={c['pearson_p']:.4f})  "
                f"Spearman ρ={c['spearman_rho']:+.3f} (p={c['spearman_p']:.4f})\n"
            )
        f.write(f"Min ρ = {out['min_rho']:+.3f}, max ρ = {out['max_rho']:+.3f}\n")
        f.write(f"Verdict: {verdict}\n")
    print(f"\nSaved → {out_path}\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
