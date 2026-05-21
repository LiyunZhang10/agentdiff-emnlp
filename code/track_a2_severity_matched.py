#!/usr/bin/env /usr/bin/python3
"""
Track A2 — Severity-matched Δ analysis (Reviewer R1-Fatal-1, R1-Fatal-4, R2-Fatal-2,
Reverse-Reviewer main attack).

Reviewer concern: if paraphrase/synonym (sem) edit-distance is systematically larger
than reorder/format/distractor (sur), then Δ may simply reflect perturbation severity,
not a semantic-vs-surface distinction.

Method:
  1. For each cell, build a severity-stratified subsample of variants:
     - Bin variants by edit_distance_norm into 10 quantile-based bins.
     - For each bin, count how many sem and sur variants fall in it.
     - The 'matched' subsample takes min(sem_count, sur_count) from each bin.
  2. Recompute IR_sem_matched and IR_sur_matched on the matched subsample.
  3. Report Δ_matched per cell vs Δ_unmatched, plus aggregate distributional gap.

Output: track_a/_a2_severity_matched.json + a markdown table.
"""
import json
import os
import math
import statistics as st
from collections import defaultdict

import numpy as np

ROOT = "."
SEM = {"paraphrase", "synonym"}
SUR = {"reorder", "format", "distractor"}
SEVERITY_FILE = os.path.join(ROOT, "track_a", "severity_per_variant.jsonl")


def load_severity():
    rows = []
    with open(SEVERITY_FILE) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("edit_distance") is None:
                continue
            if r.get("answer_inconsistent") is None:
                continue
            rows.append(r)
    return rows


def main():
    rows = load_severity()
    print(f"Loaded {len(rows)} severity rows with answer_inconsistent != null")

    # Group by cell
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)

    # For each cell, compute Δ_unmatched and Δ_matched
    results = []
    for cell, cell_rows in sorted(by_cell.items()):
        sem_rows = [r for r in cell_rows if r["side"] == "sem"]
        sur_rows = [r for r in cell_rows if r["side"] == "sur"]
        if not sem_rows or not sur_rows:
            continue
        # Unmatched (raw) IR
        ir_sem_raw = (
            sum(1 for r in sem_rows if r["answer_inconsistent"]) / len(sem_rows) * 100
        )
        ir_sur_raw = (
            sum(1 for r in sur_rows if r["answer_inconsistent"]) / len(sur_rows) * 100
        )
        delta_raw = ir_sem_raw - ir_sur_raw

        # Severity matching: bin all rows by edit_distance_norm into 10 quantile bins
        all_eds = [r["edit_distance_norm"] for r in cell_rows]
        if len(all_eds) < 10:
            continue
        # Use 10 bins; handle small samples gracefully
        n_bins = 10
        edges = np.quantile(all_eds, np.linspace(0, 1, n_bins + 1))
        # Ensure unique edges (small cells may have ties)
        edges = np.unique(edges)
        n_bins = len(edges) - 1
        if n_bins < 2:
            continue

        def bin_of(ed):
            return min(np.searchsorted(edges, ed, side="right") - 1, n_bins - 1)

        sem_by_bin = defaultdict(list)
        sur_by_bin = defaultdict(list)
        for r in sem_rows:
            sem_by_bin[bin_of(r["edit_distance_norm"])].append(r)
        for r in sur_rows:
            sur_by_bin[bin_of(r["edit_distance_norm"])].append(r)

        # Matched subsample: take min(|sem|, |sur|) from each bin (paired down-sampling)
        rng = np.random.default_rng(42)
        sem_matched = []
        sur_matched = []
        for b in range(n_bins):
            s_b = sem_by_bin.get(b, [])
            r_b = sur_by_bin.get(b, [])
            k = min(len(s_b), len(r_b))
            if k == 0:
                continue
            sem_idx = rng.choice(len(s_b), size=k, replace=False)
            sur_idx = rng.choice(len(r_b), size=k, replace=False)
            sem_matched.extend([s_b[i] for i in sem_idx])
            sur_matched.extend([r_b[i] for i in sur_idx])

        if not sem_matched or not sur_matched:
            continue
        ir_sem_matched = (
            sum(1 for r in sem_matched if r["answer_inconsistent"]) / len(sem_matched) * 100
        )
        ir_sur_matched = (
            sum(1 for r in sur_matched if r["answer_inconsistent"]) / len(sur_matched) * 100
        )
        delta_matched = ir_sem_matched - ir_sur_matched

        # Mean edit-dist of matched subsample (sanity)
        sem_ed_m = st.mean(r["edit_distance_norm"] for r in sem_matched)
        sur_ed_m = st.mean(r["edit_distance_norm"] for r in sur_matched)

        results.append({
            "cell": cell,
            "n_sem_raw": len(sem_rows),
            "n_sur_raw": len(sur_rows),
            "n_sem_matched": len(sem_matched),
            "n_sur_matched": len(sur_matched),
            "delta_raw": delta_raw,
            "delta_matched": delta_matched,
            "shrinkage": delta_raw - delta_matched,
            "edit_dist_sem_matched": sem_ed_m,
            "edit_dist_sur_matched": sur_ed_m,
            "edit_dist_gap_matched": sem_ed_m - sur_ed_m,
        })

    # Aggregate
    print(f"\n{'Cell':<55} {'Δ_raw':>7} {'Δ_match':>8} {'shrink':>7} {'sem_ed':>6} {'sur_ed':>6}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: x["cell"]):
        print(
            f"  {r['cell'][:53]:<53}  {r['delta_raw']:+6.2f} {r['delta_matched']:+7.2f} "
            f"{r['shrinkage']:+6.2f}  {r['edit_dist_sem_matched']:.3f} {r['edit_dist_sur_matched']:.3f}"
        )

    deltas_raw = [r["delta_raw"] for r in results]
    deltas_match = [r["delta_matched"] for r in results]
    pos_raw = sum(1 for d in deltas_raw if d > 0)
    pos_match = sum(1 for d in deltas_match if d > 0)
    print(f"\n=== Aggregate (n={len(results)} cells) ===")
    print(f"  Mean Δ_raw     = {st.mean(deltas_raw):+.2f}  (pos: {pos_raw}/{len(deltas_raw)})")
    print(f"  Mean Δ_matched = {st.mean(deltas_match):+.2f}  (pos: {pos_match}/{len(deltas_match)})")
    print(f"  Mean shrinkage = {st.mean([r['shrinkage'] for r in results]):+.2f}")

    # Paired t-test: is Δ_matched still > 0?
    from scipy.stats import ttest_1samp, wilcoxon
    t_stat, p_val = ttest_1samp(deltas_match, 0.0)
    print(f"  Paired t-test (Δ_matched vs 0): t={t_stat:+.3f}, p={p_val:.4f}")
    try:
        w_stat, w_p = wilcoxon(deltas_match)
        print(f"  Wilcoxon signed-rank (Δ_matched): W={w_stat:.0f}, p={w_p:.4f}")
    except Exception as e:
        w_p = None
        print(f"  Wilcoxon failed: {e}")

    # Save
    out = {
        "n_cells": len(results),
        "results_per_cell": results,
        "aggregate": {
            "mean_delta_raw": st.mean(deltas_raw),
            "mean_delta_matched": st.mean(deltas_match),
            "mean_shrinkage": st.mean([r["shrinkage"] for r in results]),
            "pos_raw": pos_raw,
            "pos_matched": pos_match,
            "ttest_matched_vs_zero": {"t": t_stat, "p": p_val},
            "wilcoxon_matched_p": w_p,
        },
    }
    out_path = os.path.join(ROOT, "track_a", "_a2_severity_matched.json")
    with open(out_path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(out_path + ".tmp", out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
