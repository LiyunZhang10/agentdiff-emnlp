#!/usr/bin/env /usr/bin/python3
"""
Track B: Robust statistical inference under small-K cluster constraints.

Inputs: existing 36-cell raw data (no new experiments).
Outputs:
  track_b/wild_cluster_bootstrap.json       — wild cluster bootstrap p-values for OLS
  track_b/hierarchical_bootstrap_cascade.json — hierarchical bootstrap on cascade-depth
  track_b/multiple_comparisons_table.json   — FDR-corrected p-value table
  track_b/cell_level_cascade_test.json      — paired cell-level GSM8K cascade test
  track_b/_b_summary.txt                     — human-readable summary

Resilience: deterministic seeded; if interrupted, just rerun. No partial state needed
because all jobs complete in seconds. We still write each artifact atomically.
"""
import json
import os
import sys
import time
import math
import random
import statistics as st
from collections import defaultdict

import numpy as np

ROOT = "/data/workspace/agentdiff_exp"
OUT = os.path.join(ROOT, "track_b")
os.makedirs(OUT, exist_ok=True)
SEED = 42

SEM = {"paraphrase", "synonym"}
SUR = {"reorder", "format", "distractor"}


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_cells():
    """Load the 36 main cells with cell-level Δ, accuracy, topology label,
    and the per-failure cascade-depth lists for hierarchical bootstrap."""
    cells = []
    for d in sorted(os.listdir(ROOT)):
        if not d.startswith("runs_real_"):
            continue
        s = d[len("runs_real_"):]
        if s.endswith("_genmimo"):
            continue
        if not (s.endswith("_fix") or s.endswith("_hpqa") or s == "mimo_v25_pro"):
            continue
        slug = s
        for suf in ("_hpqa", "_fix"):
            if slug.endswith(suf):
                slug = slug[: -len(suf)]
                break
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if not f.endswith(".jsonl"):
                continue
            parts = f.replace(".jsonl", "").split("_")
            bench = parts[0]
            scaf = parts[1]
            nc = nt = 0
            inc_per_op = defaultdict(list)
            cas_pool = {"sem": [], "sur": []}
            cas_per_q = []  # list of dicts {qid, sem_cas:[...], sur_cas:[...]}
            for ln in open(os.path.join(ROOT, d, f)):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                nt += 1
                qid = r.get("sample_id")
                if r.get("original_result", {}).get("is_correct"):
                    nc += 1
                oa = r.get("original_result", {}).get("final_answer")
                q_sem = []
                q_sur = []
                for det in r.get("propagation_details", []):
                    op = det.get("perturbation_type")
                    if op not in SEM and op not in SUR:
                        continue
                    v = det.get("variant_answer")
                    inc = (v is not None and oa is not None and v != oa)
                    inc_per_op[op].append(inc)
                    if inc:
                        depth = det.get("cascade_depth", 0)
                        if op in SEM:
                            cas_pool["sem"].append(depth)
                            q_sem.append(depth)
                        else:
                            cas_pool["sur"].append(depth)
                            q_sur.append(depth)
                cas_per_q.append({"qid": qid, "sem": q_sem, "sur": q_sur})
            if nt == 0:
                continue
            sem_inc = [sum(inc_per_op[k]) / len(inc_per_op[k]) for k in SEM if inc_per_op[k]]
            sur_inc = [sum(inc_per_op[k]) / len(inc_per_op[k]) for k in SUR if inc_per_op[k]]
            sem_mean = sum(sem_inc) / len(sem_inc) if sem_inc else 0
            sur_mean = sum(sur_inc) / len(sur_inc) if sur_inc else 0
            cells.append({
                "model": slug,
                "bench": bench,
                "scaf": scaf,
                "n": nt,
                "acc": nc / nt,
                "delta": (sem_mean - sur_mean) * 100,
                "topo_multi": 1 if bench in ("gsm8k", "hotpotqa") else 0,
                "scaf_react": 1 if scaf == "react" else 0,
                "cas_pool": cas_pool,
                "cas_per_q": cas_per_q,
            })
    return cells


# ============================================================
# Wild Cluster Bootstrap (Cameron, Gelbach & Miller 2008)
# ============================================================
def wild_cluster_bootstrap_ols(cells, x_cols, cluster_col, n_boot=10000, seed=SEED):
    """Wild cluster bootstrap with Rademacher weights.
    Returns: dict with point estimates, naive CR1 SE, wild bootstrap p-values.
    """
    rng = np.random.default_rng(seed)
    Y = np.array([c["delta"] for c in cells])
    X = np.column_stack([np.ones(len(cells))]
                        + [np.array([c[col] for c in cells]) for col in x_cols])
    cluster_ids = sorted(set(c[cluster_col] for c in cells))
    cluster_idx = [
        np.array([i for i, c in enumerate(cells) if c[cluster_col] == cid])
        for cid in cluster_ids
    ]
    K = len(cluster_ids)

    # Point estimate
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    resid = Y - X @ beta

    # CR1 cluster-robust SE (the K=6 baseline we already use)
    middle = np.zeros_like(X.T @ X)
    for idx in cluster_idx:
        u = X[idx].T @ resid[idx]
        middle += np.outer(u, u)
    G = K
    cr1_factor = G / (G - 1)
    cov_cr1 = cr1_factor * (XtX_inv @ middle @ XtX_inv)
    se_cr1 = np.sqrt(np.diag(cov_cr1))
    t_cr1 = beta / se_cr1

    # Wild cluster bootstrap under H0: beta_j = 0 (impose null per coefficient)
    p = X.shape[1]
    p_wild = np.zeros(p)
    for j in range(p):
        # Restricted: estimate model with beta_j forced to 0
        keep = [k for k in range(p) if k != j]
        Xr = X[:, keep]
        beta_r = np.linalg.lstsq(Xr, Y, rcond=None)[0]
        Y_hat_r = Xr @ beta_r
        resid_r = Y - Y_hat_r
        # Bootstrap
        boot_t = np.zeros(n_boot)
        for b in range(n_boot):
            # Rademacher weight per cluster
            w = rng.choice([-1.0, 1.0], size=K)
            Y_star = Y_hat_r.copy()
            for ci, idx in enumerate(cluster_idx):
                Y_star[idx] = Y_hat_r[idx] + w[ci] * resid_r[idx]
            beta_b = XtX_inv @ X.T @ Y_star
            resid_b = Y_star - X @ beta_b
            mid_b = np.zeros_like(X.T @ X)
            for idx in cluster_idx:
                u = X[idx].T @ resid_b[idx]
                mid_b += np.outer(u, u)
            cov_b = cr1_factor * (XtX_inv @ mid_b @ XtX_inv)
            se_b = math.sqrt(cov_b[j, j])
            boot_t[b] = beta_b[j] / se_b if se_b > 0 else 0.0
        # Two-sided p-value
        p_wild[j] = float(np.mean(np.abs(boot_t) >= abs(t_cr1[j])))

    return {
        "n_cells": len(cells),
        "n_clusters": K,
        "cluster_col": cluster_col,
        "labels": ["intercept"] + list(x_cols),
        "beta": beta.tolist(),
        "se_CR1": se_cr1.tolist(),
        "t_CR1": t_cr1.tolist(),
        "p_wild_cluster_bootstrap": p_wild.tolist(),
        "n_bootstrap": n_boot,
    }


# ============================================================
# Hierarchical bootstrap on cascade-depth (question -> cell -> model)
# ============================================================
def hierarchical_bootstrap_cascade(cells, target_bench, n_boot=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    sub_cells = [c for c in cells if c["bench"] == target_bench]
    if not sub_cells:
        return None
    # Real test statistic: pooled Welch t on sem vs sur cascade
    sem_pool_real = [d for c in sub_cells for d in c["cas_pool"]["sem"]]
    sur_pool_real = [d for c in sub_cells for d in c["cas_pool"]["sur"]]
    if not sem_pool_real or not sur_pool_real:
        return None
    real_gap = float(np.mean(sem_pool_real) - np.mean(sur_pool_real))
    # Cell-level paired statistic: per-cell mean cascade gap
    cell_gaps = []
    for c in sub_cells:
        if c["cas_pool"]["sem"] and c["cas_pool"]["sur"]:
            cell_gaps.append(np.mean(c["cas_pool"]["sem"]) - np.mean(c["cas_pool"]["sur"]))
    cell_mean_gap = float(np.mean(cell_gaps)) if cell_gaps else None
    cell_se = float(np.std(cell_gaps, ddof=1) / math.sqrt(len(cell_gaps))) if cell_gaps else None
    cell_t = cell_mean_gap / cell_se if (cell_se and cell_se > 0) else None
    # cell-level p (two-sided, t with df=K-1)
    from scipy.stats import t as tdist
    cell_p = (
        2 * (1 - tdist.cdf(abs(cell_t), len(cell_gaps) - 1))
        if cell_t is not None
        else None
    )

    # Hierarchical bootstrap: resample (model, cell, question) hierarchically
    models = sorted(set(c["model"] for c in sub_cells))
    by_model = {m: [c for c in sub_cells if c["model"] == m] for m in models}
    boot_gaps = np.zeros(n_boot)
    for b in range(n_boot):
        # Step 1: resample models
        m_sample = rng.choice(models, size=len(models), replace=True)
        boot_sem, boot_sur = [], []
        for m in m_sample:
            cells_m = by_model[m]
            # Step 2: resample cells within model
            cell_sample = rng.choice(len(cells_m), size=len(cells_m), replace=True)
            for ci in cell_sample:
                c = cells_m[ci]
                qs = c["cas_per_q"]
                # Step 3: resample questions
                q_sample = rng.choice(len(qs), size=len(qs), replace=True)
                for qi in q_sample:
                    boot_sem.extend(qs[qi]["sem"])
                    boot_sur.extend(qs[qi]["sur"])
        if boot_sem and boot_sur:
            boot_gaps[b] = np.mean(boot_sem) - np.mean(boot_sur)
        else:
            boot_gaps[b] = 0.0
    boot_gaps_sorted = np.sort(boot_gaps)
    ci_lo = float(boot_gaps_sorted[int(0.025 * n_boot)])
    ci_hi = float(boot_gaps_sorted[int(0.975 * n_boot)])
    # Two-sided percentile-bootstrap p-value: 2 * min(P(boot <= 0), P(boot >= 0)).
    # This is the standard "p-value via inversion of bootstrap CI" definition
    # (Davison & Hinkley 1997, §4.2; agrees with the CI: p < 0.05 iff CI excludes 0).
    p_le0 = float(np.mean(boot_gaps <= 0))
    p_ge0 = float(np.mean(boot_gaps >= 0))
    p_hier = float(min(2 * min(p_le0, p_ge0), 1.0))
    return {
        "benchmark": target_bench,
        "n_cells": len(sub_cells),
        "n_models": len(models),
        "real_gap_pooled": real_gap,
        "real_gap_cell_level_mean": cell_mean_gap,
        "cell_paired_t": cell_t,
        "cell_paired_p": cell_p,
        "cell_n": len(cell_gaps),
        "hierarchical_bootstrap_n": n_boot,
        "hierarchical_CI95": [ci_lo, ci_hi],
        "hierarchical_p_two_sided": p_hier,
    }


def main():
    t0 = time.time()
    cells = load_cells()
    print(f"[B] Loaded {len(cells)} cells")

    # B1. Wild cluster bootstrap on the headline OLS
    print(f"[B1] Wild cluster bootstrap (cluster=model)...")
    out_b1 = wild_cluster_bootstrap_ols(
        cells, x_cols=["topo_multi", "acc"], cluster_col="model", n_boot=10000
    )
    atomic_write_json(os.path.join(OUT, "wild_cluster_bootstrap.json"), out_b1)
    print(f"[B1] Done. Coefficients with wild bootstrap p-values:")
    for label, b, se, t_v, p_v in zip(
        out_b1["labels"], out_b1["beta"], out_b1["se_CR1"], out_b1["t_CR1"],
        out_b1["p_wild_cluster_bootstrap"]
    ):
        print(f"  {label:<20} beta={b:+8.3f}  CR1_SE={se:5.3f}  t={t_v:+.2f}  p_wild={p_v:.4f}")

    # Also run with scaffold
    out_b1b = wild_cluster_bootstrap_ols(
        cells, x_cols=["topo_multi", "acc", "scaf_react"], cluster_col="model", n_boot=10000
    )
    atomic_write_json(os.path.join(OUT, "wild_cluster_bootstrap_with_scaffold.json"), out_b1b)
    print(f"[B1b] With scaffold dummy:")
    for label, b, se, t_v, p_v in zip(
        out_b1b["labels"], out_b1b["beta"], out_b1b["se_CR1"], out_b1b["t_CR1"],
        out_b1b["p_wild_cluster_bootstrap"]
    ):
        print(f"  {label:<20} beta={b:+8.3f}  CR1_SE={se:5.3f}  t={t_v:+.2f}  p_wild={p_v:.4f}")

    # B2. Hierarchical bootstrap on cascade depth, per benchmark
    print(f"[B2] Hierarchical bootstrap on cascade depth...")
    hier = {}
    for bench in ("gsm8k", "math", "hotpotqa"):
        h = hierarchical_bootstrap_cascade(cells, bench, n_boot=5000)
        hier[bench] = h
        if h:
            print(f"  {bench:<10}  pooled_gap={h['real_gap_pooled']:+.3f}  "
                  f"cell_paired_t={h['cell_paired_t']:+.3f} cell_p={h['cell_paired_p']:.4f}  "
                  f"hier_CI=[{h['hierarchical_CI95'][0]:+.3f},{h['hierarchical_CI95'][1]:+.3f}]  "
                  f"hier_p={h['hierarchical_p_two_sided']:.4f}")
    atomic_write_json(os.path.join(OUT, "hierarchical_bootstrap_cascade.json"), hier)

    # B3. Multiple comparisons table with BH FDR
    print(f"[B3] Multiple comparisons table with BH FDR...")
    pvals = []
    pvals.append(("Topology coefficient (wild boot, cluster=model)", out_b1["p_wild_cluster_bootstrap"][1]))
    pvals.append(("Accuracy coefficient (wild boot, cluster=model)", out_b1["p_wild_cluster_bootstrap"][2]))
    pvals.append(("GSM8K cascade gap (cell-level paired t)", hier["gsm8k"]["cell_paired_p"] if hier["gsm8k"] else 1.0))
    pvals.append(("MATH cascade gap (cell-level paired t)", hier["math"]["cell_paired_p"] if hier["math"] else 1.0))
    pvals.append(("HotpotQA cascade gap (cell-level paired t)", hier["hotpotqa"]["cell_paired_p"] if hier["hotpotqa"] else 1.0))
    # Add prior tests
    from scipy.stats import pearsonr, fisher_exact
    accs = [c["acc"] for c in cells]
    deltas = [c["delta"] for c in cells]
    rho, pp = pearsonr(accs, deltas)
    pvals.append(("Pearson r(acc, Δ)", pp))
    hi = [c for c in cells if c["acc"] >= 0.65]
    lo = [c for c in cells if c["acc"] < 0.65]
    h_pos = sum(1 for c in hi if c["delta"] > 0)
    l_pos = sum(1 for c in lo if c["delta"] > 0)
    _, pf = fisher_exact([[h_pos, len(hi) - h_pos], [l_pos, len(lo) - l_pos]],
                         alternative="greater")
    pvals.append(("Fisher exact (T=0.65 split, one-sided)", pf))
    # BH FDR
    sorted_pv = sorted(enumerate(pvals), key=lambda x: x[1][1])
    m = len(pvals)
    qvals = [None] * m
    prev_q = 1.0
    for rank_minus1, (orig_idx, (label, p)) in enumerate(reversed(sorted_pv)):
        rank = m - rank_minus1
        bh = p * m / rank
        prev_q = min(prev_q, bh)
        qvals[orig_idx] = prev_q
    out_b3 = []
    for (label, p), q in zip(pvals, qvals):
        out_b3.append({"test": label, "p": p, "BH_q": q})
    atomic_write_json(os.path.join(OUT, "multiple_comparisons_table.json"), out_b3)
    print("  Test                                                    p           BH_q")
    for row in out_b3:
        print(f"    {row['test']:<55} {row['p']:.4f}    {row['BH_q']:.4f}")

    elapsed = time.time() - t0
    summary = (
        f"Track B complete in {elapsed:.1f}s\n"
        f"  Headline topology p_wild = {out_b1['p_wild_cluster_bootstrap'][1]:.4f}\n"
        f"  Headline accuracy p_wild = {out_b1['p_wild_cluster_bootstrap'][2]:.4f}\n"
        f"  GSM8K cascade cell-level p = {hier['gsm8k']['cell_paired_p']:.4f}\n"
        f"  MATH cascade cell-level p = {hier['math']['cell_paired_p']:.4f}\n"
        f"  HotpotQA cascade cell-level p = {hier['hotpotqa']['cell_paired_p']:.4f}\n"
    )
    with open(os.path.join(OUT, "_b_summary.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
