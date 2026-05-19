#!/usr/bin/env /usr/bin/python3
"""
Track C: Family-level wild cluster bootstrap (K=3 reviewer-attack response).

Reviewer 1 / 2 / 3 共识攻击点：
  - 论文用 K=6 model-level cluster 报告的 OLS p-value，但 6 个 model 中
    qwen2.5 1B/3B/7B/14B 是同一 family 的不同 size，
    llama-3.1 8B / llama-3.2 1B/3B 是同一 family 的不同 size，
    MiMo-v2.5-pro 是单独 family。真实独立 family 数 K ≈ 3。
  - K=3 cluster bootstrap 必然比 K=6 更宽（更小 power）。
  - 我们必须如实跑出来并报告。

This script:
  1. Reuses load_cells() and wild_cluster_bootstrap_ols() from track_b.
  2. Adds a "family" field to each cell:
       qwen     ← qwen25_1b qwen25_3b qwen25_7b qwen25_14b
       llama    ← llama32_1b llama32_3b llama31_8b
       mimo     ← mimo_v25_pro
  3. Runs wild bootstrap with cluster_col = "family" (K=3).
  4. Runs additional K=2 sanity (qwen vs not-qwen) for the worst case.
  5. Writes track_c/wild_cluster_bootstrap_family.json
            track_c/wild_cluster_bootstrap_qwen_vs_other.json
            track_c/_c_summary.txt

Resilience: deterministic, no network. Reruns idempotent.
"""
import json
import os
import sys
import time
import importlib.util

import numpy as np

ROOT = "/data/workspace/agentdiff_exp"
OUT = os.path.join(ROOT, "track_c")
os.makedirs(OUT, exist_ok=True)

# Import functions from track_b without running its main()
spec = importlib.util.spec_from_file_location(
    "track_b", os.path.join(ROOT, "track_b_robust_inference.py")
)
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)


# ---- model -> family map ---------------------------------------------------
def model_to_family(model_slug):
    """The model field comes from load_cells(); see track_b.load_cells.
    Inspecting the existing wild_cluster_bootstrap.json: 6 models. We need to
    map each to a family. We list the substrings we expect in the slug field.
    """
    s = model_slug.lower()
    if "qwen" in s:
        return "qwen"
    if "llama" in s:
        return "llama"
    if "mimo" in s:
        return "mimo"
    return "other"


def model_to_qwen_vs_other(model_slug):
    return "qwen" if "qwen" in model_slug.lower() else "other"


def attach_family(cells):
    for c in cells:
        c["family"] = model_to_family(c["model"])
        c["qwen_vs_other"] = model_to_qwen_vs_other(c["model"])
    return cells


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    t0 = time.time()
    cells = tb.load_cells()
    cells = attach_family(cells)
    print(f"[C] Loaded {len(cells)} cells")

    # Print family distribution
    from collections import Counter
    fam_count = Counter(c["family"] for c in cells)
    mod_per_fam = {f: sorted(set(c["model"] for c in cells if c["family"] == f)) for f in fam_count}
    print(f"[C] Family distribution:")
    for f, n in fam_count.items():
        print(f"  {f:<10} n_cells={n}  models={mod_per_fam[f]}")

    # ---- C1. K = 3 family-level wild bootstrap on the headline OLS ----
    print(f"\n[C1] Wild cluster bootstrap with cluster=family (K=3)...")
    out_c1 = tb.wild_cluster_bootstrap_ols(
        cells, x_cols=["topo_multi", "acc"], cluster_col="family", n_boot=10000
    )
    atomic_write_json(os.path.join(OUT, "wild_cluster_bootstrap_family.json"), out_c1)
    print(f"[C1] Coefficients with family-cluster wild-bootstrap p-values:")
    for label, b, se, t_v, p_v in zip(
        out_c1["labels"], out_c1["beta"], out_c1["se_CR1"], out_c1["t_CR1"],
        out_c1["p_wild_cluster_bootstrap"]
    ):
        print(f"  {label:<20} beta={b:+8.3f}  CR1_SE={se:5.3f}  t={t_v:+.2f}  p_wild={p_v:.4f}")

    # ---- C1b. With scaffold dummy ----
    print(f"\n[C1b] With scaffold dummy (cluster=family, K=3)...")
    out_c1b = tb.wild_cluster_bootstrap_ols(
        cells, x_cols=["topo_multi", "acc", "scaf_react"],
        cluster_col="family", n_boot=10000
    )
    atomic_write_json(os.path.join(OUT, "wild_cluster_bootstrap_family_with_scaffold.json"), out_c1b)
    for label, b, se, t_v, p_v in zip(
        out_c1b["labels"], out_c1b["beta"], out_c1b["se_CR1"], out_c1b["t_CR1"],
        out_c1b["p_wild_cluster_bootstrap"]
    ):
        print(f"  {label:<20} beta={b:+8.3f}  CR1_SE={se:5.3f}  t={t_v:+.2f}  p_wild={p_v:.4f}")

    # ---- C2. Worst-case K = 2 qwen-vs-other ----
    print(f"\n[C2] Wild cluster bootstrap with cluster=qwen_vs_other (K=2 worst case)...")
    out_c2 = tb.wild_cluster_bootstrap_ols(
        cells, x_cols=["topo_multi", "acc"], cluster_col="qwen_vs_other", n_boot=10000
    )
    atomic_write_json(os.path.join(OUT, "wild_cluster_bootstrap_qwen_vs_other.json"), out_c2)
    for label, b, se, t_v, p_v in zip(
        out_c2["labels"], out_c2["beta"], out_c2["se_CR1"], out_c2["t_CR1"],
        out_c2["p_wild_cluster_bootstrap"]
    ):
        print(f"  {label:<20} beta={b:+8.3f}  CR1_SE={se:5.3f}  t={t_v:+.2f}  p_wild={p_v:.4f}")

    # ---- C3. Headline-only paired-t restricted to non-qwen cells ----
    # If reviewer claims "+14.32pp is qwen-family-driven", we test it in non-qwen cells alone.
    non_qwen_cells = [c for c in cells if c["family"] != "qwen"]
    deltas_nq = [c["delta"] for c in non_qwen_cells]
    n_nq = len(deltas_nq)
    pos_nq = sum(1 for d in deltas_nq if d > 0)
    mean_nq = float(np.mean(deltas_nq)) if deltas_nq else 0.0
    se_nq = float(np.std(deltas_nq, ddof=1) / np.sqrt(n_nq)) if n_nq > 1 else 0.0
    from scipy.stats import t as tdist
    t_nq = mean_nq / se_nq if se_nq > 0 else 0.0
    p_nq = 2 * (1 - tdist.cdf(abs(t_nq), n_nq - 1)) if n_nq > 1 else 1.0
    out_c3 = {
        "n_non_qwen_cells": n_nq,
        "pos_count": pos_nq,
        "delta_mean_pp": mean_nq,
        "delta_se_pp": se_nq,
        "t_one_sample": t_nq,
        "p_two_sided": float(p_nq),
        "non_qwen_models": sorted(set(c["model"] for c in non_qwen_cells)),
    }
    atomic_write_json(os.path.join(OUT, "headline_non_qwen_only.json"), out_c3)
    print(f"\n[C3] Headline +Δ paired-t on NON-qwen cells only:")
    print(f"  n={n_nq}  pos={pos_nq}/{n_nq}  mean Δ={mean_nq:+.2f}pp  "
          f"SE={se_nq:.2f}  t={t_nq:+.2f}  p={p_nq:.4f}")

    elapsed = time.time() - t0
    summary = (
        f"Track C complete in {elapsed:.1f}s\n"
        f"  K=3 family-cluster:\n"
        f"    topology p_wild = {out_c1['p_wild_cluster_bootstrap'][1]:.4f}\n"
        f"    accuracy p_wild = {out_c1['p_wild_cluster_bootstrap'][2]:.4f}\n"
        f"  K=2 qwen-vs-other:\n"
        f"    topology p_wild = {out_c2['p_wild_cluster_bootstrap'][1]:.4f}\n"
        f"    accuracy p_wild = {out_c2['p_wild_cluster_bootstrap'][2]:.4f}\n"
        f"  Headline non-qwen Δ: n={n_nq}, mean={mean_nq:+.2f}pp, p={p_nq:.4f}\n"
    )
    with open(os.path.join(OUT, "_c_summary.txt"), "w") as f:
        f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
