#!/usr/bin/env python3
"""
code/merged_analysis_42cells.py

Merge:
  - Old 36 cells from ./results/results_conditional/per_cell_long.csv
    (6 models × 3 benches × 2 agents [cot,react])
  - New 6 cells from Qwen-2.5-14B vLLM run (cot,react × 3 benches)
=> 42 cells total.

Apply the SAME pre-registered partition used in the original dichotomy paper:
  - Group A = (tier ∈ {strong, frontier}) AND (task ∈ {shallow_arith, multi_hop})
  - Group B = (task = deep_math) OR (tier = weak)
  - Neither: mid-tier × {shallow_arith, multi_hop} (excluded by design)

For each group: report mean Δ, positive/total, run Welch t-test and
Mann-Whitney U on (A vs B). Also report Pearson r between accuracy and Δ
across capable cells (acc>=0.65), as in the threshold-based analysis.

Separately report Qwen-2.5-14B 'direct' agent (3 extra cells) as a
robustness exploration outside the partition (since old 36 cells lacked
direct).

Outputs:
  results/conditional_v2/merged_42cells.csv  (long format, all 42 cells)
  results/conditional_v2/merged_42cells.md   (full report)
  results/conditional_v2/merged_42cells.json (machine-readable headline stats)
"""
import csv
import glob
import json
import math
import os
import sys
from collections import Counter

OLD_CSV = "./results/results_conditional/per_cell_long.csv"
NEW_ROOT = "./results/runs_real_qwen25_14b_vllm"
OUT_DIR = "./results/conditional_v2"
os.makedirs(OUT_DIR, exist_ok=True)

SEM_TYPES = {"paraphrase", "synonym"}
SUR_TYPES = {"reorder", "format", "distractor"}

# Map bench name -> task category used by partition rule
BENCH_TO_TASK = {
    "gsm8k": "shallow_arith",
    "math": "deep_math",
    "hotpotqa": "multi_hop",
}


def load_old():
    rows = []
    with open(OLD_CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "slug": r["slug"],
                "display": r["display"],
                "tier": r["tier"],
                "bench": r["bench"],
                "task": r["task"],
                "agent": r["agent"],
                "n": int(r["n"]),
                "accuracy": float(r["accuracy"]),
                "sem_ir": float(r["sem_ir"]),
                "sur_ir": float(r["sur_ir"]),
                "delta": float(r["delta"]),
                "in_A": int(r["in_A"]),
                "in_B": int(r["in_B"]),
                "source": "old36",
            })
    return rows


def load_new_qwen14b():
    """Compute per-cell stats for new Qwen-2.5-14B 9 cells."""
    rows = []
    for bench in ["gsm8k", "math", "hotpotqa"]:
        for agent in ["cot", "react", "direct"]:
            files = sorted(glob.glob(os.path.join(NEW_ROOT, bench, agent, "*.json")))
            n_q = len(files)
            accs = []
            sem_qs = []
            sur_qs = []
            sem_per_q = {t: [] for t in SEM_TYPES}
            sur_per_q = {t: [] for t in SUR_TYPES}
            for fp in files:
                with open(fp) as f:
                    d = json.load(f)
                orig = d.get("original_result") or {}
                if "is_correct" in orig:
                    accs.append(1 if orig["is_correct"] else 0)
                ca = d.get("consistency_analysis") or {}
                pti = ca.get("per_type_inconsistency") or {}
                sem_vals = [pti[t] for t in SEM_TYPES if t in pti]
                sur_vals = [pti[t] for t in SUR_TYPES if t in pti]
                if sem_vals and sur_vals:
                    sem_qs.append(sum(sem_vals) / len(sem_vals))
                    sur_qs.append(sum(sur_vals) / len(sur_vals))
                for t in SEM_TYPES:
                    if t in pti:
                        sem_per_q[t].append(pti[t])
                for t in SUR_TYPES:
                    if t in pti:
                        sur_per_q[t].append(pti[t])
            acc = sum(accs) / len(accs) if accs else float("nan")
            sem_ir = sum(sem_qs) / len(sem_qs) if sem_qs else float("nan")
            sur_ir = sum(sur_qs) / len(sur_qs) if sur_qs else float("nan")
            delta = sem_ir - sur_ir

            tier = "strong"  # Qwen-2.5-14B is in the strong tier
            task = BENCH_TO_TASK[bench]
            in_A = 1 if (tier in ("strong", "frontier") and task in ("shallow_arith", "multi_hop")) else 0
            in_B = 1 if (task == "deep_math" or tier == "weak") else 0
            rows.append({
                "slug": "qwen25_14b",
                "display": "Qwen-2.5-14B",
                "tier": tier,
                "bench": bench,
                "task": task,
                "agent": agent,
                "n": n_q,
                "accuracy": acc,
                "sem_ir": sem_ir,
                "sur_ir": sur_ir,
                "delta": delta,
                "in_A": in_A,
                "in_B": in_B,
                "source": "new14b",
            })
    return rows


def welch_t(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    mx = sum(x) / nx
    my = sum(y) / ny
    vx = sum((a - mx) ** 2 for a in x) / (nx - 1) if nx > 1 else 0.0
    vy = sum((b - my) ** 2 for b in y) / (ny - 1) if ny > 1 else 0.0
    se = math.sqrt(vx / nx + vy / ny) if (vx / nx + vy / ny) > 0 else 0.0
    t = (mx - my) / se if se > 0 else 0.0
    # Welch-Satterthwaite df
    if vx > 0 and vy > 0:
        df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    else:
        df = nx + ny - 2
    try:
        from scipy import stats
        _, p = stats.ttest_ind(x, y, equal_var=False)
        method = "scipy welch"
    except Exception:
        # normal approx for df>=10
        from math import erf, sqrt as msqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / msqrt(2))))
        method = "normal_approx"
    return {"t": t, "df": df, "p": float(p), "method": method,
            "mean_x": mx, "mean_y": my}


def mannwhitney_u(x, y):
    try:
        from scipy import stats
        st, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        return {"U": float(st), "p": float(p), "method": "scipy"}
    except Exception:
        return None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    r = num / (dx * dy) if dx * dy > 0 else 0.0
    try:
        from scipy import stats
        r2, p = stats.pearsonr(xs, ys)
        return {"r": float(r2), "p": float(p), "n": n}
    except Exception:
        # t-stat for r
        t = r * math.sqrt((n - 2) / max(1 - r * r, 1e-9))
        from math import erf, sqrt as msqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / msqrt(2))))
        return {"r": r, "p": float(p), "n": n}


def fisher_exact_2x2(a, b, c, d):
    """Fisher's exact for 2x2: rows=group, cols=outcome. Returns p (two-sided)."""
    try:
        from scipy import stats
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        return {"odds_ratio": float(odds), "p": float(p)}
    except Exception:
        return None


def main():
    old_rows = load_old()
    new_rows = load_new_qwen14b()
    print("loaded old:", len(old_rows), "new:", len(new_rows))

    # write merged CSV
    out_csv = os.path.join(OUT_DIR, "merged_42cells.csv")
    cols = ["slug", "display", "tier", "bench", "task", "agent", "n",
            "accuracy", "sem_ir", "sur_ir", "delta", "in_A", "in_B", "source"]
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in old_rows + new_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print("wrote", out_csv)

    # ===== ANALYSIS =====
    # The partition uses cot/react only; new direct cells are exploration.
    main_rows = [r for r in old_rows + new_rows if r["agent"] in ("cot", "react")]
    direct_rows = [r for r in new_rows if r["agent"] == "direct"]
    print("main partition pool:", len(main_rows), "(36 old + 6 new cot/react)")
    print("direct exploration:", len(direct_rows))

    # Group A and Group B
    A_old = [r for r in old_rows if r["agent"] in ("cot", "react") and r["in_A"] == 1]
    B_old = [r for r in old_rows if r["agent"] in ("cot", "react") and r["in_B"] == 1]
    A_new = [r for r in new_rows if r["agent"] in ("cot", "react") and r["in_A"] == 1]
    B_new = [r for r in new_rows if r["agent"] in ("cot", "react") and r["in_B"] == 1]
    A_all = A_old + A_new
    B_all = B_old + B_new

    deltas_A = [r["delta"] for r in A_all]
    deltas_B = [r["delta"] for r in B_all]
    deltas_A_old = [r["delta"] for r in A_old]
    deltas_B_old = [r["delta"] for r in B_old]

    welch = welch_t(deltas_A, deltas_B)
    mwu = mannwhitney_u(deltas_A, deltas_B)
    welch_old = welch_t(deltas_A_old, deltas_B_old)

    # Pearson r between accuracy and delta (over capable: acc>=0.65) - merged
    capable_main = [r for r in main_rows if r["accuracy"] >= 0.65]
    accs_cap = [r["accuracy"] for r in capable_main]
    deltas_cap = [r["delta"] for r in capable_main]
    r_capable_42 = pearson(accs_cap, deltas_cap)
    # All cells
    accs_all = [r["accuracy"] for r in main_rows]
    deltas_all = [r["delta"] for r in main_rows]
    r_all_42 = pearson(accs_all, deltas_all)

    # Also: positive cells fraction in capable bin, Fisher exact
    cap_pos = sum(1 for r in capable_main if r["delta"] > 0)
    cap_neg = len(capable_main) - cap_pos
    weak_main = [r for r in main_rows if r["accuracy"] < 0.65]
    weak_pos = sum(1 for r in weak_main if r["delta"] > 0)
    weak_neg = len(weak_main) - weak_pos
    fisher = fisher_exact_2x2(cap_pos, cap_neg, weak_pos, weak_neg)

    # ===== WRITE REPORT =====
    out_md = os.path.join(OUT_DIR, "merged_42cells.md")
    with open(out_md, "w") as f:
        f.write("# Merged 42-cell analysis (36 old + 6 new Qwen-2.5-14B cot/react)\n\n")
        f.write("**Question:** does the original Capability×Tractability dichotomy "
                "still hold when we add 6 new high-n (n=200/cell) cells from "
                "Qwen-2.5-14B (a strong-tier, dense, instruction-tuned model "
                "absent from the original 6-model panel)?\n\n")

        f.write("## 1. Pre-registered partition (cot/react agents only)\n\n")
        f.write("Same rules as original paper:\n")
        f.write("- Group A = (tier ∈ {strong, frontier}) AND (task ∈ {shallow_arith, multi_hop})\n")
        f.write("- Group B = (task = deep_math) OR (tier = weak)\n")
        f.write("- Mid × {shallow, multi_hop}: excluded by design\n\n")

        f.write("### 1a. Old-only baseline reproduction (sanity)\n\n")
        f.write("- Group A: n={}, mean Δ = {:+.2f}pp, positive {}/{}\n".format(
            len(A_old), 100 * sum(deltas_A_old) / len(A_old) if A_old else 0,
            sum(1 for d in deltas_A_old if d > 0), len(A_old)))
        f.write("- Group B: n={}, mean Δ = {:+.2f}pp, positive {}/{}\n".format(
            len(B_old), 100 * sum(deltas_B_old) / len(B_old) if B_old else 0,
            sum(1 for d in deltas_B_old if d > 0), len(B_old)))
        if welch_old:
            f.write("- Welch t = {:.3f}, df = {:.1f}, p = {:.4g}\n\n".format(
                welch_old["t"], welch_old["df"], welch_old["p"]))

        f.write("### 1b. NEW: 6 Qwen-2.5-14B cot/react cells alone\n\n")
        f.write("| bench | task | agent | acc | sem_ir | sur_ir | Δ (pp) | group |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in [x for x in new_rows if x["agent"] in ("cot", "react")]:
            grp = "A" if r["in_A"] else ("B" if r["in_B"] else "neither")
            f.write("| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:+.2f} | {} |\n".format(
                r["bench"], r["task"], r["agent"], r["accuracy"],
                r["sem_ir"], r["sur_ir"], r["delta"] * 100, grp))
        f.write("\n")
        f.write("- Group A subset (Qwen-14B): n={}, mean Δ = {:+.2f}pp, positive {}/{}\n".format(
            len(A_new), 100 * sum(r["delta"] for r in A_new) / len(A_new) if A_new else 0,
            sum(1 for r in A_new if r["delta"] > 0), len(A_new)))
        f.write("- Group B subset (Qwen-14B): n={}, mean Δ = {:+.2f}pp, positive {}/{}\n\n".format(
            len(B_new), 100 * sum(r["delta"] for r in B_new) / len(B_new) if B_new else 0,
            sum(1 for r in B_new if r["delta"] > 0), len(B_new)))

        f.write("### 1c. MERGED 42-cell partition test (the headline)\n\n")
        f.write("- Group A: n={}, mean Δ = {:+.2f}pp, positive {}/{}\n".format(
            len(A_all), 100 * sum(deltas_A) / len(A_all) if A_all else 0,
            sum(1 for d in deltas_A if d > 0), len(A_all)))
        f.write("- Group B: n={}, mean Δ = {:+.2f}pp, positive {}/{}\n".format(
            len(B_all), 100 * sum(deltas_B) / len(B_all) if B_all else 0,
            sum(1 for d in deltas_B if d > 0), len(B_all)))
        if welch:
            f.write("- Welch t = {:.3f}, df = {:.1f}, p = {:.4g}\n".format(
                welch["t"], welch["df"], welch["p"]))
        if mwu:
            f.write("- Mann-Whitney U = {:.1f}, p = {:.4g}\n\n".format(mwu["U"], mwu["p"]))

        f.write("## 2. Capability-gating regression (acc vs Δ)\n\n")
        if r_capable_42:
            f.write("- Capable cells (acc>=0.65) in 42-pool: n={}, Pearson r = {:+.3f}, p = {:.4g}\n".format(
                r_capable_42["n"], r_capable_42["r"], r_capable_42["p"]))
        if r_all_42:
            f.write("- All 42 cells: n={}, Pearson r = {:+.3f}, p = {:.4g}\n\n".format(
                r_all_42["n"], r_all_42["r"], r_all_42["p"]))

        f.write("## 3. 2x2 Fisher exact (capable vs weak × Δ>0 vs Δ≤0)\n\n")
        f.write("|         | Δ>0 | Δ≤0 |\n|---|---|---|\n")
        f.write("| capable (acc≥0.65) | {} | {} |\n".format(cap_pos, cap_neg))
        f.write("| weak    (acc<0.65) | {} | {} |\n".format(weak_pos, weak_neg))
        if fisher:
            f.write("\n- Fisher exact two-sided p = {:.4g}, OR = {:.3f}\n\n".format(
                fisher["p"], fisher["odds_ratio"]))

        f.write("## 4. Robustness: Qwen-14B 'direct' agent (3 cells, exploration only)\n\n")
        f.write("Direct agents are NOT in the original 36-cell panel. We report them "
                "separately rather than fold into the partition test.\n\n")
        f.write("| bench | task | acc | sem_ir | sur_ir | Δ (pp) | group |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in direct_rows:
            grp = "A" if r["in_A"] else ("B" if r["in_B"] else "neither")
            f.write("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:+.2f} | {} |\n".format(
                r["bench"], r["task"], r["accuracy"], r["sem_ir"], r["sur_ir"],
                r["delta"] * 100, grp))

    print("wrote", out_md)

    # ===== HEADLINE JSON =====
    headline = {
        "old_n_AB": [len(A_old), len(B_old)],
        "new_n_AB": [len(A_new), len(B_new)],
        "merged_n_AB": [len(A_all), len(B_all)],
        "old_meanDelta_AB_pp": [100 * sum(deltas_A_old) / len(A_old) if A_old else None,
                                100 * sum(deltas_B_old) / len(B_old) if B_old else None],
        "merged_meanDelta_AB_pp": [100 * sum(deltas_A) / len(A_all) if A_all else None,
                                   100 * sum(deltas_B) / len(B_all) if B_all else None],
        "merged_positive_AB": [
            "{}/{}".format(sum(1 for d in deltas_A if d > 0), len(A_all)),
            "{}/{}".format(sum(1 for d in deltas_B if d > 0), len(B_all))],
        "merged_welch": welch,
        "merged_mwu": mwu,
        "old_welch": welch_old,
        "pearson_capable_42": r_capable_42,
        "pearson_all_42": r_all_42,
        "fisher_2x2": {
            "table": [[cap_pos, cap_neg], [weak_pos, weak_neg]],
            "result": fisher,
        },
    }
    out_json = os.path.join(OUT_DIR, "merged_42cells.json")
    with open(out_json, "w") as f:
        json.dump(headline, f, indent=2)
    print("wrote", out_json)

    # ===== CONSOLE SUMMARY =====
    print("\n" + "=" * 60)
    print("HEADLINE: merged 42-cell partition")
    print("  A: n={} mean Δ={:+.2f}pp positive {}/{}".format(
        len(A_all), 100 * sum(deltas_A) / len(A_all),
        sum(1 for d in deltas_A if d > 0), len(A_all)))
    print("  B: n={} mean Δ={:+.2f}pp positive {}/{}".format(
        len(B_all), 100 * sum(deltas_B) / len(B_all),
        sum(1 for d in deltas_B if d > 0), len(B_all)))
    if welch:
        print("  Welch t={:.3f}  df={:.1f}  p={:.4g}".format(
            welch["t"], welch["df"], welch["p"]))
    if mwu:
        print("  MWU  U={:.1f}  p={:.4g}".format(mwu["U"], mwu["p"]))
    print("\nCompared to old-only:")
    if welch_old:
        print("  old Welch t={:.3f}  p={:.4g}".format(welch_old["t"], welch_old["p"]))
    print("\nCapability regression:")
    if r_capable_42:
        print("  capable (n={}): r={:+.3f}  p={:.4g}".format(
            r_capable_42["n"], r_capable_42["r"], r_capable_42["p"]))
    print("\nFisher 2x2 (capable acc>=0.65 vs weak):")
    print("  capable [+/-] = [{}, {}]".format(cap_pos, cap_neg))
    print("  weak    [+/-] = [{}, {}]".format(weak_pos, weak_neg))
    if fisher:
        print("  p = {:.4g}".format(fisher["p"]))


if __name__ == "__main__":
    main()
