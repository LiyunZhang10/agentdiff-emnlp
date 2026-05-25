#!/usr/bin/env python3
"""
code/aggregate_qwen25_14b.py

Aggregate per-cell statistics for the new Qwen-2.5-14B 1800-trajectory run
(9 cells × 200 questions = 1800).

For each cell we compute, mirroring the existing 26-cell analysis schema:
  - accuracy (original_result.is_correct)
  - n_variants_per_q (avg)
  - sem inconsistency rate    P(answer_changes | sem variant)
  - sur inconsistency rate    P(answer_changes | sur variant)
  - delta = sem_rate - sur_rate  (in percentage points)
  - paired t-test on per-question (sem_rate_q - sur_rate_q)
  - bootstrap 95% CI on delta

Outputs:
  results/conditional_v2/qwen25_14b_per_cell.json   (machine-readable)
  results/conditional_v2/qwen25_14b_per_cell.md     (human-readable table)
"""
import json
import math
import os
import glob
import sys
from collections import defaultdict

ROOT = "/data/workspace/agentdiff-emnlp/results/runs_real_qwen25_14b_vllm"
OUT_DIR = "/data/workspace/agentdiff-emnlp/results/conditional_v2"
os.makedirs(OUT_DIR, exist_ok=True)

SEM_TYPES = {"paraphrase", "synonym"}
SUR_TYPES = {"reorder", "format", "distractor"}


def load_cell(bench, agent):
    paths = sorted(glob.glob(os.path.join(ROOT, bench, agent, "*.json")))
    rows = []
    for p in paths:
        try:
            with open(p) as f:
                rows.append(json.load(f))
        except Exception as e:
            print("skip", p, "err", e, file=sys.stderr)
    return rows


def is_correct_of(record):
    """Original answer correctness."""
    orig = record.get("original_result") or {}
    if "is_correct" in orig:
        return bool(orig["is_correct"])
    return None


def per_question_rates(record):
    """Return (sem_rate_q, sur_rate_q) for this question, or (None, None).

    Uses consistency_analysis.per_type_inconsistency which is the canonical
    field produced by AgentDiffPipelineV2. Each entry in {0.0, 1.0} since
    each question has exactly 1 variant per type.
    """
    ca = record.get("consistency_analysis") or {}
    pti = ca.get("per_type_inconsistency") or {}
    if not pti:
        return None, None
    sem_vals = [pti[t] for t in SEM_TYPES if t in pti]
    sur_vals = [pti[t] for t in SUR_TYPES if t in pti]
    sem_q = sum(sem_vals) / len(sem_vals) if sem_vals else None
    sur_q = sum(sur_vals) / len(sur_vals) if sur_vals else None
    return sem_q, sur_q


def paired_t(diffs):
    n = len(diffs)
    if n < 2:
        return {"n": n, "p": float("nan"), "t": float("nan"),
                "mean": float("nan"), "sd": float("nan")}
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    t = mean / se if se > 0 else 0.0
    try:
        from scipy import stats
        _, p = stats.ttest_1samp(diffs, 0.0)
        method = "scipy"
    except Exception:
        from math import erf, sqrt as msqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / msqrt(2))))
        method = "normal_approx"
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": float(p), "method": method}


def bootstrap_ci(diffs, B=2000, seed=42):
    if not diffs:
        return None
    import random
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(B):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    return [boots[int(0.025 * B)], boots[int(0.975 * B)]]


def analyze_cell(bench, agent, rows):
    n_total = len(rows)
    correct = [is_correct_of(r) for r in rows]
    correct = [c for c in correct if c is not None]
    acc = sum(correct) / len(correct) if correct else float("nan")

    sem_qs, sur_qs, diffs = [], [], []
    n_with_both = 0
    for r in rows:
        sem_q, sur_q = per_question_rates(r)
        if sem_q is not None and sur_q is not None:
            sem_qs.append(sem_q)
            sur_qs.append(sur_q)
            diffs.append(sem_q - sur_q)
            n_with_both += 1

    sem_mean = sum(sem_qs) / len(sem_qs) if sem_qs else float("nan")
    sur_mean = sum(sur_qs) / len(sur_qs) if sur_qs else float("nan")
    delta = sem_mean - sur_mean

    t_res = paired_t(diffs)
    ci = bootstrap_ci(diffs)

    return {
        "benchmark": bench,
        "agent": agent,
        "n_questions": n_total,
        "n_paired": n_with_both,
        "accuracy": acc,
        "sem_rate": sem_mean,
        "sur_rate": sur_mean,
        "delta_pp": delta * 100,
        "paired_t": t_res,
        "bootstrap_95ci_pp": [c * 100 for c in ci] if ci else None,
    }


def main():
    cells = []
    for bench in ["gsm8k", "math", "hotpotqa"]:
        for agent in ["cot", "react", "direct"]:
            rows = load_cell(bench, agent)
            res = analyze_cell(bench, agent, rows)
            cells.append(res)

    out_json = os.path.join(OUT_DIR, "qwen25_14b_per_cell.json")
    with open(out_json, "w") as f:
        json.dump({"cells": cells}, f, indent=2)
    print("wrote", out_json)

    # Markdown table
    out_md = os.path.join(OUT_DIR, "qwen25_14b_per_cell.md")
    with open(out_md, "w") as f:
        f.write("# Qwen-2.5-14B per-cell summary (9 cells × 200 questions)\n\n")
        f.write("| benchmark | agent | acc | sem rate | sur rate | Δ (pp) | t | p | 95% CI (pp) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for c in cells:
            tr = c["paired_t"]
            ci = c["bootstrap_95ci_pp"]
            f.write("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:+.2f} | {:+.2f} | {:.4g} | [{:+.2f}, {:+.2f}] |\n".format(
                c["benchmark"], c["agent"], c["accuracy"], c["sem_rate"], c["sur_rate"],
                c["delta_pp"], tr["t"], tr["p"],
                ci[0] if ci else float("nan"), ci[1] if ci else float("nan")))
        f.write("\n")
        # capability gating: cells with acc>=0.65
        f.write("## Capability-gating split (acc threshold = 0.65)\n\n")
        capable = [c for c in cells if c["accuracy"] >= 0.65]
        weak = [c for c in cells if c["accuracy"] < 0.65]
        f.write("- Capable cells (acc>=0.65): n={}, mean Δ={:+.2f}pp, "
                "positive Δ in {}/{}\n".format(
                    len(capable),
                    sum(c["delta_pp"] for c in capable) / len(capable) if capable else 0.0,
                    sum(1 for c in capable if c["delta_pp"] > 0), len(capable)))
        f.write("- Weak cells (acc<0.65):    n={}, mean Δ={:+.2f}pp, "
                "positive Δ in {}/{}\n".format(
                    len(weak),
                    sum(c["delta_pp"] for c in weak) / len(weak) if weak else 0.0,
                    sum(1 for c in weak if c["delta_pp"] > 0), len(weak)))
        f.write("\n")
        # overall
        all_diffs = []
        for c in cells:
            # weight by n_paired
            all_diffs.append(c["delta_pp"])
        f.write("## Overall (9 cells, mean across cells)\n\n")
        if all_diffs:
            mean_d = sum(all_diffs) / len(all_diffs)
            f.write("- mean Δ across 9 cells: {:+.2f} pp\n".format(mean_d))
            f.write("- positive Δ cells: {}/{}\n".format(
                sum(1 for d in all_diffs if d > 0), len(all_diffs)))
    print("wrote", out_md)

    # Console summary
    print("\n=== per-cell summary ===")
    for c in cells:
        print("{:8s} {:6s}  n={:3d}  acc={:.3f}  sem={:.3f}  sur={:.3f}  Δ={:+.2f}pp  p={:.3g}".format(
            c["benchmark"], c["agent"], c["n_paired"],
            c["accuracy"], c["sem_rate"], c["sur_rate"],
            c["delta_pp"], c["paired_t"]["p"]))


if __name__ == "__main__":
    main()
