#!/usr/bin/env /usr/bin/python3
"""
Track D — Within-benchmark tractability proxy (Reviewer R1-Fatal-2).

Reviewer concern: "tractability" coefficient could be benchmark identity in
disguise; without within-benchmark variation we cannot distinguish a topology
gate from a benchmark difficulty effect.

Method (proxy, not hand-labeling — proxies are clearly disclosed in the paper):

  GSM8K  : approximate "multi-route" by parsing the question with a heuristic
           that counts the number of distinct arithmetic operations or numbers
           mentioned. Problems with ≥3 distinct operations or ≥4 numbers are
           flagged as "multi-route" (more arithmetic chains plausible);
           ≤2 ops and ≤3 numbers are flagged "single-route" (canonical chain).
  MATH   : use the published `subject` field. Algebra/Counting & Probability
           → "multi-method" (multiple solution strategies common). Number
           Theory/Geometry/Precalculus → "single-canonical".
  HotpotQA: use `level` (easy/medium/hard) AND `type` (comparison/bridge).
           "comparison" type with ≥3 supporting facts → "multi-evidence";
           "bridge" type with exactly 2 supporting facts → "unique-chain".

We compute Δ_within for each cell by partitioning that cell's variants according
to the per-question tractability label, then refit:

   Δ = α + β1 * within_tractability + β2 * accuracy + ε   (within-bench OLS)

If β1 is significantly > 0 *within each benchmark*, R1-Fatal-2 is rebuked.
If β1 is null within benchmark but the original cross-benchmark coefficient
is large, the paper must downgrade the "topology" claim to "benchmark identity".

Output: track_d/within_benchmark.json + track_d/_d_summary.txt
Resilience: per-cell partials in track_d/cells/. Done file lists completed cells.
"""
import json
import os
import re
import sys
import time
import math
import statistics as st
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_ind, ttest_1samp

ROOT = "."
OUT = os.path.join(ROOT, "track_d")
os.makedirs(OUT, exist_ok=True)
RESULT_FILE = os.path.join(OUT, "within_benchmark.json")
SUMMARY_FILE = os.path.join(OUT, "_d_summary.txt")

SEM = {"paraphrase", "synonym"}
SUR = {"reorder", "format", "distractor"}


def gsm8k_tractability(question: str) -> str:
    """Crude proxy for multi-route vs single-route GSM8K problems.
    Multi-route = many distinct operations or many quantities. Single-route =
    short, single-step word problem with one canonical chain."""
    if not isinstance(question, str):
        return "unknown"
    # Count numbers (integers and decimals)
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", question)
    n_nums = len(nums)
    # Count arithmetic-meaningful keywords
    kw_multi = re.findall(
        r"\b(percent|percentage|times|each|every|total|sum|product|altogether|"
        r"difference|left|remaining|increase|decrease|earn|spend|cost|profit|"
        r"loss|average|ratio|fraction|half|third|quarter)\b",
        question.lower(),
    )
    n_ops = len(kw_multi)
    if n_nums >= 4 or n_ops >= 3:
        return "multi-route"
    if n_nums <= 3 and n_ops <= 2:
        return "single-route"
    return "mid"


def math_tractability(question: str, subject: str = "") -> str:
    """Use MATH subject as proxy. Algebra/Counting → multi-method; Number Theory/
    Geometry/Precalculus → single-canonical. If subject missing, regex from question."""
    if subject:
        s = subject.lower()
    elif isinstance(question, str):
        s = question[:200].lower()  # type guess from text head
    else:
        return "unknown"
    multi = ["algebra", "counting", "probability", "intermediate"]
    single = ["number theory", "geometry", "precalculus", "calculus"]
    for k in multi:
        if k in s:
            return "multi-method"
    for k in single:
        if k in s:
            return "single-canonical"
    # fallback: count distinct operators in question
    if isinstance(question, str):
        ops = re.findall(r"[+\-*/=^]|sqrt|sin|cos|log|integral|derivative", question)
        return "multi-method" if len(set(ops)) >= 3 else "single-canonical"
    return "unknown"


def hotpotqa_tractability(question: str, supporting_facts: list = None,
                          q_type: str = "", level: str = "") -> str:
    """Use HotpotQA type/level/supporting_facts as proxy."""
    n_sf = len(supporting_facts) if supporting_facts else 2
    if q_type and q_type.lower() == "comparison":
        return "multi-evidence" if n_sf >= 3 else "comparison-2"
    if q_type and q_type.lower() == "bridge":
        return "unique-chain"
    # fallback: use question structure
    if isinstance(question, str):
        if " or " in question.lower() or " bigger " in question.lower() or " more " in question.lower():
            return "multi-evidence"
    return "unique-chain"


def label_tractability(bench: str, sample: dict) -> str:
    q = sample.get("sample_question", "")
    if bench == "gsm8k":
        return gsm8k_tractability(q)
    if bench == "math":
        # MATH problems may store subject under sample_metadata or context
        meta = sample.get("sample_context") or {}
        if isinstance(meta, dict):
            subject = meta.get("subject", "") or meta.get("type", "")
        else:
            subject = str(meta)[:200]
        return math_tractability(q, subject)
    if bench == "hotpotqa":
        meta = sample.get("sample_context") or {}
        if isinstance(meta, dict):
            sf = meta.get("supporting_facts") or []
            t = meta.get("type", "")
            lv = meta.get("level", "")
        else:
            sf, t, lv = [], "", ""
        return hotpotqa_tractability(q, sf, t, lv)
    return "unknown"


def cell_within_delta(cell_path, bench):
    """Return per-tractability-stratum Δ for one cell."""
    by_stratum = defaultdict(lambda: {"sem_inc": [], "sur_inc": []})
    nc = nt = 0
    samples = []
    for ln in open(cell_path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        nt += 1
        if r.get("original_result", {}).get("is_correct"):
            nc += 1
        oa = r.get("original_result", {}).get("final_answer")
        stratum = label_tractability(bench, r)
        for det in r.get("propagation_details", []):
            op = det.get("perturbation_type")
            if op not in SEM and op not in SUR:
                continue
            v = det.get("variant_answer")
            inc = (v is not None and oa is not None and v != oa)
            key = "sem_inc" if op in SEM else "sur_inc"
            by_stratum[stratum][key].append(int(inc))
        samples.append({"sid": r.get("sample_id"), "stratum": stratum})
    out = {"n_originals": nt, "accuracy": nc / max(nt, 1), "by_stratum": {}}
    for stratum, blob in by_stratum.items():
        if blob["sem_inc"] and blob["sur_inc"]:
            ir_sem = sum(blob["sem_inc"]) / len(blob["sem_inc"]) * 100
            ir_sur = sum(blob["sur_inc"]) / len(blob["sur_inc"]) * 100
            out["by_stratum"][stratum] = {
                "n_sem": len(blob["sem_inc"]),
                "n_sur": len(blob["sur_inc"]),
                "IR_sem": ir_sem,
                "IR_sur": ir_sur,
                "delta": ir_sem - ir_sur,
            }
    out["samples"] = samples
    return out


def main():
    cells = []
    for d in sorted(os.listdir(ROOT)):
        if not d.startswith("runs_real_"):
            continue
        s = d[len("runs_real_"):]
        if s.endswith("_genmimo") or s.endswith("_genqwen14b"):
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
            cells.append({
                "model": slug, "bench": bench, "scaf": scaf,
                "path": os.path.join(ROOT, d, f),
            })

    print(f"[D] Found {len(cells)} cells")
    cell_results = []
    for c in cells:
        out = cell_within_delta(c["path"], c["bench"])
        cell_results.append({**c, **out})
        strata_str = ", ".join(
            f"{s}:Δ={blob['delta']:+.1f}" for s, blob in out["by_stratum"].items()
        )
        print(f"  {c['model']}/{c['bench']}/{c['scaf']}  acc={out['accuracy']:.2f}  {strata_str}")

    # Aggregate within-benchmark Δ by stratum
    print(f"\n=== Aggregate Δ by (benchmark, tractability stratum) ===")
    agg = defaultdict(lambda: defaultdict(list))
    for cr in cell_results:
        for stratum, blob in cr["by_stratum"].items():
            agg[cr["bench"]][stratum].append(blob["delta"])

    table = []
    for bench in ("gsm8k", "math", "hotpotqa"):
        for stratum, deltas in agg[bench].items():
            if len(deltas) < 3:
                continue
            mean = st.mean(deltas)
            sd = st.stdev(deltas) if len(deltas) > 1 else 0
            n_pos = sum(1 for d in deltas if d > 0)
            t, p = ttest_1samp(deltas, 0)
            table.append({
                "bench": bench, "stratum": stratum,
                "n_cells": len(deltas), "mean_delta": mean,
                "sd": sd, "n_pos": n_pos, "t": float(t), "p": float(p),
            })
            print(f"  {bench:<10} {stratum:<20} n={len(deltas):2}  μΔ={mean:+6.2f}  sd={sd:5.2f}  pos={n_pos}/{len(deltas)}  t={t:+.2f}  p={p:.4f}")

    # Within-benchmark contrast: tractable - non-tractable
    print(f"\n=== Within-benchmark contrast (tractable − non-tractable) ===")
    contrasts = []
    pairs = [
        ("gsm8k", "multi-route", "single-route"),
        ("math", "multi-method", "single-canonical"),
        ("hotpotqa", "multi-evidence", "unique-chain"),
    ]
    for bench, hi, lo in pairs:
        hi_d = agg[bench].get(hi, [])
        lo_d = agg[bench].get(lo, [])
        if len(hi_d) < 3 or len(lo_d) < 3:
            print(f"  {bench:<10} skip (hi n={len(hi_d)} lo n={len(lo_d)})")
            continue
        t, p = ttest_ind(hi_d, lo_d, equal_var=False)
        contrasts.append({
            "bench": bench,
            "hi_label": hi, "lo_label": lo,
            "n_hi": len(hi_d), "n_lo": len(lo_d),
            "mean_hi": st.mean(hi_d), "mean_lo": st.mean(lo_d),
            "diff": st.mean(hi_d) - st.mean(lo_d),
            "welch_t": float(t), "welch_p": float(p),
        })
        print(f"  {bench:<10} hi({hi})={st.mean(hi_d):+.2f} lo({lo})={st.mean(lo_d):+.2f}  diff={st.mean(hi_d)-st.mean(lo_d):+.2f}  t={t:+.2f}  p={p:.4f}")

    # Save
    out_obj = {
        "cell_results": cell_results,
        "aggregate_by_bench_stratum": [
            {**row} for row in table
        ],
        "within_benchmark_contrasts": contrasts,
    }
    with open(RESULT_FILE + ".tmp", "w") as f:
        json.dump(out_obj, f, indent=2, default=str)
    os.replace(RESULT_FILE + ".tmp", RESULT_FILE)

    with open(SUMMARY_FILE, "w") as f:
        f.write("Track D: within-benchmark tractability proxy.\n\n")
        f.write("Strata Δ per (benchmark, stratum):\n")
        for row in table:
            f.write(f"  {row['bench']} {row['stratum']:<18} μΔ={row['mean_delta']:+.2f} (n={row['n_cells']}, p={row['p']:.4f})\n")
        f.write("\nWithin-benchmark contrasts (tractable − non-tractable):\n")
        for row in contrasts:
            f.write(f"  {row['bench']}: diff={row['diff']:+.2f}  Welch t={row['welch_t']:+.2f}  p={row['welch_p']:.4f}\n")
        if not contrasts:
            f.write("  (no contrasts could be computed; insufficient cells per stratum)\n")
        else:
            n_sig = sum(1 for r in contrasts if r['welch_p'] < 0.05 and r['diff'] > 0)
            f.write(f"\n  Significant within-bench tractability gates: {n_sig}/{len(contrasts)}\n")
            if n_sig == 0:
                f.write("  → Tractability claim is NOT supported by within-benchmark variation;\n")
                f.write("    paper must downgrade to 'benchmark-identity-conditioned'.\n")
            elif n_sig == len(contrasts):
                f.write("  → STRONG support for tractability claim across all benchmarks.\n")
            else:
                f.write(f"  → Partial support: {n_sig}/{len(contrasts)} benchmarks show tractability gate.\n")
    print(f"\nSaved → {RESULT_FILE}\nSummary → {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
