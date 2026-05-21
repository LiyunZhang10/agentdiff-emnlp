#!/usr/bin/env python3
"""
code/analyze_probe_kl.py

Local-side analyzer for the KL divergence probe results pushed back from
the GPU node. Uses scipy if available for proper paired t-test (Welch DF),
falls back to normal approximation otherwise.

Usage:
    python3 code/analyze_probe_kl.py results/probe_decoding_kl/qwen25_14b_gsm8k_pilot.json
"""
import argparse
import json
import math
import sys


SEM_TYPES = {"paraphrase", "synonym"}
SUR_TYPES = {"reorder", "format", "distractor"}


def paired_t(diffs):
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    t = mean / se if se > 0 else 0.0
    cohen = mean / sd if sd > 0 else 0.0
    try:
        from scipy import stats
        t_sci, p_sci = stats.ttest_rel(
            [d for d in diffs],  # x
            [0.0] * n,           # y, equivalent to one-sample t against 0
        )
        # Note ttest_rel of x against zeros == one-sample t-test of x.
        return {
            "n": n, "mean": mean, "sd": sd, "se": se,
            "t": t, "p_two_sided": float(p_sci), "method": "scipy ttest_rel vs 0",
            "cohens_d": cohen,
        }
    except Exception:
        # normal approx
        from math import erf, sqrt as msqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / msqrt(2))))
        return {
            "n": n, "mean": mean, "sd": sd, "se": se,
            "t": t, "p_two_sided": p, "method": "normal_approx",
            "cohens_d": cohen,
        }


def wilcoxon_signed(diffs):
    try:
        from scipy import stats
        st, p = stats.wilcoxon([d for d in diffs])
        return {"statistic": float(st), "p_two_sided": float(p)}
    except Exception:
        return None


def analyze_one(out_path):
    with open(out_path) as f:
        data = json.load(f)

    per_q = data.get("per_question", [])
    sem_per_q, sur_per_q = [], []
    sem_ned, sur_ned = [], []
    for r in per_q:
        sems = [v["kl_orig_to_variant"] for v in r["variants"]
                if v["perturbation_type"] in SEM_TYPES and v.get("kl_orig_to_variant") is not None]
        surs = [v["kl_orig_to_variant"] for v in r["variants"]
                if v["perturbation_type"] in SUR_TYPES and v.get("kl_orig_to_variant") is not None]
        sem_neds = [v["norm_edit_distance"] for v in r["variants"]
                    if v["perturbation_type"] in SEM_TYPES]
        sur_neds = [v["norm_edit_distance"] for v in r["variants"]
                    if v["perturbation_type"] in SUR_TYPES]
        if sems and surs:
            sem_per_q.append(sum(sems) / len(sems))
            sur_per_q.append(sum(surs) / len(surs))
            sem_ned.append(sum(sem_neds) / len(sem_neds) if sem_neds else 0)
            sur_ned.append(sum(sur_neds) / len(sur_neds) if sur_neds else 0)

    n = len(sem_per_q)
    if n == 0:
        print("NO PAIRED DATA")
        return

    diffs = [s - u for s, u in zip(sem_per_q, sur_per_q)]
    t_res = paired_t(diffs)
    w_res = wilcoxon_signed(diffs)

    mean_sem = sum(sem_per_q) / n
    mean_sur = sum(sur_per_q) / n
    ratio = mean_sem / mean_sur if mean_sur > 0 else float("nan")

    print("=" * 60)
    print("model:           {}".format(data.get("model")))
    print("source variants: {}".format(data.get("variants_source")))
    print("n_paired:        {}".format(n))
    print("-" * 60)
    print("KL_sem mean:     {:.4f}".format(mean_sem))
    print("KL_sur mean:     {:.4f}".format(mean_sur))
    print("ratio sem/sur:   {:.3f}x".format(ratio))
    print("mean diff:       {:.4f}".format(sum(diffs) / n))
    print("-" * 60)
    print("paired t-test:")
    print("  t = {:.3f}".format(t_res["t"]))
    print("  p = {:.4g}".format(t_res["p_two_sided"]))
    print("  Cohen's d = {:.3f}".format(t_res["cohens_d"]))
    print("  method = {}".format(t_res["method"]))
    if w_res:
        print("Wilcoxon signed-rank:")
        print("  W = {:.3f}".format(w_res["statistic"]))
        print("  p = {:.4g}".format(w_res["p_two_sided"]))
    print("-" * 60)
    print("NED balance:")
    print("  ned_sem mean = {:.3f}".format(sum(sem_ned) / n))
    print("  ned_sur mean = {:.3f}".format(sum(sur_ned) / n))
    print("  diff = {:.3f}  {}".format(
        sum(sem_ned) / n - sum(sur_ned) / n,
        "OK" if abs(sum(sem_ned) / n - sum(sur_ned) / n) < 0.15
        else "WARN: KL gap may be confounded by length"))
    print("=" * 60)
    print("\nPILOT VERDICT:")
    threshold_p = 0.01
    threshold_ratio = 1.3
    pass_p = t_res["p_two_sided"] < threshold_p
    pass_ratio = ratio > threshold_ratio
    if pass_p and pass_ratio:
        print("  ✅ PASS: p<{} AND ratio>{}x — extend to all 6 models".format(
            threshold_p, threshold_ratio))
    elif pass_p and not pass_ratio:
        print("  🟡 MARGINAL: p<{} but ratio={:.2f}x <={}x".format(
            threshold_p, ratio, threshold_ratio))
        print("     → effect significant but small; consider larger n or report cautiously")
    elif not pass_p and pass_ratio:
        print("  🟡 UNDERPOWERED: ratio={:.2f}x but p={:.3g} >={}".format(
            ratio, t_res["p_two_sided"], threshold_p))
        print("     → suggestive trend; rerun with n>=50")
    else:
        print("  ❌ FAIL: p={:.3g}, ratio={:.2f}x — DO NOT WRITE INTO PAPER".format(
            t_res["p_two_sided"], ratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        analyze_one(p)
        print()


if __name__ == "__main__":
    main()
