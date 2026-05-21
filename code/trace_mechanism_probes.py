#!/usr/bin/env python3
"""
code/trace_mechanism_probes.py

Four CPU-only trace-level mechanism probes on the new Qwen-2.5-14B 1800
trajectories. All signals come directly from propagation_details produced
by AgentDiffPipelineV2 — NO hidden representations, NO logprob queries,
NO additional inference. Pure secondary analysis.

Probes (all paired sem vs sur, per question):
  M1. divergence_step:       does sem diverge EARLIER than sur?
  M2. self_correct rate:     is sem LESS likely to recover via self_correct?
  M3. cascade_depth:         is sem's cascade DEEPER once it diverges?
  M4. thought_similarity_at_step_k:  does sem decay FASTER per step?

For each: pool over all 1800 traj that have both ≥1 sem and ≥1 sur variant,
report paired test + 95% CI. Stratify by (benchmark, agent) to check
robustness; flag any cell where the sign reverses.

Outputs:
  results/conditional_v2/trace_mech_qwen25_14b.json   (full machine readable)
  results/conditional_v2/trace_mech_qwen25_14b.md     (human readable report)
"""
import glob
import json
import math
import os
from collections import Counter, defaultdict

ROOT = "./results/runs_real_qwen25_14b_vllm"
OUT_DIR = "./results/conditional_v2"
os.makedirs(OUT_DIR, exist_ok=True)

SEM_TYPES = {"paraphrase", "synonym"}
SUR_TYPES = {"reorder", "format", "distractor"}


def load_all():
    """Yield (bench, agent, record) tuples."""
    for fp in sorted(glob.glob(os.path.join(ROOT, "*", "*", "*.json"))):
        parts = fp.split(os.sep)
        bench = parts[-3]
        agent = parts[-2]
        try:
            with open(fp) as f:
                yield bench, agent, json.load(f)
        except Exception:
            continue


# ---------- statistical helpers ----------

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
        _, p = stats.ttest_1samp(diffs, 0.0)
        method = "scipy"
    except Exception:
        from math import erf, sqrt as msqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / msqrt(2))))
        method = "normal_approx"
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": float(p),
            "cohens_d": cohen, "method": method}


def wilcoxon(diffs):
    try:
        from scipy import stats
        nz = [d for d in diffs if d != 0]
        if len(nz) < 2:
            return None
        st, p = stats.wilcoxon(nz)
        return {"n_nonzero": len(nz), "W": float(st), "p": float(p)}
    except Exception:
        return None


def fisher_2x2(a, b, c, d):
    try:
        from scipy import stats
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        return {"odds_ratio": float(odds), "p": float(p)}
    except Exception:
        return None


def bootstrap_ci(values, B=2000, seed=42):
    if not values:
        return None
    import random
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(B):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    return [boots[int(0.025 * B)], boots[int(0.975 * B)]]


# ---------- per-question extraction ----------

def per_question_stats(record):
    """For one record, return dict with sem/sur lists for all probes.

    Returns None if either sem or sur side has no data for the probe.
    """
    pd_list = record.get("propagation_details") or []
    if not pd_list:
        return None
    pti = (record.get("consistency_analysis") or {}).get("per_type_inconsistency") or {}

    sem_div_steps = []
    sur_div_steps = []
    sem_cascade = []
    sur_cascade = []
    sem_self_corr = 0
    sem_total = 0
    sur_self_corr = 0
    sur_total = 0
    # collect per-step similarities by perturbation type, then average within sem/sur
    sem_step_sims = defaultdict(list)  # step_idx -> [thought_sim,...]
    sur_step_sims = defaultdict(list)

    for pd in pd_list:
        ptype = pd.get("perturbation_type")
        ds = pd.get("divergence_step")
        cd = pd.get("cascade_depth")
        pat = pd.get("propagation_pattern")
        sims = pd.get("step_similarities") or []

        target_div = sem_div_steps if ptype in SEM_TYPES else sur_div_steps if ptype in SUR_TYPES else None
        target_cas = sem_cascade if ptype in SEM_TYPES else sur_cascade if ptype in SUR_TYPES else None
        if target_div is None:
            continue
        if isinstance(ds, (int, float)):
            target_div.append(ds)
        if isinstance(cd, (int, float)):
            target_cas.append(cd)

        if ptype in SEM_TYPES:
            sem_total += 1
            if pat == "self_correct":
                sem_self_corr += 1
            for s in sims:
                k = s.get("step")
                t = s.get("thought_similarity")
                if isinstance(k, int) and isinstance(t, (int, float)):
                    sem_step_sims[k].append(t)
        elif ptype in SUR_TYPES:
            sur_total += 1
            if pat == "self_correct":
                sur_self_corr += 1
            for s in sims:
                k = s.get("step")
                t = s.get("thought_similarity")
                if isinstance(k, int) and isinstance(t, (int, float)):
                    sur_step_sims[k].append(t)

    return {
        "sem_div_step":   (sum(sem_div_steps) / len(sem_div_steps)) if sem_div_steps else None,
        "sur_div_step":   (sum(sur_div_steps) / len(sur_div_steps)) if sur_div_steps else None,
        "sem_cascade":    (sum(sem_cascade) / len(sem_cascade)) if sem_cascade else None,
        "sur_cascade":    (sum(sur_cascade) / len(sur_cascade)) if sur_cascade else None,
        "sem_total": sem_total,
        "sem_self_corr": sem_self_corr,
        "sur_total": sur_total,
        "sur_self_corr": sur_self_corr,
        "sem_step_sims": {k: sum(v) / len(v) for k, v in sem_step_sims.items()},
        "sur_step_sims": {k: sum(v) / len(v) for k, v in sur_step_sims.items()},
    }


# ---------- main ----------

def main():
    all_recs = []  # (bench, agent, record_id, q_stats)
    for bench, agent, rec in load_all():
        s = per_question_stats(rec)
        if s is not None:
            all_recs.append((bench, agent, rec.get("sample_id"), s))
    print("loaded", len(all_recs), "trajectories with propagation_details")

    # ===== M1: divergence_step (lower = earlier divergence) =====
    diffs_div = []   # sem - sur (negative = sem earlier = MORE unstable)
    for bench, agent, sid, s in all_recs:
        if s["sem_div_step"] is not None and s["sur_div_step"] is not None:
            diffs_div.append(s["sem_div_step"] - s["sur_div_step"])
    m1_t = paired_t(diffs_div)
    m1_w = wilcoxon(diffs_div)
    m1_ci = bootstrap_ci(diffs_div)

    # M1 by cell
    m1_cells = {}
    by_cell = defaultdict(list)
    for bench, agent, sid, s in all_recs:
        if s["sem_div_step"] is not None and s["sur_div_step"] is not None:
            by_cell[(bench, agent)].append(s["sem_div_step"] - s["sur_div_step"])
    for k, v in by_cell.items():
        m1_cells["{}/{}".format(*k)] = {
            "n": len(v),
            "mean_diff": sum(v) / len(v) if v else 0.0,
            "paired_t": paired_t(v),
        }

    # ===== M2: self_correct rate (sem vs sur) — pooled 2x2 Fisher =====
    sem_total = sum(s["sem_total"] for _, _, _, s in all_recs)
    sem_corr = sum(s["sem_self_corr"] for _, _, _, s in all_recs)
    sur_total = sum(s["sur_total"] for _, _, _, s in all_recs)
    sur_corr = sum(s["sur_self_corr"] for _, _, _, s in all_recs)
    m2_table = [[sem_corr, sem_total - sem_corr],
                [sur_corr, sur_total - sur_corr]]
    m2_fisher = fisher_2x2(sem_corr, sem_total - sem_corr,
                           sur_corr, sur_total - sur_corr)
    sem_rate = sem_corr / sem_total if sem_total else float("nan")
    sur_rate = sur_corr / sur_total if sur_total else float("nan")

    # M2 by cell
    m2_cells = {}
    cell_self = defaultdict(lambda: {"sem_t": 0, "sem_c": 0, "sur_t": 0, "sur_c": 0})
    for bench, agent, sid, s in all_recs:
        c = cell_self[(bench, agent)]
        c["sem_t"] += s["sem_total"]
        c["sem_c"] += s["sem_self_corr"]
        c["sur_t"] += s["sur_total"]
        c["sur_c"] += s["sur_self_corr"]
    for k, c in cell_self.items():
        sr = c["sem_c"] / c["sem_t"] if c["sem_t"] else float("nan")
        ur = c["sur_c"] / c["sur_t"] if c["sur_t"] else float("nan")
        f = fisher_2x2(c["sem_c"], c["sem_t"] - c["sem_c"],
                       c["sur_c"], c["sur_t"] - c["sur_c"])
        m2_cells["{}/{}".format(*k)] = {
            "sem_rate": sr, "sur_rate": ur,
            "diff_pp": (sr - ur) * 100 if not (math.isnan(sr) or math.isnan(ur)) else None,
            "fisher": f,
            "table": [c["sem_c"], c["sem_t"] - c["sem_c"], c["sur_c"], c["sur_t"] - c["sur_c"]],
        }

    # ===== M3: cascade_depth (sem - sur, positive = sem cascades deeper) =====
    diffs_cas = []
    for bench, agent, sid, s in all_recs:
        if s["sem_cascade"] is not None and s["sur_cascade"] is not None:
            diffs_cas.append(s["sem_cascade"] - s["sur_cascade"])
    m3_t = paired_t(diffs_cas)
    m3_w = wilcoxon(diffs_cas)
    m3_ci = bootstrap_ci(diffs_cas)

    # ===== M4: thought_similarity at step k (sem - sur, negative = sem decays more) =====
    # Aggregate per-step paired diffs across all trajectories.
    step_diffs = defaultdict(list)  # step -> [sem_sim - sur_sim, ...]
    for bench, agent, sid, s in all_recs:
        for k, sem_v in s["sem_step_sims"].items():
            sur_v = s["sur_step_sims"].get(k)
            if sur_v is not None:
                step_diffs[k].append(sem_v - sur_v)
    m4_per_step = {}
    for k in sorted(step_diffs.keys()):
        v = step_diffs[k]
        m4_per_step[str(k)] = {
            "n": len(v),
            "mean_sem_minus_sur": sum(v) / len(v) if v else 0.0,
            "paired_t": paired_t(v),
            "ci95": bootstrap_ci(v),
        }

    # ===== Build report =====
    report = {
        "n_trajectories": len(all_recs),
        "M1_divergence_step": {
            "interpretation": "sem - sur ; negative => sem diverges EARLIER",
            "n_paired": len(diffs_div),
            "mean_diff_steps": (sum(diffs_div) / len(diffs_div)) if diffs_div else None,
            "paired_t": m1_t,
            "wilcoxon": m1_w,
            "ci95": m1_ci,
            "by_cell": m1_cells,
        },
        "M2_self_correct_rate": {
            "interpretation": "Lower sem rate => sem less likely to recover",
            "sem_table": m2_table,  # [[corr, fail], ...]
            "sem_rate": sem_rate,
            "sur_rate": sur_rate,
            "diff_pp": (sem_rate - sur_rate) * 100,
            "fisher": m2_fisher,
            "by_cell": m2_cells,
        },
        "M3_cascade_depth": {
            "interpretation": "sem - sur ; positive => sem cascades deeper",
            "n_paired": len(diffs_cas),
            "mean_diff_depth": (sum(diffs_cas) / len(diffs_cas)) if diffs_cas else None,
            "paired_t": m3_t,
            "wilcoxon": m3_w,
            "ci95": m3_ci,
        },
        "M4_thought_similarity_per_step": {
            "interpretation": "sem - sur thought_similarity at step k ; "
                               "negative => sem trace diverges more from orig at step k",
            "per_step": m4_per_step,
        },
    }

    out_json = os.path.join(OUT_DIR, "trace_mech_qwen25_14b.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("wrote", out_json)

    # ===== Markdown report =====
    out_md = os.path.join(OUT_DIR, "trace_mech_qwen25_14b.md")
    with open(out_md, "w") as f:
        f.write("# Trace-level mechanism probes on Qwen-2.5-14B (n_traj={})\n\n".format(len(all_recs)))
        f.write("All signals are derived from `propagation_details` produced by "
                "the AgentDiff pipeline. No hidden representations or token-level "
                "logprobs are queried — every probe is a pure secondary analysis "
                "of step-level traces already on disk.\n\n")

        f.write("## M1. Divergence step (does sem diverge earlier?)\n\n")
        f.write("- n_paired (have both sem and sur): {}\n".format(len(diffs_div)))
        if m1_t:
            f.write("- mean(sem - sur) = {:+.3f} steps  (negative => sem earlier)\n".format(m1_t["mean"]))
            f.write("- paired t = {:.3f}, p = {:.4g}, Cohen's d = {:.3f}\n".format(
                m1_t["t"], m1_t["p"], m1_t["cohens_d"]))
        if m1_w:
            f.write("- Wilcoxon W = {:.0f}, p = {:.4g}\n".format(m1_w["W"], m1_w["p"]))
        if m1_ci:
            f.write("- bootstrap 95% CI: [{:+.3f}, {:+.3f}]\n".format(*m1_ci))
        f.write("\n### M1 by cell\n\n")
        f.write("| cell | n | mean(sem-sur) | t | p |\n|---|---|---|---|---|\n")
        for k in sorted(m1_cells):
            v = m1_cells[k]
            t = v["paired_t"] or {}
            f.write("| {} | {} | {:+.3f} | {} | {} |\n".format(
                k, v["n"], v["mean_diff"],
                "{:.3f}".format(t.get("t", float("nan"))) if t else "n/a",
                "{:.4g}".format(t.get("p", float("nan"))) if t else "n/a"))

        f.write("\n## M2. Self-correct rate (sem vs sur, pooled)\n\n")
        f.write("- sem self_correct: {} / {} = {:.4f}\n".format(sem_corr, sem_total, sem_rate))
        f.write("- sur self_correct: {} / {} = {:.4f}\n".format(sur_corr, sur_total, sur_rate))
        f.write("- diff: {:+.2f} pp (negative => sem less likely to recover)\n".format(
            (sem_rate - sur_rate) * 100))
        if m2_fisher:
            f.write("- Fisher exact two-sided p = {:.4g}, OR = {:.3f}\n".format(
                m2_fisher["p"], m2_fisher["odds_ratio"]))
        f.write("\n### M2 by cell\n\n")
        f.write("| cell | sem rate | sur rate | diff(pp) | Fisher p |\n|---|---|---|---|---|\n")
        for k in sorted(m2_cells):
            v = m2_cells[k]
            fp = (v["fisher"] or {}).get("p")
            f.write("| {} | {:.4f} | {:.4f} | {} | {} |\n".format(
                k,
                v["sem_rate"] if not math.isnan(v["sem_rate"]) else float("nan"),
                v["sur_rate"] if not math.isnan(v["sur_rate"]) else float("nan"),
                "{:+.2f}".format(v["diff_pp"]) if v["diff_pp"] is not None else "n/a",
                "{:.4g}".format(fp) if fp is not None else "n/a"))

        f.write("\n## M3. Cascade depth\n\n")
        f.write("- n_paired: {}\n".format(len(diffs_cas)))
        if m3_t:
            f.write("- mean(sem - sur) = {:+.3f}  (positive => sem cascades deeper)\n".format(m3_t["mean"]))
            f.write("- paired t = {:.3f}, p = {:.4g}, Cohen's d = {:.3f}\n".format(
                m3_t["t"], m3_t["p"], m3_t["cohens_d"]))
        if m3_w:
            f.write("- Wilcoxon W = {:.0f}, p = {:.4g}\n".format(m3_w["W"], m3_w["p"]))
        if m3_ci:
            f.write("- bootstrap 95% CI: [{:+.3f}, {:+.3f}]\n".format(*m3_ci))

        f.write("\n## M4. Thought similarity decay per step\n\n")
        f.write("- For each step k, compute mean(sem_thought_sim_k - sur_thought_sim_k) "
                "across questions that have both kinds of variant at that step.\n\n")
        f.write("| step k | n | mean(sem-sur) | t | p | 95% CI |\n|---|---|---|---|---|---|\n")
        for k in sorted(m4_per_step.keys(), key=lambda x: int(x)):
            v = m4_per_step[k]
            t = v.get("paired_t") or {}
            ci = v.get("ci95") or [float("nan"), float("nan")]
            f.write("| {} | {} | {:+.4f} | {} | {} | [{:+.4f}, {:+.4f}] |\n".format(
                k, v["n"], v["mean_sem_minus_sur"],
                "{:.3f}".format(t.get("t", float("nan"))) if t else "n/a",
                "{:.4g}".format(t.get("p", float("nan"))) if t else "n/a",
                ci[0], ci[1]))

    print("wrote", out_md)

    # console headline
    print("\n" + "=" * 60)
    print("HEADLINE (Qwen-2.5-14B trace-level mechanism, n=1800):")
    if m1_t:
        print("  M1 divergence_step:  mean(sem-sur)={:+.3f} steps  t={:.3f}  p={:.4g}  d={:.3f}".format(
            m1_t["mean"], m1_t["t"], m1_t["p"], m1_t["cohens_d"]))
    print("  M2 self_correct:     sem={:.3%}  sur={:.3%}  diff={:+.2f}pp  Fisher p={}".format(
        sem_rate, sur_rate, (sem_rate - sur_rate) * 100,
        "{:.4g}".format(m2_fisher["p"]) if m2_fisher else "n/a"))
    if m3_t:
        print("  M3 cascade_depth:    mean(sem-sur)={:+.3f}  t={:.3f}  p={:.4g}  d={:.3f}".format(
            m3_t["mean"], m3_t["t"], m3_t["p"], m3_t["cohens_d"]))
    if m4_per_step:
        print("  M4 thought_sim per-step (n questions, mean diff, p):")
        for k in sorted(m4_per_step.keys(), key=lambda x: int(x))[:8]:
            v = m4_per_step[k]
            t = v.get("paired_t") or {}
            print("    step {:>2}: n={:>4}  diff={:+.4f}  p={:.4g}".format(
                k, v["n"], v["mean_sem_minus_sur"],
                t.get("p", float("nan"))))


if __name__ == "__main__":
    main()
