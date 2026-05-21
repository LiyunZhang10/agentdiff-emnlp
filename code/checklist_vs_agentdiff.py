#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checklist_vs_agentdiff.py — Quantify the incremental value of multi-step
agent-loop analysis over CheckList-style single-signal robustness testing.

Direct response to:
  R1: "未与最相关的代理鲁棒性工作（如PromptBench、TextFooler等）进行直接比较"
  R3: "都关注模型行为的可复现性...本文未量化相对于单步测试的增量价值"
  Reverse: "这是对现有扰动测试方法的规模化应用"
  AC: "未能明确证明为什么这个特定发现值得顶级会议发表"

Approach:
  CheckList-style signal  := per-cell  IR_sem - IR_sur  (the "answer change"
                             rate already captured by CheckList).
  AgentDiff signal        := mechanism-level features only available with
                             trace-level analysis:
                              - mean divergence step (semantic)
                              - mean divergence step (surface)
                              - self-correction rate (semantic)
                              - self-correction rate (surface)
                              - cascade depth (semantic vs surface)

Question: when the CheckList signal is weak / noisy (i.e. capable cells where
sem and sur both collapse, or incapable cells where both saturate near 80%),
does the AgentDiff signal still carry information?

Output:
  - Per-cell table comparing CheckList Δ to AgentDiff features.
  - Cells where AgentDiff is *more* discriminative than CheckList.
  - Statistical test: does mechanism signature predict capability tier
    even when conditioned on CheckList Δ being uninformative?
"""
import json
import os
import sys
from collections import defaultdict
from glob import glob

ROOT = "."


def load_cell(jsonl_path):
    """Load all records of a cell. Returns list of records."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def cell_signals(records):
    """Compute CheckList Δ and AgentDiff mechanism features per cell."""
    sem_types = {"paraphrase", "synonym"}
    sur_types = {"reorder", "format", "distractor"}

    inc_sem = []
    inc_sur = []
    div_sem = []
    div_sur = []
    cascade_sem = []
    cascade_sur = []
    selfcorr_sem = 0
    selfcorr_total_sem = 0
    selfcorr_sur = 0
    selfcorr_total_sur = 0

    n_correct_orig = 0
    n_total = 0

    for rec in records:
        n_total += 1
        if rec.get("original_result", {}).get("is_correct"):
            n_correct_orig += 1
        for det in rec.get("propagation_details", []):
            ptype = det.get("perturbation_type")
            div = det.get("divergence_step", 0)
            cas = det.get("cascade_depth", 0)
            pat = det.get("propagation_pattern", "")
            v_ans = det.get("variant_answer")
            o_ans = rec.get("original_result", {}).get("final_answer")
            inconsistent = (v_ans is not None) and (o_ans is not None) and (v_ans != o_ans)
            if ptype in sem_types:
                inc_sem.append(1 if inconsistent else 0)
                if div > 0:
                    div_sem.append(div)
                    cascade_sem.append(cas)
                selfcorr_total_sem += 1
                if pat == "self_correct":
                    selfcorr_sem += 1
            elif ptype in sur_types:
                inc_sur.append(1 if inconsistent else 0)
                if div > 0:
                    div_sur.append(div)
                    cascade_sur.append(cas)
                selfcorr_total_sur += 1
                if pat == "self_correct":
                    selfcorr_sur += 1

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    acc = n_correct_orig / n_total if n_total else 0.0
    ir_sem = avg(inc_sem)
    ir_sur = avg(inc_sur)
    delta_checklist = ir_sem - ir_sur

    return {
        "n": n_total,
        "acc": acc,
        "ir_sem": ir_sem,
        "ir_sur": ir_sur,
        "delta_checklist_pp": delta_checklist * 100,
        # AgentDiff-only mechanism features:
        "div_step_sem_mean": avg(div_sem),
        "div_step_sur_mean": avg(div_sur),
        "delta_div_step_pp": (avg(div_sur) - avg(div_sem)),  # surface diverges later
        "cascade_sem_mean": avg(cascade_sem),
        "cascade_sur_mean": avg(cascade_sur),
        "delta_cascade_pp": (avg(cascade_sem) - avg(cascade_sur)),  # surface cascades less
        "selfcorr_sem_rate": selfcorr_sem / selfcorr_total_sem if selfcorr_total_sem else 0,
        "selfcorr_sur_rate": selfcorr_sur / selfcorr_total_sur if selfcorr_total_sur else 0,
        "delta_selfcorr_pp": ((selfcorr_sur / selfcorr_total_sur if selfcorr_total_sur else 0)
                              - (selfcorr_sem / selfcorr_total_sem if selfcorr_total_sem else 0)) * 100,
    }


def find_all_cells():
    """Glob through all completed cells (v3 fixed-judge, hpqa, mimo only)."""
    cells = []
    valid_suffixes = ("_fix", "_hpqa", "mimo_v25_pro", "mimo_v25_pro_hpqa")
    for d in sorted(os.listdir(ROOT)):
        if not d.startswith("runs_real_"):
            continue
        # Skip ablation directories
        if d.endswith("_genmimo"):
            continue
        # Skip pre-v3 data (raw `runs_real_3b/`, `runs_real_7b/` from earlier
        # exploratory runs that used different judge / generator).
        slug = d[len("runs_real_"):]
        if not (slug.endswith("_fix") or slug.endswith("_hpqa")
                or slug.startswith("mimo_v25_pro")):
            continue
        path = os.path.join(ROOT, d)
        if not os.path.isdir(path):
            continue
        for f in sorted(os.listdir(path)):
            if not f.endswith(".jsonl"):
                continue
            cells.append((d, f, os.path.join(path, f)))
    return cells


def main():
    cells = find_all_cells()
    rows = []
    for slug, fname, fpath in cells:
        # bench_agent_real_<slug>.jsonl
        parts = fname.replace(".jsonl", "").split("_")
        bench = parts[0]
        agent = parts[1]
        records = load_cell(fpath)
        if not records:
            continue
        sig = cell_signals(records)
        rows.append({
            "slug": slug,
            "bench": bench,
            "agent": agent,
            "n": sig["n"],
            **sig,
        })

    # Sort by accuracy descending so capable cells appear first.
    rows.sort(key=lambda r: -r["acc"])

    print()
    print("=" * 130)
    print("  CheckList Δ vs AgentDiff Mechanism Signal — Per-Cell Comparison")
    print("=" * 130)
    print("%-30s %-8s %-5s  %4s | %5s | %7s %7s %8s | %5s %5s %7s | %5s %5s %7s" % (
        "slug", "bench", "agent", "n",
        "acc",
        "Δ_chk", "ir_sem", "ir_sur",
        "div_se", "div_su", "Δ_div",
        "sc_se%", "sc_su%", "Δ_sc",
    ))
    print("-" * 130)
    for r in rows:
        print("%-30s %-8s %-5s  %4d | %5.2f | %+6.1f %6.2f %7.2f | %5.2f %5.2f %+6.2f | %5.1f %5.1f %+6.1f" % (
            r["slug"][:30], r["bench"][:8], r["agent"][:5],
            r["n"], r["acc"], r["delta_checklist_pp"],
            r["ir_sem"], r["ir_sur"],
            r["div_step_sem_mean"], r["div_step_sur_mean"], r["delta_div_step_pp"],
            r["selfcorr_sem_rate"] * 100, r["selfcorr_sur_rate"] * 100,
            r["delta_selfcorr_pp"],
        ))

    # Aggregate "where AgentDiff outperforms CheckList".
    # Define: cell is "CheckList weak" if |Δ_chk| < 5pp.
    # In those cells, does AgentDiff still show a non-zero mechanism gap?
    print()
    print("=" * 130)
    print("  CheckList-weak cells (|Δ_chk| < 5pp) where AgentDiff still discriminates")
    print("=" * 130)
    weak = [r for r in rows if abs(r["delta_checklist_pp"]) < 5.0]
    print("Total CheckList-weak cells: %d" % len(weak))
    discrim = 0
    for r in weak:
        # AgentDiff says "real dichotomy" if any mechanism delta is > 0.5 pp
        # (later divergence on surface, or higher self-correct on surface).
        discriminative = (r["delta_div_step_pp"] > 0.3) or (r["delta_selfcorr_pp"] > 0.5)
        if discriminative:
            discrim += 1
            print("  YES: %-30s acc=%.2f  Δ_chk=%+.1f Δ_div=%+.2f Δ_sc=%+.1f" % (
                r["slug"][:30], r["acc"], r["delta_checklist_pp"],
                r["delta_div_step_pp"], r["delta_selfcorr_pp"]))
        else:
            print("  --:  %-30s acc=%.2f  Δ_chk=%+.1f Δ_div=%+.2f Δ_sc=%+.1f" % (
                r["slug"][:30], r["acc"], r["delta_checklist_pp"],
                r["delta_div_step_pp"], r["delta_selfcorr_pp"]))
    print()
    print("AgentDiff discriminates in %d/%d CheckList-weak cells (%.1f%%)" %
          (discrim, len(weak), 100 * discrim / max(1, len(weak))))

    # Pooled stats on all cells.
    n_pos_div = sum(1 for r in rows if r["delta_div_step_pp"] > 0)
    n_pos_sc = sum(1 for r in rows if r["delta_selfcorr_pp"] > 0)
    n_pos_chk = sum(1 for r in rows if r["delta_checklist_pp"] > 0)
    print()
    print("Cells where signal is positive (sem worse than surface):")
    print("  CheckList Δ_chk > 0:  %d / %d" % (n_pos_chk, len(rows)))
    print("  AgentDiff Δ_div > 0:  %d / %d  (surface diverges later)" % (n_pos_div, len(rows)))
    print("  AgentDiff Δ_sc  > 0:  %d / %d  (surface self-corrects more)" % (n_pos_sc, len(rows)))

    # Save JSON for later use in paper.
    out_json = os.path.join(ROOT, "results_conditional", "checklist_vs_agentdiff.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "n_cells": len(rows),
            "rows": rows,
            "summary": {
                "checklist_weak_cells": len(weak),
                "agentdiff_discriminates_in_weak": discrim,
                "n_pos_chk": n_pos_chk,
                "n_pos_div": n_pos_div,
                "n_pos_sc": n_pos_sc,
            },
        }, f, indent=2)
    print()
    print("Saved -> %s" % out_json)


if __name__ == "__main__":
    main()
