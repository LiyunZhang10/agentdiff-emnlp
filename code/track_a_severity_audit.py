#!/usr/bin/env /usr/bin/python3
"""
Track A1+A2: Severity Audit + Severity-matched Δ
================================================
Objective:
  1. Compute, for every (original × variant) pair across all 36 main cells:
     - Levenshtein edit distance (raw + normalized by length)
     - Token-level Jaccard distance
     - (optionally) Sentence-BERT semantic distance
     - Question-length change ratio
  2. Re-derive Δ on a severity-matched subsample where surface-side variants
     are weighted to match the edit-distance distribution of paraphrase/synonym.

Resilience:
  * Output is a single jsonl file (one row per variant). Atomic append per row
    using O_APPEND.
  * Cell-level done file: if cell already in done set, skip.
  * Restart from any crash: re-read jsonl, build done set of (cell, sample_id, op).
  * No batching. One row at a time. Worst-case data loss: 1 row.
"""
import json
import os
import sys
import glob
import time
import re
from collections import defaultdict

ROOT = "/data/workspace/agentdiff_exp"
OUT_DIR = os.path.join(ROOT, "track_a")
os.makedirs(OUT_DIR, exist_ok=True)
SEVERITY_JSONL = os.path.join(OUT_DIR, "severity_per_variant.jsonl")
PROGRESS_FILE = os.path.join(OUT_DIR, "_a_progress.txt")
HEARTBEAT = os.path.join(OUT_DIR, "_a_heartbeat.txt")

SEM = {"paraphrase", "synonym"}
SUR = {"reorder", "format", "distractor"}


def levenshtein(a: str, b: str) -> int:
    """Pure-python Levenshtein, used as fallback if python-Levenshtein not installed."""
    try:
        import Levenshtein  # type: ignore
        return Levenshtein.distance(a, b)
    except ImportError:
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[n]


def tokens(s: str):
    return [t for t in re.split(r"\s+", s.strip().lower()) if t]


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta and not tb:
        return 0.0
    return 1.0 - len(ta & tb) / max(len(ta | tb), 1)


def load_done():
    """Read existing severity jsonl and return set of (cell, sample_id, op)."""
    done = set()
    if os.path.exists(SEVERITY_JSONL):
        with open(SEVERITY_JSONL) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                    done.add((r["cell"], r["sample_id"], r["op"]))
                except Exception:
                    pass  # skip malformed line
    return done


def main():
    done = load_done()
    print(f"[A] Loaded {len(done)} already-processed variants from {SEVERITY_JSONL}")

    # Iterate cells
    cells = sorted(glob.glob(os.path.join(ROOT, "runs_real_*", "*.jsonl")))
    cells = [c for c in cells if "_genmimo" not in c]

    total_added = 0
    t0 = time.time()
    fp_out = open(SEVERITY_JSONL, "a", buffering=1)  # line buffering

    for cell_path in cells:
        cell = os.path.relpath(cell_path, ROOT)
        with open(cell_path) as fp:
            for ln in fp:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                sid = r.get("sample_id")
                orig_q = r.get("sample_question", "")
                gold = r.get("sample_gold_answer")
                orig_ans = r.get("original_result", {}).get("final_answer")
                # iterate over each variant
                variants = r.get("perturbation_variants", []) or []
                # also dig from propagation_details for variant_answer
                prop = {p["perturbation_type"]: p for p in r.get("propagation_details", [])}
                for v in variants:
                    op = v.get("perturbation_type") or v.get("op") or v.get("type")
                    if op not in SEM and op not in SUR:
                        continue
                    if (cell, sid, op) in done:
                        continue
                    # variant question text: try several keys
                    variant_q = (
                        v.get("variant_question")
                        or v.get("perturbed_question")
                        or v.get("variant")
                        or v.get("text")
                        or v.get("question")
                        or ""
                    )
                    if not variant_q:
                        # cannot compute severity; record stub for completeness
                        rec = {
                            "cell": cell,
                            "sample_id": sid,
                            "op": op,
                            "side": "sem" if op in SEM else "sur",
                            "edit_distance": None,
                            "edit_distance_norm": None,
                            "jaccard": None,
                            "len_change_ratio": None,
                            "orig_len": len(orig_q),
                            "variant_len": 0,
                            "skip_reason": "no_variant_text",
                        }
                    else:
                        ed = levenshtein(orig_q, variant_q)
                        max_len = max(len(orig_q), len(variant_q), 1)
                        rec = {
                            "cell": cell,
                            "sample_id": sid,
                            "op": op,
                            "side": "sem" if op in SEM else "sur",
                            "edit_distance": ed,
                            "edit_distance_norm": ed / max_len,
                            "jaccard": jaccard(orig_q, variant_q),
                            "len_change_ratio": (len(variant_q) - len(orig_q))
                            / max(len(orig_q), 1),
                            "orig_len": len(orig_q),
                            "variant_len": len(variant_q),
                            "variant_correct": (
                                prop.get(op, {}).get("variant_answer") == gold
                                if gold is not None
                                else None
                            ),
                            "answer_inconsistent": (
                                prop.get(op, {}).get("variant_answer") != orig_ans
                                if orig_ans is not None
                                else None
                            ),
                        }
                    fp_out.write(json.dumps(rec) + "\n")
                    fp_out.flush()
                    os.fsync(fp_out.fileno())
                    done.add((cell, sid, op))
                    total_added += 1

                    # Heartbeat every 200 rows
                    if total_added % 200 == 0:
                        with open(HEARTBEAT, "w") as h:
                            h.write(
                                f"[A] {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                                f"added={total_added} done_total={len(done)} cell={cell}\n"
                            )

        # cell finished
        with open(PROGRESS_FILE, "a") as p:
            p.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{cell}\t{len(done)}\n")
        print(f"[A] Done cell {cell} cumulative_total={len(done)}")

    fp_out.close()
    elapsed = time.time() - t0
    print(
        f"[A] Track A1 complete. Added {total_added} new variants in {elapsed:.1f}s. "
        f"Total severity rows: {len(done)}"
    )

    # Final summary
    with open(os.path.join(OUT_DIR, "_a1_summary.json"), "w") as f:
        json.dump(
            {
                "total_severity_rows": len(done),
                "elapsed_sec": elapsed,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
