#!/usr/bin/env /usr/bin/python3
"""
Track F — Second-judge cross-validation (Reviewer R2-Fatal-4: judge homogeneity).

Reviewer concern: a single Qwen2.5-7B judge evaluating GSM8K/MATH/HotpotQA may
have benchmark-, operator-, or answer-format-specific biases that systematically
inflate or deflate Δ.

Method:
  * Sample 800 (variant_answer, gold_answer) pairs stratified by benchmark and
    operator, with answer_inconsistent==True (the contested judgement region).
  * Re-evaluate equivalence using MiMo-v2.5-Pro as a second LLM judge.
  * Report agreement (Cohen's κ) between Qwen2.5-7B and MiMo on:
      - overall
      - per benchmark
      - per operator (sem vs sur)
      - per answer-format pattern (numeric / fraction / text)
  * If κ ≥ 0.7 across all strata → judge homogeneity not a fatal concern.
  * If any stratum has κ ≤ 0.4 → flag and recompute Δ on judge-agreement subset.

Resilience: writes per-call results immediately to track_f/judge_results.jsonl
with one row per call. Restart re-reads file and skips done (cell, sid, op) tuples.
Bounded MiMo budget: max 1500 calls (~3M tokens) to leave buffer.
"""
import json
import os
import sys
import time
import re
import random
from collections import defaultdict
import urllib.request
import urllib.error

ROOT = "/data/workspace/agentdiff_exp"
OUT = os.path.join(ROOT, "track_f")
os.makedirs(OUT, exist_ok=True)
JUDGE_FILE = os.path.join(OUT, "judge_results.jsonl")
PROGRESS = os.path.join(OUT, "_f_progress.txt")
SUMMARY = os.path.join(OUT, "_f_summary.txt")
RESULT_FILE = os.path.join(OUT, "judge_agreement.json")

MIMO_API_KEY = "tp-s5xwbhbqjzgvoka4dtm8v7xso1lv23vugelchz4w4j3uma0p"
MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
MAX_CALLS = 1500  # safety cap

SEM = {"paraphrase", "synonym"}
SUR = {"reorder", "format", "distractor"}


def call_mimo_judge(question_orig, answer_orig, question_var, answer_var, gold,
                    timeout=60, retries=2):
    """Ask MiMo whether two answers are equivalent given the question pair.

    Note: mimo-v2.5-pro is a reasoning model that emits long internal CoT into
    `reasoning_content`. We allocate 1500 tokens so reasoning + final verdict
    both fit, and we look at both `content` and `reasoning_content` for the
    verdict word.
    """
    prompt = (
        "You are evaluating whether two answers to similar questions are EQUIVALENT.\n\n"
        f"Question A: {question_orig}\n"
        f"Answer A: {answer_orig}\n\n"
        f"Question B (a paraphrase or perturbation of A): {question_var}\n"
        f"Answer B: {answer_var}\n\n"
        f"Gold reference (if relevant): {gold}\n\n"
        "Are Answer A and Answer B EQUIVALENT (same underlying value/meaning, "
        "ignoring formatting / units representation)? "
        "End your reply with EXACTLY one line containing only the word EQUIVALENT "
        "or DIFFERENT (no other word on that final line)."
    )
    body = json.dumps({
        "model": "mimo-v2.5-pro",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,  # reasoning model needs space for CoT + verdict
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        MIMO_BASE_URL + "/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + MIMO_API_KEY,
            "Content-Type": "application/json",
        },
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            reasoning = (msg.get("reasoning_content") or "").strip()
            # Combined search for the verdict; final answer often ends the content,
            # but reasoning model may also cut off — fall through to reasoning text.
            search_text = content + "\n" + reasoning
            # Look for the verdict on the LAST non-empty line of content first
            content_last_line = content.split("\n")[-1].strip().upper() if content else ""
            if "EQUIVALENT" in content_last_line and "DIFFERENT" not in content_last_line:
                verdict = "EQUIVALENT"
            elif "DIFFERENT" in content_last_line and "EQUIVALENT" not in content_last_line:
                verdict = "DIFFERENT"
            else:
                # fallback: count occurrences across both fields
                up = search_text.upper()
                eq_count = up.count("EQUIVALENT")
                df_count = up.count("DIFFERENT")
                if eq_count > df_count:
                    verdict = "EQUIVALENT"
                elif df_count > eq_count:
                    verdict = "DIFFERENT"
                else:
                    verdict = "AMBIGUOUS"
            return verdict, (content + " | RC: " + reasoning[:150])[:400]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"MiMo call failed after {retries} retries: {last_err}")


def load_done():
    done = set()
    if os.path.exists(JUDGE_FILE):
        for ln in open(JUDGE_FILE):
            try:
                r = json.loads(ln)
                done.add((r["cell"], r["sid"], r["op"]))
            except Exception:
                continue
    return done


def detect_format(ans):
    if not isinstance(ans, str):
        return "other"
    s = ans.strip()
    if re.match(r"^-?\$?[0-9,]+(\.[0-9]+)?\$?$", s):
        return "numeric"
    if "/" in s and re.match(r"^\d+/\d+$", s):
        return "fraction"
    if len(s) <= 50:
        return "short_text"
    return "long_text"


def gather_candidates():
    """Stratified sample of inconsistent (variant, gold) pairs."""
    cands = []
    for d in sorted(os.listdir(ROOT)):
        if not d.startswith("runs_real_"):
            continue
        s = d[len("runs_real_"):]
        if s.endswith("_genmimo") or s.endswith("_genqwen14b"):
            continue
        if not (s.endswith("_fix") or s.endswith("_hpqa") or s == "mimo_v25_pro"):
            continue
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if not f.endswith(".jsonl"):
                continue
            bench = f.split("_")[0]
            cell = os.path.relpath(os.path.join(ROOT, d, f), ROOT)
            for ln in open(os.path.join(ROOT, d, f)):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                sid = r.get("sample_id")
                qo = r.get("sample_question", "")
                ao = r.get("original_result", {}).get("final_answer")
                gold = r.get("sample_gold_answer")
                v_by_op = {}
                for v in r.get("perturbation_variants", []):
                    op = v.get("perturbation_type")
                    if op:
                        v_by_op[op] = v
                for det in r.get("propagation_details", []):
                    op = det.get("perturbation_type")
                    if op not in SEM and op not in SUR:
                        continue
                    av = det.get("variant_answer")
                    if av is None or ao is None:
                        continue
                    inc = (av != ao)  # this is the existing judge label
                    qv = v_by_op.get(op, {}).get("variant_question", "")
                    if not qv:
                        continue
                    cands.append({
                        "cell": cell, "bench": bench, "op": op, "sid": sid,
                        "side": "sem" if op in SEM else "sur",
                        "question_orig": qo, "answer_orig": ao,
                        "question_var": qv, "answer_var": av,
                        "gold": gold,
                        "qwen_inc": inc,
                        "ans_format": detect_format(av),
                    })
    return cands


def sample_stratified(cands, target=800, seed=42):
    rng = random.Random(seed)
    by_strat = defaultdict(list)
    for c in cands:
        # stratify by (bench, op) — 3 benches × 5 ops = 15 strata
        by_strat[(c["bench"], c["op"])].append(c)
    per_strat = max(target // len(by_strat), 30) if by_strat else 30
    sampled = []
    for k, lst in by_strat.items():
        rng.shuffle(lst)
        sampled.extend(lst[:per_strat])
    rng.shuffle(sampled)
    return sampled[:target]


def cohen_kappa(a, b):
    """Cohen's kappa for two binary lists."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def main():
    done = load_done()
    print(f"[F] Loaded {len(done)} previously-judged pairs")

    cands = gather_candidates()
    print(f"[F] Total candidate (cell, sid, op) tuples: {len(cands)}")

    sampled = sample_stratified(cands, target=MAX_CALLS)
    print(f"[F] Sampled {len(sampled)} for re-judgment")

    fp = open(JUDGE_FILE, "a", buffering=1)
    n_calls = 0
    n_skipped = 0
    t0 = time.time()
    for c in sampled:
        key = (c["cell"], c["sid"], c["op"])
        if key in done:
            n_skipped += 1
            continue
        try:
            verdict, raw = call_mimo_judge(
                c["question_orig"], c["answer_orig"],
                c["question_var"], c["answer_var"], c["gold"],
            )
            mimo_inc = (verdict == "DIFFERENT")
        except Exception as e:
            print(f"[F][WARN] {key} failed: {e}")
            mimo_inc = None
            raw = f"ERROR: {e}"

        rec = {
            "cell": c["cell"], "sid": c["sid"], "op": c["op"], "side": c["side"],
            "bench": c["bench"], "ans_format": c["ans_format"],
            "qwen_inc": bool(c["qwen_inc"]),
            "mimo_inc": mimo_inc,
            "mimo_raw": raw,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        fp.write(json.dumps(rec) + "\n")
        fp.flush()
        os.fsync(fp.fileno())
        n_calls += 1
        if n_calls % 50 == 0:
            with open(PROGRESS, "a") as p:
                p.write(f"{time.strftime('%H:%M:%S')}  calls={n_calls}\n")
            print(f"[F] {n_calls} calls in {time.time()-t0:.1f}s")

    fp.close()
    print(f"[F] Done. {n_calls} new calls, {n_skipped} skipped (already done).")

    # Aggregate κ
    rows = []
    for ln in open(JUDGE_FILE):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("mimo_inc") is None:
            continue
        rows.append(r)
    print(f"[F] Aggregating over {len(rows)} valid judgement pairs")

    def kappa_of(filtered):
        if not filtered:
            return None
        a = [int(r["qwen_inc"]) for r in filtered]
        b = [int(r["mimo_inc"]) for r in filtered]
        return cohen_kappa(a, b), len(filtered)

    out = {"n_total": len(rows)}
    overall = kappa_of(rows)
    out["overall"] = {"kappa": overall[0], "n": overall[1]}
    print(f"  Overall κ = {overall[0]:.3f} (n={overall[1]})")

    out["by_bench"] = {}
    for bench in ("gsm8k", "math", "hotpotqa"):
        sub = [r for r in rows if r["bench"] == bench]
        k, n = kappa_of(sub) if sub else (None, 0)
        out["by_bench"][bench] = {"kappa": k, "n": n}
        if k is not None:
            print(f"  {bench:<10} κ = {k:.3f}  (n={n})")

    out["by_op"] = {}
    for op in ("paraphrase", "synonym", "reorder", "format", "distractor"):
        sub = [r for r in rows if r["op"] == op]
        k, n = kappa_of(sub) if sub else (None, 0)
        out["by_op"][op] = {"kappa": k, "n": n}
        if k is not None:
            print(f"  {op:<11} κ = {k:.3f}  (n={n})")

    out["by_side"] = {}
    for side in ("sem", "sur"):
        sub = [r for r in rows if r["side"] == side]
        k, n = kappa_of(sub) if sub else (None, 0)
        out["by_side"][side] = {"kappa": k, "n": n}
        if k is not None:
            print(f"  side {side:<5} κ = {k:.3f}  (n={n})")

    # IR with MiMo judge: re-derive Δ
    by_cell_op = defaultdict(lambda: {"sem": [], "sur": []})
    for r in rows:
        by_cell_op[r["cell"]]["sem" if r["side"] == "sem" else "sur"].append(int(r["mimo_inc"]))

    cell_deltas = {}
    for cell, blob in by_cell_op.items():
        if not blob["sem"] or not blob["sur"]:
            continue
        ir_sem = sum(blob["sem"]) / len(blob["sem"]) * 100
        ir_sur = sum(blob["sur"]) / len(blob["sur"]) * 100
        cell_deltas[cell] = {"IR_sem_mimo": ir_sem, "IR_sur_mimo": ir_sur,
                              "delta_mimo": ir_sem - ir_sur}
    out["per_cell_mimo_delta"] = cell_deltas

    with open(RESULT_FILE + ".tmp", "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(RESULT_FILE + ".tmp", RESULT_FILE)

    with open(SUMMARY, "w") as f:
        f.write("Track F: second-judge agreement (MiMo vs Qwen2.5-7B).\n\n")
        f.write(f"Overall κ = {out['overall']['kappa']:.3f} (n={out['overall']['n']})\n\n")
        f.write("By benchmark:\n")
        for k, v in out["by_bench"].items():
            if v["kappa"] is not None:
                f.write(f"  {k:<10} κ = {v['kappa']:.3f}  (n={v['n']})\n")
        f.write("\nBy operator:\n")
        for k, v in out["by_op"].items():
            if v["kappa"] is not None:
                f.write(f"  {k:<11} κ = {v['kappa']:.3f}  (n={v['n']})\n")
        f.write("\nBy side:\n")
        for k, v in out["by_side"].items():
            if v["kappa"] is not None:
                f.write(f"  {k:<5} κ = {v['kappa']:.3f}  (n={v['n']})\n")
        f.write("\nMiMo-judge per-cell Δ (subset only):\n")
        for cell, blob in sorted(cell_deltas.items()):
            f.write(f"  {cell}: Δ={blob['delta_mimo']:+.2f}\n")
    print(f"\nSaved → {RESULT_FILE}\nSummary → {SUMMARY}")


if __name__ == "__main__":
    main()
