#!/usr/bin/env /usr/bin/python3
"""
Track D: Embedding-based severity audit (Reviewer 1 / 3 attack response).

Reviewer 1 共识攻击点：
  "Edit distance does not measure semantic offset. Matching paraphrase (ED=0.480)
   to distractor (ED=0.485) is not severity control —— what we need is
   information-content control."

This script:
  1. Reads track_a/severity_per_variant.jsonl (8350 rows, edit_distance + jaccard already)
  2. Adds three new severity proxies per variant:
       - sbert_dist     : 1 - cosine(SBERT(orig), SBERT(variant))
       - prompt_len_chg : abs((variant_len - orig_len) / max(orig_len,1))
       - tfidf_dist     : 1 - cosine(TFIDF(orig), TFIDF(variant)) over the cell question pool
  3. Stratifies severity-matched Δ on each new proxy. If Δ shrinks but stays >0
     across all four severity definitions, the headline phenomenon is robust.
     If Δ collapses on sbert_dist, the phenomenon is an edit-distance artefact.

Embedding via ollama nomic-embed-text (768-d, no GPU, local).

Resilience: append-only jsonl with done set. Idempotent rerun.
"""
import json
import os
import sys
import math
import time
import re
import urllib.request
import urllib.error
from collections import defaultdict

import numpy as np

ROOT = "."
SRC = os.path.join(ROOT, "track_a/severity_per_variant.jsonl")
OUT_DIR = os.path.join(ROOT, "track_d")
os.makedirs(OUT_DIR, exist_ok=True)
DST = os.path.join(OUT_DIR, "severity_with_embeddings.jsonl")
SUMMARY = os.path.join(OUT_DIR, "_d_summary.txt")
SEVMATCH = os.path.join(OUT_DIR, "severity_matched_delta_4proxies.json")
HEARTBEAT = os.path.join(OUT_DIR, "_d_heartbeat.txt")

OLLAMA = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"

SEM = {"paraphrase", "synonym"}
SUR = {"reorder", "format", "distractor"}


def embed(text):
    body = json.dumps({"model": EMBED_MODEL, "prompt": text or " "}).encode()
    req = urllib.request.Request(
        OLLAMA + "/api/embeddings", data=body,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode())
            return np.array(d["embedding"], dtype=np.float32)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def cosine(u, v):
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def load_done():
    """Read DST and return set of (cell, sample_id, op) already embedded."""
    done = set()
    if os.path.exists(DST):
        with open(DST) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                    done.add((r["cell"], r["sample_id"], r["op"]))
                except Exception:
                    pass
    return done


def load_orig_questions():
    """Build (cell, sample_id) -> orig_question map by re-reading runs_real_*."""
    import glob
    omap = {}
    for cell_path in glob.glob(os.path.join(ROOT, "runs_real_*", "*.jsonl")):
        if "_genmimo" in cell_path:
            continue
        cell = os.path.relpath(cell_path, ROOT)
        for ln in open(cell_path):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            sid = r.get("sample_id")
            oq = r.get("sample_question", "")
            for v in r.get("perturbation_variants", []) or []:
                op = v.get("perturbation_type")
                if op not in SEM and op not in SUR:
                    continue
                vq = v.get("variant_question", "")
                omap[(cell, sid, op)] = (oq, vq)
    return omap


def stratified_delta(rows, severity_key, n_bins=10):
    """For each cell, compute IR_sem - IR_sur where surface variants are
    weighted by edit-distance bin to match the severity distribution of
    sem variants. Returns: dict cell -> matched_delta (pp).
    Logic mirrors track_a2 but with arbitrary severity_key.
    """
    by_cell = defaultdict(lambda: {"sem": [], "sur": []})
    for r in rows:
        side = r.get("side")
        if side not in ("sem", "sur"):
            continue
        sev = r.get(severity_key)
        if sev is None or not isinstance(sev, (int, float)):
            continue
        inc = 1.0 if r.get("answer_inconsistent") else 0.0
        by_cell[r["cell"]][side].append((sev, inc))

    deltas = {}
    for cell, sides in by_cell.items():
        sem = sides["sem"]
        sur = sides["sur"]
        if not sem or not sur:
            continue
        # Build edges from union of sem severities (paraphrase+synonym)
        sem_sevs = np.array([s for s, _ in sem])
        # Quantile bin edges (handle small-cell case)
        n_bins_eff = min(n_bins, max(2, len(sem_sevs) // 5))
        if n_bins_eff < 2:
            edges = np.array([sem_sevs.min() - 1e-9, sem_sevs.max() + 1e-9])
        else:
            edges = np.quantile(sem_sevs, np.linspace(0, 1, n_bins_eff + 1))
            edges[0] -= 1e-9
            edges[-1] += 1e-9
        # sem distribution over bins
        sem_bin = np.digitize(sem_sevs, edges) - 1
        sem_bin = np.clip(sem_bin, 0, len(edges) - 2)
        sem_dist = np.bincount(sem_bin, minlength=len(edges) - 1) / len(sem_sevs)
        # sur per-bin IR
        sur_sevs = np.array([s for s, _ in sur])
        sur_incs = np.array([inc for _, inc in sur])
        sur_bin = np.digitize(sur_sevs, edges) - 1
        sur_bin = np.clip(sur_bin, 0, len(edges) - 2)
        # weighted IR for sur using sem_dist as importance weights
        weighted_ir_sur = 0.0
        total_w = 0.0
        for b in range(len(edges) - 1):
            mask = sur_bin == b
            if mask.sum() == 0 or sem_dist[b] == 0:
                continue
            ir_b = sur_incs[mask].mean()
            weighted_ir_sur += sem_dist[b] * ir_b
            total_w += sem_dist[b]
        if total_w == 0:
            continue
        weighted_ir_sur = weighted_ir_sur / total_w
        ir_sem = float(np.mean([inc for _, inc in sem]))
        deltas[cell] = (ir_sem - weighted_ir_sur) * 100.0  # pp
    return deltas


def main():
    t0 = time.time()
    print("[D] Loading already-embedded done set ...")
    done = load_done()
    print(f"[D] {len(done)} variants already embedded")

    print("[D] Building orig_question map from runs_real_* ...")
    omap = load_orig_questions()
    print(f"[D] {len(omap)} (cell, sid, op) pairs in orig map")

    print(f"[D] Streaming {SRC} and embedding the ones not yet done ...")
    n_total = 0
    n_emb = 0
    fp_out = open(DST, "a", buffering=1)
    with open(SRC) as fp:
        for ln_no, ln in enumerate(fp, 1):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            n_total += 1
            key = (r["cell"], r["sample_id"], r["op"])
            if key in done:
                continue
            pair = omap.get(key)
            if pair is None:
                # variant not present in original .jsonl; skip
                continue
            oq, vq = pair
            try:
                eo = embed(oq)
                ev = embed(vq)
            except Exception as e:
                # network issue, save heartbeat and skip
                with open(HEARTBEAT, "w") as h:
                    h.write(f"embed_error at row {ln_no}: {e}\n")
                continue
            sb = 1.0 - cosine(eo, ev)
            r["sbert_dist"] = sb
            r["orig_len"] = len(oq)
            r["variant_len"] = len(vq)
            r["len_ratio_abs"] = abs(len(vq) - len(oq)) / max(len(oq), 1)
            fp_out.write(json.dumps(r) + "\n")
            n_emb += 1
            if n_emb % 50 == 0:
                elapsed = time.time() - t0
                rate = n_emb / max(elapsed, 1e-3)
                with open(HEARTBEAT, "w") as h:
                    h.write(
                        f"[D] {n_emb} embedded, total seen {n_total}, "
                        f"rate {rate:.1f}/s, elapsed {elapsed:.1f}s\n"
                    )
                print(f"  ... {n_emb} embedded ({rate:.1f}/s)")

    fp_out.close()
    elapsed = time.time() - t0
    print(f"[D] Embedding pass done. {n_emb} new embeddings in {elapsed:.1f}s")

    # ---- Now do severity-matched Δ on 4 severity proxies ----
    print("[D] Loading combined embedded jsonl for matched-Δ ...")
    rows = []
    with open(DST) as fp:
        for ln in fp:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    print(f"[D] Loaded {len(rows)} embedded rows")

    proxies = [
        ("edit_distance_norm", "Edit distance (normalized)"),
        ("jaccard", "Token Jaccard distance"),
        ("sbert_dist", "Sentence-BERT (nomic-embed) cosine distance"),
        ("len_ratio_abs", "Absolute length ratio change"),
    ]

    all_results = {}
    for key, label in proxies:
        deltas = stratified_delta(rows, key, n_bins=10)
        if not deltas:
            all_results[key] = {"label": label, "n_cells": 0}
            continue
        d_arr = np.array(list(deltas.values()))
        n = len(d_arr)
        mean = float(d_arr.mean())
        sd = float(d_arr.std(ddof=1)) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        t = mean / se if se > 0 else 0.0
        from scipy.stats import t as tdist
        p = float(2 * (1 - tdist.cdf(abs(t), n - 1))) if n > 1 else 1.0
        pos = int((d_arr > 0).sum())
        all_results[key] = {
            "label": label,
            "n_cells": n,
            "mean_pp": mean,
            "se_pp": se,
            "paired_t": t,
            "p_two_sided": p,
            "pos_count": pos,
            "delta_per_cell": {c: float(v) for c, v in deltas.items()},
        }
        print(f"  {key:<28} mean Δ = {mean:+.2f} pp  t={t:+.2f}  p={p:.4f}  ({pos}/{n} positive)")

    with open(SEVMATCH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    with open(SUMMARY, "w") as f:
        f.write(f"Track D complete in {time.time() - t0:.1f}s\n")
        f.write(f"Embedded {len(rows)} variants via {EMBED_MODEL}\n\n")
        f.write("Severity-matched Δ on 4 severity proxies:\n")
        for key, label in proxies:
            r = all_results.get(key, {})
            if not r or r.get("n_cells", 0) == 0:
                f.write(f"  {label}: NO DATA\n")
                continue
            f.write(
                f"  {label:<45} mean={r['mean_pp']:+.2f}pp  "
                f"t={r['paired_t']:+.2f}  p={r['p_two_sided']:.4f}  "
                f"{r['pos_count']}/{r['n_cells']} positive\n"
            )
    print("\n--- _d_summary.txt ---")
    print(open(SUMMARY).read())


if __name__ == "__main__":
    main()
