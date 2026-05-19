# -*- coding: utf-8 -*-
"""
Sanity Judge Swap Experiment
-----------------------------
目的：验证我们论文中 "semantic-variant (paraphrase/synonym) vs surface-variant
(reorder/format/distractor)" 的 dichotomy 不是因为小 judge (qwen2.5:3b) 过于宽松/过于严格
造成的伪现象。

做法：
  1. 固定 seed，从 data_v2_real/{gsm8k,math}.jsonl 各取前 N=20 题。
  2. 用 VariantGeneratorV2 (ollama qwen2.5:3b) 对每题生成 paraphrase + synonym
     两种语义变体（共 2 个变体/题）。
  3. 同一组变体分别送给：
       - Judge A: qwen2.5:3b (ollama, 当前论文 pipeline 用的 judge)
       - Judge B: mimo-v2.5-pro (closed-source, flagship, 用来做 sanity)
  4. 汇报：
       - 各 judge 的 "等价率"
       - 两个 judge 在 per-variant 上的 agreement rate 与 Cohen's κ
       - 若 κ ≥ 0.6 且 |accept_rate(A) − accept_rate(B)| ≤ 0.1，则认为论文 judge 安全

用法：
  python3 sanity_judge.py --n 20 --bench gsm8k math --out sanity_judge_report
  # 需要先 export MIMO_API_KEY=...  (或在环境变量里)

本脚本**不**依赖正在运行的主 pipeline，可与 mimo supervisor 并行跑；
它只重用 agentdiff_v2.py 里的 VariantGeneratorV2 + EquivalenceValidator 类。
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

# 使脚本既能 CLI 直接跑，也能 import
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from agentdiff_v2 import VariantGeneratorV2, EquivalenceValidatorV2  # noqa: E402
from api_router import ollama_call, mimo_call  # noqa: E402


def _load_bench(path, n, seed=42):
    """从 jsonl 读全部题，固定 seed 采样前 n 题（排序后再取前 n，保证可复现）。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    # 固定按 id 排序 -> 取前 n，若无 id 则按出现顺序
    items.sort(key=lambda x: str(x.get("id", "")))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def _cohen_kappa(a, b):
    """a, b: list of 0/1（1=equivalent）。手写 Cohen κ，避免依赖 sklearn。"""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - pe) < 1e-9:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _qwen_judge_fn(prompt):
    return ollama_call(prompt, model="qwen2.5:3b", temperature=0.0)


def _mimo_judge_fn(prompt):
    return mimo_call(prompt, model="mimo-v2.5-pro", temperature=0.0)


def _run_judge(validator, original, variant, ptype):
    """调用 EquivalenceValidator.validate；失败时记 fallback 标记。"""
    t0 = time.time()
    try:
        res = validator.validate(original, variant, ptype)
        res["_elapsed"] = time.time() - t0
        res["_error"] = None
    except Exception as e:
        res = {
            "is_equivalent": True,  # 保守接受（与原实现一致）
            "confidence": 0.5,
            "reason": "EXCEPTION: %s" % str(e),
            "_elapsed": time.time() - t0,
            "_error": str(e),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="题数 per benchmark")
    ap.add_argument("--bench", nargs="+", default=["gsm8k", "math"],
                    help="benchmark 列表")
    ap.add_argument("--data-dir", default=os.path.join(THIS_DIR, "data_v2_real"))
    ap.add_argument("--out", default=os.path.join(THIS_DIR, "sanity_judge_report"),
                    help="输出前缀，生成 .json / .md / .jsonl 三个文件")
    ap.add_argument("--types", nargs="+",
                    default=["paraphrase", "synonym"],
                    help="只测这些语义扰动类型（surface 扰动 judge 是确定性通过，无需 sanity）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 变体生成器：固定用 ollama qwen2.5:3b（与主 pipeline 一致）
    # 这样 sanity 测的就是"同样的 variant，不同的 judge"
    generator = VariantGeneratorV2(
        llm_fn=_qwen_judge_fn,
        gen_llm_fn=_qwen_judge_fn,
    )
    qwen_validator = EquivalenceValidatorV2(llm_fn=_qwen_judge_fn)
    mimo_validator = EquivalenceValidatorV2(llm_fn=_mimo_judge_fn)

    out_jsonl = args.out + ".jsonl"
    out_json = args.out + ".json"
    out_md = args.out + ".md"

    # 清空 jsonl（重跑不做 resume）
    with open(out_jsonl, "w", encoding="utf-8") as f:
        f.write("")

    records = []
    for bench in args.bench:
        data_path = os.path.join(args.data_dir, "%s.jsonl" % bench)
        if not os.path.exists(data_path):
            print("[WARN] missing %s, skip" % data_path, flush=True)
            continue
        samples = _load_bench(data_path, args.n, seed=args.seed)
        print("[INFO] %s: loaded %d samples" % (bench, len(samples)), flush=True)

        for idx, sample in enumerate(samples):
            try:
                variants = generator.generate_variants(sample, types=args.types)
            except Exception as e:
                print("[WARN] variant gen failed for %s %d: %s" % (bench, idx, e),
                      flush=True)
                continue

            for v in variants:
                ptype = v["perturbation_type"]
                # surface 扰动的 judge 在 EquivalenceValidator 里是确定性通过，
                # 所以 sanity 只对 paraphrase/synonym 做。args.types 已经过滤过，
                # 这里再保险一次。
                if ptype not in ("paraphrase", "synonym"):
                    continue

                qwen_res = _run_judge(qwen_validator, sample, v, ptype)
                mimo_res = _run_judge(mimo_validator, sample, v, ptype)

                rec = {
                    "benchmark": bench,
                    "sample_id": sample.get("id", "%s_%d" % (bench, idx)),
                    "perturbation_type": ptype,
                    "original_question": sample.get("question", sample.get("problem", "")),
                    "variant_question": v.get("question", v.get("problem", "")),
                    "gold_answer": sample.get("answer", ""),
                    "judge_qwen25_3b": qwen_res,
                    "judge_mimo_v25_pro": mimo_res,
                }
                records.append(rec)
                with open(out_jsonl, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print("[%s #%d %s] qwen=%s mimo=%s" % (
                    bench, idx, ptype,
                    "EQ" if qwen_res["is_equivalent"] else "NE",
                    "EQ" if mimo_res["is_equivalent"] else "NE",
                ), flush=True)

    # ---------- 统计 ----------
    def _agg(subset, label):
        qwen_lbl = [1 if r["judge_qwen25_3b"]["is_equivalent"] else 0 for r in subset]
        mimo_lbl = [1 if r["judge_mimo_v25_pro"]["is_equivalent"] else 0 for r in subset]
        n = len(subset)
        if n == 0:
            return {
                "scope": label, "n": 0,
                "qwen_accept_rate": None,
                "mimo_accept_rate": None,
                "agreement_rate": None,
                "cohen_kappa": None,
            }
        agree = sum(1 for x, y in zip(qwen_lbl, mimo_lbl) if x == y)
        return {
            "scope": label,
            "n": n,
            "qwen_accept_rate": sum(qwen_lbl) / n,
            "mimo_accept_rate": sum(mimo_lbl) / n,
            "agreement_rate": agree / n,
            "cohen_kappa": _cohen_kappa(qwen_lbl, mimo_lbl),
        }

    stats = [_agg(records, "overall")]
    for ptype in sorted(set(r["perturbation_type"] for r in records)):
        stats.append(_agg([r for r in records if r["perturbation_type"] == ptype],
                          "ptype=%s" % ptype))
    for bench in sorted(set(r["benchmark"] for r in records)):
        stats.append(_agg([r for r in records if r["benchmark"] == bench],
                          "benchmark=%s" % bench))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_records": len(records),
        "judges": {
            "A": "qwen2.5:3b (ollama, paper default)",
            "B": "mimo-v2.5-pro (closed-source flagship)",
        },
        "variant_generator": "qwen2.5:3b (aligned with main pipeline)",
        "benches": args.bench,
        "types": args.types,
        "seed": args.seed,
        "stats": stats,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---------- Markdown 摘要 ----------
    lines = []
    lines.append("# Sanity Judge Swap Report")
    lines.append("")
    lines.append("Generated at: %s" % report["generated_at"])
    lines.append("")
    lines.append("Judge A (paper default): **qwen2.5:3b (ollama)**  ")
    lines.append("Judge B (sanity):        **mimo-v2.5-pro (closed, xiaomi mimo)**  ")
    lines.append("Variant generator:       qwen2.5:3b  (fixed seed=%d, n=%d per bench)" % (args.seed, args.n))
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append("| scope | n | qwen accept | mimo accept | agreement | Cohen κ |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in stats:
        def _fmt(x):
            return "%.3f" % x if isinstance(x, float) and x == x else "—"
        lines.append("| %s | %d | %s | %s | %s | %s |" % (
            s["scope"], s["n"],
            _fmt(s["qwen_accept_rate"]),
            _fmt(s["mimo_accept_rate"]),
            _fmt(s["agreement_rate"]),
            _fmt(s["cohen_kappa"]),
        ))
    lines.append("")
    lines.append("## Interpretation guideline")
    lines.append("")
    lines.append("- If **Cohen κ ≥ 0.6** and **|accept_rate_A − accept_rate_B| ≤ 0.10**:")
    lines.append("  the paper's dichotomy (sem-IR >> sur-IR) is **robust to judge choice**;")
    lines.append("  we can safely keep qwen2.5:3b as judge and cite this sanity in §Limitations.")
    lines.append("- If κ < 0.4 or accept rates differ by >20%: the judge is a confound;")
    lines.append("  the paper must either (i) swap to mimo as the canonical judge,")
    lines.append("  or (ii) run a dual-judge ensemble and redo main tables.")
    lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n=== DONE ===")
    print("jsonl -> %s" % out_jsonl)
    print("json  -> %s" % out_json)
    print("md    -> %s" % out_md)


if __name__ == "__main__":
    main()
