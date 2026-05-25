#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgentDiff — Cross-Model Replication (model-parameterized version).

Runs N samples × {benchmarks} × {agent_types} on an arbitrary ollama model tag,
writing to runs_real_<slug>/ and results_real_<slug>/.

Supports resume (skips sample_ids already in the output jsonl).

Usage:
    # single config
    python3 run_cross_model.py -m qwen2.5:7b -b gsm8k -a react -n 200
    # full 6-cell matrix for one model (gsm8k+math × react/cot/direct)
    python3 run_cross_model.py -m qwen2.5:7b -n 200
    # custom slug (default: auto-derived from model tag)
    python3 run_cross_model.py -m llama3.1:8b --slug llama31_8b -n 200
"""
import json
import os
import re
import sys
import time
import logging
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentdiff_v2 import AgentDiffPipelineV2
from api_router import call_llm

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EXP_DIR, "data_v2_real")


def model_slug(model_tag):
    """qwen2.5:7b -> qwen25_7b ; llama3.1:8b -> llama31_8b"""
    s = model_tag.lower()
    s = re.sub(r"[:\-./]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_data(benchmark, n_samples=200):
    path = os.path.join(DATA_DIR, "%s.jsonl" % benchmark)
    samples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples[:n_samples]


def load_done_ids(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        done.add(rec.get("sample_id", ""))
                    except Exception:
                        pass
    return done


def run_single_config(model_tag, slug, benchmark, agent_type, n_samples, runs_dir, do_patch=False, logger=None, gen_model_tag=None, test_provider="ollama", gen_provider="ollama"):
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("[%s] %s / %s (n=%d) provider=%s", slug, benchmark, agent_type, n_samples, test_provider)
    if gen_model_tag and gen_model_tag != model_tag:
        logger.info("[%s] variant generator: %s/%s", slug, gen_provider, gen_model_tag)
    logger.info("=" * 60)

    samples = load_data(benchmark, n_samples)
    logger.info("Loaded %d samples", len(samples))

    out_file = "%s_%s_real_%s.jsonl" % (benchmark, agent_type, slug)
    out_path = os.path.join(runs_dir, out_file)
    done_ids = load_done_ids(out_path)
    logger.info("Already completed: %d samples", len(done_ids))

    remaining = [s for s in samples if s.get("id", "unknown") not in done_ids]
    if not remaining:
        logger.info("All samples already completed.")
        return out_path

    def llm_fn(prompt):
        return call_llm(prompt, provider=test_provider, model=model_tag, temperature=0.0)

    if gen_model_tag and gen_model_tag != model_tag:
        def gen_llm_fn(prompt):
            return call_llm(prompt, provider=gen_provider, model=gen_model_tag, temperature=0.0)
    else:
        gen_llm_fn = None

    pipeline = AgentDiffPipelineV2(
        llm_fn=llm_fn,
        agent_type=agent_type,
        perturbation_types=["paraphrase", "synonym", "reorder", "format", "distractor"],
        gen_llm_fn=gen_llm_fn,
        val_llm_fn=gen_llm_fn,
    )

    n_done = len(done_ids)
    n_total = len(samples)
    t_start = time.time()

    with open(out_path, 'a') as f:
        for sample in remaining:
            sid = sample.get("id", "unknown")
            t0 = time.time()
            try:
                result = pipeline.run_single(sample, do_patch=do_patch, validate_variants=True)
                result["benchmark"] = benchmark
                result["provider"] = test_provider
                result["model"] = model_tag
            except Exception as e:
                logger.error("Error on sample %s: %s", sid, str(e))
                result = {
                    "sample_id": sid,
                    "agent_type": agent_type,
                    "benchmark": benchmark,
                    "provider": test_provider,
                    "model": model_tag,
                    "error": str(e),
                }

            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            n_done += 1
            elapsed = time.time() - t0

            if n_done % 5 == 0 or n_done == n_total:
                total_elapsed = time.time() - t_start
                n_new = n_done - len(done_ids)
                avg = total_elapsed / max(n_new, 1)
                eta = (n_total - n_done) * avg
                logger.info(
                    "[%s][%s/%s] %d/%d (%.1fs/sample, ETA %.0fm)",
                    slug, benchmark, agent_type, n_done, n_total, elapsed, eta / 60
                )

    logger.info("[%s] Complete: %s / %s (%d samples)", slug, benchmark, agent_type, n_done)
    return out_path


def aggregate_config(slug, benchmark, agent_type, runs_dir):
    out_file = "%s_%s_real_%s.jsonl" % (benchmark, agent_type, slug)
    out_path = os.path.join(runs_dir, out_file)

    if not os.path.exists(out_path):
        return None

    results = []
    with open(out_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not valid:
        return {"n_samples": 0, "n_errors": len(errors)}

    n = len(valid)
    accuracy = sum(1 for r in valid if r.get("original_result", {}).get("is_correct", False)) / n
    inconsistency_rate = sum(1 for r in valid if not r.get("consistency_analysis", {}).get("is_consistent", True)) / n
    avg_consistency = sum(r.get("consistency_analysis", {}).get("consistency_rate", 1.0) for r in valid) / n

    per_type = defaultdict(lambda: {"total": 0, "inconsistent": 0})
    for r in valid:
        pti = r.get("consistency_analysis", {}).get("per_type_inconsistency", {})
        for ptype, rate in pti.items():
            per_type[ptype]["total"] += 1
            per_type[ptype]["inconsistent"] += rate
    per_type_rates = {pt: c["inconsistent"] / c["total"] if c["total"] > 0 else 0
                      for pt, c in per_type.items()}

    all_patterns = Counter()
    for r in valid:
        patterns = r.get("consistency_analysis", {}).get("propagation_patterns", {})
        for p, c in patterns.items():
            all_patterns[p] += c

    return {
        "n_samples": n,
        "n_errors": len(errors),
        "accuracy": accuracy,
        "inconsistency_rate": inconsistency_rate,
        "avg_consistency": avg_consistency,
        "per_type_inconsistency": per_type_rates,
        "propagation_patterns": dict(all_patterns),
    }


def run_all(model_tag, slug, n_samples, benchmarks, agent_types, runs_dir, results_dir, logger, gen_model_tag=None, test_provider="ollama", gen_provider="ollama"):
    logger.info("=" * 60)
    logger.info("[%s] Cross-model suite", slug)
    logger.info("Model: %s", model_tag)
    logger.info("Benchmarks: %s", benchmarks)
    logger.info("Agents: %s", agent_types)
    logger.info("n per cell: %d", n_samples)
    logger.info("=" * 60)

    all_stats = {}
    for bm in benchmarks:
        for at in agent_types:
            key = "%s_%s_real_%s" % (bm, at, slug)
            try:
                run_single_config(model_tag, slug, bm, at, n_samples, runs_dir,
                                  logger=logger,
                                  gen_model_tag=gen_model_tag,
                                  test_provider=test_provider,
                                  gen_provider=gen_provider)
                summary = aggregate_config(slug, bm, at, runs_dir)
                if summary:
                    all_stats[key] = summary
                    logger.info("[%s] %s: acc=%.1f%%, ir=%.1f%%, cr=%.3f",
                                slug, key,
                                summary.get("accuracy", 0) * 100,
                                summary.get("inconsistency_rate", 0) * 100,
                                summary.get("avg_consistency", 0))
            except Exception as e:
                logger.error("[%s] %s failed: %s", slug, key, str(e))

    stats_path = os.path.join(results_dir, "all_stats_real_%s.json" % slug)
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    logger.info("[%s] All stats -> %s", slug, stats_path)
    return all_stats


def main():
    parser = argparse.ArgumentParser(description="AgentDiff cross-model runner")
    parser.add_argument("-m", "--model", required=True, help="model tag (e.g. qwen2.5:7b, gemini-2.0-flash, mimo-v2.5-pro)")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "groq", "gemini", "mimo", "deepseek"],
                        help="LLM provider for the test model")
    parser.add_argument("--gen-model", default=None,
                        help="independent variant-generator model tag (recommended: qwen2.5:3b)")
    parser.add_argument("--gen-provider", default="ollama", choices=["ollama", "groq", "gemini", "mimo", "deepseek"],
                        help="LLM provider for the variant generator (default: ollama)")
    parser.add_argument("--slug", default=None, help="output slug; auto-derived if omitted")
    parser.add_argument("-b", "--benchmark", default=None)
    parser.add_argument("-a", "--agent", default=None)
    parser.add_argument("-n", "--n-samples", type=int, default=200)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math"],
                        help="default: gsm8k math (2x3 matrix)")
    parser.add_argument("--agents", nargs="+", default=["react", "cot", "direct"])
    args = parser.parse_args()

    slug = args.slug or model_slug(args.model)
    runs_dir = os.path.join(EXP_DIR, "runs_real_%s" % slug)
    results_dir = os.path.join(EXP_DIR, "results_real_%s" % slug)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(EXP_DIR, "real_%s_exp.log" % slug)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ]
    )
    logger = logging.getLogger("cross_model")

    if args.benchmark and args.agent:
        run_single_config(args.model, slug, args.benchmark, args.agent,
                          args.n_samples, runs_dir, logger=logger,
                          gen_model_tag=args.gen_model,
                          test_provider=args.provider,
                          gen_provider=args.gen_provider)
        s = aggregate_config(slug, args.benchmark, args.agent, runs_dir)
        if s:
            print(json.dumps(s, indent=2))
    else:
        bms = [args.benchmark] if args.benchmark else args.benchmarks
        ats = [args.agent] if args.agent else args.agents
        run_all(args.model, slug, args.n_samples, bms, ats, runs_dir, results_dir, logger,
                gen_model_tag=args.gen_model,
                test_provider=args.provider,
                gen_provider=args.gen_provider)


if __name__ == "__main__":
    main()
