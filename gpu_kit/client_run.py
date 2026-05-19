#!/usr/bin/env python3
"""
gpu_kit/client_run.py

Self-contained driver to run AgentDiff perturbation experiments on a GPU node
that hosts a vllm server. Reuses code/agentdiff_v2.py for all perturbation,
validation, and analysis logic — only the LLM backend changes.

Typical invocation on the GPU node (after `vllm serve` is up):

    cd agentdiff-emnlp
    python3 gpu_kit/client_run.py \
        --model /jizhicfs/stephnialuo/models/Qwen2.5-14B-Instruct \
        --slug qwen25_14b_vllm \
        --benchmarks gsm8k math hotpotqa \
        --agents react cot direct \
        --n-samples 200

Outputs:
    results/runs_real_<slug>_vllm/<benchmark>/<agent>/...
    results/results_real_<slug>_vllm/<benchmark>_<agent>.json
"""
import argparse
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = os.path.join(ROOT, "code")
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

# reuse the existing pipeline (no modifications)
from agentdiff_v2 import AgentDiffPipelineV2  # noqa: E402
from vllm_llm_fn import make_llm_fn  # noqa: E402


BENCHMARK_FILES = {
    "gsm8k":    "data/gsm8k_test.jsonl",
    "math":     "data/math_test.jsonl",
    "hotpotqa": "data/hotpotqa_test.jsonl",
}


def load_data(benchmark, n_samples):
    path = os.path.join(ROOT, BENCHMARK_FILES[benchmark])
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if len(samples) >= n_samples:
                break
    return samples


def run_one(pipeline, sample, idx, out_dir, logger):
    sid = sample.get("id", f"item_{idx}")
    out_path = os.path.join(out_dir, f"{sid}.json")
    if os.path.exists(out_path):
        return  # resume support
    try:
        result = pipeline.run(sample)
        with open(out_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"sample {sid} failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="vllm model name as served (e.g. /path/to/Qwen2.5-14B-Instruct)")
    ap.add_argument("--slug", required=True,
                    help="output suffix, e.g. qwen25_14b_vllm")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math"])
    ap.add_argument("--agents", nargs="+", default=["react", "cot", "direct"])
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--gen-model", default=None,
                    help="optional independent generator model (defaults to same as test model)")
    ap.add_argument("--gen-base-url", default=None)
    args = ap.parse_args()

    runs_dir    = os.path.join(ROOT, "results", f"runs_real_{args.slug}")
    results_dir = os.path.join(ROOT, "results", f"results_real_{args.slug}")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(ROOT, f"gpu_kit/_logs")
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_path, f"{args.slug}.log")),
        ],
    )
    logger = logging.getLogger("gpu_client")

    # build llm_fn closures
    test_fn = make_llm_fn(model=args.model, base_url=args.base_url,
                          api_key=args.api_key, temperature=args.temperature,
                          max_tokens=args.max_tokens)
    if args.gen_model:
        gen_fn = make_llm_fn(model=args.gen_model,
                             base_url=args.gen_base_url or args.base_url,
                             api_key=args.api_key, temperature=0.7,
                             max_tokens=args.max_tokens)
    else:
        gen_fn = test_fn

    # smoke probe
    logger.info("smoke probe...")
    pong = test_fn("Reply with exactly: PONG")
    logger.info(f"smoke -> {pong[:80]!r}")

    for bm in args.benchmarks:
        samples = load_data(bm, args.n_samples)
        logger.info(f"benchmark={bm}  loaded {len(samples)} samples")
        for agent in args.agents:
            out_dir = os.path.join(runs_dir, bm, agent)
            os.makedirs(out_dir, exist_ok=True)
            pipeline = AgentDiffPipelineV2(
                llm_fn=test_fn,
                agent_type=agent,
                gen_llm_fn=gen_fn,
            )
            for i, s in enumerate(samples):
                if i % 25 == 0:
                    logger.info(f"  [{bm}/{agent}] {i}/{len(samples)}")
                run_one(pipeline, s, i, out_dir, logger)
            # aggregate this cell
            try:
                from run_cross_model import aggregate_config
                summary = aggregate_config(args.slug, bm, agent, runs_dir)
                if summary:
                    sp = os.path.join(results_dir, f"{bm}_{agent}.json")
                    with open(sp, "w") as f:
                        json.dump(summary, f, indent=2)
                    logger.info(f"  -> {sp}")
            except Exception as e:
                logger.exception(f"aggregate failed: {e}")

    logger.info("ALL DONE.")


if __name__ == "__main__":
    main()
