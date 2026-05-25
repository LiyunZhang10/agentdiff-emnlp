# Overnight Plan — 2026-05-13 18:07 → 2026-05-14 morning

## Decision (locked in)
Option **C**: extend with HotpotQA × 6 models × {cot, react}, **n=30**, v3 fixed-judge.
mimo track runs in parallel right away (API, no GPU contention).
ollama track waits for the main v3 supervisor to fully drain, then runs smaller-first.

## Three concurrent tracks

| # | Track | Process | Started | Models × cells | Status |
|---|-------|---------|---------|----------------|--------|
| 1 | Main v3 (gsm8k+math, n=20) | PID 1758839, `_supervisor_rerun_v2.sh` | 15:05 | 6 × 4 = 24 cells | running on Llama-3.2-1B math/react |
| 2 | HotpotQA — MiMo  (parallel) | PID 3025556, `_supervisor_hotpotqa_v3.sh --mimo-only` | 18:07 | 1 × 2 = 2 cells | live |
| 3 | HotpotQA — Ollama (waits)   | PID 3026831, `_supervisor_hotpotqa_v3.sh --ollama`    | 18:07 | 5 × 2 = 10 cells | parked, polling for track 1 to finish |

Total new cells across the night: **24 (track 1 already in flight) + 12 (tracks 2+3) = 36 cells, n_total = 24×20 + 12×30 = 840 samples.**

## Smoke validation already done (18:03 → 18:04)
- mimo × hotpotqa × cot, n=1, full pipeline (gen on ollama qwen2.5:3b, agent on mimo, judge on qwen2.5:3b).
- Result: `IR=1.0`, per-type clean (paraphrase=1.0, synonym=0.0, reorder=0.0, format=0.0, distractor=1.0), propagation=`{consistent:3, early_diverge:2}`. End-to-end pipeline confirmed working on hotpotqa.

## Output directory layout
- gsm8k+math (track 1): `runs_real_<slug>_fix/` (existing convention)
- HotpotQA new (tracks 2+3): `runs_real_<slug>_hpqa/`
  - `runs_real_mimo_v25_pro_hpqa/`
  - `runs_real_llama32_1b_hpqa/`
  - `runs_real_qwen25_3b_hpqa/`
  - `runs_real_llama32_3b_hpqa/`
  - `runs_real_qwen25_7b_hpqa/`
  - `runs_real_llama31_8b_hpqa/`

## Per-cell timeouts (HotpotQA, n=30)
1B: 6h, 3B: 9h, 7B: 12h, 8B: 12h, mimo: 6h. All cells resume on restart (sample_id-based dedup in run_cross_model.py).

## ETA (rough)
- Track 2 (mimo hpqa): ≈95s/sample × 30 × 2 = ~95 min → done by ≈ **19:45**.
- Track 1 (main v3): currently 1B math/react in progress; remaining 1 + 4×4 = 17 cells. Conservative 19h → finish around **2026-05-14 ≈ 13:00**.
- Track 3 (ollama hpqa): starts after track 1; smaller-first; should be done by **2026-05-14 ≈ 24:00 ~ 2026-05-15 morning** (this is the long pole).

If you want hotpotqa results sooner, we can later split track-3 into a heat-aware scheduler that interleaves with main v3 cells, but that risks contention so I left the safe sequential path.

## How to inspect progress
```bash
# Heartbeat (consolidated log, all 3 supervisors)
tail -50 /data/workspace/agentdiff_exp/_heartbeat.txt

# Track 1 (gsm8k+math)
tail /data/workspace/agentdiff_exp/_supervisor_rerun_v3.out

# Track 2 (mimo hpqa)
tail /data/workspace/agentdiff_exp/_supervisor_hotpotqa_mimo.out

# Track 3 (ollama hpqa, currently parked)
tail /data/workspace/agentdiff_exp/_supervisor_hotpotqa_ollama.out

# Per-cell completion counts (HotpotQA)
for d in runs_real_*_hpqa; do
  echo "== $d =="; wc -l $d/*.jsonl 2>/dev/null;
done
```

## Recovery
If supervisor dies, just relaunch the exact same nohup command — every cell resumes from where it stopped (dedup on `sample_id`).

## Aggregation tomorrow
After everything finishes, run:
```bash
/usr/bin/python3 -u aggregate_fix_models.py   # extends to *_hpqa slugs (TODO: confirm)
/usr/bin/python3 -u plot_dichotomy_heatmap.py # may need slug whitelist update
```
We'll inspect / patch these aggregators when results land.
