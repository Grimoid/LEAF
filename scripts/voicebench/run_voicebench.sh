#!/usr/bin/env bash
# VoiceBench open-ended QA (alpacaeval / commoneval / wildvoice) for LEAF vs GRPO adapters.
#   Phase 1 (training env): generate responses per (method, subset)
#   Phase 2 (judge env, vLLM): score with M-Prometheus-14B using VoiceBench's open-QA prompt
#   Phase 3: aggregate into OUT_DIR/VOICEBENCH_RESULTS.md
#
#   LEAF_ADAPTER=<run>/_actor_global_step<N> GRPO_ADAPTER=<run>/_actor_global_step<M> \
#   OUT_DIR=results/voicebench bash scripts/voicebench/run_voicebench.sh
# Optional: SUBSETS="alpacaeval commoneval wildvoice"  PHASES="gen judge agg"  VLLM_TP=1  LIMIT=0
#           JUDGE_PYTHON=<python with vllm> (defaults to $PYTHON_BIN / python)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

LEAF_ADAPTER="${LEAF_ADAPTER:?set LEAF_ADAPTER}"
GRPO_ADAPTER="${GRPO_ADAPTER:?set GRPO_ADAPTER}"
OUT_DIR="${OUT_DIR:-results/voicebench}"
SUBSETS="${SUBSETS:-alpacaeval commoneval wildvoice}"
PHASES="${PHASES:-gen judge agg}"
VLLM_TP="${VLLM_TP:-1}"
LIMIT="${LIMIT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JUDGE_PYTHON="${JUDGE_PYTHON:-$PYTHON_BIN}"
mkdir -p "$OUT_DIR"

if [[ " $PHASES " == *" gen "* ]]; then
  for subset in $SUBSETS; do
    for method in leaf grpo; do
      adapter="$LEAF_ADAPTER"; [[ "$method" == grpo ]] && adapter="$GRPO_ADAPTER"
      out="$OUT_DIR/${method}__${subset}.jsonl"
      [[ -s "$out" ]] && { echo "[gen] $out exists, skipping"; continue; }
      "$PYTHON_BIN" scripts/voicebench/voicebench_generate.py --data "$subset" --adapter "$adapter" \
        --limit "$LIMIT" --output "$out"
    done
  done
fi
if [[ " $PHASES " == *" judge "* ]]; then
  for f in "$OUT_DIR"/*__*.jsonl; do
    [[ "$f" == *.scored.jsonl ]] && continue
    [[ -s "${f%.jsonl}.scored.jsonl" ]] && { echo "[judge] ${f%.jsonl}.scored.jsonl exists, skipping"; continue; }
    "$JUDGE_PYTHON" scripts/voicebench/voicebench_judge_prometheus.py --src "$f" --tp "$VLLM_TP"
  done
fi
if [[ " $PHASES " == *" agg "* ]]; then
  "$PYTHON_BIN" scripts/voicebench/voicebench_aggregate.py --dir "$OUT_DIR"
fi
