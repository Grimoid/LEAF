#!/usr/bin/env bash
set -euo pipefail

# Pairwise head-to-head judging with JudgeLM (BAAI/JudgeLM-*-v1.0).
# Drop-in replacement for run_judge_predictions_pairwise.sh — same env vars,
# same outputs, but uses the JudgeLM model + prompt template.
#
# Required: PRED_A, PRED_B, RESULTS_JSONL
# Optional: LABEL_A (default leaf), LABEL_B (default grpo), DETAILS_JSONL,
#           JUDGE_MODEL_NAME (default BAAI/JudgeLM-13B-v1.0), VLLM_TP,
#           VLLM_MAX_MODEL_LEN (default 2048 — JudgeLM-13B-v1.0 LLaMA-1 cap), LIMIT

PRED_A="${PRED_A:?set PRED_A=path to model-A step_*.jsonl}"
PRED_B="${PRED_B:?set PRED_B=path to model-B step_*.jsonl}"
LABEL_A="${LABEL_A:-leaf}"
LABEL_B="${LABEL_B:-grpo}"
RESULTS_JSONL="${RESULTS_JSONL:?set RESULTS_JSONL=output aggregate jsonl}"
DETAILS_JSONL="${DETAILS_JSONL:-${RESULTS_JSONL%.jsonl}_details.jsonl}"

JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-BAAI/JudgeLM-13B-v1.0}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-512}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-1.0}"
JUDGE_DTYPE="${JUDGE_DTYPE:-bfloat16}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
LIMIT="${LIMIT:-0}"

# Cache HF models under scratch.
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}}"
mkdir -p "$HF_CACHE_ROOT/hub"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/hub}"

python scripts/judge/judge_predictions_pairwise_judgelm.py \
  --pred_a "$PRED_A" --pred_b "$PRED_B" \
  --label_a "$LABEL_A" --label_b "$LABEL_B" \
  --results_jsonl "$RESULTS_JSONL" \
  --details_jsonl "$DETAILS_JSONL" \
  --judge_model_name "$JUDGE_MODEL_NAME" \
  --judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
  --judge_temperature "$JUDGE_TEMPERATURE" \
  --judge_top_p "$JUDGE_TOP_P" \
  --judge_dtype "$JUDGE_DTYPE" \
  --use_vllm 1 \
  --vllm_tensor_parallel_size "$VLLM_TP" \
  --vllm_gpu_memory_utilization "$VLLM_GPU_UTIL" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --limit "$LIMIT"
