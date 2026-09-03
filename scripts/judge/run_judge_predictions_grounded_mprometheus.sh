#!/usr/bin/env bash
set -euo pipefail

# Per-prediction-file M-Prometheus Likert-5 runner (call once per predictions jsonl;
# judge_checkpoint.sh drives it for the standard per-checkpoint protocol).
#
# Required: PREDICTIONS_JSONL, DETAILS_JSONL, RESULTS_JSONL
# Optional: RUBRIC_KEY (qa|mt, default qa), RUBRIC_FILE (overrides RUBRIC_KEY),
#           JUDGE_MODEL_NAME (default Unbabel/M-Prometheus-14B), VLLM_TP, etc.

PREDICTIONS_JSONL="${PREDICTIONS_JSONL:?set PREDICTIONS_JSONL}"
DETAILS_JSONL="${DETAILS_JSONL:?set DETAILS_JSONL}"
RESULTS_JSONL="${RESULTS_JSONL:?set RESULTS_JSONL}"
RUBRIC_KEY="${RUBRIC_KEY:-qa}"
RUBRIC_FILE="${RUBRIC_FILE:-}"

JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Unbabel/M-Prometheus-14B}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-768}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-1.0}"
JUDGE_DTYPE="${JUDGE_DTYPE:-bfloat16}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
LIMIT="${LIMIT:-0}"
SCORE_MODE="${SCORE_MODE:-expected}"   # argmax|expected (default expected gives continuous 0-100)
LOGPROBS_TOPK="${LOGPROBS_TOPK:-5}"     # vLLM default engine cap is 5; the python script raises max_logprobs if you set higher

# Cache HF models under scratch.
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${HF_HOME:-$HOME/.cache/huggingface}}"
mkdir -p "$HF_CACHE_ROOT/hub"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_ROOT/hub}"

extra=()
if [[ -n "$RUBRIC_FILE" ]]; then
  extra+=(--rubric_file "$RUBRIC_FILE")
fi

python scripts/judge/judge_predictions_grounded_mprometheus.py \
  --predictions_jsonl "$PREDICTIONS_JSONL" \
  --judge_details_jsonl "$DETAILS_JSONL" \
  --results_jsonl "$RESULTS_JSONL" \
  --rubric_key "$RUBRIC_KEY" \
  --judge_model_name "$JUDGE_MODEL_NAME" \
  --judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
  --judge_temperature "$JUDGE_TEMPERATURE" \
  --judge_top_p "$JUDGE_TOP_P" \
  --judge_dtype "$JUDGE_DTYPE" \
  --use_vllm 1 \
  --vllm_tensor_parallel_size "$VLLM_TP" \
  --vllm_gpu_memory_utilization "$VLLM_GPU_UTIL" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --limit "$LIMIT" \
  --score_mode "$SCORE_MODE" \
  --logprobs_topk "$LOGPROBS_TOPK" \
  "${extra[@]}"
