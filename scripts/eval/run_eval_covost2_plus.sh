#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
DATA_DIR="${DATA_DIR:-${SPEECH_DATA_ROOT:?set SPEECH_DATA_ROOT}/covost2_en_de}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-0}"
RANDOM_SUBSET_SEED="${RANDOM_SUBSET_SEED:--1}"
NUM_RUNS="${NUM_RUNS:-5}"
BASE_SEED="${BASE_SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
DO_SAMPLE="${DO_SAMPLE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
WITH_BERTSCORE="${WITH_BERTSCORE:-1}"
BERTSCORE_LANG="${BERTSCORE_LANG:-de}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
LOWERCASE_BLEU="${LOWERCASE_BLEU:-1}"
EVAL_LOWERCASE="${EVAL_LOWERCASE:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
PREDICTIONS_JSONL="${PREDICTIONS_JSONL:-}"
JUDGE_DETAILS_JSONL="${JUDGE_DETAILS_JSONL:-}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-}"
JUDGE_RUN_MODE="${JUDGE_RUN_MODE:-first}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-4}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-256}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-1.0}"
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda}"
JUDGE_DTYPE="${JUDGE_DTYPE:-bfloat16}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/$USER/hf-datasets}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-/tmp/$USER/hf-evaluate}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-/tmp/$USER/hf-metrics}"
export NLTK_DATA="${NLTK_DATA:-/tmp/$USER/nltk_data}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_EVALUATE_CACHE" "$HF_METRICS_CACHE" "$NLTK_DATA"

if [[ -n "$OUTPUT_JSON" ]]; then
  mkdir -p "$(dirname "$OUTPUT_JSON")"
  LOG_FILE="${OUTPUT_JSON%.json}.log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  [[ -z "$PREDICTIONS_JSONL" ]] && PREDICTIONS_JSONL="${OUTPUT_JSON%.json}_predictions.jsonl"
  if [[ -n "$JUDGE_MODEL_NAME" && -z "$JUDGE_DETAILS_JSONL" ]]; then
    JUDGE_DETAILS_JSONL="${OUTPUT_JSON%.json}_judge_details.jsonl"
  fi
  echo "LOG_FILE=$LOG_FILE"
fi

extra_args=()
[[ -n "$OUTPUT_JSON" ]] && extra_args+=(--output_json "$OUTPUT_JSON")
[[ -n "$PREDICTIONS_JSONL" ]] && extra_args+=(--predictions_jsonl "$PREDICTIONS_JSONL")
[[ -n "$JUDGE_DETAILS_JSONL" ]] && extra_args+=(--judge_details_jsonl "$JUDGE_DETAILS_JSONL")
[[ -n "$JUDGE_MODEL_NAME" ]] && extra_args+=(--judge_model_name "$JUDGE_MODEL_NAME")

torchrun --nproc_per_node="$NUM_GPUS" scripts/eval/eval_covost2_plus.py \
  --model_name "$MODEL_NAME" \
  --data_dir "$DATA_DIR" \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --random_subset_seed "$RANDOM_SUBSET_SEED" \
  --num_runs "$NUM_RUNS" \
  --base_seed "$BASE_SEED" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --do_sample "$DO_SAMPLE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --with_bertscore "$WITH_BERTSCORE" \
  --bertscore_lang "$BERTSCORE_LANG" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --lowercase_bleu "$LOWERCASE_BLEU" \
  --eval_lowercase "$EVAL_LOWERCASE" \
  --judge_run_mode "$JUDGE_RUN_MODE" \
  --judge_batch_size "$JUDGE_BATCH_SIZE" \
  --judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
  --judge_temperature "$JUDGE_TEMPERATURE" \
  --judge_top_p "$JUDGE_TOP_P" \
  --judge_device "$JUDGE_DEVICE" \
  --judge_dtype "$JUDGE_DTYPE" \
  "${extra_args[@]}"
