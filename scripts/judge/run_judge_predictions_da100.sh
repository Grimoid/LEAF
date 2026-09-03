#!/usr/bin/env bash
set -euo pipefail

# LibriSQA grounded judging on 0-100 Direct Assessment scale, with optional
# self-consistency (N_SAMPLES>1 + JUDGE_TEMPERATURE>0 => median of N).

RUN_DIR="${RUN_DIR:?set RUN_DIR}"
PRED_DIR="${PRED_DIR:-$RUN_DIR/librisqa_judge_predictions}"
DETAILS_DIR="${DETAILS_DIR:-$RUN_DIR/librisqa_judge_da100_details}"
RESULTS_JSONL="${RESULTS_JSONL:-$RUN_DIR/eval_all_ckpts_librisqa_judge_da100_only.jsonl}"

LIBRISQA_TRAIN_JSON="${LIBRISQA_TRAIN_JSON:-hf://datasets/ZihanZhao/LibriSQA/LibriSQA-PartI/LibriSQA-PartI-train.json}"
LIBRISQA_TEST_JSON="${LIBRISQA_TEST_JSON:-hf://datasets/ZihanZhao/LibriSQA/LibriSQA-PartI/LibriSQA-PartI-test.json}"

JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-512}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-1.0}"
JUDGE_DTYPE="${JUDGE_DTYPE:-bfloat16}"
N_SAMPLES="${N_SAMPLES:-1}"
SEED="${SEED:-0}"
STEP_FILTER="${STEP_FILTER:-}"
OVERWRITE="${OVERWRITE:-0}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"

python scripts/judge/judge_predictions_da100.py \
  --predictions_dir "$PRED_DIR" \
  --judge_details_dir "$DETAILS_DIR" \
  --results_jsonl "$RESULTS_JSONL" \
  --librisqa_source_jsons "$LIBRISQA_TRAIN_JSON" "$LIBRISQA_TEST_JSON" \
  --judge_model_name "$JUDGE_MODEL_NAME" \
  --judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
  --judge_temperature "$JUDGE_TEMPERATURE" \
  --judge_top_p "$JUDGE_TOP_P" \
  --judge_dtype "$JUDGE_DTYPE" \
  --n_samples "$N_SAMPLES" \
  --seed "$SEED" \
  --step_filter "$STEP_FILTER" \
  --overwrite "$OVERWRITE" \
  --use_vllm 1 \
  --vllm_tensor_parallel_size "$VLLM_TP" \
  --vllm_gpu_memory_utilization "$VLLM_GPU_UTIL" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
