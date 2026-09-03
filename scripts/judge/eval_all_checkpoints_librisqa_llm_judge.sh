#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-${RUN_DIR:?set RUN_DIR}}"
RESULTS_JSONL="${RESULTS_JSONL:-${RUN_DIR}/eval_all_ckpts_librisqa_judge.jsonl}"
DATA_DIR="${DATA_DIR:-${SPEECH_DATA_ROOT:?set SPEECH_DATA_ROOT}/librisqa_part1}"
SPLIT="${SPLIT:-test}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
BASE_SEED="${BASE_SEED:-42}"
LIMIT="${LIMIT:-0}"
RANDOM_SUBSET_SEED="${RANDOM_SUBSET_SEED:--1}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
DO_SAMPLE="${DO_SAMPLE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WITH_BERTSCORE="${WITH_BERTSCORE:-0}"
BERTSCORE_LANG="${BERTSCORE_LANG:-en}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
LOWERCASE_BLEU="${LOWERCASE_BLEU:-1}"
EVAL_LOWERCASE="${EVAL_LOWERCASE:-0}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME-}"   # empty = stage predictions only (judging is a separate phase)
JUDGE_RUN_MODE="${JUDGE_RUN_MODE:-first}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-32}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-384}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-1.0}"
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda}"
JUDGE_DTYPE="${JUDGE_DTYPE:-bfloat16}"
STEP_FILTER="${STEP_FILTER:-}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/$USER/hf-datasets}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-/tmp/$USER/hf-evaluate}"
export HF_METRICS_CACHE="${HF_METRICS_CACHE:-/tmp/$USER/hf-metrics}"
export NLTK_DATA="${NLTK_DATA:-/tmp/$USER/nltk_data}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_EVALUATE_CACHE" "$HF_METRICS_CACHE" "$NLTK_DATA"

mapfile -t CKPT_DIRS < <(
  find "$RUN_DIR" -maxdepth 1 -type d -name '_actor_global_step*' \
    | sed 's|.*/||' \
    | sort -t'p' -k2 -n \
    | while read -r name; do
        step="${name#_actor_global_step}"
        if [[ -z "$STEP_FILTER" ]] || echo "$step" | grep -qE "$STEP_FILTER"; then
          echo "$RUN_DIR/$name"
        fi
      done
)

TOTAL="${#CKPT_DIRS[@]}"
[[ "$TOTAL" -eq 0 ]] && { echo "ERROR: No checkpoints found in $RUN_DIR"; exit 1; }

declare -A DONE_STEPS
if [[ -f "$RESULTS_JSONL" ]]; then
  while IFS= read -r line; do
    s=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('global_step',''))" 2>/dev/null || true)
    [[ -n "$s" ]] && DONE_STEPS["$s"]=1
  done < "$RESULTS_JSONL"
fi

echo "Evaluating $TOTAL checkpoints from $RUN_DIR"
echo "Results -> $RESULTS_JSONL"

TMP_JSON=$(mktemp /tmp/eval_librisqa_judge_XXXXXX.json)
trap 'rm -f "$TMP_JSON"' EXIT

IDX=0
for CKPT_PATH in "${CKPT_DIRS[@]}"; do
  IDX=$((IDX + 1))
  STEP="$(basename "$CKPT_PATH")"
  STEP="${STEP#_actor_global_step}"
  if [[ -n "${DONE_STEPS[$STEP]+_}" ]]; then
    echo "[$IDX/$TOTAL] step=$STEP already evaluated, skipping"
    continue
  fi

  echo "[$IDX/$TOTAL] step=$STEP  $CKPT_PATH"
  rm -f "$TMP_JSON"
  MASTER_PORT=$((29500 + RANDOM % 10000))
  if [[ "$NUM_GPUS" -gt 1 ]]; then
    EVAL_CMD=(torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT")
  else
    EVAL_CMD=(python)
  fi

  extra_args=()
  [[ -n "$JUDGE_MODEL_NAME" ]] && extra_args+=(--judge_model_name "$JUDGE_MODEL_NAME")
  if [[ "$SAVE_PREDICTIONS" -eq 1 ]]; then
    pred_dir="$RUN_DIR/librisqa_judge_predictions"
    judge_dir="$RUN_DIR/librisqa_judge_details"
    mkdir -p "$pred_dir" "$judge_dir"
    extra_args+=(--predictions_jsonl "$pred_dir/step_${STEP}.jsonl")
    if [[ -n "$JUDGE_MODEL_NAME" ]]; then
      extra_args+=(--judge_details_jsonl "$judge_dir/step_${STEP}.jsonl")
    fi
  fi

  if "${EVAL_CMD[@]}" scripts/judge/eval_librisqa_llm_judge.py \
      --model_name "$CKPT_PATH" \
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
      --output_json "$TMP_JSON" \
      "${extra_args[@]}"; then
    python3 -c "
import json
with open('$TMP_JSON') as f:
    metrics = json.load(f)
metrics['global_step'] = $STEP
metrics['checkpoint'] = '$CKPT_PATH'
print(json.dumps(metrics))
" >> "$RESULTS_JSONL"
  else
    echo "  ERROR: eval failed for step=$STEP"
  fi
done

echo "Done. Results in $RESULTS_JSONL"
