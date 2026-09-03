#!/usr/bin/env bash
# ==========================================================================
#  Paper-protocol LLM-as-judge for ONE checkpoint (spoken QA), three phases:
#    [1] predictions + text metrics  (training env): stochastic decoding T=0.9 / top-p 0.9 /
#        max 200 tokens, one pass, predictions staged as step_<N>.jsonl
#    [2] GEMBA-style DA-100 (0-100 direct assessment, with the source passage):
#        Qwen2.5-14B-Instruct, N_SAMPLES=5 samples at T=0.7, median per item    (judge env, vLLM)
#    [3] Likert-5 absolute grading: Unbabel/M-Prometheus-14B, greedy,
#        SCORE_MODE=expected (logprob-weighted continuous score)               (judge env, vLLM)
#
#    RUN_DIR=<run> STEP=<N> DATA_DIR=$SPEECH_DATA_ROOT/librisqa_part1 \
#    TRAIN_PYTHON=<python of training env> JUDGE_PYTHON=<python with vllm> \
#    bash scripts/judge/judge_checkpoint.sh
#  Optional: SPLIT=test|validation  LIMIT=0  PHASES="pred da100 likert"  VLLM_GPU_UTIL=0.85
#            LOWERCASE_BLEU=1 EVAL_LOWERCASE=1 (text-metric casing for phase 1)
#            LIBRISQA_TRAIN_JSON / LIBRISQA_TEST_JSON (LibriSQA source JSONs with the passages,
#            default: the HF hub files; needed by DA-100 for the passage lookup)
#  Outputs (under RUN_DIR, suffixed _val for SPLIT=validation):
#    librisqa_judge_predictions/step_<N>.jsonl          staged predictions (question/reference/prediction)
#    test_metrics.jsonl                                 BLEU / ROUGE / METEOR / (BERTScore) for the step
#    librisqa_judge_da100_details/step_<N>.jsonl        per-item DA-100 + eval_all_ckpts_librisqa_judge_da100_only.jsonl
#    librisqa_judge_mprometheus_details/step_<N>.jsonl  per-item Likert + mprometheus_results/step_<N>.jsonl
# ==========================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

RUN_DIR="${RUN_DIR:?set RUN_DIR}"; STEP="${STEP:?set STEP}"
DATA_DIR="${DATA_DIR:-${SPEECH_DATA_ROOT:?set SPEECH_DATA_ROOT or DATA_DIR}/librisqa_part1}"
SPLIT="${SPLIT:-test}"; LIMIT="${LIMIT:-0}"; GU="${VLLM_GPU_UTIL:-0.85}"
PHASES="${PHASES:-pred da100 likert}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${PYTHON_BIN:-python}}"
JUDGE_PYTHON="${JUDGE_PYTHON:-$TRAIN_PYTHON}"
DA_JUDGE="${DA_JUDGE:-Qwen/Qwen2.5-14B-Instruct}"
LK_JUDGE="${LK_JUDGE:-Unbabel/M-Prometheus-14B}"
SUF=""; [ "$SPLIT" = "validation" ] && SUF="_val"
[ -d "$RUN_DIR/_actor_global_step${STEP}" ] || { echo "ERROR: $RUN_DIR/_actor_global_step${STEP} missing"; exit 1; }

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_DIR/librisqa_judge_predictions${SUF}"
echo "=== JUDGE $(basename "$RUN_DIR") @ step $STEP  split=$SPLIT limit=$LIMIT  $(date) ==="

RAW_PF="$RUN_DIR/librisqa_judge_predictions/step_${STEP}.jsonl"
PF="$RUN_DIR/librisqa_judge_predictions${SUF}/step_${STEP}.jsonl"
if [[ " $PHASES " == *" pred "* ]]; then
  if [ -s "$PF" ]; then echo "[1/3] predictions exist ($(wc -l <"$PF") lines) -> skip"; else
    echo "[1/3] predictions + text metrics…"
    PATH="$(dirname "$TRAIN_PYTHON"):$PATH" \
    RUN_DIR="$RUN_DIR" DATA_DIR="$DATA_DIR" SPLIT="$SPLIT" LIMIT="$LIMIT" \
    RESULTS_JSONL="$RUN_DIR/test_metrics${SUF}.jsonl" \
    JUDGE_MODEL_NAME="" SAVE_PREDICTIONS=1 STEP_FILTER="^${STEP}$" \
    DO_SAMPLE=1 TEMPERATURE=0.9 TOP_P=0.9 MAX_NEW_TOKENS=200 NUM_RUNS="${NUM_RUNS:-1}" BASE_SEED=42 \
    NUM_GPUS="${NUM_GPUS:-1}" EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}" \
    WITH_BERTSCORE="${WITH_BERTSCORE:-0}" LOWERCASE_BLEU="${LOWERCASE_BLEU:-1}" EVAL_LOWERCASE="${EVAL_LOWERCASE:-1}" \
    bash scripts/judge/eval_all_checkpoints_librisqa_llm_judge.sh
    [ -n "$SUF" ] && [ -s "$RAW_PF" ] && mv "$RAW_PF" "$PF"
  fi
  [ -s "$PF" ] || { echo "ERROR: predictions not produced ($PF)"; exit 2; }
fi

if [[ " $PHASES " == *" da100 "* ]]; then
  echo "[2/3] DA-100 ($DA_JUDGE, N=${N_SAMPLES:-5} @ T=${DA_TEMPERATURE:-0.7})…"
  PATH="$(dirname "$JUDGE_PYTHON"):$PATH" \
  RUN_DIR="$RUN_DIR" PRED_DIR="$RUN_DIR/librisqa_judge_predictions${SUF}" \
  DETAILS_DIR="$RUN_DIR/librisqa_judge_da100_details${SUF}" \
  RESULTS_JSONL="$RUN_DIR/eval_all_ckpts_librisqa_judge_da100_only${SUF}.jsonl" \
  JUDGE_MODEL_NAME="$DA_JUDGE" \
  N_SAMPLES="${N_SAMPLES:-5}" JUDGE_TEMPERATURE="${DA_TEMPERATURE:-0.7}" STEP_FILTER="^${STEP}$" OVERWRITE=1 \
  VLLM_TP="${VLLM_TP:-1}" VLLM_GPU_UTIL="$GU" VLLM_MAX_MODEL_LEN=6144 \
  bash scripts/judge/run_judge_predictions_da100.sh
fi

if [[ " $PHASES " == *" likert "* ]]; then
  echo "[3/3] Likert-5 ($LK_JUDGE, greedy, score_mode=${SCORE_MODE:-expected})…"
  PATH="$(dirname "$JUDGE_PYTHON"):$PATH" \
  PREDICTIONS_JSONL="$PF" \
  DETAILS_JSONL="$RUN_DIR/librisqa_judge_mprometheus_details${SUF}/step_${STEP}.jsonl" \
  RESULTS_JSONL="$RUN_DIR/mprometheus_results${SUF}/step_${STEP}.jsonl" \
  RUBRIC_KEY="${RUBRIC_KEY:-qa}" JUDGE_MODEL_NAME="$LK_JUDGE" SCORE_MODE="${SCORE_MODE:-expected}" \
  VLLM_TP="${VLLM_TP:-1}" VLLM_GPU_UTIL="$GU" VLLM_MAX_MODEL_LEN=8192 \
  bash scripts/judge/run_judge_predictions_grounded_mprometheus.sh
fi
echo "=== DONE $(basename "$RUN_DIR")@$STEP split=$SPLIT ==="
