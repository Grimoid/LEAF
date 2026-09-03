# LLM-as-judge (`scripts/judge/`)

Reference-based judging of spoken-QA / translation outputs with open judge models served by
vLLM (run these in the judge environment, `requirements-judge.txt`).

## Two phases

```
  [1] stage predictions            eval_librisqa_llm_judge.py / eval_all_checkpoints_librisqa_llm_judge.sh
      (training env, GPU)          -> <run>/librisqa_judge_predictions/step_<N>.jsonl
                                      rows: {prompt, question, reference, prediction}
                    |
  [2] judge the staged rows        judge_predictions_*.py  (+ eval_speech_judge_utils*.py)
      (judge env, vLLM)            -> per-item details jsonl + per-file aggregate jsonl
```

## Judges

| Producer | Judge | Protocol |
|---|---|---|
| `judge_predictions_da100.py` | GEMBA-style DA-100 (0–100 direct assessment, with the source passage; Qwen2.5-14B-Instruct) | `--n_samples 5 --judge_temperature 0.7`, median per item (self-consistency) |
| `judge_predictions_grounded_mprometheus.py` | Rubric-grounded Likert-5 (Unbabel/M-Prometheus-14B, Prometheus-2 prompt, `qa`/`mt` rubric) | greedy; `--score_mode expected` = logprob-weighted continuous score |
| `judge_predictions_pairwise_judgelm.py` | Pairwise A vs B for spoken QA (BAAI/JudgeLM-13B-v1.0) | both candidate orders judged; a verdict counts only when they agree |
| `judge_predictions_pairwise_mprometheus.py` | Pairwise A vs B for CoVoST2 translation (M-Prometheus-14B, relative-grading prompt, `mt` rubric) | same swap protocol |

## Drivers

| Script | Runs |
|---|---|
| `judge_checkpoint.sh` | **paper protocol for one checkpoint**: predictions → DA-100 → Likert-5 (`RUN_DIR`, `STEP`, `DATA_DIR`, `TRAIN_PYTHON`, `JUDGE_PYTHON`) |
| `run_judge_predictions_da100.sh` | DA-100 over every `step_*.jsonl` of a predictions dir (`RUN_DIR`, `PRED_DIR`, `STEP_FILTER`) |
| `run_judge_predictions_grounded_mprometheus.sh` | Likert-5 for one predictions file (`PREDICTIONS_JSONL`, `DETAILS_JSONL`, `RESULTS_JSONL`, `RUBRIC_KEY`) |
| `run_judge_predictions_pairwise_judgelm.sh` / `..._mprometheus.sh` | one A/B pair (`PRED_A`, `PRED_B`, `LABEL_A`, `LABEL_B`, `RESULTS_JSONL`) |
| `eval_all_checkpoints_librisqa_llm_judge.sh` | prediction staging over a run's checkpoints (`SAVE_PREDICTIONS=1`, `JUDGE_MODEL_NAME` empty) |

DA-100 needs the LibriSQA source JSONs (with the passages) for the grounded prompt:
`LIBRISQA_TRAIN_JSON` / `LIBRISQA_TEST_JSON` (defaults: the `ZihanZhao/LibriSQA` hub files).

