"""Rubric-grounded Likert-5 judging with M-Prometheus (Unbabel/M-Prometheus-14B).

Reads staged prediction JSONLs (one row per item, with at least `question`,
`reference`, `prediction` keys) and writes:
  - Per-item details JSONL: input row + judge_raw + parsed score
  - Aggregate JSONL: one line per input file with mean/std/CVaR-friendly stats

Protocol: the Prometheus-2 absolute-grading prompt with a task rubric (`qa` or `mt`),
a single 1-5 score per item, greedy decoding by default; `--score_mode expected`
returns the logprob-weighted continuous score.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_speech_judge_utils import save_jsonl
from eval_speech_judge_utils_mprometheus import (
    DEFAULT_MT_RUBRIC,
    DEFAULT_QA_RUBRIC,
    build_mprometheus_messages,
    parse_mprometheus,
)
from eval_speech_judge_utils import load_jsonl


RUBRIC_PRESETS = {"qa": DEFAULT_QA_RUBRIC, "mt": DEFAULT_MT_RUBRIC}


def _apply_chat_template(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return messages[0]["content"]


def _resolve_score_token_ids(tokenizer):
    """Map score {1..5} → single token id (preferred: with leading space,
    since that's how the model emits 'Feedback: ... [RESULT] 4')."""
    ids: Dict[int, int] = {}
    for s in range(1, 6):
        chosen = None
        for cand in (f" {s}", str(s)):  # try space-prefixed first
            toks = tokenizer.encode(cand, add_special_tokens=False)
            if len(toks) == 1:
                chosen = toks[0]
                break
        if chosen is None:
            # Fallback: take the first token of the multi-token encoding.
            toks = tokenizer.encode(f" {s}", add_special_tokens=False)
            chosen = toks[-1] if toks else -1
        ids[s] = chosen
    return ids


def _find_score_token_position(text: str, token_ids: list, tokenizer, score_token_set: set):
    """Find the position (in the generated token list) of the integer score
    token that comes after '[RESULT]'. Returns -1 if not found.

    Strategy: incrementally decode tokens and track when '[RESULT]' has appeared.
    The next token whose id is one of the score token ids is our score token.
    """
    decoded = ""
    saw_result = False
    for i, tid in enumerate(token_ids):
        decoded += tokenizer.decode([tid])
        if not saw_result and "[RESULT]" in decoded:
            saw_result = True
            continue
        if saw_result and tid in score_token_set:
            return i
    return -1


def _expected_score_from_logprobs(logprobs_at_pos, score_token_ids: Dict[int, int]):
    """Given vLLM's logprobs dict at the score-token position, compute the
    softmax-weighted expected score over {1..5}. Tokens not present in the
    top-K logprobs are treated as having logprob ≈ -inf (probability ≈ 0).

    Returns (expected_5pt, expected_pct, probs_dict).
    """
    import math
    raw_lps: Dict[int, float] = {}
    for score, tid in score_token_ids.items():
        if tid in logprobs_at_pos:
            obj = logprobs_at_pos[tid]
            # vLLM Logprob object exposes .logprob; raw float also possible.
            lp = obj.logprob if hasattr(obj, "logprob") else float(obj)
            raw_lps[score] = lp
    if not raw_lps:
        return None, None, {}
    # Softmax-normalise across the 5 score logprobs only.
    max_lp = max(raw_lps.values())
    exps = {s: math.exp(lp - max_lp) for s, lp in raw_lps.items()}
    z = sum(exps.values())
    probs = {s: exps[s] / z for s in exps}
    exp5 = sum(s * p for s, p in probs.items())
    exp_pct = (exp5 - 1.0) * 25.0   # linear map [1,5] -> [0,100]
    return exp5, exp_pct, probs


def _generate_vllm(llm, tokenizer, records, rubric, max_new_tokens, temperature, top_p,
                   score_mode="argmax", logprobs_topk=20):
    """score_mode:
       - 'argmax'   : only the integer score is parsed (existing behaviour).
       - 'expected' : also extracts logprobs at the score-token position and
                      computes the softmax-weighted expected score in [1,5]
                      and the percentage in [0,100].
    """
    from vllm import SamplingParams
    prompts = [
        _apply_chat_template(tokenizer, build_mprometheus_messages(r, rubric=rubric))
        for r in records
    ]
    sp_kwargs = dict(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    if score_mode == "expected":
        sp_kwargs["logprobs"] = logprobs_topk
    sp = SamplingParams(**sp_kwargs)
    outputs = llm.generate(prompts, sp, use_tqdm=True)

    if score_mode == "expected":
        score_token_ids = _resolve_score_token_ids(tokenizer)
        score_token_set = set(score_token_ids.values())
    else:
        score_token_ids = score_token_set = None

    results = []
    for rec, o in zip(records, outputs):
        comp = o.outputs[0] if o.outputs else None
        text = comp.text if comp else ""
        parsed = parse_mprometheus(text)
        row = dict(rec)
        row["judge_raw"] = text
        row.update(parsed)

        if score_mode == "expected" and comp is not None and comp.logprobs is not None:
            pos = _find_score_token_position(text, list(comp.token_ids), tokenizer, score_token_set)
            if pos >= 0 and pos < len(comp.logprobs):
                exp5, exp_pct, probs = _expected_score_from_logprobs(
                    comp.logprobs[pos], score_token_ids
                )
                if exp5 is not None:
                    row["score_expected_5pt"] = round(exp5, 6)
                    row["score_expected_pct"] = round(exp_pct, 6)
                    row["score_probs"] = {str(k): round(v, 6) for k, v in probs.items()}
                else:
                    row["score_expected_5pt"] = None
                    row["score_expected_pct"] = None
                    row["score_probs"] = {}
            else:
                row["score_expected_5pt"] = None
                row["score_expected_pct"] = None
                row["score_probs"] = {}

        results.append(row)
    return results


def _aggregate(rows: List[Dict[str, Any]], predictions_file: str, judge_model: str,
               rubric_key: str, score_mode: str):
    scores = [r["score"] for r in rows if r.get("score", -1) >= 1]
    parse_fails = sum(1 for r in rows if r.get("parse_failed"))
    n = len(rows)
    valid = len(scores)
    out: Dict[str, Any] = {
        "judge_protocol": "mprometheus_likert5",
        "judge_scale": "1-5",
        "judge_score_mode": score_mode,
        "judge_model_name": judge_model,
        "judge_rubric_key": rubric_key,
        "predictions_file": predictions_file,
        "judge_num_examples": n,
        "judge_valid_examples": valid,
        "judge_parse_fail_rate": round(parse_fails / max(n, 1), 6),
    }
    if scores:
        mean = statistics.mean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        out["judge_overall_score_mean"] = round(mean, 6)
        out["judge_overall_score_std"] = round(std, 6)
        out["judge_overall_pct"] = round(100.0 * mean / 5.0, 4)
        for k in range(1, 6):
            out[f"judge_score_frac_{k}"] = round(
                sum(1 for s in scores if s == k) / len(scores), 6
            )

    # Continuous (logprob-weighted) score stats, if available.
    cont = [r["score_expected_5pt"] for r in rows
            if r.get("score_expected_5pt") is not None]
    if cont:
        mean_c = statistics.mean(cont)
        std_c = statistics.pstdev(cont) if len(cont) > 1 else 0.0
        out["judge_expected_5pt_mean"] = round(mean_c, 6)
        out["judge_expected_5pt_std"] = round(std_c, 6)
        out["judge_expected_pct_mean"] = round((mean_c - 1.0) * 25.0, 6)
        out["judge_expected_valid_examples"] = len(cont)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_jsonl", required=True,
                    help="Path to one staged predictions JSONL (single step's predictions)")
    ap.add_argument("--judge_details_jsonl", required=True,
                    help="Path where per-item details JSONL will be written")
    ap.add_argument("--results_jsonl", required=True,
                    help="Aggregate JSONL — one line is appended per invocation")
    ap.add_argument("--rubric_key", choices=sorted(RUBRIC_PRESETS.keys()), default="qa",
                    help="Which built-in rubric to use (qa = open-ended QA, mt = translation)")
    ap.add_argument("--rubric_file", default="",
                    help="Optional path to a text file containing a custom rubric. "
                         "Overrides --rubric_key if set.")
    ap.add_argument("--judge_model_name", default="Unbabel/M-Prometheus-14B")
    ap.add_argument("--judge_max_new_tokens", type=int, default=768)
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_top_p", type=float, default=1.0)
    ap.add_argument("--judge_dtype", default="bfloat16")
    ap.add_argument("--use_vllm", type=int, default=1)
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--vllm_max_model_len", type=int, default=8192,
                    help="M-Prometheus is Qwen2.5-14B based (32k native); 8k is plenty for our prompts.")
    ap.add_argument("--score_mode", choices=("argmax", "expected"), default="argmax",
                    help="'argmax': integer score only. "
                         "'expected': also extracts top-K logprobs at the score token "
                         "and computes the softmax-weighted expected score in [1,5] "
                         "(and the equivalent 0-100 percentage).")
    ap.add_argument("--logprobs_topk", type=int, default=5,
                    help="vLLM top-K logprobs to request when score_mode=expected. "
                         "vLLM's default engine cap is 5; if you raise it, also pass "
                         "max_logprobs on LLM() init (we do that automatically).")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rubric = RUBRIC_PRESETS[args.rubric_key]
    if args.rubric_file:
        with open(args.rubric_file, "r", encoding="utf-8") as f:
            rubric = f.read().strip()

    records = load_jsonl(Path(args.predictions_jsonl))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"Loaded {len(records)} prediction rows from {args.predictions_jsonl}")

    if not args.use_vllm:
        print("ERROR: this script requires --use_vllm 1.", file=sys.stderr)
        sys.exit(2)

    from vllm import LLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model_name, trust_remote_code=True)
    print(f"Loading judge {args.judge_model_name} via vLLM (TP={args.vllm_tensor_parallel_size}) ...")
    llm_kwargs = dict(
        model=args.judge_model_name,
        dtype=args.judge_dtype,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        trust_remote_code=True,
    )
    # If we're going to request > 5 logprobs, also raise the engine cap.
    if args.score_mode == "expected" and args.logprobs_topk > 5:
        llm_kwargs["max_logprobs"] = args.logprobs_topk
    llm = LLM(**llm_kwargs)

    rows = _generate_vllm(
        llm, tokenizer, records, rubric,
        args.judge_max_new_tokens, args.judge_temperature, args.judge_top_p,
        score_mode=args.score_mode, logprobs_topk=args.logprobs_topk,
    )

    Path(args.judge_details_jsonl).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.judge_details_jsonl, rows)
    print(f"Wrote details -> {args.judge_details_jsonl}")

    summary = _aggregate(rows, args.predictions_jsonl, args.judge_model_name,
                         args.rubric_key, args.score_mode)
    Path(args.results_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Appended aggregate -> {args.results_jsonl}")


if __name__ == "__main__":
    main()
