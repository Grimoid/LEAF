"""Pairwise A/B judging with JudgeLM (BAAI/JudgeLM-13B-v1.0) — the paper's primary
pairwise judge for the spoken-QA tasks.

JudgeLM was trained specifically as a judge with swap-augmentation; on top of that,
every pair is judged in BOTH candidate orders (AB and BA) and a verdict counts only
when the two orders agree. The prompt template comes from
eval_speech_judge_utils_judgelm (Vicuna format, "Assistant 1/2" markers,
with-reference variant); the model emits a pair of scores (s1, s2) on the first
line, parsed via parse_judgelm, and the winner is sign(s1 - s2).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_speech_judge_utils import save_jsonl
from eval_speech_judge_utils_judgelm import build_judgelm_prompt, parse_judgelm
from eval_speech_judge_utils import (
    _lookup_passage,
    build_passage_lookup,
    load_jsonl,
)

DEFAULT_LIBRISQA_JSONS = [
    "hf://datasets/ZihanZhao/LibriSQA/LibriSQA-PartI/LibriSQA-PartI-train.json",
    "hf://datasets/ZihanZhao/LibriSQA/LibriSQA-PartI/LibriSQA-PartI-test.json",
]
DIMS = ("correctness_winner", "grounding_winner", "completeness_winner", "hallucination_winner")


def _align(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    key = lambda r: (str(r.get("question") or "").strip(), str(r.get("reference") or "").strip())
    idx_b = {key(r): r for r in rows_b}
    return [(ra, idx_b[key(ra)]) for ra in rows_a if key(ra) in idx_b]


def _build_order_records(pairs, passage_lookup, order: str):
    recs = []
    for ra, rb in pairs:
        passage = _lookup_passage(passage_lookup, ra.get("question"), ra.get("reference"))
        if order == "ab":
            cand_a, cand_b = ra.get("prediction", ""), rb.get("prediction", "")
        else:
            cand_a, cand_b = rb.get("prediction", ""), ra.get("prediction", "")
        recs.append({
            "question": ra.get("question"),
            "reference": ra.get("reference"),
            "passage": passage,  # JudgeLM ignores this — we just preserve it for downstream
            "candidate_a": cand_a,
            "candidate_b": cand_b,
            "order": order,
        })
    return recs


def _generate_vllm(llm, records, max_new_tokens, temperature, top_p):
    from vllm import SamplingParams
    prompts = [build_judgelm_prompt(r) for r in records]
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sp, use_tqdm=True)
    results = []
    for rec, o in zip(records, outputs):
        text = o.outputs[0].text if o.outputs else ""
        parsed = parse_judgelm(text)
        row = dict(rec)
        row["judge_raw"] = text
        row.update(parsed)
        results.append(row)
    return results


def _resolve_to_model(verdict: str, order: str, label_a: str, label_b: str) -> str:
    """Map 'a'/'b'/'tie' (relative to prompt order) back to model labels."""
    if verdict == "tie":
        return "tie"
    if order == "ab":
        return label_a if verdict == "a" else label_b
    else:
        return label_b if verdict == "a" else label_a


def aggregate(rows_ab, rows_ba, label_a, label_b):
    out_examples = []
    counts = {k: {label_a: 0, label_b: 0, "tie": 0, "inconsistent": 0} for k in ("winner",) + DIMS}
    n = min(len(rows_ab), len(rows_ba))
    for r_ab, r_ba in zip(rows_ab, rows_ba):
        ex = {
            "question": r_ab.get("question"),
            "reference": r_ab.get("reference"),
            "pred_a": r_ab.get("candidate_a"),
            "pred_b": r_ab.get("candidate_b"),
            "ab_raw": r_ab.get("judge_raw"),
            "ba_raw": r_ba.get("judge_raw"),
            "ab_scores": [r_ab.get("score_1"), r_ab.get("score_2")],
            "ba_scores": [r_ba.get("score_1"), r_ba.get("score_2")],
        }
        for k in ("winner",) + DIMS:
            v_ab = _resolve_to_model(r_ab.get(k, "tie"), "ab", label_a, label_b)
            v_ba = _resolve_to_model(r_ba.get(k, "tie"), "ba", label_a, label_b)
            final = v_ab if v_ab == v_ba else "inconsistent"
            ex[f"{k}_ab"] = v_ab
            ex[f"{k}_ba"] = v_ba
            ex[f"{k}_final"] = final
            counts[k][final] += 1
        out_examples.append(ex)

    summary = {
        "judge_num_pairs": n,
        "label_a": label_a,
        "label_b": label_b,
        "judge_family": "judgelm",
        "parse_fail_rate_ab": sum(1 for r in rows_ab if r.get("parse_failed")) / max(n, 1),
        "parse_fail_rate_ba": sum(1 for r in rows_ba if r.get("parse_failed")) / max(n, 1),
    }
    for k, c in counts.items():
        tot = max(n, 1)
        wins_a, wins_b = c[label_a], c[label_b]
        ties, inc = c["tie"], c["inconsistent"]
        decided = wins_a + wins_b
        summary[f"{k}_win_rate_{label_a}"] = round(wins_a / tot, 6)
        summary[f"{k}_win_rate_{label_b}"] = round(wins_b / tot, 6)
        summary[f"{k}_tie_rate"] = round(ties / tot, 6)
        summary[f"{k}_inconsistent_rate"] = round(inc / tot, 6)
        summary[f"{k}_consistency"] = round((decided + ties) / tot, 6)
        summary[f"{k}_net_adv_{label_a}"] = round((wins_a - wins_b) / tot, 6)
        if decided:
            summary[f"{k}_win_rate_decided_{label_a}"] = round(wins_a / decided, 6)
    return summary, out_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_a", required=True, help="step_*.jsonl for model A")
    ap.add_argument("--pred_b", required=True, help="step_*.jsonl for model B")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--details_jsonl", default="")
    ap.add_argument("--librisqa_source_jsons", nargs="*", default=DEFAULT_LIBRISQA_JSONS,
                    help="Used for record bookkeeping only; JudgeLM does not see the passage.")
    ap.add_argument("--judge_model_name", default="BAAI/JudgeLM-13B-v1.0")
    ap.add_argument("--judge_max_new_tokens", type=int, default=512)
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_top_p", type=float, default=1.0)
    ap.add_argument("--judge_dtype", default="bfloat16")
    ap.add_argument("--use_vllm", type=int, default=1)
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--vllm_max_model_len", type=int, default=2048,
                    help="JudgeLM-13B-v1.0 is LLaMA-1 based; max_position_embeddings=2048.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows_a = load_jsonl(Path(args.pred_a))
    rows_b = load_jsonl(Path(args.pred_b))
    pairs = _align(rows_a, rows_b)
    if args.limit > 0:
        random.Random(args.seed).shuffle(pairs)
        pairs = pairs[:args.limit]
    print(f"Aligned pairs: {len(pairs)} (A={len(rows_a)} B={len(rows_b)})")

    print("Building passage lookup ...")
    passage_lookup = build_passage_lookup(args.librisqa_source_jsons)

    recs_ab = _build_order_records(pairs, passage_lookup, "ab")
    recs_ba = _build_order_records(pairs, passage_lookup, "ba")

    if not args.use_vllm:
        print("ERROR: JudgeLM pairwise judging requires --use_vllm 1.", file=sys.stderr)
        sys.exit(2)

    from vllm import LLM
    print(f"Loading judge {args.judge_model_name} via vLLM (TP={args.vllm_tensor_parallel_size}) ...")
    llm = LLM(
        model=args.judge_model_name,
        dtype=args.judge_dtype,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        trust_remote_code=False,
    )

    print(f"Judging order A-B ({len(recs_ab)}) ...")
    rows_ab = _generate_vllm(llm, recs_ab, args.judge_max_new_tokens, args.judge_temperature, args.judge_top_p)
    print(f"Judging order B-A ({len(recs_ba)}) ...")
    rows_ba = _generate_vllm(llm, recs_ba, args.judge_max_new_tokens, args.judge_temperature, args.judge_top_p)

    summary, examples = aggregate(rows_ab, rows_ba, args.label_a, args.label_b)
    summary["pred_a"] = args.pred_a
    summary["pred_b"] = args.pred_b
    summary["judge_model_name"] = args.judge_model_name

    Path(args.results_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    if args.details_jsonl:
        save_jsonl(args.details_jsonl, examples)
    print(json.dumps({k: v for k, v in summary.items() if "rate" in k or "net_adv" in k or "consistency" in k}, indent=2))
    print(f"Done. Aggregate -> {args.results_jsonl}")


if __name__ == "__main__":
    main()
