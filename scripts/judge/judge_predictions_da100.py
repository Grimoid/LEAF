"""LibriSQA judge on continuous 0-100 Direct Assessment scale, with passage.

Optional --n_samples > 1 + --judge_temperature > 0 enables self-consistency:
each example is scored N times and the MEDIAN is kept (robust to Likert-style
noise; reports std across samples too).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_speech_judge_utils import parse_judge_json, safe_mean, safe_std, save_jsonl
from eval_speech_judge_utils_da100 import build_judge_messages_da100
from eval_speech_judge_utils import _lookup_passage, build_passage_lookup, load_jsonl

STEP_RE = re.compile(r"step_(\d+)\.jsonl$")
DEFAULT_LIBRISQA_JSONS = [
    "hf://datasets/ZihanZhao/LibriSQA/LibriSQA-PartI/LibriSQA-PartI-train.json",
    "hf://datasets/ZihanZhao/LibriSQA/LibriSQA-PartI/LibriSQA-PartI-test.json",
]


def _apply(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"


def _gen_vllm(llm, tokenizer, records, max_new_tokens, temperature, top_p, n_samples, seed, rubric_kind="qa"):
    from vllm import SamplingParams
    prompts = [_apply(tokenizer, build_judge_messages_da100(r, rubric_kind=rubric_kind)) for r in records]
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, n=n_samples, seed=seed)
    outputs = llm.generate(prompts, sp, use_tqdm=True)
    rows = []
    for rec, o in zip(records, outputs):
        scores, texts = [], []
        for cand in (o.outputs or []):
            texts.append(cand.text)
            parsed = parse_judge_json(cand.text, scale="da100")
            if not parsed.get("parse_failed"):
                scores.append(float(parsed["overall_score"]))
        row = dict(rec)
        if scores:
            row["overall_score"] = float(statistics.median(scores))
            row["overall_score_mean"] = safe_mean(scores)
            row["overall_score_std"] = safe_std(scores)
            row["n_samples_parsed"] = len(scores)
            row["parse_failed"] = False
        else:
            row["overall_score"] = 0.0
            row["n_samples_parsed"] = 0
            row["parse_failed"] = True
        row["judge_raw"] = texts[0] if texts else ""
        rows.append(row)
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [float(r["overall_score"]) for r in rows if not r.get("parse_failed")]
    summary: Dict[str, Any] = {
        "judge_num_examples": len(rows),
        "judge_parse_fail_rate": sum(1 for r in rows if r.get("parse_failed")) / max(len(rows), 1),
        "judge_passage_missing_rate": sum(1 for r in rows if not str(r.get("passage") or "").strip()) / max(len(rows), 1),
        "judge_scale": "da100",
        "judge_protocol": "da100_grounded_v1",
    }
    if vals:
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        def pct(p): return vals_sorted[min(n - 1, int(p * n))]
        summary["judge_overall_score_mean"] = round(safe_mean(vals), 4)
        summary["judge_overall_score_std"] = round(safe_std(vals), 4)
        summary["judge_overall_p25"] = round(pct(0.25), 4)
        summary["judge_overall_p50"] = round(pct(0.50), 4)
        summary["judge_overall_p75"] = round(pct(0.75), 4)
        summary["judge_frac_below_50"] = round(sum(1 for v in vals if v < 50) / len(vals), 4)
        summary["judge_frac_below_70"] = round(sum(1 for v in vals if v < 70) / len(vals), 4)
        summary["judge_frac_90_plus"] = round(sum(1 for v in vals if v >= 90) / len(vals), 4)
        summary["judge_overall_pct"] = summary["judge_overall_score_mean"]
    intra = [float(r.get("overall_score_std", 0.0)) for r in rows if r.get("n_samples_parsed", 0) > 1]
    if intra:
        summary["judge_intra_item_std_mean"] = round(safe_mean(intra), 4)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_dir", required=True)
    ap.add_argument("--judge_details_dir", default="")
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--librisqa_source_jsons", nargs="*", default=DEFAULT_LIBRISQA_JSONS)
    ap.add_argument("--judge_model_name", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--judge_max_new_tokens", type=int, default=512)
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_top_p", type=float, default=1.0)
    ap.add_argument("--judge_dtype", default="bfloat16")
    ap.add_argument("--n_samples", type=int, default=1, help=">1 enables self-consistency; use temperature>0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step_filter", default="")
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--use_vllm", type=int, default=1)
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--vllm_max_model_len", type=int, default=6144)
    ap.add_argument("--rubric_kind", default="qa", choices=["qa", "mt"],
                    help="qa = LibriSQA-style grounded QA rubric (default); "
                         "mt = GEMBA-DA MT rubric for translation tasks (CoVoST2 en→de). "
                         "MT mode skips the passage lookup (uses 'source' field instead).")
    args = ap.parse_args()

    pred_dir = Path(args.predictions_dir)
    entries = []
    for f in sorted(pred_dir.glob("step_*.jsonl")):
        m = STEP_RE.search(f.name)
        if not m:
            continue
        step = int(m.group(1))
        if args.step_filter and not re.search(args.step_filter, str(step)):
            continue
        entries.append((step, f))
    if not entries:
        print(f"ERROR: no prediction files in {pred_dir}", file=sys.stderr); sys.exit(1)

    if args.rubric_kind == "mt":
        # Translation rubric uses source/reference/prediction directly; no QA passage needed.
        print("Rubric kind = mt (GEMBA-DA MT); skipping passage lookup.")
        passage_lookup = None
    else:
        print("Building passage lookup ...")
        passage_lookup = build_passage_lookup(args.librisqa_source_jsons)

    results_path = Path(args.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if results_path.exists() and not args.overwrite:
        for line in open(results_path):
            try: done.add(int(json.loads(line).get("global_step", -1)))
            except Exception: pass

    details_dir = Path(args.judge_details_dir) if args.judge_details_dir else None
    if details_dir:
        details_dir.mkdir(parents=True, exist_ok=True)

    if not args.use_vllm:
        print("ERROR: DA100 judge requires --use_vllm 1", file=sys.stderr); sys.exit(2)

    from vllm import LLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading judge {args.judge_model_name} via vLLM ...")
    llm = LLM(
        model=args.judge_model_name,
        dtype=args.judge_dtype,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        trust_remote_code=True,
        enforce_eager=False,
    )

    for idx, (step, pred_file) in enumerate(entries, 1):
        if step in done:
            print(f"[{idx}/{len(entries)}] step={step} already judged, skipping"); continue
        recs = load_jsonl(pred_file)
        if args.rubric_kind == "mt":
            print(f"[{idx}/{len(entries)}] step={step} judging {len(recs)} (mt rubric; source/reference/prediction) n_samples={args.n_samples} T={args.judge_temperature}")
        else:
            missing = 0
            for r in recs:
                p = _lookup_passage(passage_lookup, r.get("question"), r.get("reference"))
                r["passage"] = p
                if not p: missing += 1
            print(f"[{idx}/{len(entries)}] step={step} judging {len(recs)} (passage missing {missing}) n_samples={args.n_samples} T={args.judge_temperature}")
        rows = _gen_vllm(llm, tokenizer, recs,
                         args.judge_max_new_tokens, args.judge_temperature, args.judge_top_p,
                         args.n_samples, args.seed, rubric_kind=args.rubric_kind)
        summary = summarize(rows)
        summary["global_step"] = step
        summary["predictions_file"] = str(pred_file)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        if details_dir:
            save_jsonl(str(details_dir / f"step_{step}.jsonl"), rows)
        print(f"    mean={summary.get('judge_overall_score_mean')} p25={summary.get('judge_overall_p25')} "
              f"<70={summary.get('judge_frac_below_70')} 90+={summary.get('judge_frac_90_plus')} "
              f"parse_fail={summary['judge_parse_fail_rate']:.3f}")
    print(f"Done. Results in {results_path}")


if __name__ == "__main__":
    main()
