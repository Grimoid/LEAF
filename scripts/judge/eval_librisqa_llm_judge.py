from __future__ import annotations

import argparse
import json
import math
import os
import sys
import sysconfig
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def _prefer_site_packages_for_hf_datasets() -> None:
    candidates = []
    for key in ("platlib", "purelib"):
        path = sysconfig.get_paths().get(key)
        if path and path not in candidates:
            candidates.append(path)
    for path in reversed(candidates):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


_prefer_site_packages_for_hf_datasets()

from datasets import Audio, load_from_disk


def _prefer_repo_root_for_local_openrlhf() -> None:
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
    # sibling script directories (scripts/eval, scripts/judge) for cross-imports
    for sub in ("eval", "judge"):
        sibling = str(Path(__file__).resolve().parents[1] / sub)
        if sibling not in sys.path:
            sys.path.insert(0, sibling)


_prefer_repo_root_for_local_openrlhf()

from eval_librisqa import _batched_examples, _cleanup_dist, _dist_info, _format_prompt, _shard_dataset
from eval_speech_judge_utils import JudgeConfig, run_local_llm_judge, save_jsonl
from openrlhf.speech_leaf import (
    compute_corpus_metrics,
    load_audio_array,
    load_model_maybe_peft,
    load_processor,
    sync_audio_features,
)
from openrlhf.speech_leaf.text_utils import normalize_text

import torch.distributed as dist
from tqdm import tqdm


@torch.no_grad()
def _run_one_collect(
    model,
    processor,
    tokenizer,
    split_ds,
    device,
    args,
    seed: int,
    rank: int,
    world_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    shard_ds = _shard_dataset(split_ds, rank, world_size)
    rows: List[Dict[str, Any]] = []
    audio_token = getattr(processor, "audio_token", "<|audio|>")
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    iterator = _batched_examples(shard_ds, args.eval_batch_size)
    total_batches = math.ceil(len(shard_ds) / args.eval_batch_size) if len(shard_ds) else 0
    progress = tqdm(
        iterator,
        total=total_batches,
        desc=f"librisqa eval seed={seed}",
        leave=False,
        disable=rank != 0,
    )

    for batch in progress:
        prompts_raw = [example["prompt"] for example in batch]
        prompts = [_format_prompt(tokenizer, p) for p in prompts_raw]
        refs = [normalize_text(example["reference"]) for example in batch]
        audios = [load_audio_array(example["audio"], target_sampling_rate=16000)[0] for example in batch]
        # For datasets without a separate 'question' column (e.g. dailytalk_longaudio,
        # speech_longaudio), fall back to the prompt with the audio token stripped.
        questions = [
            str(example.get("question") or example.get("prompt", "")).replace("<|audio|>", "").strip()
            for example in batch
        ]

        inputs = processor(text=prompts, audio=audios, return_tensors="pt", padding=True)
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        inputs = sync_audio_features(inputs, audio_token_id)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            return_dict_in_generate=True,
        )
        if args.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty

        outputs = model.generate(**gen_kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs.sequences[:, prompt_len:]
        preds = [normalize_text(tokenizer.decode(row, skip_special_tokens=True)) for row in generated]

        for prompt, question, ref, pred in zip(prompts_raw, questions, refs, preds):
            rows.append(
                {
                    "prompt": prompt,
                    "question": question,
                    "reference": ref,
                    "prediction": pred,
                }
            )

    if world_size > 1:
        gathered_rows = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_rows, rows)
        if rank != 0:
            return {}, []
        rows = [item for sublist in gathered_rows for item in sublist]

    predictions = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    metrics = compute_corpus_metrics(
        predictions=predictions,
        references=references,
        with_text_metrics=True,
        with_bertscore=bool(args.with_bertscore),
        bertscore_lang=args.bertscore_lang,
        lowercase_bleu=bool(args.lowercase_bleu),
        lowercase=bool(args.eval_lowercase),
    )

    return metrics, rows


def _aggregate(all_runs: List[Dict[str, float]]) -> Dict[str, object]:
    if not all_runs:
        return {}
    metrics: Dict[str, object] = {}
    for key in all_runs[0].keys():
        values = [run[key] for run in all_runs if key in run]
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        metrics[key] = round(mean, 6)
        metrics[f"{key}_std"] = round(std, 6)
        metrics[f"{key}_runs"] = [round(value, 6) for value in values]
    metrics["num_runs"] = len(all_runs)
    return metrics


def _summarize_metrics(metrics: Dict[str, Any]) -> str:
    keys = [k for k, v in metrics.items() if isinstance(v, (int, float)) and not k.endswith("_std")]
    priority = [
        "bleu",
        "rougeL",
        "meteor",
        "bertscore_f1",
        "judge_overall_score_mean",
        "judge_overall_pct",
    ]
    ordered = [k for k in priority if k in keys] + [k for k in keys if k not in priority]
    return " | ".join(f"{k}={metrics[k]:.2f}" for k in ordered[:8])


def main():
    parser = argparse.ArgumentParser(description="Evaluate LibriSQA and optionally score with a local LLM judge")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--do_sample", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--dump_samples", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--random_subset_seed", type=int, default=-1)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--with_bertscore", type=int, default=0)
    parser.add_argument("--bertscore_lang", type=str, default="en")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--predictions_jsonl", type=str, default="")
    parser.add_argument("--judge_details_jsonl", type=str, default="")
    parser.add_argument("--lowercase_bleu", type=int, default=1)
    parser.add_argument("--eval_lowercase", type=int, default=0)
    parser.add_argument("--judge_model_name", type=str, default="")
    parser.add_argument("--judge_run_mode", type=str, default="first", choices=["none", "first", "all"])
    parser.add_argument("--judge_batch_size", type=int, default=4)
    parser.add_argument("--judge_max_new_tokens", type=int, default=256)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_top_p", type=float, default=1.0)
    parser.add_argument("--judge_device", type=str, default="cuda")
    parser.add_argument("--judge_dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    rank, world_size, _, device = _dist_info()

    dataset = load_from_disk(str(Path(args.data_dir) / "dataset"))
    split_ds = dataset[args.split]
    if "audio" in split_ds.column_names:
        split_ds = split_ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    if args.limit > 0:
        if args.random_subset_seed >= 0:
            split_ds = split_ds.shuffle(seed=args.random_subset_seed)
        split_ds = split_ds.select(range(min(args.limit, len(split_ds))))

    processor = load_processor(args.model_name)
    tokenizer = processor.tokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_model_maybe_peft(args.model_name, dtype=dtype)
    model.to(device)
    model.eval()

    if rank == 0:
        print(
            f"Running {args.num_runs} LibriSQA eval pass(es) over {len(split_ds)} samples "
            f"(judge_run_mode={args.judge_run_mode}, judge_model={'set' if args.judge_model_name else 'off'})",
            flush=True,
        )

    all_runs: List[Dict[str, float]] = []
    last_rows: List[Dict[str, Any]] = []
    judge_rows_to_save: List[Dict[str, Any]] = []
    judge_summary_final: Dict[str, Any] = {}

    for run_idx in range(args.num_runs):
        seed = args.base_seed + run_idx
        metrics, rows = _run_one_collect(model, processor, tokenizer, split_ds, device, args, seed, rank, world_size)
        if rank == 0:
            last_rows = rows

            should_judge = (
                bool(args.judge_model_name)
                and args.judge_run_mode != "none"
                and (args.judge_run_mode == "all" or run_idx == 0)
            )
            if should_judge:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                judge_cfg = JudgeConfig(
                    model_name=args.judge_model_name,
                    batch_size=args.judge_batch_size,
                    max_new_tokens=args.judge_max_new_tokens,
                    temperature=args.judge_temperature,
                    top_p=args.judge_top_p,
                    device=args.judge_device,
                    dtype=args.judge_dtype,
                )
                judge_summary, judge_rows = run_local_llm_judge(task="librisqa", records=rows, config=judge_cfg)
                metrics.update(judge_summary)
                judge_summary_final = judge_summary
                if args.judge_run_mode == "first":
                    judge_rows_to_save = judge_rows
                elif args.judge_run_mode == "all":
                    for row in judge_rows:
                        row["judge_run_index"] = run_idx
                    judge_rows_to_save.extend(judge_rows)
                print(f"    Judge: {_summarize_metrics(judge_summary)}", flush=True)
                if run_idx + 1 < args.num_runs:
                    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    model = load_model_maybe_peft(args.model_name, dtype=dtype)
                    model.to(device)
                    model.eval()

            all_runs.append(dict(metrics))
            print(f"  Run {run_idx + 1}/{args.num_runs} (seed={seed}): {_summarize_metrics(metrics)}", flush=True)

    if rank == 0:
        aggregate = _aggregate(all_runs)
        if judge_summary_final and args.judge_run_mode == "first":
            aggregate.update(judge_summary_final)
        print(json.dumps(aggregate, indent=2))

        if args.predictions_jsonl and last_rows:
            save_jsonl(args.predictions_jsonl, last_rows)
            print(f"Saved predictions: {args.predictions_jsonl}", flush=True)
        if args.judge_details_jsonl and judge_rows_to_save:
            save_jsonl(args.judge_details_jsonl, judge_rows_to_save)
            print(f"Saved judge details: {args.judge_details_jsonl}", flush=True)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(aggregate, f, indent=2)
            print(f"Saved metrics: {output_path}", flush=True)

    _cleanup_dist(world_size)


if __name__ == "__main__":
    main()
