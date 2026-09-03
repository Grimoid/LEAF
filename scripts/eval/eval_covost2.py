"""Evaluation script for CoVoST2 en->de (AST) experiments.

This is a thin wrapper around eval_librisqa.py with German-appropriate defaults
(bertscore_lang=de). The core evaluation logic is identical since both tasks
use the same dataset format (prompt, reference, audio).
"""
from __future__ import annotations

import sys

# Reuse the generic eval logic from eval_librisqa — it already handles
# any dataset in {prompt, reference, audio} format.
from eval_librisqa import main as _eval_main, _dist_info, _run_one, _aggregate, _cleanup_dist

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Audio, load_from_disk

from openrlhf.speech_leaf import (
    load_model_maybe_peft,
    load_processor,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on CoVoST2 en->de (AST)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--do_sample", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--random_subset_seed", type=int, default=-1)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--with_bertscore", type=int, default=0)
    parser.add_argument("--bertscore_lang", type=str, default="de")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty (Granite Speech recommends 3.0)")
    parser.add_argument("--dump_samples", type=int, default=0,
                        help="Print this many pred/ref pairs for debugging")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--lowercase_bleu", type=int, default=1,
                        help="Use lowercase BLEU (1=yes, 0=no). Standard sacrebleu is case-sensitive.")
    parser.add_argument("--eval_lowercase", type=int, default=0,
                        help="Lowercase all predictions and references before computing ALL metrics "
                             "(exact match, BLEU, ROUGE, METEOR, BERTScore). 1=yes, 0=no.")
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
            f"[CoVoST2 en->de] Running {args.num_runs} eval passes over {len(split_ds)} samples "
            f"(do_sample={args.do_sample}, temp={args.temperature}, top_p={args.top_p}, "
            f"batch={args.eval_batch_size}, world_size={world_size}, "
            f"bertscore_lang={args.bertscore_lang})",
            flush=True,
        )

    all_runs = []
    for run_idx in range(args.num_runs):
        seed = args.base_seed + run_idx
        metrics = _run_one(model, processor, tokenizer, split_ds, device, args, seed, rank, world_size)
        if rank == 0:
            all_runs.append(metrics)
            summary = " | ".join(f"{k}={v:.2f}" for k, v in metrics.items())
            print(f"  Run {run_idx + 1}/{args.num_runs} (seed={seed}): {summary}", flush=True)

    if rank == 0:
        aggregate = _aggregate(all_runs)
        print(json.dumps(aggregate, indent=2))

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(aggregate, f, indent=2)
            print(f"Saved metrics: {output_path}", flush=True)

    _cleanup_dist(world_size)


if __name__ == "__main__":
    main()
