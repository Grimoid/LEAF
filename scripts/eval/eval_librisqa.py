from __future__ import annotations

import argparse
import json
import math
import os
import sys
import sysconfig
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm


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


_prefer_repo_root_for_local_openrlhf()

from openrlhf.speech_leaf import (
    compute_corpus_metrics,
    load_audio_array,
    load_model_maybe_peft,
    load_processor,
    sync_audio_features,
)
from openrlhf.speech_leaf.text_utils import normalize_text


def _dist_info() -> Tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def _cleanup_dist(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _format_prompt(tokenizer, raw_prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return raw_prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _shard_dataset(split_ds, rank: int, world_size: int):
    if world_size <= 1:
        return split_ds
    indices = list(range(rank, len(split_ds), world_size))
    return split_ds.select(indices)


def _batched_examples(dataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        stop = min(len(dataset), start + batch_size)
        yield [dataset[idx] for idx in range(start, stop)]


@torch.no_grad()
def _run_one(
    model,
    processor,
    tokenizer,
    split_ds,
    device,
    args,
    seed: int,
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    shard_ds = _shard_dataset(split_ds, rank, world_size)
    predictions: List[str] = []
    references: List[str] = []
    raw_prompts: List[str] = []
    audio_token = getattr(processor, "audio_token", "<|audio|>")
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    iterator = _batched_examples(shard_ds, args.eval_batch_size)
    total_batches = math.ceil(len(shard_ds) / args.eval_batch_size) if len(shard_ds) else 0
    progress = tqdm(
        iterator,
        total=total_batches,
        desc=f"eval seed={seed}",
        leave=False,
        disable=rank != 0,
    )

    for batch in progress:
        batch_raw_prompts = [example["prompt"] for example in batch]
        prompts = [_format_prompt(tokenizer, p) for p in batch_raw_prompts]
        batch_refs = [normalize_text(example["reference"]) for example in batch]
        audios = [load_audio_array(example["audio"], target_sampling_rate=16000)[0] for example in batch]

        inputs = processor(
            text=prompts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
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
        batch_preds = [
            normalize_text(tokenizer.decode(row, skip_special_tokens=True))
            for row in generated
        ]
        predictions.extend(batch_preds)
        references.extend(batch_refs)
        raw_prompts.extend(batch_raw_prompts)

    if world_size > 1:
        gathered_preds = [None for _ in range(world_size)]
        gathered_refs = [None for _ in range(world_size)]
        gathered_prompts = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_preds, predictions)
        dist.all_gather_object(gathered_refs, references)
        dist.all_gather_object(gathered_prompts, raw_prompts)
        if rank != 0:
            return {}
        predictions = [item for sublist in gathered_preds for item in sublist]
        references = [item for sublist in gathered_refs for item in sublist]
        raw_prompts = [item for sublist in gathered_prompts for item in sublist]

    if getattr(args, "dump_samples", 0) > 0 and rank == 0:
        n = min(args.dump_samples, len(predictions))
        print(f"\n--- Sample predictions (first {n}) ---")
        for i in range(n):
            print(f"[{i}] QUESTION: {raw_prompts[i]}")
            print(f"[{i}] EXPECTED: {references[i]}")
            print(f"[{i}]      GOT: {predictions[i]}")
            match = "EXACT MATCH" if predictions[i] == references[i] else "no match"
            print(f"[{i}]   STATUS: {match}")
            print()

    return compute_corpus_metrics(
        predictions=predictions,
        references=references,
        with_text_metrics=True,
        with_bertscore=bool(args.with_bertscore),
        bertscore_lang=args.bertscore_lang,
        lowercase_bleu=bool(args.lowercase_bleu),
        lowercase=bool(args.eval_lowercase),
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--do_sample", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty (Granite Speech recommends 3.0)")
    parser.add_argument("--dump_samples", type=int, default=0,
                        help="Print this many pred/ref pairs for debugging")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--random_subset_seed", type=int, default=-1)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--with_bertscore", type=int, default=0)
    parser.add_argument("--bertscore_lang", type=str, default="en")
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
            f"Running {args.num_runs} eval passes over {len(split_ds)} samples "
            f"(do_sample={args.do_sample}, temp={args.temperature}, top_p={args.top_p}, "
            f"batch={args.eval_batch_size}, world_size={world_size}, "
            f"random_subset_seed={args.random_subset_seed})",
            flush=True,
        )

    all_runs: List[Dict[str, float]] = []
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
