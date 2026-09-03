"""Evaluation script for LibriSQA with Qwen2-Audio models.

Drop-in equivalent of eval_librisqa.py but uses Qwen2AudioForConditionalGeneration
(via AutoModel) and the Qwen2Audio processor interface.

Key differences from the Granite Speech eval script:
  - Model loaded via AutoModel with trust_remote_code=True (transformers 4.52+ removed AutoModelForConditionalGeneration)
  - Processor called with audios= (plural) instead of audio=
  - Dataset prompts contain <|audio|> (Granite token) which is replaced with
    Qwen's <|audio_bos|><|AUDIO|><|audio_eos|> placeholder
  - No sync_audio_features (Qwen2-Audio doesn't use input_features_mask)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from datasets import Audio, load_from_disk
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from openrlhf.speech_leaf import compute_corpus_metrics, load_audio_array
from openrlhf.speech_leaf.text_utils import normalize_text

# Granite Speech uses <|audio|>; Qwen2-Audio uses this triplet.
_GRANITE_AUDIO_TOKEN = "<|audio|>"
_QWEN_AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"


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


def _load_processor(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif tokenizer.bos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.bos_token_id
    return processor


def _load_model(model_name: str, dtype):
    return Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )


def _qwen_prompt(tokenizer, raw_prompt: str) -> str:
    """Replace Granite audio token with Qwen's placeholder and apply chat template."""
    text = raw_prompt.replace(_GRANITE_AUDIO_TOKEN, _QWEN_AUDIO_PLACEHOLDER)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


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
        prompts = [_qwen_prompt(tokenizer, p) for p in batch_raw_prompts]
        batch_refs = [normalize_text(example["reference"]) for example in batch]
        audios = [
            load_audio_array(example["audio"], target_sampling_rate=16000)[0]
            for example in batch
        ]

        inputs = processor(
            text=prompts,
            audios=audios,  # Qwen2-Audio uses "audios" (plural)
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

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
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        metrics[key] = round(mean, 6)
        metrics[f"{key}_std"] = round(std, 6)
        metrics[f"{key}_runs"] = [round(v, 6) for v in values]
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
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--dump_samples", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--random_subset_seed", type=int, default=-1)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size for inference (Qwen2-Audio is 7B; default smaller than Granite)")
    parser.add_argument("--with_bertscore", type=int, default=0)
    parser.add_argument("--bertscore_lang", type=str, default="en")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--lowercase_bleu", type=int, default=1)
    parser.add_argument("--eval_lowercase", type=int, default=0)
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

    processor = _load_processor(args.model_name)
    tokenizer = processor.tokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = _load_model(args.model_name, dtype=dtype)
    model.to(device)
    model.eval()

    if rank == 0:
        print(
            f"Running {args.num_runs} eval passes over {len(split_ds)} samples "
            f"(do_sample={args.do_sample}, temp={args.temperature}, top_p={args.top_p}, "
            f"batch={args.eval_batch_size}, world_size={world_size})",
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
