#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import sysconfig
from collections import defaultdict
from pathlib import Path

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

from datasets import Audio, load_from_disk  # noqa: E402
from tqdm import tqdm  # noqa: E402


def _prefer_repo_root_for_local_openrlhf() -> None:
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)


_prefer_repo_root_for_local_openrlhf()

from openrlhf.speech_leaf import load_audio_array, load_model_maybe_peft, load_processor, sync_audio_features  # noqa: E402
from openrlhf.speech_leaf.fork_analysis import analyze_trajectories  # noqa: E402
from openrlhf.speech_leaf.reward import RewardConfig, compute_reward  # noqa: E402
from openrlhf.speech_leaf.sampler import PrefixTreeConfig, GenConfig, generate_batch  # noqa: E402
from openrlhf.speech_leaf.text_utils import normalize_text  # noqa: E402


def format_prompt(tokenizer, raw_prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return raw_prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_example_inputs(processor, tokenizer, example: dict, prompt_max_len: int, device: torch.device):
    audio_array, _ = load_audio_array(example["audio"], target_sampling_rate=16000)
    prompt = format_prompt(tokenizer, example["prompt"])
    inputs = processor(
        text=prompt,
        audio=audio_array,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_max_len,
    )
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    audio_token = getattr(processor, "audio_token", "<|audio|>")
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
    return sync_audio_features(inputs, audio_token_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--random_subset_seed", type=int, default=-1)
    parser.add_argument("--prompt_max_len", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--do_sample", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--reward_mode", type=str, default="bleu", choices=["bleu", "binary"])
    parser.add_argument("--lowercase_bleu", type=int, default=1)
    parser.add_argument("--rollout_budget_K", type=int, default=8)   # rollout budget K
    parser.add_argument("--fork_budget_B", type=int, default=2)   # fork budget B
    parser.add_argument("--fork_min_prefix_tokens", type=int, default=1)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    model_path = Path(args.model_name)
    if ("/" in args.model_name or args.model_name.startswith(".")) and not model_path.exists():
        raise SystemExit(f"Model/checkpoint path does not exist: {args.model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_from_disk(str(Path(args.data_dir) / "dataset"))
    split_ds = dataset[args.split]
    if "audio" in split_ds.column_names:
        split_ds = split_ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    indices = list(range(len(split_ds)))
    if args.random_subset_seed >= 0:
        rng = random.Random(args.random_subset_seed)
        rng.shuffle(indices)
    indices = indices[args.start_index:]
    if args.limit > 0:
        indices = indices[: args.limit]

    processor = load_processor(args.model_name)
    tokenizer = processor.tokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_model_maybe_peft(args.model_name, dtype=dtype)
    model.to(device)
    model.eval()

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=bool(args.do_sample),
    )
    tree_cfg = PrefixTreeConfig(
        K=args.rollout_budget_K,
        B=args.fork_budget_B,
        min_prefix_tokens=args.fork_min_prefix_tokens,
    )
    total_samples = max(1, args.rollout_budget_K)
    reward_cfg = RewardConfig(mode=args.reward_mode, lowercase_bleu=bool(args.lowercase_bleu))

    selected_counts = defaultdict(int)
    usable_counts = defaultdict(int)
    group_sizes_by_pos = defaultdict(list)
    reward_spreads_by_pos = defaultdict(list)
    total_examples = len(indices)

    # Build a tqdm description tag that identifies this run (model/data/split) so
    # multiple parallel cells writing to separate log files can be told apart easily.
    tqdm_desc = (
        f"{Path(args.model_name).name} / {Path(args.data_dir).name} / {args.split}"
    )
    pbar = tqdm(
        enumerate(indices, start=1),
        total=total_examples,
        desc=tqdm_desc,
        mininterval=2.0,    # update at most every 2s so log files stay tidy
        miniters=1,
        dynamic_ncols=True,
        leave=True,
    )
    running_usable = 0  # cumulative usable fork-fires; show as live "fire" stat in the bar
    for example_idx, dataset_idx in pbar:
        example = split_ds[dataset_idx]
        reference = normalize_text(example["reference"])
        model_inputs = prepare_example_inputs(processor, tokenizer, example, args.prompt_max_len, device)
        trajectories = generate_batch(
            model=model,
            tokenizer=tokenizer,
            model_inputs=model_inputs,
            cfg=gen_cfg,
            num_return_sequences=total_samples,
            compute_surprisal=True,
        )
        analysis = analyze_trajectories(
            trajectories,
            reward_fn=lambda hypothesis, ref=reference: float(compute_reward(ref, hypothesis, reward_cfg)),
            prefix_tree_cfg=tree_cfg,
        )

        for pos in analysis["selected_positions"]:
            selected_counts[int(pos)] += 1
        n_usable_this = 0
        for pos_entry in analysis["positions"]:
            if not pos_entry["usable"]:
                continue
            pos = int(pos_entry["position"])
            usable_counts[pos] += 1
            n_usable_this += 1
            for group in pos_entry["groups"]:
                group_sizes_by_pos[pos].append(int(group["group_size"]))
                reward_spreads_by_pos[pos].append(float(group["reward_spread"]))
        running_usable += n_usable_this
        # Live stats in the tqdm bar: cumulative usable_fraction.
        total_sel = sum(selected_counts.values())
        if total_sel:
            pbar.set_postfix_str(
                f"usable={running_usable}/{total_sel} ({100*running_usable/total_sel:.1f}%)",
                refresh=False,
            )

    positions = sorted(set(selected_counts.keys()) | set(usable_counts.keys()))
    position_stats = []
    for pos in positions:
        selected = selected_counts[pos]
        usable = usable_counts[pos]
        group_sizes = group_sizes_by_pos[pos]
        reward_spreads = reward_spreads_by_pos[pos]
        position_stats.append(
            {
                "position": pos,
                "selected_count": selected,
                "usable_count": usable,
                "fork_fire_rate": (usable / selected) if selected else 0.0,
                "mean_group_size": (sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0,
                "mean_reward_spread": (sum(reward_spreads) / len(reward_spreads)) if reward_spreads else 0.0,
            }
        )

    payload = {
        "task_name": Path(args.data_dir).name,
        "data_dir": args.data_dir,
        "split": args.split,
        "num_examples": len(indices),
        "model_name": args.model_name,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": bool(args.do_sample),
        },
        "budgets": {
            "K": args.rollout_budget_K,
            "B": args.fork_budget_B,
            "min_prefix_tokens": args.fork_min_prefix_tokens,
        },
        "position_stats": position_stats,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
