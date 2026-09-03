#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# `--fork-stats` writes fork_position_stats_rank*.jsonl; fork_group_stats_rank*.jsonl
# (an older file name with a subset of the fields) is accepted as well.
_STATS_GLOBS = ("fork_position_stats_rank*.jsonl", "fork_group_stats_rank*.jsonl")


def load_records(run_dir: Path) -> list[dict]:
    records = []
    paths = sorted(p for pattern in _STATS_GLOBS for p in run_dir.glob(pattern))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def summarize(records: list[dict]) -> dict:
    if not records:
        return {}

    total_batches = len(records)
    total_prompts = sum(int(r["num_prompts"]) for r in records)
    total_attempted = sum(int(r["attempted_fork_positions"]) for r in records)
    total_usable_positions = sum(int(r["usable_fork_positions"]) for r in records)
    attempted_per_prompt = records[0]["attempted_fork_positions_per_prompt"]

    group_sizes = [size for r in records for size in r.get("group_sizes", [])]
    reward_spreads = [spread for r in records for spread in r.get("reward_spreads", [])]
    usable_positions = [pos for r in records for pos in r.get("usable_position_indices", [])]

    return {
        "task_name": records[0].get("task_name", "unknown"),
        "batches": total_batches,
        "prompts": total_prompts,
        "attempted_per_prompt": attempted_per_prompt,
        "avg_usable_positions_per_batch": total_usable_positions / total_batches if total_batches else 0.0,
        "avg_usable_positions_per_prompt": total_usable_positions / total_prompts if total_prompts else 0.0,
        "usable_fraction": total_usable_positions / total_attempted if total_attempted else 0.0,
        "mean_group_size": sum(group_sizes) / len(group_sizes) if group_sizes else 0.0,
        "max_group_size": max(group_sizes) if group_sizes else 0,
        "mean_reward_spread": sum(reward_spreads) / len(reward_spreads) if reward_spreads else 0.0,
        "max_reward_spread": max(reward_spreads) if reward_spreads else 0.0,
        "mean_usable_position": sum(usable_positions) / len(usable_positions) if usable_positions else 0.0,
        "max_usable_position": max(usable_positions) if usable_positions else -1,
        "usable_group_count": len(group_sizes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    records = load_records(Path(args.run_dir))
    if not records:
        raise SystemExit(f"No fork_position_stats_rank*.jsonl (or fork_group_stats_rank*.jsonl) files found in {args.run_dir}")

    by_task = defaultdict(list)
    for record in records:
        by_task[record.get("task_name", "unknown")].append(record)

    for task_name, task_records in by_task.items():
        summary = summarize(task_records)
        print(f"task={task_name}")
        print(f"  batches={summary['batches']}")
        print(f"  prompts={summary['prompts']}")
        print(
            f"  usable_forks={summary['avg_usable_positions_per_prompt']:.3f} / "
            f"{summary['attempted_per_prompt']} per prompt "
            f"({summary['usable_fraction']:.2%} of attempted positions)"
        )
        print(
            f"  mean_group_size={summary['mean_group_size']:.3f} "
            f"max_group_size={summary['max_group_size']}"
        )
        print(
            f"  mean_reward_spread={summary['mean_reward_spread']:.6f} "
            f"max_reward_spread={summary['max_reward_spread']:.6f}"
        )
        print(
            f"  mean_usable_position={summary['mean_usable_position']:.3f} "
            f"max_usable_position={summary['max_usable_position']}"
        )
        print(
            f"  paper_sentence=On {task_name}, an average of "
            f"{summary['avg_usable_positions_per_prompt']:.3f} / "
            f"{summary['attempted_per_prompt']} fork positions per prompt yielded groups "
            f"of size >= 2 (mean group size {summary['mean_group_size']:.3f}, "
            f"max group size {summary['max_group_size']}), and the average within-group "
            f"reward spread was {summary['mean_reward_spread']:.6f} BLEU."
        )
        print()


if __name__ == "__main__":
    main()
