#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_records(run_dir: Path):
    paths = sorted(run_dir.glob("fork_position_stats_rank*.jsonl"))
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return paths, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    paths, records = load_records(run_dir)
    if not records:
        raise SystemExit(f"No fork_position_stats_rank*.jsonl files found in {run_dir}")

    selected_counts = Counter()
    usable_counts = Counter()
    total_selected_positions = 0
    total_usable_positions = 0
    total_usable_groups = 0
    total_group_size_sum = 0.0
    reward_spreads = []
    usable_position_indices = []

    for record in records:
        for pos in record.get("selected_position_indices", []):
            selected_counts[int(pos)] += 1
        for pos in record.get("usable_position_indices", []):
            usable_counts[int(pos)] += 1
        total_selected_positions += int(record.get("selected_fork_positions", 0))
        total_usable_positions += int(record.get("usable_fork_positions", 0))
        total_usable_groups += int(record.get("usable_group_count", 0))
        total_group_size_sum += sum(float(x) for x in record.get("group_sizes", []))
        reward_spreads.extend(float(x) for x in record.get("reward_spreads", []))
        usable_position_indices.extend(int(x) for x in record.get("usable_position_indices", []))

    positions = sorted(set(selected_counts.keys()) | set(usable_counts.keys()))
    position_stats = []
    for pos in positions:
        selected = int(selected_counts[pos])
        usable = int(usable_counts[pos])
        position_stats.append(
            {
                "position": pos,
                "selected_count": selected,
                "usable_count": usable,
                "fork_fire_rate": (usable / selected) if selected else 0.0,
            }
        )

    payload = {
        "task_name": records[0].get("task_name", run_dir.name),
        "run_dir": str(run_dir),
        "num_records": len(records),
        "source_files": [str(path) for path in paths],
        "total_selected_positions": total_selected_positions,
        "total_usable_positions": total_usable_positions,
        "usable_fraction": (total_usable_positions / total_selected_positions) if total_selected_positions else 0.0,
        "usable_positions_per_record": (total_usable_positions / len(records)) if records else 0.0,
        "usable_groups_per_record": (total_usable_groups / len(records)) if records else 0.0,
        "mean_group_size": (total_group_size_sum / total_usable_groups) if total_usable_groups else 0.0,
        "mean_reward_spread": (sum(reward_spreads) / len(reward_spreads)) if reward_spreads else 0.0,
        "mean_usable_position": (
            sum(usable_position_indices) / len(usable_position_indices) if usable_position_indices else 0.0
        ),
        "position_stats": position_stats,
    }

    text = json.dumps(payload, indent=2)
    print(text)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
