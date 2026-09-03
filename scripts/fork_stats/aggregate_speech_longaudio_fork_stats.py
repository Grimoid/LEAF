#!/usr/bin/env python3
"""Aggregate speech_longaudio fork-stats training runs into a paper_artifacts-style directory.

For each Granite backbone (granite3_2b, granite3_8b, granite4_1b), this script:
  * locates the training run dir (auto-discovered from the standard OUTPUT_ROOT
    pattern, or supplied explicitly), reads its fork_position_stats_rank*.jsonl
    and fork_group_stats_rank*.jsonl, and writes a paper_artifacts-style JSON
    (granite<x>_speech_longaudio_fork_fire_stats.json) with the same schema
    plot_fork_fire_rate.py and generate_advisor_summary.py expect;
  * computes the table-friendly summary fields (usable_fraction, mean_usable_position,
    early_fire = pos<=10, late_fire = pos>=20, mean_group_size, mean_reward_spread,
    Table 2 batch counts);
  * if --include_paper_artifacts is set, copies the existing LibriSQA / CoVoST2
    fork_fire_stats.json files for the same three models so all three datasets
    sit side-by-side; the per-model plots then become 3-series figures.

Outputs (default: results/fork_stats/longaudio_<timestamp>/):
  granite3_2b_speech_longaudio_fork_fire_stats.json
  granite3_8b_speech_longaudio_fork_fire_stats.json
  granite4_1b_speech_longaudio_fork_fire_stats.json
  granite<x>_fork_fire_rate__speech_longaudio.{png,svg}            per-model, longaudio only
  granite<x>_fork_fire_rate__all_tasks.{png,svg}                   per-model, 3-task plot (if --include_paper_artifacts)
  speech_longaudio_fork_fire_rate__all_models.{png,svg}            longaudio, 3 backbones
  checkpoint_fork_fire_summary.csv                                 paper Table-3-style summary
  batch_fork_group_summary.csv                                     paper Table-2-style batch counts
  README.md                                                        index of files

Typical usage):
  python scripts/fork_stats/aggregate_speech_longaudio_fork_stats.py \\
      --auto_discover \\
      --include_paper_artifacts <dir with {backbone}_{task}_fork_fire_stats.json> \\
      --output_dir results/fork_stats/speech_longaudio_$(date +%Y%m%d_%H%M%S)

Or pass run dirs explicitly:
  python scripts/fork_stats/aggregate_speech_longaudio_fork_stats.py \\
      --run_dir granite3_2b=/path/to/run_xxx \\
      --run_dir granite3_8b=/path/to/run_yyy \\
      --run_dir granite4_1b=/path/to/run_zzz \\
      --output_dir /tmp/longaudio_artifacts
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Run-dir auto-discovery: for each backbone tag, look for LongAudio training runs (launched
# with `speechrl train --fork-stats`) under $FORK_STATS_RUN_ROOT (default: <repo>/ckpts).
# Explicit --run_dir tag=path arguments always take precedence.
import os as _os
_RUN_ROOT = Path(_os.environ.get("FORK_STATS_RUN_ROOT", REPO_ROOT / "ckpts"))
DEFAULT_OUTPUT_ROOTS = {
    "granite3_2b": [_RUN_ROOT / "leaf_granite3_2b_longaudio"],
    "granite3_8b": [_RUN_ROOT / "leaf_granite3_8b_longaudio"],
    "granite4_1b": [_RUN_ROOT / "leaf_granite4_1b_longaudio"],
}

MODEL_NAMES = {
    "granite3_2b": "ibm-granite/granite-speech-3.3-2b",
    "granite3_8b": "ibm-granite/granite-speech-3.3-8b",
    "granite4_1b": "ibm-granite/granite-4.0-1b-speech",
}

PLOT_SCRIPT = Path(__file__).resolve().parent / "plot_fork_fire_rate.py"

# ---------- jsonl ingestion ----------

def load_position_records(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(run_dir.glob("fork_position_stats_rank*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_group_records(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(run_dir.glob("fork_group_stats_rank*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ---------- per-position aggregation ----------

def build_position_stats(records: list[dict]) -> tuple[list[dict], dict]:
    """Aggregate per-record records into per-position summary and overall summary.

    The training jsonl flattens group_sizes / reward_spreads across positions, so
    per-position group_size / reward_spread cannot be reconstructed exactly.
    We therefore include only the fields plot_fork_fire_rate.py / generate_advisor_summary.py
    actually need per position (selected_count, usable_count, fork_fire_rate),
    and report mean_group_size + mean_reward_spread at the top level.
    """
    selected_counts: dict[int, int] = defaultdict(int)
    usable_counts: dict[int, int] = defaultdict(int)
    all_group_sizes: list[int] = []
    all_reward_spreads: list[float] = []
    all_usable_positions: list[int] = []

    for rec in records:
        for pos in rec.get("selected_position_indices", []):
            selected_counts[int(pos)] += 1
        for pos in rec.get("usable_position_indices", []):
            usable_counts[int(pos)] += 1
        all_group_sizes.extend(int(x) for x in rec.get("group_sizes", []))
        all_reward_spreads.extend(float(x) for x in rec.get("reward_spreads", []))
        all_usable_positions.extend(int(x) for x in rec.get("usable_position_indices", []))

    positions = sorted(set(selected_counts.keys()) | set(usable_counts.keys()))
    position_stats = []
    for pos in positions:
        sel = int(selected_counts[pos])
        use = int(usable_counts[pos])
        position_stats.append(
            {
                "position": pos,
                "selected_count": sel,
                "usable_count": use,
                "fork_fire_rate": (use / sel) if sel else 0.0,
            }
        )

    total_selected = sum(p["selected_count"] for p in position_stats)
    total_usable = sum(p["usable_count"] for p in position_stats)
    early_rows = [p for p in position_stats if p["position"] <= 10]
    late_rows = [p for p in position_stats if p["position"] >= 20]
    early_selected = sum(p["selected_count"] for p in early_rows)
    early_usable = sum(p["usable_count"] for p in early_rows)
    late_selected = sum(p["selected_count"] for p in late_rows)
    late_usable = sum(p["usable_count"] for p in late_rows)

    summary = {
        "total_selected": total_selected,
        "total_usable": total_usable,
        "usable_fraction": (total_usable / total_selected) if total_selected else 0.0,
        "mean_usable_position": (
            sum(all_usable_positions) / len(all_usable_positions) if all_usable_positions else 0.0
        ),
        "early_fire": (early_usable / early_selected) if early_selected else 0.0,
        "late_fire": (late_usable / late_selected) if late_selected else 0.0,
        "mean_group_size": (sum(all_group_sizes) / len(all_group_sizes)) if all_group_sizes else 0.0,
        "max_group_size": max(all_group_sizes) if all_group_sizes else 0,
        "mean_reward_spread": (sum(all_reward_spreads) / len(all_reward_spreads)) if all_reward_spreads else 0.0,
        "max_reward_spread": max(all_reward_spreads) if all_reward_spreads else 0.0,
        "usable_group_count": len(all_group_sizes),
        "num_records": len(records),
    }
    return position_stats, summary


def summarize_group_records(records: list[dict]) -> dict:
    """Compute Table 2-style stats from fork_group_stats_rank*.jsonl."""
    if not records:
        return {}
    total_batches = len(records)
    total_prompts = sum(int(r.get("num_prompts", 0)) for r in records)
    total_attempted = sum(int(r.get("attempted_fork_positions", 0)) for r in records)
    total_usable_positions = sum(int(r.get("usable_fork_positions", 0)) for r in records)
    fired_batches = sum(1 for r in records if int(r.get("usable_fork_positions", 0)) > 0)
    group_sizes = [size for r in records for size in r.get("group_sizes", [])]
    reward_spreads = [spread for r in records for spread in r.get("reward_spreads", [])]
    usable_positions = [pos for r in records for pos in r.get("usable_position_indices", [])]
    nonzero_spread = sum(1 for s in reward_spreads if float(s) > 0.0)
    median_pos = sorted(usable_positions)[len(usable_positions) // 2] if usable_positions else 0
    return {
        "batches_logged": total_batches,
        "batch_fire_rate": (fired_batches / total_batches) if total_batches else 0.0,
        "usable_over_attempted": (total_usable_positions / total_attempted) if total_attempted else 0.0,
        "usable_groups_per_prompt": (len(group_sizes) / total_prompts) if total_prompts else 0.0,
        "mean_group_size": (sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0,
        "nonzero_spread_rate": (nonzero_spread / len(reward_spreads)) if reward_spreads else 0.0,
        "mean_reward_spread": (sum(reward_spreads) / len(reward_spreads)) if reward_spreads else 0.0,
        "median_usable_position": median_pos,
    }


# ---------- run-dir discovery ----------

def latest_run_dir(roots: list[Path] | Path) -> Path | None:
    if isinstance(roots, Path):
        roots = [roots]
    for root in roots:
        if not root.exists():
            continue
        candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
        if not candidates:
            continue
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def parse_run_dir_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--run_dir expects tag=/path, got: {spec}")
    tag, path = spec.split("=", 1)
    if tag not in DEFAULT_OUTPUT_ROOTS:
        raise SystemExit(
            f"--run_dir tag must be one of {sorted(DEFAULT_OUTPUT_ROOTS)}, got '{tag}'"
        )
    return tag, Path(path).expanduser().resolve()


# ---------- artifact writers ----------

def write_paper_artifact_json(
    output_path: Path,
    *,
    tag: str,
    task_name: str,
    run_dir: Path,
    position_stats: list[dict],
    summary: dict,
    group_summary: dict,
    train_split: str,
) -> None:
    payload = {
        "task_name": task_name,
        "data_dir": str(REPO_ROOT / "data" / task_name),
        "split": train_split,
        "num_examples": summary.get("num_records", 0),
        "model_name": MODEL_NAMES[tag],
        "source": {
            "kind": "training_run",
            "run_dir": str(run_dir),
        },
        "generation": {
            "max_new_tokens": 100,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 0,
            "do_sample": True,
        },
        "budgets": {"K": 8, "B": 2, "min_prefix_tokens": 1},
        "summary": summary,
        "batch_summary": group_summary,
        "position_stats": position_stats,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def call_plot(series: list[tuple[str, Path]], png_path: Path, svg_path: Path, title: str, subtitle: str) -> None:
    if not series:
        return
    args = [sys.executable, str(PLOT_SCRIPT)]
    for label, path in series:
        args += ["--series", f"{label}={path}"]
    args += ["--output_png", str(png_path), "--output_svg", str(svg_path)]
    if title:
        args += ["--title", title]
    if subtitle:
        args += ["--subtitle", subtitle]
    print(f"[longaudio-aggregate] plot -> {png_path.name}", flush=True)
    subprocess.run(args, check=True)


# ---------- driver ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        action="append",
        default=[],
        help="Explicit per-model run dir, format tag=/path (tag in granite3_2b|granite3_8b|granite4_1b). Repeatable.",
    )
    parser.add_argument(
        "--auto_discover",
        action="store_true",
        help="If set, pick the latest run_<tag> dir under the standard OUTPUT_ROOT for each backbone not specified by --run_dir.",
    )
    parser.add_argument(
        "--include_stats_dir",
        type=str,
        default="",
        help="Optional path to an existing fork-fire stats dir; its "
        "granite<x>_{librisqa,covost2}_fork_fire_stats.json files are copied into the output and "
        "added as extra series in the per-model plots.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Where to write the aggregated artifacts. Default: results/fork_stats/longaudio_<timestamp>.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="speech_longaudio",
        help="Task tag used in output JSON / file names.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Split label recorded in the output JSON (records were collected from this split).",
    )
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / f"results/fork_stats/longaudio_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[longaudio-aggregate] output_dir={out_dir}", flush=True)

    explicit_dirs = dict(parse_run_dir_arg(spec) for spec in args.run_dir)
    resolved_dirs: dict[str, Path] = {}
    for tag, default_roots in DEFAULT_OUTPUT_ROOTS.items():
        if tag in explicit_dirs:
            resolved_dirs[tag] = explicit_dirs[tag]
        elif args.auto_discover:
            disc = latest_run_dir(default_roots)
            if disc is None:
                roots_str = ", ".join(str(r) for r in default_roots)
                print(f"[longaudio-aggregate] WARN no run dir found for tag={tag} under any of: {roots_str}", file=sys.stderr)
                continue
            resolved_dirs[tag] = disc
        else:
            print(f"[longaudio-aggregate] skip tag={tag}: no --run_dir and --auto_discover not set")

    if not resolved_dirs:
        raise SystemExit("[longaudio-aggregate] no run dirs to aggregate; pass --run_dir or --auto_discover")

    include_stats_dir_path = (
        Path(args.include_stats_dir).expanduser().resolve() if args.include_stats_dir else None
    )

    csv_rows: list[dict] = []
    batch_csv_rows: list[dict] = []
    longaudio_jsons: dict[str, Path] = {}
    librisqa_jsons: dict[str, Path] = {}
    covost2_jsons: dict[str, Path] = {}

    for tag, run_dir in resolved_dirs.items():
        print(f"[longaudio-aggregate] tag={tag} run_dir={run_dir}", flush=True)
        position_records = load_position_records(run_dir)
        group_records = load_group_records(run_dir)
        group_summary = summarize_group_records(group_records)

        if position_records:
            position_stats, summary = build_position_stats(position_records)
            out_json = out_dir / f"{tag}_{args.task_name}_fork_fire_stats.json"
            write_paper_artifact_json(
                out_json,
                tag=tag,
                task_name=args.task_name,
                run_dir=run_dir,
                position_stats=position_stats,
                summary=summary,
                group_summary=group_summary,
                train_split=args.train_split,
            )
            longaudio_jsons[tag] = out_json
            csv_rows.append(
                {
                    "variant": tag,
                    "task": args.task_name,
                    "usable_fraction": summary["usable_fraction"],
                    "mean_usable_position": summary["mean_usable_position"],
                    "early_fire": summary["early_fire"],
                    "late_fire": summary["late_fire"],
                    "mean_group_size": summary["mean_group_size"],
                    "mean_reward_spread": summary["mean_reward_spread"],
                    "total_selected": summary["total_selected"],
                    "total_usable": summary["total_usable"],
                    "path": str(out_json),
                }
            )
        else:
            print(
                f"[longaudio-aggregate] tag={tag}: no fork_position_stats_rank*.jsonl found "
                f"(per-position fork-fire requires the FLAVOR=position trainer); writing Table 2 summary only.",
                file=sys.stderr,
            )

        if group_summary:
            batch_csv_rows.append(
                {
                    "variant": tag,
                    "task": args.task_name,
                    **{k: group_summary[k] for k in group_summary},
                    "run_dir": str(run_dir),
                }
            )
        elif not position_records:
            print(
                f"[longaudio-aggregate] tag={tag}: no jsonl logs in {run_dir} -- skipping",
                file=sys.stderr,
            )

    # Pull in existing librisqa / covost2 artifacts so per-model plots have all 3 datasets.
    if include_stats_dir_path and include_stats_dir_path.exists():
        for tag in resolved_dirs:
            for task_label, dataset_short, registry in (
                ("LibriSQA", "librisqa", librisqa_jsons),
                ("CoVoST2", "covost2", covost2_jsons),
            ):
                src = include_stats_dir_path / f"{tag}_{dataset_short}_fork_fire_stats.json"
                if not src.exists():
                    continue
                dst = out_dir / src.name
                shutil.copyfile(src, dst)
                registry[tag] = dst
                with open(src, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                rows = payload.get("position_stats", [])
                total_selected = sum(int(r["selected_count"]) for r in rows)
                total_usable = sum(int(r["usable_count"]) for r in rows)
                early_rows = [r for r in rows if int(r["position"]) <= 10]
                late_rows = [r for r in rows if int(r["position"]) >= 20]
                early_selected = sum(int(r["selected_count"]) for r in early_rows)
                early_usable = sum(int(r["usable_count"]) for r in early_rows)
                late_selected = sum(int(r["selected_count"]) for r in late_rows)
                late_usable = sum(int(r["usable_count"]) for r in late_rows)
                mean_usable_pos = (
                    sum(int(r["position"]) * int(r["usable_count"]) for r in rows) / total_usable
                    if total_usable
                    else 0.0
                )
                gs_total_groups = sum(int(r["usable_count"]) for r in rows)
                if rows and "mean_group_size" in rows[0] and gs_total_groups:
                    mean_group_size = (
                        sum(float(r["mean_group_size"]) * int(r["usable_count"]) for r in rows) / gs_total_groups
                    )
                    mean_reward_spread = (
                        sum(float(r["mean_reward_spread"]) * int(r["usable_count"]) for r in rows) / gs_total_groups
                    )
                else:
                    mean_group_size = 0.0
                    mean_reward_spread = 0.0
                csv_rows.append(
                    {
                        "variant": tag,
                        "task": task_label,
                        "usable_fraction": (total_usable / total_selected) if total_selected else 0.0,
                        "mean_usable_position": mean_usable_pos,
                        "early_fire": (early_usable / early_selected) if early_selected else 0.0,
                        "late_fire": (late_usable / late_selected) if late_selected else 0.0,
                        "mean_group_size": mean_group_size,
                        "mean_reward_spread": mean_reward_spread,
                        "total_selected": total_selected,
                        "total_usable": total_usable,
                        "path": str(dst),
                    }
                )

    # Write CSVs.
    if csv_rows:
        csv_path = out_dir / "checkpoint_fork_fire_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[longaudio-aggregate] wrote {csv_path}")
    if batch_csv_rows:
        batch_csv_path = out_dir / "batch_fork_group_summary.csv"
        all_keys = sorted({k for row in batch_csv_rows for k in row.keys()})
        ordered_keys = (
            ["variant", "task", "batches_logged", "batch_fire_rate", "usable_over_attempted",
             "usable_groups_per_prompt", "mean_group_size", "nonzero_spread_rate",
             "mean_reward_spread", "median_usable_position", "run_dir"]
        )
        ordered_keys += [k for k in all_keys if k not in ordered_keys]
        with open(batch_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(batch_csv_rows)
        print(f"[longaudio-aggregate] wrote {batch_csv_path}")

    # Combined plot: speech_longaudio across all 3 backbones.
    if len(longaudio_jsons) >= 1:
        png = out_dir / f"{args.task_name}_fork_fire_rate__all_models.png"
        svg = out_dir / f"{args.task_name}_fork_fire_rate__all_models.svg"
        series = [
            (label.replace("granite3_", "Granite 3.3 ").replace("granite4_", "Granite 4.0 ").replace("_2b", " 2B").replace("_8b", " 8B").replace("_1b", " 1B"), path)
            for label, path in longaudio_jsons.items()
        ]
        call_plot(
            series,
            png,
            svg,
            title=f"Fork-fire rate on {args.task_name} (training-time)",
            subtitle="Three Granite backbones; positions with >=3 selected forks shown",
        )

    # Per-model plots: speech_longaudio alone, and 3-task combined if paper_artifacts available.
    for tag, longaudio_json in longaudio_jsons.items():
        single_png = out_dir / f"{tag}_fork_fire_rate__{args.task_name}.png"
        single_svg = out_dir / f"{tag}_fork_fire_rate__{args.task_name}.svg"
        call_plot(
            [(args.task_name, longaudio_json)],
            single_png,
            single_svg,
            title=f"Fork-fire rate -- {tag} on {args.task_name}",
            subtitle="Training-time stats (positions with >=3 selected forks shown)",
        )
        combined_series = [(args.task_name, longaudio_json)]
        if tag in librisqa_jsons:
            combined_series.append(("LibriSQA", librisqa_jsons[tag]))
        if tag in covost2_jsons:
            combined_series.append(("CoVoST2", covost2_jsons[tag]))
        if len(combined_series) > 1:
            multi_png = out_dir / f"{tag}_fork_fire_rate__all_tasks.png"
            multi_svg = out_dir / f"{tag}_fork_fire_rate__all_tasks.svg"
            call_plot(
                combined_series,
                multi_png,
                multi_svg,
                title=f"Fork-fire rate -- {tag}",
                subtitle="LibriSQA / CoVoST2 from validation-time fork-fire stats; speech_longaudio from training-time stats",
            )

    # README / index.
    readme_lines = [
        f"# speech_longaudio fork-stats artifacts",
        "",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Per-model fork-fire JSON (Table 3 inputs)",
        "",
    ]
    for tag, run_dir in resolved_dirs.items():
        readme_lines.append(f"- `{tag}_{args.task_name}_fork_fire_stats.json` (from `{run_dir}`)")
    if librisqa_jsons or covost2_jsons:
        readme_lines += ["", "## Copied from --include_paper_artifacts"]
        for tag in resolved_dirs:
            if tag in librisqa_jsons:
                readme_lines.append(f"- `{librisqa_jsons[tag].name}`")
            if tag in covost2_jsons:
                readme_lines.append(f"- `{covost2_jsons[tag].name}`")
    readme_lines += [
        "",
        "## Summary CSVs",
        "- `checkpoint_fork_fire_summary.csv` (variant, task, usable_fraction, mean_usable_position, early_fire/late_fire, group/spread; one row per (model, dataset))",
        "- `batch_fork_group_summary.csv` (Table 2 style; one row per (model, dataset) with batch counts)",
        "",
        "## Plots",
        f"- `{args.task_name}_fork_fire_rate__all_models.{{png,svg}}` -- 3 backbones on {args.task_name}",
        f"- `<tag>_fork_fire_rate__{args.task_name}.{{png,svg}}` -- per-model {args.task_name}-only plot",
        f"- `<tag>_fork_fire_rate__all_tasks.{{png,svg}}` -- per-model 3-task plot (when --include_paper_artifacts is given)",
        "",
        "Re-render any plot manually with `scripts/speech/plot_fork_fire_rate.py --series ...`.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"[longaudio-aggregate] done -> {out_dir}")


if __name__ == "__main__":
    main()
