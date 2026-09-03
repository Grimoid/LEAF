"""Merge VoxPopuli-LongAudio and Europarl-LongAudio into a single dataset.

Produces speech_longaudio = voxpopuli_longaudio + europarl_longaudio.
DailyTalk is intentionally excluded; it remains a separate dataset.

Both source datasets must already be prepared (run the individual prepare
scripts first).  This script simply loads them from disk and concatenates
the train/validation/test splits.

Usage
-----
    python scripts/data_prep/prepare_speech_longaudio.py \\
        --voxpopuli_dir data/voxpopuli_longaudio \\
        --europarl_dir  data/europarl_longaudio \\
        --out_dir       data/speech_longaudio

The output has the same schema as each source:
    task (string), id (string), prompt (string), reference (string),
    audio (Audio @ 16 kHz), duration (float64)

The 'task' column distinguishes the source ('voxpopuli_longaudio' vs
'europarl_longaudio') and is preserved as-is from each source dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge VoxPopuli-LongAudio and Europarl-LongAudio datasets."
    )
    ap.add_argument("--voxpopuli_dir", required=True,
                    help="Root output dir of prepare_voxpopuli_longaudio.py")
    ap.add_argument("--europarl_dir", required=True,
                    help="Root output dir of prepare_europarl_longaudio.py")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for the merged dataset")
    args = ap.parse_args()

    vox_dir = Path(args.voxpopuli_dir)
    ep_dir  = Path(args.europarl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VoxPopuli-LongAudio ...")
    vox_ds = load_from_disk(str(vox_dir / "dataset"))
    print(f"  {vox_ds}")

    print("Loading Europarl-LongAudio ...")
    ep_ds = load_from_disk(str(ep_dir / "dataset"))
    print(f"  {ep_ds}")

    splits = set(vox_ds.keys()) | set(ep_ds.keys())
    merged: dict = {}
    for split in sorted(splits):
        parts = []
        if split in vox_ds and len(vox_ds[split]) > 0:
            parts.append(vox_ds[split])
        if split in ep_ds and len(ep_ds[split]) > 0:
            parts.append(ep_ds[split])
        if parts:
            merged[split] = concatenate_datasets(parts)
        else:
            merged[split] = vox_ds[split] if split in vox_ds else ep_ds[split]

    merged_ds = DatasetDict(merged)

    print("\nMerged dataset:")
    print(f"  {merged_ds}")
    for split, ds in merged_ds.items():
        task_counts: dict = {}
        for t in ds["task"]:
            task_counts[t] = task_counts.get(t, 0) + 1
        print(f"  {split}: {len(ds):,} rows  " +
              "  ".join(f"{k}={v:,}" for k, v in sorted(task_counts.items())))

    merged_ds.save_to_disk(str(out_dir / "dataset"))
    print(f"\nSaved to: {out_dir / 'dataset'}")


if __name__ == "__main__":
    main()
