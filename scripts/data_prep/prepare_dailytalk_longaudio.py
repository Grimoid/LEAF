"""Prepare DailyTalk + nvidia/LongAudio QA dataset for the LEAF speech experiments.

Task: model listens to a full spoken dialogue (DailyTalk session audio) and
answers an open-ended question about it.  Format matches LibriSQA:
{prompt, reference, audio}.

Answer lengths are ~19 words (same as LibriSQA), making BLEU a suitable reward.

Data sources
------------
- QA pairs : nvidia/LongAudio  (daily_talk split, DailyTalkConnectingQA)
             45 951 pairs over 2 541 unique sessions; ~78 % are <= 40 s.
- Audio    : DailyTalk corpus (Google Drive, see download script)
             Raw directory contains d{session_id:04d}/ subdirectories, each
             holding utterance WAVs named {turn}_{speaker}_d{id:04d}.wav.
             We concatenate them in turn order to reproduce the LongAudio
             session WAV for each dialogue.

Session ID mapping
------------------
nvidia/LongAudio uses integer IDs 0-2540 (matching DailyTalk's 2 541 dialogues).
The prepare script tries d{id:04d}, d{id+1:04d}, and bare {id} as fallbacks.

Usage
-----
    python scripts/data_prep/prepare_dailytalk_longaudio.py \\
        --dailytalk_dir /path/to/DailyTalk/raw \\
        --out_dir /path/to/output/dailytalk_longaudio

Override defaults:
    --max_duration_secs 40   # skip examples whose audio exceeds this
    --allow_missing_audio 1  # skip missing sessions instead of failing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from tqdm import tqdm

SAMPLE_RATE = 16_000
SOUND_TOKEN = "<sound>"
AUDIO_TOKEN = "<|audio|>"

LONGAUDIO_REPO = "nvidia/LongAudio"
LONGAUDIO_JSON = "longaudio_xl/DailyTalk_LongAudio.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def _stable_split(uid: str, val_frac: float, test_frac: float) -> str:
    """Deterministic train/val/test assignment from ID hash."""
    h = int(hashlib.md5(uid.encode()).hexdigest()[:8], 16) / (16 ** 8)
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "validation"
    return "train"


def _resample(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return array
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(orig_sr, target_sr)
        return resample_poly(array, target_sr // g, orig_sr // g).astype(np.float32)
    except ImportError:
        n = int(len(array) * target_sr / orig_sr)
        return np.interp(np.linspace(0, len(array) - 1, n),
                         np.arange(len(array)), array).astype(np.float32)


def _turn_id(wav_path: Path) -> int:
    """Extract the leading turn index from a DailyTalk utterance filename."""
    try:
        return int(wav_path.stem.split("_")[0])
    except (ValueError, IndexError):
        return 0


def _find_data_root(dailytalk_dir: Path) -> Path:
    """Walk up to 3 levels into dailytalk_dir to find the directory that
    directly contains integer-named session subdirectories with WAV files.
    E.g. resolves  DailyTalk/  →  DailyTalk/dailytalk/data/
    """
    def _looks_like_data_root(d: Path) -> bool:
        # Contains at least one subdirectory whose name is a bare integer
        try:
            return any(
                child.is_dir() and child.name.isdigit()
                for child in d.iterdir()
            )
        except PermissionError:
            return False

    if _looks_like_data_root(dailytalk_dir):
        return dailytalk_dir

    for depth in range(3):
        for child in dailytalk_dir.rglob("*"):
            if child.is_dir() and _looks_like_data_root(child):
                return child

    return dailytalk_dir  # fallback: return as-is


def _find_dialogue_dir(data_root: Path, session_id: int) -> Optional[Path]:
    """Locate the session directory under data_root, trying naming variants."""
    def _check(d: Path) -> bool:
        return d.is_dir() and any(d.glob("*.wav"))

    for name in (str(session_id), f"{session_id:04d}",
                 f"d{session_id}", f"d{session_id:04d}",
                 f"d{session_id + 1:04d}"):
        c = data_root / name
        if _check(c):
            return c
    return None


def _build_session_audio(dialogue_dir: Path) -> Optional[np.ndarray]:
    """Concatenate utterance WAVs in turn order to form one session array."""
    wav_files = sorted(dialogue_dir.glob("*.wav"), key=_turn_id)
    if not wav_files:
        return None
    arrays: List[np.ndarray] = []
    for wf in wav_files:
        try:
            data, sr = sf.read(str(wf), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            arrays.append(_resample(data, sr, SAMPLE_RATE))
        except Exception:
            continue
    return np.concatenate(arrays) if arrays else None


def _format_prompt(human_value: str) -> str:
    """Replace <sound> with <|audio|> wherever it appears in the question."""
    return human_value.replace(SOUND_TOKEN, AUDIO_TOKEN).strip()


def _parse_convs(raw) -> list:
    if isinstance(raw, list):
        return raw
    try:
        import ast
        return ast.literal_eval(raw)
    except Exception:
        return json.loads(raw)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare DailyTalk-LongAudio QA dataset for the LEAF speech experiments."
    )
    ap.add_argument("--dailytalk_dir", required=True,
                    help="Root dir of DailyTalk raw audio (contains d0000/, d0001/, ...)")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory (dataset/ created inside)")
    ap.add_argument("--max_duration_secs", type=float, default=40.0,
                    help="Exclude examples whose audio duration exceeds this (default 40)")
    ap.add_argument("--allow_missing_audio", type=int, default=0,
                    help="Skip missing sessions instead of raising (default 0)")
    ap.add_argument("--val_frac",  type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit",   type=int, default=0)
    ap.add_argument("--test_limit",  type=int, default=0)
    args = ap.parse_args()

    dailytalk_dir   = Path(args.dailytalk_dir)
    out_dir         = Path(args.out_dir)
    audio_cache_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_cache_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover the subdirectory that contains bare-integer session dirs
    data_root = _find_data_root(dailytalk_dir)
    print(f"  DailyTalk data root: {data_root}")

    # ── load LongAudio metadata ───────────────────────────────────────────
    print("Loading nvidia/LongAudio DailyTalk metadata from HuggingFace Hub ...")
    json_path = hf_hub_download(
        repo_id=LONGAUDIO_REPO,
        filename=LONGAUDIO_JSON,
        repo_type="dataset",
    )
    with open(json_path) as f:
        entries: list = json.load(f)
    print(f"  {len(entries)} total entries")

    entries = [e for e in entries if float(e["duration"]) <= args.max_duration_secs]
    print(f"  {len(entries)} after duration filter (<= {args.max_duration_secs}s)")

    # ── build session-level audio cache ──────────────────────────────────
    unique_sounds: Dict[str, int] = {
        e["sound"]: int(Path(e["sound"]).stem)
        for e in entries
    }
    print(f"\nBuilding audio cache for {len(unique_sounds)} unique sessions ...")

    session_paths: Dict[str, Optional[str]] = {}
    missing = 0
    for sound_file, session_id in tqdm(unique_sounds.items(), desc="sessions"):
        cache_path = audio_cache_dir / sound_file
        if cache_path.exists():
            session_paths[sound_file] = str(cache_path)
            continue

        dialogue_dir = _find_dialogue_dir(data_root, session_id)
        if dialogue_dir is None:
            missing += 1
            if not args.allow_missing_audio:
                raise FileNotFoundError(
                    f"No DailyTalk directory found for session {session_id} "
                    f"(tried d{session_id:04d}, d{session_id+1:04d}, etc.) "
                    f"under {dailytalk_dir}.\n"
                    "Pass --allow_missing_audio 1 to skip."
                )
            session_paths[sound_file] = None
            continue

        audio = _build_session_audio(dialogue_dir)
        if audio is None:
            missing += 1
            if not args.allow_missing_audio:
                raise FileNotFoundError(f"No WAV files found in {dialogue_dir}")
            session_paths[sound_file] = None
            continue

        sf.write(str(cache_path), audio, SAMPLE_RATE)
        session_paths[sound_file] = str(cache_path)

    if missing:
        print(f"  WARNING: {missing} sessions skipped (missing audio)")

    # ── convert entries to rows ───────────────────────────────────────────
    print("\nConverting entries ...")
    split_rows: Dict[str, List[dict]] = {"train": [], "validation": [], "test": []}

    for entry in tqdm(entries, desc="entries"):
        audio_path = session_paths.get(entry["sound"])
        if audio_path is None:
            continue

        convs = _parse_convs(entry["conversations"])
        human_val = next((c["value"] for c in convs if c["from"] == "human"), "")
        gpt_val   = next((c["value"] for c in convs if c["from"] == "gpt"),   "")
        if not human_val or not gpt_val:
            continue

        split = _stable_split(str(entry["id"]), args.val_frac, args.test_frac)
        split_rows[split].append({
            "task":      "dailytalk_longaudio",
            "id":        str(entry["id"]),
            "prompt":    _format_prompt(human_val),
            "reference": gpt_val.strip(),
            "audio":     audio_path,
            "duration":  float(entry["duration"]),
        })

    for split_name, limit in [("train", args.train_limit),
                               ("validation", args.val_limit),
                               ("test", args.test_limit)]:
        if limit > 0:
            split_rows[split_name] = split_rows[split_name][:limit]

    # ── build HF DatasetDict ──────────────────────────────────────────────
    def _to_ds(rows: List[dict]) -> Dataset:
        if not rows:
            return Dataset.from_dict(
                {"task": [], "id": [], "prompt": [], "reference": [], "audio": [], "duration": []}
            )
        non_audio  = [{k: v for k, v in r.items() if k != "audio"} for r in rows]
        audio_vals = [r["audio"] for r in rows]
        ds = Dataset.from_list(non_audio)
        ds = ds.add_column("audio", audio_vals)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        return ds

    dataset = DatasetDict({k: _to_ds(v) for k, v in split_rows.items()})
    dataset.save_to_disk(str(out_dir / "dataset"))

    sizes = {k: len(v) for k, v in split_rows.items()}
    print(f"\nSaved to: {out_dir / 'dataset'}")
    print("Sizes: " + "  ".join(f"{k}={n}" for k, n in sizes.items()))
    print(f"max_duration_secs={args.max_duration_secs}")


if __name__ == "__main__":
    main()
