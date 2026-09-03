"""Prepare Europarl-ST + nvidia/LongAudio QA dataset for the LEAF speech experiments.

Task: model listens to a European Parliament speech segment and answers an
open-ended question about it.  Format matches LibriSQA / DailyTalk-LongAudio:
{prompt, reference, audio}.

Answer lengths are ~31 words (mean), making BLEU a suitable reward.
MCQ entries (~22%) are excluded — only open-ended comprehension questions are kept.

Data sources
------------
- QA pairs : nvidia/LongAudio  (Europarl_LongAudio.json)
             56,054 pairs; after duration + MCQ filter: ~7,181 usable.
- Audio    : Europarl-ST v1.1 corpus (MLLP, Universitat Politècnica de València)
             Download: https://www.mllp.upv.es/europarl-st/v1.1.tar.gz  (~20 GB)
             License : CC BY-NC 4.0

Audio reconstruction
--------------------
nvidia/LongAudio references pre-segmented clips with timestamp offsets:
  en.20080925.16.4-121_0.0_6.74.wav   <- segment 0.0–6.74 s
  en.20080925.16.4-121_7.11_11.58.wav <- segment 7.11–11.58 s

The base M4A file in Europarl-ST is:
  <europarl_st_dir>/en/audios/en.20080925.16.4-121.m4a

The prepare script:
  1. Parses the base ID and timestamps from each sound filename.
  2. Loads the M4A (or WAV if already converted) using soundfile + pydub fallback.
  3. Extracts the exact time slice and resamples to 16 kHz.
  4. Concatenates slices for entries with multiple sound paths.
  5. Caches each segment as a WAV under out_dir/audio/.

Download Europarl-ST
--------------------
    mkdir -p /path/to/EuroparlST
    wget -O /path/to/EuroparlST/v1.1.tar.gz https://www.mllp.upv.es/europarl-st/v1.1.tar.gz
    tar -xzf /path/to/EuroparlST/v1.1.tar.gz -C /path/to/EuroparlST

The extracted directory should contain:  en/audios/en.YYYYMMDD.*.m4a  etc.

Usage
-----
    python scripts/data_prep/prepare_europarl_longaudio.py \\
        --europarl_st_dir /path/to/EuroparlST \\
        --out_dir data/europarl_longaudio

Override defaults:
    --max_duration_secs 40   # skip examples whose audio exceeds this
    --filter_mcq 1           # exclude multiple-choice questions (default: 1)
    --allow_missing_audio 1  # skip entries with missing audio instead of failing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from tqdm import tqdm

SAMPLE_RATE = 16_000
SOUND_TOKEN = "<sound>"
AUDIO_TOKEN = "<|audio|>"

LONGAUDIO_REPO = "nvidia/LongAudio"
LONGAUDIO_JSON = "longaudio_xl/Europarl_LongAudio.json"

# Regex to parse: en.20080925.16.4-121_0.0_6.74.wav
# Groups: (base_id, start_sec, end_sec)
_SOUND_RE = re.compile(
    r"^(.+?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)(?:\.wav)?$"
)


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
        return np.interp(
            np.linspace(0, len(array) - 1, n),
            np.arange(len(array)), array,
        ).astype(np.float32)


def _parse_sound_path(sound_path: str) -> Optional[Tuple[str, float, float]]:
    """Parse a LongAudio sound path into (base_id, start_sec, end_sec).

    'en.20080925.16.4-121_0.0_6.74.wav' -> ('en.20080925.16.4-121', 0.0, 6.74)
    Returns None if the path does not match the expected pattern.
    """
    stem = Path(sound_path).stem if sound_path.endswith(".wav") else sound_path
    m = _SOUND_RE.match(stem)
    if not m:
        return None
    return m.group(1), float(m.group(2)), float(m.group(3))


def _find_source_audio(base_id: str, europarl_st_dir: Path) -> Optional[Path]:
    """Locate the Europarl-ST source audio file for a given base_id.

    Tries M4A and WAV extensions, and walks the language-specific audios subdirs.
    base_id example: 'en.20080925.16.4-121'
    """
    # Language code is the first component before the first dot
    lang = base_id.split(".")[0]

    candidate_dirs = [
        europarl_st_dir / lang / "audios",
        europarl_st_dir / "v1.1" / lang / "audios",
        europarl_st_dir,
    ]
    for ext in (".m4a", ".wav", ".ogg", ".flac"):
        for d in candidate_dirs:
            p = d / f"{base_id}{ext}"
            if p.exists():
                return p
    return None


def _load_audio_slice(
    source_path: Path,
    start_sec: float,
    end_sec: float,
) -> Optional[np.ndarray]:
    """Load a time slice [start_sec, end_sec) from source_path at SAMPLE_RATE.

    Tries soundfile first (WAV/FLAC/OGG); falls back to PyAV which handles
    M4A/AAC and other formats via its bundled ffmpeg libraries.
    """
    # ── Try soundfile (works for WAV, FLAC, OGG; NOT M4A) ──
    try:
        info = sf.info(str(source_path))
        sr = info.samplerate
        start_frame = int(start_sec * sr)
        end_frame   = int(end_sec   * sr)
        arr, sr = sf.read(
            str(source_path),
            start=start_frame,
            stop=end_frame,
            dtype="float32",
            always_2d=False,
        )
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return _resample(arr, sr, SAMPLE_RATE)
    except Exception:
        pass

    # ── Fallback: PyAV (handles M4A/AAC via bundled ffmpeg libraries) ──
    try:
        import av
        with av.open(str(source_path)) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            sr = stream.sample_rate
            pcm = []
            for frame in container.decode(stream):
                arr = np.array(frame.to_ndarray(), dtype=np.float32)
                # fltp planar: (channels, samples) → mean to mono
                # packed (s16 etc.): (1, samples*ch) — rare for M4A but handle
                if frame.format.is_planar:
                    arr = arr.mean(axis=0) if arr.ndim > 1 else arr
                else:
                    nch = max(1, arr.size // max(frame.samples, 1))
                    arr = arr.reshape(-1, nch).mean(axis=1) if nch > 1 else arr.reshape(-1)
                # Normalize integer formats; fltp is already [-1, 1]
                fmt = frame.format.name
                if "s16" in fmt:
                    arr /= 32768.0
                elif "s32" in fmt:
                    arr /= 2147483648.0
                elif "u8" in fmt:
                    arr = (arr - 128.0) / 128.0
                pcm.append(arr)
        if not pcm:
            return None
        full = np.concatenate(pcm)
        s_idx = int(start_sec * sr)
        e_idx = min(int(end_sec * sr), len(full))
        sliced = full[s_idx:e_idx]
        if len(sliced) == 0:
            return None
        return _resample(sliced, sr, SAMPLE_RATE)
    except Exception:
        pass

    # ── Fallback 2: pydub (requires system ffmpeg) ──
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(str(source_path))
        seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE)
        start_ms = int(start_sec * 1000)
        end_ms   = int(end_sec   * 1000)
        chunk = seg[start_ms:end_ms]
        arr = np.array(chunk.get_array_of_samples(), dtype=np.float32)
        arr /= float(2 ** (chunk.sample_width * 8 - 1))
        return arr
    except Exception:
        pass

    return None


def _is_mcq(entry: dict) -> bool:
    """Return True if this entry is a multiple-choice question."""
    q = entry["conversations"][0]["value"]
    a = entry["conversations"][-1]["value"]
    return (
        "Choose the correct option" in q
        or "Choose the correct answer" in q
        or a.strip().startswith("(")
    )


def _format_prompt(human_value: str) -> str:
    """Replace <sound> with <|audio|> wherever it appears in the question."""
    return human_value.replace(SOUND_TOKEN, AUDIO_TOKEN).strip()


def _find_europarl_st_root(europarl_st_dir: Path) -> Path:
    """Auto-discover the root that contains language subdirectories with audios/."""
    def _looks_like_root(d: Path) -> bool:
        try:
            return any(
                (child / "audios").is_dir()
                for child in d.iterdir()
                if child.is_dir() and len(child.name) <= 3
            )
        except PermissionError:
            return False

    if _looks_like_root(europarl_st_dir):
        return europarl_st_dir

    for child in europarl_st_dir.rglob("*"):
        if child.is_dir() and _looks_like_root(child):
            return child

    return europarl_st_dir  # fallback


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare Europarl-ST-LongAudio QA dataset for the LEAF speech experiments."
    )
    ap.add_argument("--europarl_st_dir", required=True,
                    help="Root dir of extracted Europarl-ST v1.1 (contains en/, de/, ...)")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory (dataset/ and audio/ created inside)")
    ap.add_argument("--max_duration_secs", type=float, default=40.0,
                    help="Exclude examples whose total audio duration exceeds this (default 40)")
    ap.add_argument("--filter_mcq", type=int, default=1,
                    help="Exclude multiple-choice questions (default 1)")
    ap.add_argument("--allow_missing_audio", type=int, default=0,
                    help="Skip entries with missing audio instead of failing (default 0)")
    ap.add_argument("--val_frac",  type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit",   type=int, default=0)
    ap.add_argument("--test_limit",  type=int, default=0)
    args = ap.parse_args()

    europarl_st_dir = Path(args.europarl_st_dir)
    out_dir         = Path(args.out_dir)
    audio_cache_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_cache_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover the Europarl-ST root
    ep_root = _find_europarl_st_root(europarl_st_dir)
    print(f"  Europarl-ST root: {ep_root}")

    # ── load LongAudio metadata ───────────────────────────────────────────
    print("Loading nvidia/LongAudio Europarl metadata from HuggingFace Hub ...")
    json_path = hf_hub_download(
        repo_id=LONGAUDIO_REPO,
        filename=LONGAUDIO_JSON,
        repo_type="dataset",
    )
    with open(json_path) as f:
        entries: list = json.load(f)
    print(f"  {len(entries)} total entries")

    # Duration filter
    entries = [e for e in entries if float(e["duration"]) <= args.max_duration_secs]
    print(f"  {len(entries)} after duration filter (<= {args.max_duration_secs}s)")

    # MCQ filter
    if args.filter_mcq:
        entries = [e for e in entries if not _is_mcq(e)]
        print(f"  {len(entries)} after MCQ filter")

    # ── build per-entry audio cache ───────────────────────────────────────
    print("\nBuilding audio cache (slicing Europarl-ST source files) ...")
    entry_audio_paths: Dict[str, Optional[str]] = {}
    missing_entries = 0

    for entry in tqdm(entries, desc="entries"):
        entry_id = str(entry["id"])
        sounds = entry["sound"] if isinstance(entry["sound"], list) else [entry["sound"]]

        # Cache key: concatenation of all segment cache paths
        cache_name = f"{entry_id.replace('/', '_')}.wav"
        cache_path = audio_cache_dir / cache_name
        if cache_path.exists():
            entry_audio_paths[entry_id] = str(cache_path)
            continue

        # Extract and concatenate all segments
        arrays: List[np.ndarray] = []
        ok = True
        for sound_path in sounds:
            parsed = _parse_sound_path(sound_path)
            if parsed is None:
                print(f"  WARNING: could not parse sound path: {sound_path}")
                ok = False
                break
            base_id, start_sec, end_sec = parsed

            # Check per-segment cache first
            seg_name = Path(sound_path).stem + ".wav"
            seg_cache = audio_cache_dir / seg_name
            if seg_cache.exists():
                arr, _ = sf.read(str(seg_cache), dtype="float32", always_2d=False)
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                arrays.append(arr)
                continue

            # Load from Europarl-ST source
            src = _find_source_audio(base_id, ep_root)
            if src is None:
                if not args.allow_missing_audio:
                    raise FileNotFoundError(
                        f"No Europarl-ST audio found for base_id='{base_id}' "
                        f"under {ep_root}.\n"
                        "Pass --allow_missing_audio 1 to skip.\n"
                        "Download Europarl-ST from: "
                        "https://www.mllp.upv.es/europarl-st/v1.1.tar.gz"
                    )
                ok = False
                break

            arr = _load_audio_slice(src, start_sec, end_sec)
            if arr is None:
                if not args.allow_missing_audio:
                    raise RuntimeError(
                        f"Could not decode audio slice {start_sec}-{end_sec}s "
                        f"from {src}.\n"
                        "Ensure ffmpeg is installed for M4A support (pydub fallback)."
                    )
                ok = False
                break

            # Cache segment
            sf.write(str(seg_cache), arr, SAMPLE_RATE)
            arrays.append(arr)

        if not ok or not arrays:
            missing_entries += 1
            entry_audio_paths[entry_id] = None
            continue

        concat = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        sf.write(str(cache_path), concat, SAMPLE_RATE)
        entry_audio_paths[entry_id] = str(cache_path)

    if missing_entries:
        print(f"  WARNING: {missing_entries} entries skipped (missing audio)")

    # ── convert entries to rows ───────────────────────────────────────────
    print("\nConverting entries to dataset rows ...")
    split_rows: Dict[str, List[dict]] = {"train": [], "validation": [], "test": []}

    for entry in tqdm(entries, desc="rows"):
        entry_id = str(entry["id"])
        audio_path = entry_audio_paths.get(entry_id)
        if audio_path is None:
            continue

        convs = entry["conversations"]
        human_val = next((c["value"] for c in convs if c["from"] == "human"), "")
        gpt_val   = next((c["value"] for c in convs if c["from"] == "gpt"),   "")
        if not human_val or not gpt_val:
            continue

        split = _stable_split(entry_id, args.val_frac, args.test_frac)
        split_rows[split].append({
            "task":      "europarl_longaudio",
            "id":        entry_id,
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
    print("Sizes: " + "  ".join(f"{k}={n:,}" for k, n in sizes.items()))
    print(f"max_duration_secs={args.max_duration_secs}  filter_mcq={args.filter_mcq}")


if __name__ == "__main__":
    main()
