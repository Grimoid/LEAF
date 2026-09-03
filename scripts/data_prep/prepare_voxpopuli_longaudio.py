"""Prepare VoxPopuli + nvidia/LongAudio QA dataset for the LEAF speech experiments.

Task: model listens to a European Parliament speech (VoxPopuli session audio) and
answers an open-ended question about it.  Format matches LibriSQA / DailyTalk-LongAudio:
{prompt, reference, audio}.

Answer lengths are ~27 words (mean), making BLEU a suitable reward.
MCQ entries (~25%) are excluded — only open-ended comprehension questions are kept.

Data sources
------------
- QA pairs : nvidia/LongAudio  (VoxPopuli_LongAudio.json, split "VoxPopuliConnectingQA")
             81,952 pairs (all English); after duration + MCQ filter: ~9,607 usable.
- Audio    : Two-stage retrieval:
    Stage 1 — facebook/voxpopuli HF parquet shards (labeled English subset, ~543 h).
              Each row has an audio_id that matches the stem of the LongAudio sound
              path.  Covers ~15% of needed IDs.
    Stage 2 — dl.fbaipublicfiles.com/voxpopuli/audios/en_{year}.tar
              VoxPopuli per-year tarballs containing FULL SESSION recordings
              (one OGG per plenary agenda item, ~5-10 GB per year, 2009-2020).
              The VoxPopuli annotation TSV (asr_en.tsv.gz, ~63 MB, one-time
              download) provides VAD timestamps to slice each speaker-turn segment
              from the session OGG.  Only years containing missing IDs are downloaded.
              Tarballs are deleted after extraction to reclaim disk space.

Tarball structure
-----------------
  en/{year}/{session_id}_en.ogg   (e.g. en/2013/20131007-0900-PLENARY-19_en.ogg)

Audio_id mapping
----------------
  audio_id  = "{session_id}-{id_}"
             = "20131007-0900-PLENARY-19-en_20131007-21:26:04_1"
  session_id = audio_id.split("-en_")[0] = "20131007-0900-PLENARY-19"
  id_        = "en_20131007-21:26:04_1"  (= "{lang}_{timestamp}_{seg}")

Audio cache
-----------
Audio is cached segment-by-segment in out_dir/audio/{audio_id}.wav.
Multiple segments belonging to one LongAudio entry are concatenated in listed order.
Caching is idempotent; already-cached WAVs are never re-extracted.

Usage
-----
    python scripts/data_prep/prepare_voxpopuli_longaudio.py \\
        --out_dir data/voxpopuli_longaudio

Skip parquet scan (faster if you already cached from HF, or want tarballs only):
    --skip_parquet 1

Set tarball download directory (default: out_dir/tarballs):
    --tarball_dir /path/to/scratch/vp_tarballs

Other overrides:
    --max_duration_secs 40   # skip examples whose audio exceeds this
    --filter_mcq 1           # exclude multiple-choice questions (default: 1)
    --allow_missing_audio 1  # skip entries with missing audio instead of failing
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

SAMPLE_RATE = 16_000
SOUND_TOKEN = "<sound>"
AUDIO_TOKEN = "<|audio|>"

LONGAUDIO_REPO = "nvidia/LongAudio"
LONGAUDIO_JSON = "longaudio_xl/VoxPopuli_LongAudio.json"
VOXPOPULI_REPO = "facebook/voxpopuli"
VOXPOPULI_LANG = "en"

FBAIPUBLICFILES_BASE = (
    "https://dl.fbaipublicfiles.com/voxpopuli/audios/en_{year}.tar"
)
ANNOTATION_TSV_URL = (
    "https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_en.tsv.gz"
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


def _decode_audio_bytes(audio_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw audio bytes (ogg/flac/wav) to float32 at SAMPLE_RATE."""
    try:
        arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return _resample(arr, sr, SAMPLE_RATE)
    except Exception:
        return None


def _sound_path_to_audio_id(sound_path: str) -> str:
    """Extract audio_id from a LongAudio sound path.

    'en/2017/20171116-0900-PLENARY-5-en_20171116-10:38:16_0.wav'
    -> '20171116-0900-PLENARY-5-en_20171116-10:38:16_0'
    """
    return Path(sound_path).stem


def _audio_id_to_session_id(audio_id: str) -> str:
    """Extract session_id from audio_id.

    '20131007-0900-PLENARY-19-en_20131007-21:26:04_1' -> '20131007-0900-PLENARY-19'

    The session_id is everything before '-{lang}_' in the audio_id.
    """
    return audio_id.split(f"-{VOXPOPULI_LANG}_")[0]


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


def _get_voxpopuli_parquet_paths() -> List[str]:
    """Return sorted list of parquet shard paths for facebook/voxpopuli English.

    Scans both the labeled English config (en/) and the accented English config
    (en_accented/) — both contain audio_ids ending in '-en' that may match
    nvidia/LongAudio entries.  The multilang config is a re-sharding of the
    same labeled data and adds no new English audio_ids.
    """
    configs = (f"{VOXPOPULI_LANG}/", f"{VOXPOPULI_LANG}_accented/")
    all_files = list(list_repo_files(VOXPOPULI_REPO, repo_type="dataset"))
    parquet_files = sorted(
        f for f in all_files
        if any(f.startswith(c) for c in configs) and f.endswith(".parquet")
    )
    return parquet_files


def _delete_parquet_from_hf_cache(local_path: str) -> None:
    """Delete a parquet shard from the HF hub cache (blob + snapshot symlink)."""
    try:
        blob_path = os.path.realpath(local_path)
        if os.path.islink(local_path):
            os.unlink(local_path)
        if os.path.isfile(blob_path):
            os.unlink(blob_path)
    except Exception as e:
        print(f"    WARNING: could not delete cached parquet {local_path}: {e}")


def _scan_parquet_shards(
    missing: Set[str],
    audio_cache_dir: Path,
    cached: Dict[str, str],
    delete_parquet_after: bool,
) -> Set[str]:
    """Scan HF parquet shards for missing audio_ids; write WAVs to cache.

    Modifies `cached` in place.  Returns the set of IDs still not found.
    """
    import pyarrow.parquet as pq

    shard_paths = _get_voxpopuli_parquet_paths()
    print(f"  Scanning {len(shard_paths)} parquet shards ...")

    remaining = set(missing)
    for shard_name in tqdm(shard_paths, desc="parquet shards"):
        if not remaining:
            break
        try:
            local = hf_hub_download(VOXPOPULI_REPO, shard_name, repo_type="dataset")
        except Exception as e:
            print(f"    WARNING: could not download {shard_name}: {e}")
            continue

        try:
            table = pq.read_table(local)
        except Exception as e:
            print(f"    WARNING: could not read {shard_name}: {e}")
            if delete_parquet_after:
                _delete_parquet_from_hf_cache(local)
            continue

        # Process in record batches to avoid PyArrow int32 offset overflow
        # that occurs when combine_chunks() is called on large binary columns.
        found_in_shard = 0
        for batch in table.to_batches():
            if not remaining:
                break
            batch_audio_ids = batch["audio_id"].to_pylist()
            audio_col = batch.column("audio")
            for i, aid in enumerate(batch_audio_ids):
                if aid not in remaining:
                    continue
                raw = audio_col[i].as_py()["bytes"]
                arr = _decode_audio_bytes(raw)
                if arr is None:
                    print(f"    WARNING: could not decode audio for {aid}")
                    continue
                cache_path = audio_cache_dir / f"{aid}.wav"
                sf.write(str(cache_path), arr, SAMPLE_RATE)
                cached[aid] = str(cache_path)
                remaining.discard(aid)
                found_in_shard += 1

        del table
        if delete_parquet_after:
            _delete_parquet_from_hf_cache(local)
            tqdm.write(f"    deleted shard ({found_in_shard} IDs extracted)")

    return remaining


def _load_asr_annotation_tsv(
    cache_dir: Path,
    needed_ids: Set[str],
) -> Dict[str, List[Tuple[float, float]]]:
    """Download and parse asr_en.tsv.gz to get VAD timestamps for each audio_id.

    The TSV maps audio_id -> [(start_sec, end_sec), ...] within the session OGG.
    audio_id is reconstructed as "{session_id}-{id_}" (e.g.
    "20131007-0900-PLENARY-19-en_20131007-21:26:04_1").

    Only entries whose audio_id is in `needed_ids` are loaded to save memory.
    Returns: {audio_id: [(start_sec, end_sec), ...]}
    """
    tsv_path = cache_dir / "asr_en.tsv.gz"
    if not tsv_path.exists():
        print(f"  Downloading VoxPopuli annotation TSV ({ANNOTATION_TSV_URL}) ...")
        subprocess.run(
            ["wget", "-c", "--show-progress", "-O", str(tsv_path), ANNOTATION_TSV_URL],
            check=True,
        )

    print("  Parsing annotation TSV ...")
    vad_lookup: Dict[str, List[Tuple[float, float]]] = {}
    with gzip.open(str(tsv_path), "rt") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            session_id = row["session_id"]
            id_ = row["id_"]
            audio_id = f"{session_id}-{id_}"
            if audio_id not in needed_ids:
                continue
            try:
                vad = literal_eval(row["vad"])  # [[start, end], ...]
                vad_lookup[audio_id] = [(float(s), float(e)) for s, e in vad]
            except Exception:
                continue

    found = len(vad_lookup)
    total = len(needed_ids)
    print(f"  Found VAD timestamps for {found:,} / {total:,} needed audio_ids.")
    return vad_lookup


def _scan_fbaipublicfiles_tarballs(
    missing: Set[str],
    audio_cache_dir: Path,
    tarball_dir: Path,
    cached: Dict[str, str],
    vad_lookup: Dict[str, List[Tuple[float, float]]],
    delete_tarball_after: bool,
) -> Set[str]:
    """Download per-year VoxPopuli tarballs and extract speaker-turn WAVs.

    Each tarball contains FULL SESSION recordings (one OGG per plenary agenda item):
      en/{year}/{session_id}_en.ogg

    VAD timestamps from vad_lookup are used to slice individual speaker turns from
    the session OGG.  Multiple VAD segments per turn are concatenated.

    Downloads only the years that contain at least one missing audio_id.
    Each tarball (~5-10 GB) is deleted after extraction when delete_tarball_after=True.

    Modifies `cached` in place.  Returns the set of IDs still not found.
    """
    tarball_dir.mkdir(parents=True, exist_ok=True)

    # Group missing audio_ids by year -> session_id -> [audio_id]
    year_to_sessions: Dict[str, Dict[str, List[str]]] = {}
    for aid in missing:
        session_id = _audio_id_to_session_id(aid)
        year = session_id[:4]
        year_to_sessions.setdefault(year, {}).setdefault(session_id, []).append(aid)

    years_sorted = sorted(year_to_sessions)
    print(f"  Need to scan {len(years_sorted)} year tarballs: {years_sorted}")

    remaining = set(missing)
    lang_suffix = f"_{VOXPOPULI_LANG}"  # "_en"

    for year in years_sorted:
        session_map = year_to_sessions[year]  # session_id -> [audio_id]
        still_needed = {
            sid: [a for a in aids if a in remaining]
            for sid, aids in session_map.items()
        }
        still_needed = {k: v for k, v in still_needed.items() if v}
        if not still_needed:
            continue

        n_segs = sum(len(v) for v in still_needed.values())
        url = FBAIPUBLICFILES_BASE.format(year=year)
        tarball_path = tarball_dir / f"audio_{year}.tar"

        # Download with wget -c (resumable)
        print(f"\n  [{year}] Downloading {url} ...")
        try:
            subprocess.run(
                ["wget", "-c", "--show-progress", "-O", str(tarball_path), url],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: wget failed for {url}: {e}")
            continue

        # Stream through tarball — find session OGGs and slice segments
        print(f"  [{year}] Slicing {n_segs} segments from "
              f"{len(still_needed)} session file(s) ...")
        found_in_year = 0
        try:
            with tarfile.open(str(tarball_path), "r:") as tf:
                for member in tqdm(tf, desc=f"tar {year}", leave=False):
                    if not member.isfile():
                        continue
                    stem = Path(member.name).stem  # "20120328-0900-PLENARY-10_en"
                    if not stem.endswith(lang_suffix):
                        continue
                    session_id = stem[: -len(lang_suffix)]  # strip "_en"

                    if session_id not in still_needed:
                        continue

                    audio_ids_for_session = [
                        a for a in still_needed[session_id] if a in remaining
                    ]
                    if not audio_ids_for_session:
                        continue

                    # Load the full session OGG into memory
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    session_bytes = fobj.read()
                    try:
                        session_arr, session_sr = sf.read(
                            io.BytesIO(session_bytes), dtype="float32", always_2d=False
                        )
                    except Exception as exc:
                        print(f"    WARNING: could not decode session {session_id}: {exc}")
                        continue
                    if session_arr.ndim > 1:
                        session_arr = session_arr.mean(axis=1)
                    session_len = len(session_arr)

                    # Slice each speaker turn using VAD timestamps
                    for aid in audio_ids_for_session:
                        vad_segs = vad_lookup.get(aid)
                        if not vad_segs:
                            print(f"    WARNING: no VAD entry for {aid}")
                            continue
                        parts: List[np.ndarray] = []
                        for start_s, end_s in vad_segs:
                            s_idx = int(start_s * session_sr)
                            e_idx = min(int(end_s * session_sr), session_len)
                            if s_idx < e_idx:
                                parts.append(session_arr[s_idx:e_idx])
                        if not parts:
                            print(f"    WARNING: empty VAD segments for {aid}")
                            continue
                        arr = np.concatenate(parts)
                        arr = _resample(arr, session_sr, SAMPLE_RATE)
                        cache_path = audio_cache_dir / f"{aid}.wav"
                        sf.write(str(cache_path), arr, SAMPLE_RATE)
                        cached[aid] = str(cache_path)
                        remaining.discard(aid)
                        found_in_year += 1

        except Exception as e:
            print(f"  WARNING: error reading tarball {tarball_path}: {e}")

        print(f"  [{year}] Sliced {found_in_year} / {n_segs} segments.")

        if delete_tarball_after:
            try:
                tarball_path.unlink()
                print(f"  [{year}] Deleted tarball.")
            except Exception as e:
                print(f"  WARNING: could not delete {tarball_path}: {e}")

    return remaining


def _build_audio_cache(
    needed_ids: Set[str],
    audio_cache_dir: Path,
    tarball_dir: Path,
    allow_missing: bool,
    skip_parquet: bool = False,
    delete_parquet_after: bool = True,
    delete_tarball_after: bool = True,
) -> Dict[str, str]:
    """Build a complete audio cache for all needed_ids.

    Stage 1 (optional): scan facebook/voxpopuli HF parquet shards.
    Stage 2: download per-year tarballs from dl.fbaipublicfiles.com for any
             IDs still missing after stage 1.

    Returns: {audio_id: str(cache_path)}.
    """
    # Check which IDs are already cached
    cached: Dict[str, str] = {}
    missing: Set[str] = set()
    for aid in needed_ids:
        cp = audio_cache_dir / f"{aid}.wav"
        if cp.exists():
            cached[aid] = str(cp)
        else:
            missing.add(aid)

    if not missing:
        print(f"  All {len(needed_ids)} audio segments already cached.")
        return cached

    print(f"  {len(cached)} already cached, {len(missing)} to fetch.")

    # Stage 1: HF parquet shards (covers labeled English subset, ~15% of total)
    if skip_parquet:
        print("\n  Skipping parquet scan (--skip_parquet 1).")
        remaining_after_parquet = missing
    else:
        print("\n  Stage 1: scanning facebook/voxpopuli HF parquet shards ...")
        remaining_after_parquet = _scan_parquet_shards(
            missing=missing,
            audio_cache_dir=audio_cache_dir,
            cached=cached,
            delete_parquet_after=delete_parquet_after,
        )
        n_found = len(missing) - len(remaining_after_parquet)
        print(f"  Stage 1 complete: {n_found} found, {len(remaining_after_parquet)} still missing.")

    # Stage 2: fbaipublicfiles year tarballs (session-level recordings + VAD slicing)
    if remaining_after_parquet:
        print(f"\n  Stage 2: downloading fbaipublicfiles year tarballs for "
              f"{len(remaining_after_parquet)} missing IDs ...")
        # Load VAD timestamp metadata (needed to slice segments from session OGGs)
        vad_lookup = _load_asr_annotation_tsv(
            cache_dir=audio_cache_dir.parent,
            needed_ids=remaining_after_parquet,
        )
        remaining_after_tarballs = _scan_fbaipublicfiles_tarballs(
            missing=remaining_after_parquet,
            audio_cache_dir=audio_cache_dir,
            tarball_dir=tarball_dir,
            cached=cached,
            vad_lookup=vad_lookup,
            delete_tarball_after=delete_tarball_after,
        )
        n_found = len(remaining_after_parquet) - len(remaining_after_tarballs)
        print(f"  Stage 2 complete: {n_found} found, {len(remaining_after_tarballs)} still missing.")
    else:
        remaining_after_tarballs = set()

    if remaining_after_tarballs:
        msg = (f"{len(remaining_after_tarballs)} audio_ids not found in parquet shards "
               f"or fbaipublicfiles tarballs.")
        if allow_missing:
            print(f"  WARNING: {msg}")
        else:
            raise FileNotFoundError(
                f"{msg}\n"
                "Pass --allow_missing_audio 1 to skip missing entries.\n"
                f"Missing IDs (first 5): {list(remaining_after_tarballs)[:5]}"
            )

    return cached


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare VoxPopuli-LongAudio QA dataset for the LEAF speech experiments."
    )
    ap.add_argument("--out_dir", required=True,
                    help="Output directory (dataset/ and audio/ created inside)")
    ap.add_argument("--tarball_dir", default="",
                    help="Directory for downloading fbaipublicfiles year tarballs "
                         "(default: out_dir/tarballs)")
    ap.add_argument("--max_duration_secs", type=float, default=40.0,
                    help="Exclude examples whose total audio duration exceeds this (default 40)")
    ap.add_argument("--filter_mcq", type=int, default=1,
                    help="Exclude multiple-choice questions (default 1)")
    ap.add_argument("--allow_missing_audio", type=int, default=0,
                    help="Skip entries with missing audio instead of failing (default 0)")
    ap.add_argument("--skip_parquet", type=int, default=0,
                    help="Skip HF parquet scan and go straight to fbaipublicfiles (default 0)")
    ap.add_argument("--delete_parquet_after", type=int, default=1,
                    help="Delete each parquet shard from HF cache after extracting audio "
                         "to reclaim disk space (default 1)")
    ap.add_argument("--delete_tarball_after", type=int, default=1,
                    help="Delete each year tarball after extracting audio (default 1)")
    ap.add_argument("--val_frac",  type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit",   type=int, default=0)
    ap.add_argument("--test_limit",  type=int, default=0)
    args = ap.parse_args()

    out_dir         = Path(args.out_dir)
    audio_cache_dir = out_dir / "audio"
    tarball_dir     = Path(args.tarball_dir) if args.tarball_dir else out_dir / "tarballs"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── load LongAudio metadata ───────────────────────────────────────────
    print("Loading nvidia/LongAudio VoxPopuli metadata from HuggingFace Hub ...")
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

    # ── collect all needed audio_ids ─────────────────────────────────────
    needed_ids: Set[str] = set()
    for entry in entries:
        sounds = entry["sound"] if isinstance(entry["sound"], list) else [entry["sound"]]
        for s in sounds:
            needed_ids.add(_sound_path_to_audio_id(s))
    print(f"\nUnique audio segments needed: {len(needed_ids)}")

    # ── build audio cache (parquet → fbaipublicfiles tarballs) ───────────
    print("\nBuilding audio cache ...")
    audio_cache = _build_audio_cache(
        needed_ids=needed_ids,
        audio_cache_dir=audio_cache_dir,
        tarball_dir=tarball_dir,
        allow_missing=bool(args.allow_missing_audio),
        skip_parquet=bool(args.skip_parquet),
        delete_parquet_after=bool(args.delete_parquet_after),
        delete_tarball_after=bool(args.delete_tarball_after),
    )
    print(f"\n  Total cached: {len(audio_cache)} / {len(needed_ids)} segments")

    # ── build per-entry concatenated audio cache ──────────────────────────
    # For entries with multiple sound segments, concatenate and cache as a single file.
    print("\nBuilding entry-level concatenated audio cache ...")
    entry_audio_paths: Dict[str, Optional[str]] = {}
    missing_entries = 0

    for entry in tqdm(entries, desc="entries"):
        entry_id = str(entry["id"])
        sounds = entry["sound"] if isinstance(entry["sound"], list) else [entry["sound"]]
        audio_ids = [_sound_path_to_audio_id(s) for s in sounds]

        # Check if all segments are available (guard empty sound list too)
        if not audio_ids or not all(aid in audio_cache for aid in audio_ids):
            missing_entries += 1
            entry_audio_paths[entry_id] = None
            continue

        if len(audio_ids) == 1:
            # Single segment — reuse directly
            entry_audio_paths[entry_id] = audio_cache[audio_ids[0]]
            continue

        # Multiple segments — concatenate and cache
        concat_name = f"__concat__{entry_id}.wav"
        concat_path = audio_cache_dir / concat_name
        if concat_path.exists():
            entry_audio_paths[entry_id] = str(concat_path)
            continue

        arrays: List[np.ndarray] = []
        for aid in audio_ids:
            arr, _ = sf.read(audio_cache[aid], dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            arrays.append(arr)
        concat = np.concatenate(arrays)
        sf.write(str(concat_path), concat, SAMPLE_RATE)
        entry_audio_paths[entry_id] = str(concat_path)

    if missing_entries:
        print(f"  WARNING: {missing_entries} entries skipped (missing audio segments)")

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
            "task":      "voxpopuli_longaudio",
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
    print("Sizes: " + "  ".join(f"{k}={len(v):,}" for k, v in split_rows.items()))
    print(f"max_duration_secs={args.max_duration_secs}  filter_mcq={args.filter_mcq}")


if __name__ == "__main__":
    main()
