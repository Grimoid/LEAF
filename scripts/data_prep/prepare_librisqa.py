from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

from datasets import Audio, Dataset, DatasetDict, load_dataset
from tqdm import tqdm


DEFAULT_LIBRISQA_TRAIN = "hf://datasets/ZihanZhao/LibriSQA/LibriSQA-PartI/LibriSQA-PartI-train.json"
DEFAULT_LIBRISQA_TEST = "hf://datasets/ZihanZhao/LibriSQA/LibriSQA-PartI/LibriSQA-PartI-test.json"

PROMPT_TEMPLATES = [
    "Listen to the audio <|audio|> and answer the following question: {question}",
    "<|audio|> Please answer this question about the spoken passage: {question}",
    "You will hear speech in <|audio|>. Answer: {question}",
]


def pick_key(example: Dict, candidates: List[str]) -> str:
    for key in candidates:
        if key in example and example[key] is not None:
            return key
    raise KeyError(f"None of keys exist: {candidates}")


def stable_template_idx(text: str, n_templates: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_templates


def resolve_audio_path(librispeech_root: Path, speech_path: str) -> str:
    speech_path = speech_path.strip()
    stripped = speech_path[len("LibriSpeech/") :] if speech_path.startswith("LibriSpeech/") else speech_path

    bases = [
        librispeech_root / speech_path,
        librispeech_root / "LibriSpeech" / stripped,
        librispeech_root / stripped,
    ]

    candidates = []
    for path in bases:
        candidates.append(path)
        if path.suffix == ".wav":
            candidates.append(path.with_suffix(".flac"))
        elif path.suffix == ".flac":
            candidates.append(path.with_suffix(".wav"))

    for path in candidates:
        if path.exists():
            return str(path.resolve())

    return str((librispeech_root / speech_path).resolve())


def _split_candidates(librispeech_root: Path, split_name: str) -> List[Path]:
    return [
        librispeech_root / split_name,
        librispeech_root / "LibriSpeech" / split_name,
    ]


def _has_split(librispeech_root: Path, split_name: str) -> bool:
    return any(path.exists() for path in _split_candidates(librispeech_root, split_name))


def preflight_librispeech_root(librispeech_root: Path, allow_missing_audio: bool) -> None:
    if not librispeech_root.exists():
        raise FileNotFoundError(
            f"--librispeech_root does not exist: {librispeech_root}\n"
            "Point this to the directory containing the LibriSpeech splits."
        )

    required = ["train-clean-360", "test-clean"]
    missing = [split for split in required if not _has_split(librispeech_root, split)]
    if not missing:
        return

    details = []
    for split in missing:
        candidates = _split_candidates(librispeech_root, split)
        details.append(f"  - {split}: expected {candidates[0]} or {candidates[1]}")
    message = (
        "Missing required LibriSpeech split(s) for LibriSQA Part-I:\n"
        + "\n".join(details)
        + "\nLibriSQA train uses train-clean-360 and test uses test-clean."
    )
    if allow_missing_audio:
        print("WARNING:", message)
        return
    raise FileNotFoundError(message)


def convert_split(split_ds, librispeech_root: Path, allow_missing_audio: bool) -> Dataset:
    rows = []
    for example in tqdm(split_ds, desc="convert"):
        question_key = pick_key(example, ["question"])
        answer_key = pick_key(example, ["answer"])
        audio_key = pick_key(example, ["speech_path", "audio_path", "path"])

        question = str(example[question_key]).strip()
        answer = str(example[answer_key]).strip()
        audio_path = resolve_audio_path(librispeech_root, str(example[audio_key]).strip())

        if not Path(audio_path).exists():
            if not allow_missing_audio:
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}\n"
                    "Pass --allow_missing_audio 1 to skip missing files."
                )
            continue

        prompt = PROMPT_TEMPLATES[stable_template_idx(question, len(PROMPT_TEMPLATES))].format(question=question)
        rows.append(
            {
                "task": "librisqa",
                "prompt": prompt,
                "reference": answer,
                "question": question,
                "audio": audio_path,
            }
        )

    dataset = Dataset.from_list(rows) if rows else Dataset.from_dict(
        {"task": [], "prompt": [], "reference": [], "question": [], "audio": []}
    )
    return dataset.cast_column("audio", Audio(sampling_rate=16000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--librispeech_root", type=str, required=True)
    parser.add_argument("--train_json", type=str, default=DEFAULT_LIBRISQA_TRAIN)
    parser.add_argument("--test_json", type=str, default=DEFAULT_LIBRISQA_TEST)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_missing_audio", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("json", data_files={"train": args.train_json, "test": args.test_json})
    train_full = raw["train"]
    test_raw = raw["test"]

    split = train_full.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_raw = split["train"]
    val_raw = split["test"]

    if args.train_limit > 0:
        train_raw = train_raw.select(range(min(args.train_limit, len(train_raw))))
    if args.val_limit > 0:
        val_raw = val_raw.select(range(min(args.val_limit, len(val_raw))))
    if args.test_limit > 0:
        test_raw = test_raw.select(range(min(args.test_limit, len(test_raw))))

    librispeech_root = Path(args.librispeech_root)
    allow = bool(args.allow_missing_audio)
    preflight_librispeech_root(librispeech_root, allow_missing_audio=allow)

    train = convert_split(train_raw, librispeech_root, allow_missing_audio=allow)
    val = convert_split(val_raw, librispeech_root, allow_missing_audio=allow)
    test = convert_split(test_raw, librispeech_root, allow_missing_audio=allow)

    if len(train) == 0 and len(test) > 0:
        print(
            "WARNING: train split is empty after filtering. "
            f"Re-splitting {len(test)} available samples into 70/15/15."
        )
        split1 = test.train_test_split(test_size=0.30, seed=args.seed)
        train = split1["train"]
        rest = split1["test"]
        split2 = rest.train_test_split(test_size=0.50, seed=args.seed)
        val = split2["train"]
        test = split2["test"]

    dataset = DatasetDict({"train": train, "validation": val, "test": test})
    dataset.save_to_disk(str(out_dir / "dataset"))

    print(f"Saved LibriSQA dataset to: {out_dir / 'dataset'}")
    print(f"Sizes: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
