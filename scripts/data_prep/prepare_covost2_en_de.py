from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from datasets import Audio, DatasetDict, load_dataset


PROMPT_TEMPLATES = [
    "<|audio|> Listen to the speech and translate it into German.",
    "Translate the audio <|audio|> from English to German.",
    "You will hear English speech in <|audio|>. Provide the German translation.",
]


def stable_template_idx(text: str, n: int) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


def prepare_split(ds):
    def _map(ex):
        src = str(ex.get("sentence", "")).strip()
        tgt = str(ex.get("translation", "")).strip()
        prompt = PROMPT_TEMPLATES[stable_template_idx(src, len(PROMPT_TEMPLATES))]
        ex["task"] = "ast"
        ex["prompt"] = prompt
        ex["reference"] = tgt
        return ex

    return ds.map(_map)


def ensure_audio_column(ds):
    cols = set(ds.column_names)
    if "audio" in cols:
        return ds
    for candidate in ("path", "file", "audio_path"):
        if candidate in cols:
            ds = ds.rename_column(candidate, "audio")
            return ds.cast_column("audio", Audio(sampling_rate=16000))
    raise KeyError(
        "No audio column found in CoVoST2 split. Expected one of: "
        "'audio', 'path', 'file', 'audio_path'."
    )


def main():
    ap = argparse.ArgumentParser(description="Prepare CoVoST2 en->de dataset for the LEAF speech experiments.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory (dataset/ will be created inside)")
    ap.add_argument("--data_dir", type=str, required=True,
                    help="Path to unpacked Common Voice Corpus 4 English directory "
                         "(download from https://commonvoice.mozilla.org/en/datasets)")
    ap.add_argument("--config", type=str, default="en_de", help="CoVoST2 language pair config")
    ap.add_argument("--train_limit", type=int, default=0, help="Limit training samples (0=all)")
    ap.add_argument("--val_limit", type=int, default=0, help="Limit validation samples (0=all)")
    ap.add_argument("--test_limit", type=int, default=0, help="Limit test samples (0=all)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CoVoST2 ({args.config}) from HuggingFace (data_dir={args.data_dir})...", flush=True)
    train = load_dataset("facebook/covost2", args.config, split="train", data_dir=args.data_dir, trust_remote_code=True, verification_mode="no_checks")
    val = load_dataset("facebook/covost2", args.config, split="validation", data_dir=args.data_dir, trust_remote_code=True, verification_mode="no_checks")
    test = load_dataset("facebook/covost2", args.config, split="test", data_dir=args.data_dir, trust_remote_code=True, verification_mode="no_checks")

    if args.train_limit > 0:
        train = train.select(range(min(args.train_limit, len(train))))
    if args.val_limit > 0:
        val = val.select(range(min(args.val_limit, len(val))))
    if args.test_limit > 0:
        test = test.select(range(min(args.test_limit, len(test))))

    train = ensure_audio_column(train)
    val = ensure_audio_column(val)
    test = ensure_audio_column(test)

    train = prepare_split(train)
    val = prepare_split(val)
    test = prepare_split(test)

    ds = DatasetDict({"train": train, "validation": val, "test": test})
    ds.save_to_disk(str(out_dir / "dataset"))
    print(f"Saved CoVoST2 en->de dataset to: {out_dir / 'dataset'}")
    print(f"Sizes: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
