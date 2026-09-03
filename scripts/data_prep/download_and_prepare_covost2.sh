#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Download and prepare CoVoST2 en->de dataset for the LEAF speech experiments.
#
# CoVoST2 is built on Mozilla Common Voice audio. This script handles:
#   1. Downloading Common Voice English audio (via Mozilla Data Collective)
#   2. Downloading CoVoST2 translation metadata (from Facebook, public)
#   3. Building the final dataset in our standard format
#
# === SETUP (one-time) ===
#
#   pip install datacollective
#
#   Then get an API key:
#     1. Sign up at https://datacollective.mozillafoundation.org/
#     2. Go to Account -> Credentials -> create API key
#     3. Export it:  export MDC_API_KEY="your-key-here"
#
# === USAGE ===
#
#   # Full automatic download + preparation:
#   MDC_API_KEY="your-key" bash scripts/data_prep/download_and_prepare_covost2.sh
#
#   # If you already have Common Voice extracted somewhere:
#   CV_ROOT=/path/to/cv-corpus/en bash scripts/data_prep/download_and_prepare_covost2.sh
#
#   # Quick smoke test with limited samples:
#   TRAIN_LIMIT=100 VAL_LIMIT=20 TEST_LIMIT=20 \
#     MDC_API_KEY="your-key" bash scripts/data_prep/download_and_prepare_covost2.sh
#
# === ENVIRONMENT VARIABLES ===
#   MDC_API_KEY   - Mozilla Data Collective API key (needed for CV download)
#   CV_ROOT       - Skip download: path to already-extracted Common Voice dir
#                   (must contain validated.tsv and clips/ directory)
#   CV_ARCHIVE    - Skip download: path to Common Voice .tar.gz (will extract)
#   OUT_DIR       - Output directory (default: data/covost2_en_de)
#   WORK_DIR      - Temp working directory
#                   (default: ${TMPDIR:-/tmp}/covost2_prep)
#   TRAIN_LIMIT   - Limit training samples, 0=all (default: 0)
#   VAL_LIMIT     - Limit validation samples, 0=all (default: 0)
#   TEST_LIMIT    - Limit test samples, 0=all (default: 0)
# =============================================================================

CV_ROOT="${CV_ROOT:-}"
CV_ARCHIVE="${CV_ARCHIVE:-}"
OUT_DIR="${OUT_DIR:-${SPEECH_DATA_ROOT:-$(pwd)/data}/covost2_en_de}"
WORK_DIR="${WORK_DIR:-${TMPDIR:-/tmp}/covost2_prep}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
VAL_LIMIT="${VAL_LIMIT:-0}"
TEST_LIMIT="${TEST_LIMIT:-0}"

# Common Voice Scripted Speech 24.0 - English on Mozilla Data Collective
# https://datacollective.mozillafoundation.org/datasets/cmj8u3p1w0075nxxbe8bedl00
MDC_DATASET_ID="cmj8u3p1w0075nxxbe8bedl00"

mkdir -p "$WORK_DIR" "$OUT_DIR"

echo "============================================="
echo "  CoVoST2 en->de Download & Preparation"
echo "============================================="
echo "OUT_DIR:  $OUT_DIR"
echo "WORK_DIR: $WORK_DIR"
echo ""

# ---- Resolve Common Voice directory ----
if [[ -n "$CV_ROOT" ]]; then
  # User provided an already-extracted directory
  if [[ ! -f "$CV_ROOT/validated.tsv" ]]; then
    echo "ERROR: CV_ROOT=$CV_ROOT does not contain validated.tsv"
    echo "Expected: $CV_ROOT/validated.tsv and $CV_ROOT/clips/*.mp3"
    exit 1
  fi
  echo "[Common Voice] Using pre-extracted directory: $CV_ROOT"

elif [[ -n "$CV_ARCHIVE" ]]; then
  # User provided a .tar.gz — extract it
  if [[ ! -f "$CV_ARCHIVE" ]]; then
    echo "ERROR: CV_ARCHIVE=$CV_ARCHIVE does not exist"
    exit 1
  fi
  CV_EXTRACT_DIR="$WORK_DIR/common_voice"
  echo "[Common Voice] Extracting $CV_ARCHIVE ..."
  mkdir -p "$CV_EXTRACT_DIR"
  tar xzf "$CV_ARCHIVE" -C "$CV_EXTRACT_DIR"
  CV_ROOT=$(find "$CV_EXTRACT_DIR" -name "validated.tsv" -printf '%h\n' -quit 2>/dev/null || true)
  if [[ -z "$CV_ROOT" ]]; then
    echo "ERROR: Could not find validated.tsv after extracting."
    exit 1
  fi
  echo "[Common Voice] Found data at: $CV_ROOT"

else
  # Download via Mozilla Data Collective
  echo "[Common Voice] Downloading via Mozilla Data Collective..."
  echo "  Dataset: Common Voice Scripted Speech 24.0 - English (87.74 GB)"
  echo "  ID: $MDC_DATASET_ID"
  echo ""

  if [[ -z "${MDC_API_KEY:-}" ]]; then
    echo "ERROR: MDC_API_KEY is not set."
    echo ""
    echo "To download Common Voice automatically, you need a Mozilla Data Collective API key:"
    echo "  1. Sign up at https://datacollective.mozillafoundation.org/"
    echo "  2. Go to Account -> Credentials -> create API key"
    echo "  3. Re-run with: MDC_API_KEY=\"your-key\" bash $0"
    echo ""
    echo "Alternatively, if you already have Common Voice audio:"
    echo "  CV_ROOT=/path/to/extracted/en bash $0"
    echo "  CV_ARCHIVE=/path/to/en.tar.gz bash $0"
    exit 1
  fi

  # Download via MDC REST API (curl-based, as recommended by MDC website)
  CV_DOWNLOAD_DIR="$WORK_DIR/mdc_download"
  mkdir -p "$CV_DOWNLOAD_DIR"
  CV_TAR="$CV_DOWNLOAD_DIR/mcv-scripted-en-v24.0.tar.gz"

  if [[ -f "$CV_TAR" ]]; then
    echo "  Archive already downloaded: $CV_TAR"
  else
    echo "  Getting presigned download URL..."
    RESPONSE=$(curl -s -X POST \
      "https://datacollective.mozillafoundation.org/api/datasets/$MDC_DATASET_ID/download" \
      -H "Authorization: Bearer $MDC_API_KEY" \
      -H "Content-Type: application/json")

    # Check for errors
    if echo "$RESPONSE" | grep -q '"error"'; then
      echo "ERROR: MDC API returned an error:"
      echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
      echo ""
      echo "Make sure you have accepted the dataset terms at:"
      echo "  https://datacollective.mozillafoundation.org/datasets/$MDC_DATASET_ID"
      exit 1
    fi

    DOWNLOAD_URL=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['downloadUrl'])" 2>/dev/null)
    if [[ -z "$DOWNLOAD_URL" || "$DOWNLOAD_URL" == "null" ]]; then
      echo "ERROR: Could not extract download URL from API response:"
      echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
      exit 1
    fi

    echo "  Downloading (~88GB, this will take a while)..."
    wget -q --show-progress -O "$CV_TAR" "$DOWNLOAD_URL"
    echo "  -> Downloaded: $CV_TAR"
  fi

  # Extract the archive
  echo "[Common Voice] Extracting archive..."
  CV_EXTRACT_DIR="$WORK_DIR/common_voice"
  mkdir -p "$CV_EXTRACT_DIR"
  tar xf "$CV_TAR" -C "$CV_EXTRACT_DIR"
  CV_ROOT=$(find "$CV_EXTRACT_DIR" -name "validated.tsv" -printf '%h\n' -quit 2>/dev/null || true)
  if [[ -z "$CV_ROOT" ]]; then
    echo "ERROR: Could not find validated.tsv after extracting."
    echo "Contents:"
    find "$CV_EXTRACT_DIR" -maxdepth 3 | head -20
    exit 1
  fi
  echo "[Common Voice] Found data at: $CV_ROOT"
fi

# Verify Common Voice directory
if [[ ! -f "$CV_ROOT/validated.tsv" ]]; then
  echo "ERROR: $CV_ROOT/validated.tsv not found"
  exit 1
fi
echo "[Common Voice] validated.tsv: OK"
if [[ -d "$CV_ROOT/clips" ]]; then
  NUM_CLIPS=$(find "$CV_ROOT/clips" -name "*.mp3" 2>/dev/null | head -5 | wc -l)
  echo "[Common Voice] clips/ directory: $([ "$NUM_CLIPS" -gt 0 ] && echo 'OK (has mp3 files)' || echo 'WARNING: no mp3 files found')"
else
  echo "WARNING: $CV_ROOT/clips/ not found — audio may be missing"
fi
echo ""

# ---- Step 1: Download CoVoST2 translation TSV ----
COVOST_TSV_URL="https://dl.fbaipublicfiles.com/covost/covost_v2.en_de.tsv.tar.gz"
COVOST_TSV_TAR="$WORK_DIR/covost_v2.en_de.tsv.tar.gz"
COVOST_TSV="$WORK_DIR/covost_v2.en_de.tsv"

if [[ -f "$COVOST_TSV" ]]; then
  echo "[Step 1/2] CoVoST2 TSV already downloaded."
else
  echo "[Step 1/2] Downloading CoVoST2 en->de translation metadata..."
  wget -q --show-progress -O "$COVOST_TSV_TAR" "$COVOST_TSV_URL"
  tar xzf "$COVOST_TSV_TAR" -C "$WORK_DIR"
  echo "  -> Done"
fi
echo ""

# ---- Step 2: Build dataset using facebook/covost2 HF loader ----
echo "[Step 2/2] Building dataset with facebook/covost2 loader..."
echo ""

python3 - "$CV_ROOT" "$OUT_DIR" "$TRAIN_LIMIT" "$VAL_LIMIT" "$TEST_LIMIT" <<'PYEOF'
from __future__ import annotations

import hashlib
import sys
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


def add_fields(example):
    src = str(example.get("sentence", "")).strip()
    tgt = str(example.get("translation", "")).strip()
    prompt = PROMPT_TEMPLATES[stable_template_idx(src, len(PROMPT_TEMPLATES))]
    example["task"] = "ast"
    example["prompt"] = prompt
    example["reference"] = tgt
    return example


def main():
    cv_root = sys.argv[1]
    out_dir = sys.argv[2]
    train_limit = int(sys.argv[3])
    val_limit = int(sys.argv[4])
    test_limit = int(sys.argv[5])

    print(f"Loading facebook/covost2 en_de (data_dir={cv_root})...", flush=True)
    train = load_dataset("facebook/covost2", "en_de", split="train",
                         data_dir=cv_root, trust_remote_code=True)
    val = load_dataset("facebook/covost2", "en_de", split="validation",
                       data_dir=cv_root, trust_remote_code=True)
    test = load_dataset("facebook/covost2", "en_de", split="test",
                        data_dir=cv_root, trust_remote_code=True)
    print(f"  Raw sizes: train={len(train)} val={len(val)} test={len(test)}", flush=True)

    if train_limit > 0:
        train = train.select(range(min(train_limit, len(train))))
    if val_limit > 0:
        val = val.select(range(min(val_limit, len(val))))
    if test_limit > 0:
        test = test.select(range(min(test_limit, len(test))))

    print("Adding prompt/reference fields...", flush=True)
    train = train.map(add_fields)
    val = val.map(add_fields)
    test = test.map(add_fields)

    ds = DatasetDict({"train": train, "validation": val, "test": test})
    save_path = str(Path(out_dir) / "dataset")
    ds.save_to_disk(save_path)
    print(f"\nSaved CoVoST2 en->de dataset to: {save_path}", flush=True)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
PYEOF

echo ""
echo "============================================="
echo "  Preparation complete!"
echo "  Dataset saved to: $OUT_DIR/dataset"
echo "============================================="
