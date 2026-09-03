#!/usr/bin/env bash
# Download Europarl-ST v1.1 and prepare the Europarl-LongAudio dataset.
#
# Audio source: https://www.mllp.upv.es/europarl-st/v1.1.tar.gz  (~20 GB)
# License     : CC BY-NC 4.0
#
# Usage:
#   bash scripts/data_prep/download_and_prepare_europarl_longaudio.sh
#
# Key environment variables:
#   EUROPARL_ST_DIR      (default: $SPEECH_DATA_ROOT/raw/EuroparlST)
#   OUT_DIR              (default: data/europarl_longaudio)
#   SKIP_DOWNLOAD        set to 1 if Europarl-ST is already downloaded
#   MAX_DURATION_SECS    (default: 40)
#   FILTER_MCQ           (default: 1)
#   ALLOW_MISSING_AUDIO  (default: 0)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

EUROPARL_ST_DIR="${EUROPARL_ST_DIR:-${SPEECH_DATA_ROOT:-$(pwd)/data}/raw/EuroparlST}"
OUT_DIR="${OUT_DIR:-${SPEECH_DATA_ROOT:-$REPO_ROOT/data}/europarl_longaudio}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
MAX_DURATION_SECS="${MAX_DURATION_SECS:-40}"
FILTER_MCQ="${FILTER_MCQ:-1}"
ALLOW_MISSING_AUDIO="${ALLOW_MISSING_AUDIO:-0}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$EUROPARL_ST_DIR"

echo "========================================"
echo "Europarl-LongAudio dataset preparation"
echo "  EUROPARL_ST_DIR=$EUROPARL_ST_DIR"
echo "  OUT_DIR=$OUT_DIR"
echo "  MAX_DURATION_SECS=$MAX_DURATION_SECS"
echo "  FILTER_MCQ=$FILTER_MCQ"
echo "========================================"

# ── Step 1: Download Europarl-ST ─────────────────────────────────────────────
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    echo "SKIP_DOWNLOAD=1 — skipping Europarl-ST download."
else
    TARBALL="$EUROPARL_ST_DIR/v1.1.tar.gz"
    if [[ -f "$TARBALL" ]]; then
        echo "Tarball already exists: $TARBALL — skipping download."
    else
        echo "Downloading Europarl-ST v1.1 (~20 GB) ..."
        wget -c -O "$TARBALL" "https://www.mllp.upv.es/europarl-st/v1.1.tar.gz"
        echo "Download complete."
    fi

    echo "Extracting Europarl-ST ..."
    tar -xzf "$TARBALL" -C "$EUROPARL_ST_DIR"
    echo "Extraction complete."
fi

# ── Step 2: Prepare dataset ───────────────────────────────────────────────────
echo ""
echo "Preparing Europarl-LongAudio dataset ..."
python scripts/data_prep/prepare_europarl_longaudio.py \
    --europarl_st_dir "$EUROPARL_ST_DIR" \
    --out_dir "$OUT_DIR" \
    --max_duration_secs "$MAX_DURATION_SECS" \
    --filter_mcq "$FILTER_MCQ" \
    --allow_missing_audio "$ALLOW_MISSING_AUDIO"

echo ""
echo "Done. Dataset saved to: $OUT_DIR/dataset"
