#!/usr/bin/env bash
# Prepare the merged speech_longaudio dataset (VoxPopuli + Europarl).
#
# Runs both individual prepare scripts (if datasets not already built),
# then merges them into speech_longaudio.
#
# DailyTalk is intentionally excluded from this merged dataset.
#
# Usage:
#   bash scripts/data_prep/download_and_prepare_speech_longaudio.sh
#
# Key environment variables:
#   VOXPOPULI_DIR        (default: data/voxpopuli_longaudio)
#   EUROPARL_DIR         (default: data/europarl_longaudio)
#   OUT_DIR              (default: data/speech_longaudio)
#   SKIP_VOXPOPULI       set to 1 if voxpopuli_longaudio already prepared
#   SKIP_EUROPARL        set to 1 if europarl_longaudio already prepared
#   EUROPARL_ST_DIR      (default: $SPEECH_DATA_ROOT/raw/EuroparlST)
#   SKIP_DOWNLOAD        set to 1 to skip Europarl-ST download
#   MAX_DURATION_SECS    (default: 40)
#   FILTER_MCQ           (default: 1)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

VOXPOPULI_DIR="${VOXPOPULI_DIR:-${SPEECH_DATA_ROOT:-$REPO_ROOT/data}/voxpopuli_longaudio}"
EUROPARL_DIR="${EUROPARL_DIR:-${SPEECH_DATA_ROOT:-$REPO_ROOT/data}/europarl_longaudio}"
OUT_DIR="${OUT_DIR:-${SPEECH_DATA_ROOT:-$REPO_ROOT/data}/speech_longaudio}"
SKIP_VOXPOPULI="${SKIP_VOXPOPULI:-0}"
SKIP_EUROPARL="${SKIP_EUROPARL:-0}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

echo "========================================================"
echo "speech_longaudio = voxpopuli_longaudio + europarl_longaudio"
echo "  VOXPOPULI_DIR=$VOXPOPULI_DIR"
echo "  EUROPARL_DIR=$EUROPARL_DIR"
echo "  OUT_DIR=$OUT_DIR"
echo "========================================================"

# ── Step 1: Prepare VoxPopuli ─────────────────────────────────────────────────
if [[ "$SKIP_VOXPOPULI" == "1" ]] || [[ -d "$VOXPOPULI_DIR/dataset" ]]; then
    echo "VoxPopuli-LongAudio already prepared — skipping."
else
    echo ""
    echo "=== Preparing VoxPopuli-LongAudio ==="
    OUT_DIR="$VOXPOPULI_DIR" bash "$SCRIPT_DIR/download_and_prepare_voxpopuli_longaudio.sh"
fi

# ── Step 2: Prepare Europarl ──────────────────────────────────────────────────
if [[ "$SKIP_EUROPARL" == "1" ]] || [[ -d "$EUROPARL_DIR/dataset" ]]; then
    echo "Europarl-LongAudio already prepared — skipping."
else
    echo ""
    echo "=== Preparing Europarl-LongAudio ==="
    OUT_DIR="$EUROPARL_DIR" bash "$SCRIPT_DIR/download_and_prepare_europarl_longaudio.sh"
fi

# ── Step 3: Merge ─────────────────────────────────────────────────────────────
echo ""
echo "=== Merging into speech_longaudio ==="
python scripts/data_prep/prepare_speech_longaudio.py \
    --voxpopuli_dir "$VOXPOPULI_DIR" \
    --europarl_dir  "$EUROPARL_DIR" \
    --out_dir       "$OUT_DIR"

echo ""
echo "Done. Merged dataset saved to: $OUT_DIR/dataset"
