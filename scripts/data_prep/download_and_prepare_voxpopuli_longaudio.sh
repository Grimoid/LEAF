#!/usr/bin/env bash
# Download and prepare the VoxPopuli-LongAudio dataset.
#
# Audio retrieval is two-stage:
#   Stage 1 — facebook/voxpopuli HF parquet shards (labeled English subset, ~15% of IDs)
#   Stage 2 — dl.fbaipublicfiles.com per-year tarballs (~5-10 GB each, 2009-2020)
#             Only years containing missing IDs are downloaded.
#
# Usage:
#   bash scripts/data_prep/download_and_prepare_voxpopuli_longaudio.sh
#
# Key environment variables:
#   OUT_DIR              (default: data/voxpopuli_longaudio)
#   TARBALL_DIR          (default: OUT_DIR/tarballs)   temp storage for year tarballs
#   MAX_DURATION_SECS    (default: 40)
#   FILTER_MCQ           (default: 1)
#   ALLOW_MISSING_AUDIO  (default: 0)
#   SKIP_PARQUET         (default: 0)  set to 1 to skip HF parquet scan entirely
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

OUT_DIR="${OUT_DIR:-${SPEECH_DATA_ROOT:-$REPO_ROOT/data}/voxpopuli_longaudio}"
TARBALL_DIR="${TARBALL_DIR:-$OUT_DIR/tarballs}"
MAX_DURATION_SECS="${MAX_DURATION_SECS:-40}"
FILTER_MCQ="${FILTER_MCQ:-1}"
ALLOW_MISSING_AUDIO="${ALLOW_MISSING_AUDIO:-0}"
SKIP_PARQUET="${SKIP_PARQUET:-0}"

# Use scratch for HF hub cache — parquet shards are ~3 GB each and /tmp is too small.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TARBALL_DIR"

echo "========================================"
echo "VoxPopuli-LongAudio dataset preparation"
echo "  OUT_DIR=$OUT_DIR"
echo "  TARBALL_DIR=$TARBALL_DIR"
echo "  MAX_DURATION_SECS=$MAX_DURATION_SECS"
echo "  FILTER_MCQ=$FILTER_MCQ"
echo "  SKIP_PARQUET=$SKIP_PARQUET"
echo "========================================"

python scripts/data_prep/prepare_voxpopuli_longaudio.py \
    --out_dir "$OUT_DIR" \
    --tarball_dir "$TARBALL_DIR" \
    --max_duration_secs "$MAX_DURATION_SECS" \
    --filter_mcq "$FILTER_MCQ" \
    --allow_missing_audio "$ALLOW_MISSING_AUDIO" \
    --skip_parquet "$SKIP_PARQUET"

echo ""
echo "Done. Dataset saved to: $OUT_DIR/dataset"
