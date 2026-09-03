#!/usr/bin/env bash
# Download DailyTalk audio from Google Drive and prepare the LongAudio QA dataset.
#
# DailyTalk audio is hosted on Google Drive.  This script uses gdown.
# Install gdown first if needed:
#   pip install gdown
#
# Usage:
#   bash scripts/data_prep/download_and_prepare_dailytalk_longaudio.sh
#
# Key env overrides:
#   DAILYTALK_DIR      – where to store downloaded DailyTalk raw audio
#   OUT_DIR            – prepared HF dataset output directory
#   SKIP_DOWNLOAD      – set to 1 if audio already downloaded
#   MAX_DURATION_SECS  – filter threshold in seconds (default 40)
#   ALLOW_MISSING_AUDIO – set to 1 to skip missing sessions (default 0)
set -euo pipefail

DAILYTALK_DIR="${DAILYTALK_DIR:-${SPEECH_DATA_ROOT:-$(pwd)/data}/raw/DailyTalk}"
OUT_DIR="${OUT_DIR:-${SPEECH_DATA_ROOT:-$(pwd)/data}/dailytalk_longaudio}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
MAX_DURATION_SECS="${MAX_DURATION_SECS:-40}"
ALLOW_MISSING_AUDIO="${ALLOW_MISSING_AUDIO:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Google Drive folder ID from the DailyTalk GitHub README
# https://github.com/keonlee9420/DailyTalk
GDRIVE_FOLDER_ID="1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL"

echo "=== DailyTalk-LongAudio Download + Prepare ==="
echo "DAILYTALK_DIR=$DAILYTALK_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "MAX_DURATION_SECS=$MAX_DURATION_SECS"

# ── Step 1: Download DailyTalk audio from Google Drive ───────────────────────
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  echo ""
  echo "SKIP_DOWNLOAD=1 — skipping download."
elif [[ -d "$DAILYTALK_DIR" ]] && [[ -n "$(ls -A "$DAILYTALK_DIR" 2>/dev/null)" ]]; then
  echo ""
  echo "DailyTalk directory already populated ($DAILYTALK_DIR) — skipping download."
else
  echo ""
  echo "Downloading DailyTalk audio from Google Drive ..."
  echo "  Folder ID: $GDRIVE_FOLDER_ID"
  mkdir -p "$DAILYTALK_DIR"

  # gdown --folder downloads all files in the Google Drive folder
  "$PYTHON_BIN" -m gdown \
    --folder "$GDRIVE_FOLDER_ID" \
    -O "$DAILYTALK_DIR"

  echo "Download complete: $DAILYTALK_DIR"

  # The folder may contain a zip; unzip if present
  for zipfile in "$DAILYTALK_DIR"/*.zip; do
    [[ -f "$zipfile" ]] || continue
    echo "Unzipping $zipfile ..."
    unzip -q "$zipfile" -d "$DAILYTALK_DIR"
    echo "Unzip complete."
  done
fi

# ── Step 2: Run prepare script ───────────────────────────────────────────────
echo ""
echo "Running prepare_dailytalk_longaudio.py ..."
"$PYTHON_BIN" scripts/data_prep/prepare_dailytalk_longaudio.py \
  --dailytalk_dir   "$DAILYTALK_DIR" \
  --out_dir         "$OUT_DIR" \
  --max_duration_secs "$MAX_DURATION_SECS" \
  --allow_missing_audio "$ALLOW_MISSING_AUDIO"

echo ""
echo "=== Done. Dataset saved to: ${OUT_DIR}/dataset ==="
