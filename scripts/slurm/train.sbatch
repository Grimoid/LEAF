#!/bin/bash
# Example Slurm submission for one LEAF / GRPO training run (3 GPUs: 2 actors + 1 reference).
# Submit with:   METHOD=leaf MODEL=granite3_2b DATASET=librisqa sbatch scripts/slurm/train.sbatch
# Resubmitting with the same OUT dir resumes from the latest DeepSpeed checkpoint.
#SBATCH --job-name=leaf_speech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gpus=3
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x_%j.log
#SBATCH --open-mode=append
set -euo pipefail

METHOD="${METHOD:-leaf}"            # leaf | grpo
MODEL="${MODEL:-granite3_2b}"
DATASET="${DATASET:-librisqa}"
OUT="${OUT:-ckpts/${METHOD}_${MODEL}_${DATASET}}"    # fixed dir so resubmissions resume
EXTRA="${EXTRA:-}"                  # e.g. "--set num_episodes=3 --set prefix_tree_B=4"

export NCCL_TIMEOUT=7200000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export OPENRLHF_DIST_TIMEOUT_MINUTES=180
export CONDA_ENV="${CONDA_ENV:-$PWD/.conda}"
export SPEECH_DATA_ROOT="${SPEECH_DATA_ROOT:-$PWD/data}"
export JOB_TMP_ROOT="${JOB_TMP_ROOT:-${TMPDIR:-/tmp}/$USER/leaf_${SLURM_JOB_ID}}"
mkdir -p slurm_logs

bash scripts/speechrl.sh train --method "$METHOD" --model "$MODEL" --dataset "$DATASET" \
  --out "$OUT" --log $EXTRA
