#!/usr/bin/env bash
# ===========================================================================
# Launcher entrypoint: environment setup + dispatch to scripts/speechrl.py.
#
#   bash scripts/speechrl.sh train --method leaf --model granite3_2b --dataset librisqa
#   bash scripts/speechrl.sh train --method grpo --model granite3_2b --dataset librisqa
#   bash scripts/speechrl.sh eval  --model granite3_2b --dataset librisqa --run <run_dir> --mode all
#   <any train/eval> --dry-run     # print the command; no env setup, no run
#
# Env overrides: CONDA_ENV (conda prefix to activate; optional), PYTHON_BIN, CUDA_MODULE,
#                SPEECH_DATA_ROOT (prepared datasets; default <repo>/data), JOB_TMP_ROOT.
# For train/eval this activates the conda env (if given), sets NCCL/temp dirs and puts THIS
# repo first on PYTHONPATH so the driver and the Ray workers import `openrlhf` from here.
# ===========================================================================
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"     # scripts/
REPO_ROOT="$(cd -- "$HERE/.." && pwd)"
export SPEECH_DATA_ROOT="${SPEECH_DATA_ROOT:-$REPO_ROOT/data}"

LIGHT=0
for a in "$@"; do case "$a" in --dry-run) LIGHT=1 ;; esac; done

if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
  export PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV/bin/python}"
fi
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [[ "$LIGHT" == "0" ]]; then
  if command -v module >/dev/null 2>&1 && [[ -n "${CUDA_MODULE:-}" ]]; then
    module load "$CUDA_MODULE" || true
  fi
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}"
  fi
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  JOB_TMP_ROOT="${JOB_TMP_ROOT:-${TMPDIR:-/tmp}/$USER/speechrl_${SLURM_JOB_ID:-$$}}"
  export RAY_TMPDIR="${RAY_TMPDIR:-$JOB_TMP_ROOT/ray}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$JOB_TMP_ROOT/triton-cache}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$JOB_TMP_ROOT/torch_extensions}"
  mkdir -p "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
  unset RAY_ADDRESS
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "[speechrl] python=$PYTHON_BIN  openrlhf -> $("$PYTHON_BIN" -c 'import openrlhf; print(openrlhf.__file__)' 2>/dev/null || echo '?')"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" scripts/speechrl.py "$@"
