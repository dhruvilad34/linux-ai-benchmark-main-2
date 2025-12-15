#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_benchmark.sh --num_tasks 3 --num_agents 10
# Exports CUDA_VISIBLE_DEVICES ahead of time so the Python process
# only sees the selected GPUs.

GPU_LIST=${CUDA_VISIBLE_DEVICES:-"1,2,3"}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

echo "[run_benchmark] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

source "./venv/bin/activate"

# Keep external services quiet for performance-sensitive runs.
export DISABLE_WEAVE=true
export WANDB_MODE=offline

python main.py --config config/config.yaml "$@"



