#!/usr/bin/env bash
# Run training inside WSL2 (conda env: ai)
# Called by Task Scheduler via reg-service.ps1
set -euo pipefail

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate ai
exec gpt_nipc_train "$@"
