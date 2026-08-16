#!/usr/bin/env bash
# Run training inside WSL2 (conda env: ai)
# Called by start.ps1 via Task Scheduler (SYSTEM account)

set -euo pipefail

PROJECT_DIR="/home/george/source/ai/word-gpt-mini"
CONDA_BASE="/home/george/miniconda3"
CONFIG="$PROJECT_DIR/wsl/train_gpt.json"
LOG_DIR="/mnt/e/training/logs"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

echo "[$(date)] Training start" >> "$LOG_FILE"
echo "Config: $CONFIG" >> "$LOG_FILE"
echo "Log:    $LOG_FILE" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ai

cd "$PROJECT_DIR"

python train_noipc_ddp.py "$CONFIG" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "---" >> "$LOG_FILE"
echo "[$(date)] Training exited with code $EXIT_CODE" >> "$LOG_FILE"

exit $EXIT_CODE