# Run training on Windows natively (conda env: ai)
# Called by Task Scheduler (SYSTEM account)

$ErrorActionPreference = "Stop"

$PROJECT_DIR = "C:\Users\george\source\ai\word-gpt-mini"
$CONFIG = "$PROJECT_DIR\train_gpt.json"
$LOG_DIR = "E:\training\logs"
$LOG_FILE = "$LOG_DIR\train_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null
}

"[$(Get-Date)] Training start" | Add-Content $LOG_FILE
"Config: $CONFIG" | Add-Content $LOG_FILE
"Log:    $LOG_FILE" | Add-Content $LOG_FILE
"---" | Add-Content $LOG_FILE

conda run -n ai --no-capture-output -p $PROJECT_DIR python train_noipc_ddp.py $CONFIG *>> $LOG_FILE
$exitCode = $LASTEXITCODE

"---" | Add-Content $LOG_FILE
"[$(Get-Date)] Training exited with code $exitCode" | Add-Content $LOG_FILE

exit $exitCode