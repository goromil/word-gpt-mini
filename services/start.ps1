# Run training on Windows natively (conda env: ai)
# Called by Task Scheduler via reg-service.ps1

$ErrorActionPreference = "Stop"

$CONFIG = "$env:USERPROFILE\.config\wordgpt\gpt_train.json"

conda run -n ai --no-capture-output gpt_nipc_train $CONFIG
