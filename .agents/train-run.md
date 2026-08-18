# Training Lifecycle Manager

## Description
Manages the full training lifecycle: pre-flight checks, cache building, trainer selection, launch, monitoring, and resume.

## Triggers
"start training", "resume training", "launch trainer", "check training status", "run training", "begin training"

## Workflow

### Step 1 — Verify config exists and is valid
```bash
python -m py_compile gpt_train.json  # not needed, it's JSON
python -c "import json; json.load(open('gpt_train.json'))"  # validate
```
If missing or broken, suggest `python train_designer.py --no-interact --force` to regenerate.

### Step 2 — Check cache status
```bash
# List existing cache files
Get-ChildItem "E:\training\cache" -Filter "vocab-*.json" | Select-Object Name, Length, LastWriteTime
Get-ChildItem "E:\training\cache" -Filter "data-*.npy" | Select-Object Name, Length, LastWriteTime
```
- If both exist and are recent → cache is warm, skip rebuild
- If either missing → need to build cache first

### Step 3 — Build cache if needed
```bash
python gpt_nipc_train.py --ensure-cache-only
```
Or for IPC systems:
```bash
python gpt_ipc_train.py --ensure-cache-only
```
This calls `ensure_cache_ready()` which builds vocab + data cache if missing.

### Step 4 — Select trainer
| GPU Topology | Trainer | Reason |
|---|---|---|
| RTX 3090 / PCIe bridge (PXB) | `gpt_nipc_train.py` | No P2P, CPU grad sync |
| V100 SXM2 / A100 / H100 (NVLink) | `gpt_ipc_train.py` | Full DDP GPU all_reduce |
| Windows, any GPU (experimental) | `experimental/train_rpc.py` | TCP sockets, no DDP |

**Default choice:** `gpt_nipc_train.py` (RTX 3090s in this project).

### Step 5 — Launch training
```bash
# Standard multi-GPU launch
python gpt_nipc_train.py -d 0,1

# Custom config, single GPU
python gpt_nipc_train.py -d 0 gpt_train.json

# Override epochs
python gpt_nipc_train.py -d 0,1 --epochs 20

# Force cache rebuild before training
python gpt_nipc_train.py -d 0,1 --force-cache
```

### Step 6 — Monitor training
```bash
# Tail checkpoint status log
$ckpt_dir = "E:\training\checkpoints"
$hash_dirs = Get-ChildItem $ckpt_dir -Directory | Sort-Object LastWriteTime -Descending
$latest = $hash_dirs[0]
if ($latest) {
    Get-Content "$ckpt_dir\$latest\checkpoint_status.txt" -Tail 20
}

# Check resume metadata
Get-Content "$ckpt_dir\$latest\resume.json"
```

### Step 7 — Resume (automatic, no extra action)
Trainer auto-resumes from latest checkpoint. Just re-run the same command:
```bash
python gpt_nipc_train.py -d 0,1
```
It reads `resume.json`, finds latest `model.{slot}.pth`, and continues from `global_batch`.

### Step 8 — Emergency stop
```bash
# Graceful: Ctrl+C triggers signal handlers, checkpoints current batch
# Force kill if hung:
taskkill /F /IM python.exe
```
After force-kill, next launch will resume from last saved checkpoint (not current batch).

## Key Files
| File | Purpose |
|------|---------|
| `gpt_train.json` | Working config with model/training/tokenizer/paths |
| `gpt_nipc_train.py` | CPU-sync trainer (RTX 3090) |
| `gpt_ipc_train.py` | Full DDP trainer (NVLink) |
| `experimental/train_rpc.py` | TCP-sync trainer (experimental) |
| `E:\training\cache\` | Cached vocab and dataset |
| `E:\training\checkpoints\` | Checkpoint directory |

## Common Issues
| Symptom | Fix |
|---------|-----|
| Segfault (0xC0000005) | Switch from `gpt_ipc_train.py` to `gpt_nipc_train.py` |
| OOM at startup | Reduce `batch_size` or `seq_length` in config |
| "Cache incomplete" warning | Trainer builds cache automatically; wait |
| Loss stuck at 5.0+ | Check `lr` is non-zero, verify dataset isn't all `<pad>` |
