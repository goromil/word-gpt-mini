# Model & Training Debugger

## Description
Diagnoses training problems: loss anomalies, crashes, OOM errors, gradient issues, and hardware compatibility.

## Triggers
"loss not decreasing", "NaN loss", "OOM", "crash", "segfault", "analyze training", "debug training", "training problem", "gradient issue"

## Workflow

### Step 1 — Read checkpoint status log
```bash
$ckpt_dir = "E:\training\checkpoints"
$hash_dirs = Get-ChildItem $ckpt_dir -Directory | Sort-Object LastWriteTime -Descending
$latest = $hash_dirs[0]
if ($latest) {
    Write-Host "=== Latest checkpoint: $latest ==="
    Get-Content "$ckpt_dir\$latest\checkpoint_status.txt" -Tail 50
    Write-Host "=== Resume metadata ==="
    Get-Content "$ckpt_dir\$latest\resume.json"
}
```

### Step 2 — Diagnose by symptom

#### NaN or infinite loss
```bash
# Verify forward/backward pass on a single batch
python test_one_step.py
```
- If test_one_step works → issue is in training loop or multi-GPU sync
- If test_one_step produces NaN → issue is in model or config

Check for:
- `lr` too high (> 0.001 for BPE models)
- Gradient overflow → check `sync.gradient_accumulation_steps`
- Corrupt cache → delete `E:\training\cache\data-*.npy`, rebuild

#### Loss stuck at 5.0+ (log(32768) ≈ 10.4, log(65536) ≈ 11.1)
Expected loss curve:
- Epoch 1: 7-10 (random weights)
- Epoch 5: 4-6
- Epoch 10+: 3-4 (depends on corpus quality)

If stuck:
1. Check `lr` is not zero
2. Check `batch_size` is reasonable (16-128)
3. Verify dataset has real tokens, not all `<pad>`:
```bash
python -c "
import numpy as np, json
cfg = json.load(open('gpt_train.json'))
d = np.load(cfg['paths']['cache_dir'] + '/data-*.npy')
print(f'Unique tokens: {len(np.unique(d))}')
print(f'Zeros (pad): {(d == 0).sum() / d.size * 100:.1f}%')
print(f'Non-zero: {(d != 0).sum() / d.size * 100:.1f}%')
"
```

#### OOM (CUDA out of memory)
```bash
# Check VRAM usage vs config
python train_designer.py --scan-vocab  # shows memory analysis
```
Formula from `train_designer.py`:
```
GPU_mem = fixed_gb + var_gb
fixed_gb = params * 12 / world_size / 1024^3  (weights + grad + Adam)
var_gb   = bs_per_gpu * seq_length * K * n_layer * 1.45 / 1024^3
K        = 13 * n_embd + 4 * n_heads * seq_length
```
Quick fixes (in order of impact):
1. Reduce `seq_length` (biggest impact — attention matrix is O(seq^2))
2. Reduce `batch_size`
3. Reduce `n_layer`
4. Increase `sync.gradient_accumulation_steps` to compensate for smaller batches

#### Segfault (0xC0000005) on startup
**Root cause:** CUDA IPC on PXB/PCIe-bridge GPUs.
**Fix:** Switch from `gpt_ipc_train.py` to `gpt_nipc_train.py`.

```bash
# Wrong (will segfault on RTX 3090):
python gpt_ipc_train.py -d 0,1

# Correct for RTX 3090:
python gpt_nipc_train.py -d 0,1
```

#### Gradient sync failures
Check sync method in config:
- `"method": "cpu"` — CPU all_reduce (noIPC trainer)
- `"method": "gpu"` — GPU all_reduce (IPC trainer, requires P2P)
- `"chunks": 4` — gradient tensor split into chunks for memory efficiency

If sync hangs:
```bash
# Check port conflicts
netstat -ano | findstr "29500"
```
Trainer uses `127.0.0.1:29500` as master address on Windows.

### Step 3 — Verify model architecture
```bash
# Check parameter count
python calc_params.py gpt_train.json

# Verify model loads
python -c "
import json, torch
from gpt_train import GPTMini
cfg = json.load(open('gpt_train.json'))
model_cfg = dict(cfg['model'])
model_cfg.pop('tokenizer', None)
model = GPTMini(model_cfg, 65536)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
print(f'Layers: {model_cfg[\"n_layer\"]}')
print(f'Heads: {model_cfg[\"n_head\"]}')
print(f'Head dim: {model_cfg[\"head_dim\"]}')
print(f'Seq len: {model_cfg[\"seq_length\"]}')
"
```

### Step 4 — Run validation tests
```bash
python test_gpt_train.py
```
Expected: 71/71 pass, 0 fail.

## Diagnostic Checklist
| Symptom | First Check | Likely Fix |
|---------|-------------|-----------|
| Segfault on launch | GPU topology | Switch to `gpt_nipc_train.py` |
| OOM at batch N | `seq_length`, `batch_size` | Reduce one, keep other |
| NaN loss after epoch K | `lr` too high | Halve `lr`, resume |
| Loss stuck at start | Dataset content | Rebuild cache, check for corrupt data |
| Slow training (low tok/s) | `batch_size` vs GPU count | Increase batch, reduce grad_accum |
| Gradient sync hang | Port 29500, world_size | Kill stale processes, re-launch |
| Checkpoint not resuming | `ckpt_hash` mismatch | Regenerate config, verify hash |
