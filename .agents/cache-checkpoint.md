# Cache & Checkpoint Inspector

## Description
Inspects, validates, and manages cache files and checkpoints: hash chain verification, tier inspection, stale cache detection, cleanup.

## Triggers
"check cache", "checkpoint info", "resume from", "cache invalidation", "list checkpoints", "cache status", "checkpoint analysis"

## Workflow

### Step 1 — List cache files with sizes
```bash
Write-Host "=== Vocab Cache ==="
Get-ChildItem "E:\training\cache" -Filter "vocab-*.json" | ForEach-Object {
    $size = [math]::Round($_.Length / 1KB, 2)
    Write-Host "$($_.Name) ($size KB, $($_.LastWriteTime))"
}

Write-Host "`n=== Dataset Cache ==="
Get-ChildItem "E:\training\cache" -Filter "data-*.npy" | ForEach-Object {
    $size = [math]::Round($_.Length / 1GB, 2)
    Write-Host "$($_.Name) ($size GB, $($_.LastWriteTime))"
}
```

### Step 2 — Decode cache hash chain
```bash
python -c "
import json
from gpt_mini3 import get_vocab_hash, compute_corpus_hash
from pathlib import Path

cfg = json.load(open('gpt_mini3.json'))
dirs = [cfg['paths']['data_dir']]

# Compute current hashes
vh = get_vocab_hash(cfg['tokenizer'], dirs)
ch = compute_corpus_hash(dirs)
print(f'Current vocab_hash: {vh}')
print(f'Current corpus_hash: {ch}')

# Match against cached files
cache_dir = Path(cfg['paths']['cache_dir'])
vocab_match = list(cache_dir.glob(f'vocab-{vh}.json'))
data_match = list(cache_dir.glob(f'data-{vh}-{ch}.npy'))
print(f'Vocab cache hit: {bool(vocab_match)}')
print(f'Data cache hit: {bool(data_match)}')

# List all cache hashes for comparison
print(f'`nAll vocab caches:')
for f in sorted(cache_dir.glob('vocab-*.json')):
    h = f.stem.replace('vocab-', '')
    print(f'  {h}')
print(f'All data caches:')
for f in sorted(cache_dir.glob('data-*.npy')):
    # data-{vocab_hash}-{corpus_hash}.npy
    stem = f.stem
    parts = stem.split('-')
    vh_cached = parts[1]
    ch_cached = parts[2]
    sz = f'{f.stat().st_size / 1e9:.1f}GB'
    print(f'  vocab={vh_cached} corpus={ch_cached} ({sz})')
"
```

### Step 3 — Inspect checkpoint tiers
```bash
$ckpt_dir = "E:\training\checkpoints"
Get-ChildItem $ckpt_dir -Directory | ForEach-Object {
    $hash = $_.Name
    $tiers = Get-ChildItem $_.FullName -Directory | ForEach-Object { $_.Name }
    $latest = Get-ChildItem $_.FullName -Filter "model.*.pth" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $resume = Get-Content "$($_.FullName)\resume.json" -ErrorAction SilentlyContinue
    Write-Host "=== $hash ==="
    Write-Host "  Tiers: $($tiers -join ', ')"
    Write-Host "  Latest model: $($latest.Name if $latest else 'none')"
    Write-Host "  Resume: $resume"
    Write-Host ""
}
```

### Step 4 — Checkpoint tier structure
| Tier | Save Interval | Directory |
|------|--------------|-----------|
| 0 (base) | Every checkpoint | `checkpoints/{hash}/` |
| 1 | Every 10 epochs | `checkpoints/{hash}/1/` |
| 2 | Every 100 epochs | `checkpoints/{hash}/2/` |
| 3 | Every 1000 epochs | `checkpoints/{hash}/3/` |
| 4 | Every 10000 epochs | `checkpoints/{hash}/4/` |

Base tier uses slot rotation: `model.0.pth` / `model.1.pth` (slot = epoch & 1).
Higher tiers use static `model.pth`.

### Step 5 — Validate hash chain integrity
```bash
python -c "
import json, hashlib
from gpt_mini3 import get_vocab_hash, compute_corpus_hash, get_model_hash
import torch
from gpt_mini3 import GPTMini

cfg = json.load(open('gpt_mini3.json'))
model_cfg = dict(cfg['model'])
model_cfg.pop('tokenizer', None)
dirs = [cfg['paths']['data_dir']]

# Recompute all hashes
vh = get_vocab_hash(cfg['tokenizer'], dirs)
ch = compute_corpus_hash(dirs)

# Load model and compute ckpt_hash
model = GPTMini(model_cfg, cfg['tokenizer']['max_vocab_size'])
ckpt_h = get_model_hash(model, vh)

print(f'Hash chain:')
print(f'  vocab_hash: {vh}')
print(f'  corpus_hash: {ch}')
print(f'  ckpt_hash: {ckpt_h}')
print(f'')
print(f'Expected cache: vocab-{vh}.json, data-{vh}-{ch}.npy')
print(f'Expected ckpt dir: checkpoints/{ckpt_h}/')
"
```

### Step 6 — Stale cache detection
```bash
python -c "
from pathlib import Path
import json, os
from gpt_mini3 import get_vocab_hash, compute_corpus_hash

cfg = json.load(open('gpt_mini3.json'))
dirs = [cfg['paths']['data_dir']]

current_vh = get_vocab_hash(cfg['tokenizer'], dirs)
current_ch = compute_corpus_hash(dirs)

cache_dir = Path(cfg['paths']['cache_dir'])

# Check each cache file
for vocab_file in cache_dir.glob('vocab-*.json'):
    vh = vocab_file.stem.replace('vocab-', '')
    status = 'CURRENT' if vh == current_vh else 'STALE'
    age = (Path().cwd().stat().st_mtime - vocab_file.stat().st_mtime) / 86400
    print(f'{vocab_file.name}: {status} ({age:.0f} days old)')

for data_file in cache_dir.glob('data-*.npy'):
    stem = data_file.stem
    parts = stem.split('-')
    vh, ch = parts[1], parts[2]
    vh_ok = vh == current_vh
    ch_ok = ch == current_ch
    if vh_ok and ch_ok:
        status = 'CURRENT'
    elif vh_ok:
        status = 'STALE (corpus changed)'
    else:
        status = 'STALE (vocab changed)'
    sz = f'{data_file.stat().st_size / 1e9:.1f}GB'
    print(f'{data_file.name}: {status} ({sz})')
"
```

### Step 7 — Cleanup old artifacts
```bash
# Delete all stale caches (keep current)
python -c "
from pathlib import Path
import json
from gpt_mini3 import get_vocab_hash, compute_corpus_hash

cfg = json.load(open('gpt_mini3.json'))
dirs = [cfg['paths']['data_dir']]
current_vh = get_vocab_hash(cfg['tokenizer'], dirs)
current_ch = compute_corpus_hash(dirs)
cache_dir = Path(cfg['paths']['cache_dir'])

# Delete stale vocab caches
for f in cache_dir.glob('vocab-*.json'):
    vh = f.stem.replace('vocab-', '')
    if vh != current_vh:
        print(f'Deleting {f}')
        # f.unlink()  # Uncomment to actually delete

# Delete stale dataset caches
for f in cache_dir.glob('data-*.npy'):
    stem = f.stem.split('-')
    if stem[1] != current_vh or stem[2] != current_ch:
        print(f'Deleting {f}')
        # f.unlink()  # Uncomment to actually delete
"

# Delete old checkpoint tiers (keep tier 3+)
Get-ChildItem "E:\training\checkpoints" -Recurse -Directory | Where-Object { $_.Name -match "^[0-1]$" } | ForEach-Object {
    Write-Host "Low-tier checkpoint: $($_.FullName)"
}
```

## Key Files
| File | Purpose |
|------|---------|
| `E:\training\cache\vocab-{hash}.json` | Cached BPE vocabulary |
| `E:\training\cache\data-{vh}-{ch}.npy` | Cached tokenized dataset |
| `E:\training\checkpoints\{hash}/` | Latest checkpoint |
| `E:\training\checkpoints\{hash}/resume.json` | Resume metadata |
| `E:\training\checkpoints\{hash}/checkpoint_status.txt` | Training log (tab-separated) |
| `E:\training\checkpoints\{hash}/1/..4/` | Tiered checkpoints |

## Hash Chain
```
corpus files (*.txt) + .meta.json (tier)
  → vocab_hash (size, tier, sample_cap, NO mtime)
    → vocab-{hash}.json
    → data-{hash}-{corpus_hash}.npy
      ← corpus_hash (16 location 1KB samples, NO mtime)
  → ckpt_hash (model dims + vocab_hash)
    → checkpoints/{hash}/
```
