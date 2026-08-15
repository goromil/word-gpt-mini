# Common Patterns & Conventions

## Description
Shared patterns used across all skills: GPU detection, config loading, WSL path translation, environment setup.

## Conventions

### Python environment
```bash
# Training machine (WSL):
wsl /home/george/miniconda3/envs/ai/bin/python <script.py>

# Dev machine (Windows PowerShell):
python <script.py>
```
Environment is `ai` under miniconda3 in WSL. `sentencepiece` must be installed.

### Config loading (matches `gpt_mini3.py:947-957`)
```python
import json
cfg = json.load(open('gpt_mini3.json'))

# Model config (without tokenizer sub-keys)
model_cfg = dict(cfg['model'])
model_cfg.pop('tokenizer', None)

# Tokenizer config (top-level key in current format)
vocab_cfg = dict(cfg['tokenizer'])

# Training config
train_cfg = cfg['training']

# Paths
paths = cfg['paths']
```

### Data directories
```python
data_dirs = [paths["data_dir"]]
if "extra_data_dirs" in paths:
    data_dirs.extend(paths["extra_data_dirs"])
```
In current config, `extra_data_dirs` is absent; all data is in `data_dir`.

### Hash functions
```python
from gpt_mini3 import get_vocab_hash, compute_corpus_hash, get_model_hash

# Vocab hash: tokenizer config + file metadata (size, tier, NO mtime)
vh = get_vocab_hash(vocab_cfg, data_dirs)

# Corpus hash: 16 evenly-spaced 1KB samples per file (NO mtime)
ch = compute_corpus_hash(data_dirs)

# Checkpoint hash: model tensor dims + vocab_hash
ckpt_h = get_model_hash(model_instance, vh)
```

### GPU detection
```python
import torch
gpus = []
for i in range(torch.cuda.device_count()):
    gpus.append({
        'idx': i,
        'name': torch.cuda.get_device_name(i),
        'gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 1),
    })
```

### Path translation
| Windows Path | WSL Path |
|---|---|
| `C:\Users\gorom\source\ai\word-gpt-mini` | `/mnt/c/Users/gorom/source/ai/word-gpt-mini` |
| `E:\training\data` | `/mnt/e/training/data` |
| `E:\training\cache` | `/mnt/e/training/cache` |
| `E:\training\checkpoints` | `/mnt/e/training/checkpoints` |

### Trainer selection logic
```
if GPU has P2P (NVLink/SXM2):
    → train_ipc_ddp.py
elif GPU is PXB/PCIe bridge (RTX 3090):
    → train_noipc_ddp.py
elif Windows, any GPU (experimental):
    → experimental/train_rpc.py
```

### Source filtering
`sources` arrays filter `.txt` files by **name prefix**:
- `"tinystories"` matches `tinystories.txt`
- `"wikipedia_en_corpus"` matches `wikipedia_en_corpus.txt`
- `"chitanka_epub_corpus"` matches `chitanka_epub_corpus.txt`

### Checkpoint resume
Trainer always resumes from latest checkpoint:
1. Find latest `checkpoints/{hash}/` directory
2. Read `resume.json` for `epoch`, `loss`, `global_batch`
3. Load `model.{epoch & 1}.pth` (slot rotation)
4. Load `optimizer.{epoch & 1}.pt` (rank 0 only)

### Status file format
`checkpoint_status.txt` is tab-separated:
```
time    epoch    batch    loss    tok/s    batch/s    total_samples
12:34:56    1    1000    4.5234    125000    15.3    64000
```
