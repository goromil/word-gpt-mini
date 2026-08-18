# GPT-mini

A GPT implementation in PyTorch with self-training and configurable architecture.

## Package Setup (one-time)

```bash
# In WSL, activate conda env ai:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai
pip install -e /path/to/word-gpt-mini

# Create config in ~/.config/wordgpt/ (auto-detects WSL/Linux paths):
gpt_setup_config

# For Windows config (converts /mnt/e/ -> E:\):
gpt_setup_config --platform windows

# Show config location without creating:
gpt_setup_config --show
```

Config lives in `~/.config/wordgpt/gpt_train.json` — **not** in the repo.
Edit paths there to point to your data, checkpoints, and cache directories.

## Quick Start

After `pip install -e .` and `gpt_setup_config`:

```bash
gpt_nipc_train -d 0,1                 # train on 2 RTX 3090s (no-P2P)
gpt_ipc_train -d 0,1                   # train on 2 V100 SXM2 (P2P/NVLink)
```

### Commands

After installation these commands are available on PATH:

| Command | Description |
|---|---|
| `gpt_nipc_train` | Multi-GPU training (CPU grad sync, no P2P needed) |
| `gpt_ipc_train` | Multi-GPU training (GPU all_reduce, requires P2P) |
| `gpt_train` | Single-GPU training |
| `gpt_info` | Info: checkpoints, vocab, cache, data status |
| `gpt_dataset_dl` | Web scraper for chitanka.info |
| `gpt_dataset_cumulative_dl` | Archive/HuggingFace dataset downloader |
| `gpt_dataset_builder` | Corpus post-processor (lang filter, dedup) |
| `gpt_setup_config` | Config setup utility |

All training commands default to `~/.config/wordgpt/gpt_train.json`. Override with path arg or use checkpoint flags:

```bash
gpt_nipc_train --checkpoint-hash ca3a8182d56d8845
gpt_nipc_train --checkpoint-path /mnt/e/training/checkpoints/xxx
gpt_nipc_train -d 0,1 --epochs 50 --save_every 3
```

## Config Files

- `gpt_train_draft.json.tmpl` — draft architecture template (committed to repo)
- `~/.config/wordgpt/gpt_train.json` — working config (gitignored, per-user)

## Workflow

### Design Training Config (GPU-Aware, internal)

Run `python -m wordgpt.train_designer` directly (not a public command):

```bash
python -m wordgpt.train_designer                      # interactive: select GPUs, choose config
python -m wordgpt.train_designer --no-interact        # auto-select best config
python -m wordgpt.train_designer --no-interact --force  # overwrite existing config
python -m wordgpt.train_designer --scan-vocab         # scan corpus for vocab size
```

**GPU Memory Analysis**: `train_designer` calculates per-GPU memory usage and proposes valid
(seq_length, batch_size) combinations that fit your hardware. Memory formula:

```
GPU_mem = fixed_gb + var_gb

fixed_gb = params × 12 / world_size / 1024³  (weights + grad + Adam states)
var_gb   = bs_per_gpu × seq_length × K × n_layer × 1.45 / 1024³
K        = 13 × n_embd + 4 × n_heads × seq_length
```

Where:
- **×12**: 2B (FP16 weights) + 4B (grad) + 4B (Adam m) + 2B (Adam v) per parameter
- **K breakdown**: 13×n_embd (QKV+MLP+residuals) + 4×n_heads×seq_length (attention matrix)
- **×1.45**: backward pass overhead (activations + gradients + optimizer scratch)

The attention term `4 × n_heads × seq_length²` is the dominant variable cost — reducing seq_length
has the biggest impact on memory. The MLP feedforward `13 × n_embd` is second.

**Breaking point example** (32L/12H/hd=128, n_embd=1536, 2xRTX 3090):
```
seq_length=512, bs_per_gpu=32:  44.6 GB  [OOM]
seq_length=512, bs_per_gpu=8:   15.1 GB  [OK]
seq_length=256, bs_per_gpu=16:  12.9 GB  [OK]
seq_length=64,  bs_per_gpu=128: 17.7 GB  [OK]
```

### Parameter Calculator (internal)

```bash
python -m wordgpt.calc_params                                   # from default config
python -m wordgpt.calc_params /path/to/config.json              # custom config path
```

Interactive mode prompts for GPU selection, shows memory analysis, and proposes valid
(seq_length, batch_size) combinations that fit your hardware.

### Checkpoint Strategy

Three triggers, all writing to the same base tier (`checkpoints/<hash>/`):

| Setting | Default | Meaning |
|---|---|---|
| `checkpoint_interval` | 10,000 | save every N batches (~5 min on GPU) |
| `checkpoint_every_min` | 30 | save every N minutes (wall-clock) |
| `checkpoint_every` | 1 | save every N epochs |

Resume reads `checkpoints/<hash>/train.log` once, extracts `global_batch`, and continues from that exact batch.

### Available Corpora

| Corpus | Size | Est. Tokens | Vocab (sampled) |
|---|---|---|---|
| TinyStories-v2 GPT4 | 2.1 GB | 1.8B | ~24,800 (32,768 rounded) |
| TinyStories-train (full) | 2.4 GB | 2.2B | — |
| Bulgarian Corpus 33B | 84.8 GB | ~29B | — |
| Russian Cleared Wikipedia | 153 MB | — | — |

### Multi-GPU Training

Two trainers, choose based on your GPU topology:

| Trainer | GPU Sync | Requires | Use On |
|---|---|---|---|
| `gpt_ipc_train` | DDP GPU all_reduce | P2P (NVLink, SXM2) | V100, A100, H100 |
| `gpt_nipc_train` | CPU all_reduce (gloo) | No P2P needed | RTX 3090, consumer GPUs |

**How they differ**: DDP's `all_reduce` on CUDA tensors uses CUDA IPC, which requires P2P
(peer-to-peer) access between GPUs. Our RTX 3090s have PXB (PCIe bridge) topology — no P2P —
so DDP crashes with segfault (exit code 0xC0000005). The no-IPC trainer syncs gradients via CPU
where gloo works fine on localhost.

```bash
# P2P GPUs (NVLink):
gpt_ipc_train -d 0,1               # train on 2 GPUs
gpt_ipc_train -d 0,1,2,3           # train on 4 GPUs

# No-P2P GPUs (PCIe bridge):
gpt_nipc_train -d 0,1             # train on 2 GPUs, CPU grad sync
```

**How it works**: Uses `torch.multiprocessing.spawn` with one process per GPU.
Each rank runs the same training loop but processes different data via `LazyDistributedSampler`
(avoiding `torch.randperm` MemoryError on 453M-token dataset).

**Master address**: `127.0.0.1:29500` — set BEFORE `mp.spawn` to avoid Windows hostname resolution issues.

**Ctrl+C handling**: Signal handlers on each rank call `dist.destroy_process_group()` before exit.
Main process waits 2s for graceful shutdown, then force-kills children.

**Memory**: Each GPU loads full model weights but only processes `batch_size / world_size`
samples. On no-P2P systems, gradients are copied to CPU for all_reduce and copied back — adds
~50-100ms overhead per step, negligible compared to forward/backward compute.

## Draft Config Format (`gpt_train_draft.json`)

```json
{
  "model": {
    "n_layer": 32,
    "n_head": 12,
    "head_dim": 128,
    "seq_length": 512,
    "vocab": { "max_vocab_size": 32768, "max_word_len": 20 }
  },
  "training-defaults": {
    "batch_size": 64,
    "lr": 0.0003,
    "checkpoint_interval": 10000,
    "checkpoint_every_min": 30
  },
  "paths": {
    "data_dir": "E:\\training\\data",
    "extra_data_dirs": ["E:\\training\\data2\\bulgarian-corpus-33b"],
    "checkpoint_dir": "E:\\training\\checkpoints",
    "cache_dir": "E:\\training\\cache"
  }
}
```

`training-defaults` provides default training parameters that `train_designer` merges with calculated `epochs` and `checkpoint_every` into the final working config.

### Dataset Download & Build

Data pipeline consists of three stages — download, extract, build. Each tool is independent.

**Pipeline**: `gpt_dataset_dl` or `gpt_dataset_cumulative_dl` → raw files → `gpt_dataset_builder` → `corpus.txt` → training

#### Stage 1: Download

Two downloaders, choose based on your data source:

**`gpt_dataset_dl`** — Web scraper for chitanka.info
- Uses Playwright (headless Chrome) to bypass Cloudflare protection
- XML search API for metadata, individual page fetch for content
- Auto-retry with rate limit handling, checkpoint-based resume

```bash
gpt_dataset_dl --api chitanka-xml                              # single pass
gpt_dataset_dl --api chitanka-xml --max-cycles 100 --cycle-interval 1800  # auto-retry
gpt_dataset_dl --retry --api chitanka-xml                      # retry blocked
gpt_dataset_dl --status --api chitanka-xml                     # check status
gpt_dataset_dl --list                                          # list sources
```

**`gpt_dataset_cumulative_dl`** — Archive and HuggingFace downloader
- Handles zip archives with EPUBs inside
- Downloads HuggingFace datasets (parquet shards)
- Configured from `dataset_dl.json`

```bash
gpt_dataset_cumulative_dl                              # all entries
gpt_dataset_cumulative_dl chitanka-epub                # specific entry
gpt_dataset_cumulative_dl wiki-en                      # HF entry
gpt_dataset_cumulative_dl --skip-dl                    # extract only
```

#### Stage 2: Build Corpus

**`gpt_dataset_builder`** — Post-process downloaded files into training format
- Language detection: filters by cyrillic ratio (>30% = Bulgarian)
- Deduplication: MD5 hash of text lines
- Output: single clean text file (1 line per text, 15–2048 chars)

```bash
gpt_dataset_builder                                             # auto-detect from dataset_dl.json
gpt_dataset_builder --input E:\training\data2                   # specify input dir
gpt_dataset_builder --input E:\training\data2 --combine corpus.txt  # specify output
```

#### Rate Limiting

chitanka.info blocks IPs after ~10–20 requests. Playwright bypasses Cloudflare but not IP ban. Use 30-min intervals for auto-retry mode.

#### Resume & Recovery

- Checkpoints saved every 10 texts. Safe to Ctrl+C and resume.
- Registry tracks: downloaded IDs, failed IDs, blocked queries, cycle count.
- Run same command to resume from where you left off.

#### Multi-Corpus Training

Set `corpora` in config to train on multiple datasets:

```json
"corpora": [
  {"path": "E:\\training\\data\\tinystories.txt", "weight": 1.0},
  {"path": "E:\\training\\data\\wiki_train.txt", "weight": 0.5},
  {"path": "E:\\training\\data2\\chitanka_corpus.txt", "weight": 0.3}
]
```

Or use `extra_data_dirs` for automatic discovery:
```json
"paths": {
  "data_dir": "E:\\training\\data",
  "extra_data_dirs": ["E:\\training\\data2"]
}
```

#### Info & Monitoring

```bash
gpt_info                              # checkpoints, cache, data status
gpt_info --checkpoints                # checkpoint list with epochs, loss, age
gpt_info --cache                      # vocab and data cache files
gpt_info --data                       # training data files and sizes
```

## Working Config Format (`gpt_train.json`)

```json
{
  "model": { "n_layer": 36, "n_head": 20, "head_dim": 64, "seq_length": 1024, "tokenizer": "BPE" },
  "training": {
    "epochs": 31,
    "batch_size": 16,
    "lr": 0.0002,
    "checkpoint": { "every_batch": 250, "every_min": 30, "every_epoch": 1 },
    "sync": { "gradient_accumulation_steps": 512, "method": "cpu", "chunks": 4 },
    "sources": ["tinystories", "wikipedia_en_corpus", "chitanka_epub_corpus"]
  },
  "tokenizer": {
    "max_vocab_size": 65536,
    "max_word_len": 20,
    "vocab_sample_cap": 25000000,
    "pre_sample_per_source": 500,
    "sources": ["tinystories", "wikipedia_en_corpus", "chitanka_epub_corpus"]
  },
  "paths": {
    "data_dir": "E:\\training\\data",
    "checkpoint_dir": "E:\\training\\checkpoints",
    "cache_dir": "E:\\training\\cache"
  }
}
```

**Source filtering**: `sources` arrays filter corpus `.txt` files by name prefix.
`tokenizer.sources` controls which corpora contribute to vocab training.
`training.sources` controls which corpora are used for actual model training.
Omit either array to use all available corpus files.

**Sentence sampling**: `vocab_sample_cap` (default 25M) caps the number of
sentences fed to SentencePiece. For multi-terabyte corpora, sentences are
sampled proportionally per source using file-size-based rasterization.
`pre_sample_per_source` (default 500) keeps a small in-memory buffer for
estimation.

### Checkpoint Info

List all checkpoints, vocab/cache files, and training data:

```bash
gpt_info                              # show everything
gpt_info --checkpoints                # checkpoints only
gpt_info --cache                      # vocab/data cache only
gpt_info --data                       # training data files only
gpt_info --config                     # config summary only
gpt_info --config /path/to/config.json  # custom config path
```

Checkpoints are stored in `checkpoint_dir/<model_hash>/` with tiered snapshots:
- Tier 0: latest checkpoint (alternates model.0.pth / model.1.pth)
- Tier 1: every 10th epoch
- Tier 2: every 100th epoch
- Tier 3: every 1000th epoch, etc.

Each checkpoint contains `resume.json` (epoch, loss, global_batch), `config.json`, `cache_lock.json` (vocab/data cache basenames), and `checkpoint_status.txt` (training progress log).

## Key Features

- **BPE Tokenization** — SentencePiece-based tokenizer trained on sampled corpus
- **Streaming Pipeline** — `SentenceIterator` yields sentences lazily; no full corpus in memory
- **Source Filtering** — `tokenizer.sources` and `training.sources` config arrays select corpus subsets by filename prefix
- **Auto Vocab Sizing** — corpus-sampled via `vocab_sample_cap`, streaming rasterization across tiers
- **Auto Epoch Sizing** — Chinchilla scaling law calculator (N=50 default)
- **Text Generation** — next-token prediction with temperature sampling
- **Transformer Architecture** — multi-head attention, layer norm, feed-forward
- **Weight Tying** — output projection tied to input embeddings
- **CUDA Support** — automatically uses GPU if available
- **Content-Based Caching** — deterministic hashes (no mtime) for vocab and dataset; stable across file copies
- **Checkpointing** — batch/time/epoch triggers with global batch resume
- **Resumable Downloads** — handles SSL interruptions, resumes from partial file

## Services — Auto-Start Training (WSL2 + Task Scheduler)

Automate training startup on PC boot via Windows Task Scheduler. No private paths
are committed to git — all paths live in `~/.config/wordgpt/gpt_train.json`.

### Files

| File | Purpose |
|---|---|
| `services/start.sh` | Bash: activates conda `ai`, exec `gpt_nipc_train "$@"` |
| `services/start.ps1` | PowerShell native: runs `gpt_nipc_train` directly |
| `services/reg-service.ps1` | Registers Task Scheduler task with inline logging |

### How It Works

1. PC boots → Task Scheduler fires `reg-service.ps1` command as SYSTEM
2. For WSL: `wsl.exe` launches `bash services/start.sh` which activates conda and runs training
3. For native: PowerShell runs `conda run -n ai gpt_nipc_train`
4. Logging is handled inline by the scheduler command (like `llama-service`)

### Setup (one-time)

Run in **elevated PowerShell** (Admin), replace `$Username` with your Windows username:

```powershell
# WSL mode (default):
powershell -ExecutionPolicy Bypass -File services/reg-service.ps1 -Username george -Type wsl

# Native Windows mode:
powershell -ExecutionPolicy Bypass -File services/reg-service.ps1 -Username george -Type native
```

Run immediately without rebooting:

```powershell
schtasks /Run /TN "WordGPT-Training"
```

### Passing Extra Arguments

Edit the Task Scheduler task > Actions > Add arguments. Append to the existing command:

```
--epochs 50 -d 0,1
--checkpoint-hash ca3a8182d56d8845
--cache-renew
```

### Logs

Logs are written to `repo/logs/`:
- `wordgpt.log` — stdout
- `wordgpt-error.log` — stderr

Directory is gitignored.

## Notes

- BPE tokenization via SentencePiece. Vocab is trained on a sampled subset of the corpus.
- Adjust `n_layer`, `n_head`, `head_dim` based on hardware and dataset size.
- Chinchilla: optimal tokens = ~200x params; 50x is practical minimum.
- Run `python -m wordgpt.benchmark_streaming` on the training machine to measure pipeline throughput.
