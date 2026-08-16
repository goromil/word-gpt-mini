# Architecture & Hyperparameter Design

## Description
Designs and tunes model architecture: layer/head/dim sizing, GPU memory fitting, Chinchilla scaling, and config generation.

## Triggers
"change architecture", "tune config", "fit model to GPU", "Chinchilla calc", "model design", "architecture change", "hyperparameter tuning", "resize model"

## Workflow

### Step 1 — Current config analysis
```bash
# Parameter count
python calc_params.py train_gpt.json

# Current config
python -c "
import json
cfg = json.load(open('train_gpt.json'))
m = cfg['model']
n_embd = m['n_head'] * m['head_dim']
print(f'Architecture: {m[\"n_layer\"]}L / {m[\"n_head\"]}H / hd={m[\"head_dim\"]}')
print(f'n_embd: {n_embd}')
print(f'Seq length: {m[\"seq_length\"]}')
print(f'Vocab: {cfg[\"tokenizer\"][\"max_vocab_size\"]}')
"
```

### Step 2 — GPU-aware config generation
```bash
# Auto-select best config for detected GPUs
python train_designer.py --no-interact --force

# Interactive mode (select GPUs, choose config)
python train_designer.py

# Scan corpus for vocab size
python train_designer.py --scan-vocab
```

### Step 3 — Memory calculation

Memory formula (from `train_designer.py`):
```
GPU_mem = fixed_gb + var_gb

fixed_gb = params * 12 / world_size / 1024^3  (weights + grad + Adam m + Adam v)
var_gb   = bs_per_gpu * seq_length * K * n_layer * 1.45 / 1024^3

K = 13 * n_embd + 4 * n_heads * seq_length

Where:
  x12 = 2B(FP16 weights) + 4B(grad) + 4B(Adam m) + 2B(Adam v)
  K breakdown: 13xn_embd (QKV+MLP+residuals) + 4xn_headsxseq_length (attention matrix)
  x1.45 = backward pass overhead (activations + gradients + optimizer scratch)
```

**Dominant costs (in order):**
1. `4 * n_heads * seq_length^2` — attention matrix (reduce `seq_length` first)
2. `13 * n_embd` — MLP feedforward (reduce `head_dim` next)
3. `n_layer` — scales all activations linearly

**Example** (36L/20H/hd=64, n_embd=1280, 2x RTX 3090):
```
seq_length=1024, bs_per_gpu=8:   ~18 GB  [OK for 24GB VRAM]
seq_length=1024, bs_per_gpu=16:  ~24 GB  [edge, may OOM]
seq_length=512, bs_per_gpu=16:   ~14 GB  [OK]
seq_length=256, bs_per_gpu=32:   ~10 GB  [comfortable]
```

### Step 4 — Chinchilla scaling
```
Optimal tokens ≈ 200 * params
Practical minimum ≈ 50 * params

epochs = total_tokens_in_corpus / (batch_size * tokens_per_step)

Where tokens_per_step = batch_size * seq_length
```

For current config (36L/20H/hd=64 ≈ 126M params):
- Optimal: 25B tokens
- Minimum: 6.3B tokens

### Step 5 — Adjust config parameters

#### Reduce memory (in order of impact):
1. `seq_length`: 1024 → 512 → 256 (biggest impact)
2. `batch_size`: reduce by half
3. `head_dim`: 64 → 48 (reduces n_embd)
4. `n_layer`: 36 → 24 (scales activations)
5. `n_head`: 20 → 12 (reduces n_embd and attention)

#### Increase capacity (in order of impact):
1. `n_layer`: +4 at a time (check VRAM headroom)
2. `n_head`: +2 (must divide n_embd evenly)
3. `head_dim`: +8 (must be multiple of 8)

### Step 6 — Source filtering configuration
```json
"tokenizer": {
    "sources": ["tinystories", "wikipedia_en_corpus", "chitanka_epub_corpus"]
},
"training": {
    "sources": ["tinystories", "wikipedia_en_corpus", "chitanka_test_corpus", "chitanka_epub_corpus"]
}
```
- `tokenizer.sources` — corpora for vocab building (fewer = faster training)
- `training.sources` — corpora for actual model training (all = best quality)
- Different sets allow: vocab from high-quality data, training on broader data

### Step 7 — Vocab sizing
```json
"tokenizer": {
    "max_vocab_size": 65536,
    "vocab_sample_cap": 25000000,
    "pre_sample_per_source": 500
}
```
- `max_vocab_size`: 32768 for small corpora, 65536 for multi-language
- `vocab_sample_cap`: 25M sentences max fed to SentencePiece
- `pre_sample_per_source`: 500 sentences kept in memory for estimation

### Step 8 — Validate changes
```bash
# Verify config is valid JSON
python -c "import json; json.load(open('train_gpt.json'))"

# Check parameter count matches expectation
python calc_params.py train_gpt.json

# Verify hash stability
python -c "
from train_gpt import get_vocab_hash
import json
cfg = json.load(open('train_gpt.json'))
h = get_vocab_hash(cfg['tokenizer'], [cfg['paths']['data_dir']])
print(f'vocab_hash: {h}')
"
```

## Config Template
```json
{
  "model": {
    "n_layer": 36,
    "n_head": 20,
    "head_dim": 64,
    "seq_length": 1024,
    "tokenizer": "BPE"
  },
  "training": {
    "epochs": 31,
    "batch_size": 16,
    "lr": 0.0002,
    "checkpoint": {
      "every_batch": 250,
      "every_min": 30,
      "every_epoch": 1
    },
    "sync": {
      "gradient_accumulation_steps": 512,
      "method": "cpu",
      "chunks": 4
    },
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

## Common Adjustments
| Goal | Change | Effect |
|------|--------|--------|
| Fit smaller GPU | Reduce `seq_length` | Biggest memory savings |
| More capacity | Increase `n_layer` by 4 | +3% memory, +~5% accuracy |
| Faster training | Reduce `n_head` | Smaller attention matrix |
| Better multilingual | `max_vocab_size` 65536 | More room for non-Latin words |
| Smaller model | `head_dim` 48, `n_head` 12 | 768 n_embd → 13B param model |
