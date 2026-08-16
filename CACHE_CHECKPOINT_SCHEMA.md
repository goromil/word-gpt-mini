# Cache & Checkpoint Schema

## Overview

All cache and checkpoint files are named by **deterministic hashes** that form a
causal dependency chain.  Identical inputs always produce identical names; any
change invalidates only the affected downstream artifacts.

Each artifact carries a **metadata sidecar** (`.meta.json`) documenting its
provenance: which data sources contributed, what tier allocation was used, how
many tokens were produced.

---

## Data File Meta

Each corpus `.txt` file may have an adjacent `.txt.meta.json` that declares its
**tier** and **language**:

```
data/tinystories.txt
data/tinystories.txt.meta.json
data/chitanka.txt
data/chitanka.txt.meta.json
```

### `datafile.meta.json` Schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tier` | int | `1` | Vocab allocation tier: 1, 2, or 3 |
| `language` | string | `null` | ISO language code (documentation only) |

**Template:** `templates/cache/datafile.meta.json.tmpl`

**Example — Bulgarian corpus at tier 3:**
```json
{ "tier": 3, "language": "bg" }
```

**Missing `.meta.json` defaults to tier 1.**

### Tier Allocation

When `BPETokenizer.train()` runs, it allocates BPE merges proportionally per tier:

| Tier | Default Ratio | Description |
|------|---------------|-------------|
| T1   | 66%           | Primary language (English) |
| T2   | 22%           | Secondary language |
| T3   | 12%           | Tertiary language (Bulgarian) |

Ratio configurable via `tier_ratios` parameter. Sentence sample cap
(`vocab_sample_cap`) controls how many sentences are fed to SentencePiece,
keeping training fast and memory-efficient even for multi-terabyte corpora.

---

## Hash Definitions

### `vocab_hash`

```python
get_vocab_hash(vocab_cfg, data_dirs) -> str    # 16-char hex
```

**Inputs**

| Component | Details |
|-----------|---------|
| Tokenizer config | `max_vocab_size`, `max_word_len`, `sentence_sample_cap` / `vocab_sample_cap` |
| Data file metadata | For every `*.txt` in `data_dirs`: filename, file size (`st_size`) — **mtime excluded** |
| **File tier** | From adjacent `.txt.meta.json` (or default `1`) |

**Used for**

- `vocab-{vocab_hash}.json` — cached BPE vocabulary (SentencePiece model)
- `data-{vocab_hash}-{corpus_h}.npy` — cached tokenized data array

**Why it works:**  Changing any source file (new content → new size),
adding a file, altering tokenizer params, changing sentence sample cap, or
changing a file's tier all produce a new hash.  Modification time (`mtime`) is
intentionally excluded for stability across file copies and transfers.
The model architecture is intentionally excluded — the vocabulary
and tokenizer are independent of `n_layer`, `n_head`, etc.

---

### `corpus_h`

```python
compute_corpus_hash(data_dirs) -> str    # 16-char hex (shared function)
```

**Inputs** — Per file: relative path, size, plus **16 evenly-spaced 1 KB samples**
across the file. Excludes `mtime` for stability.

For files smaller than 16 KB, samples every kilobyte. For files larger than 16 KB,
samples 1 KB at offsets spaced evenly across the file.

**Used for** — Second segment of data cache: `data-{vocab_hash}-{corpus_h}.npy`.

**Why separate from `vocab_hash`:**  `vocab_hash` uses lightweight file metadata
(name, size, tier) to detect whether the vocabulary might have changed.
`corpus_h` samples actual file content at multiple locations, ensuring the
`.npy` cache is invalidated even if file content changes without a size change
(e.g., in-place editing that preserves size). The sampling approach keeps
computation fast while catching real content changes.

---

### `ckpt_hash`

```python
get_model_hash(model, vocab_hash) -> str    # 16-char hex
```

**Inputs**

| Component | Source |
|-----------|--------|
| `vocab_size` | `model.transformer.wte.num_embeddings` |
| `n_embd` | `model.n_embd` (= `n_head × head_dim`) |
| `n_layer` | `len(model.transformer.h)` |
| `n_head` | `model.transformer.h[0].attn.n_head` |
| `head_dim` | `model.transformer.h[0].attn.head_dim` |
| `seq_length` | `model.wpe.size(1)` |
| **`vocab_hash`** | passed as argument |

All tensor attributes are read from the **instantiated model object**, not from
the config dict, so the hash always reflects the actual graph dimensions.

**Used for** — `checkpoints/{ckpt_hash}/` (checkpoint directory)

**Why `vocab_hash` is included:**  Two models with identical architecture but
different vocabularies have embedding weights that are meaningless to each
other.  Embedding the `vocab_hash` in the checkpoint hash guarantees they
never collide.

---

## Cache Files

```
E:\training\cache\vocab-{vocab_conf_hash}-{vocab_hash}.json
E:\training\cache\vocab-{vocab_conf_hash}-{vocab_hash}.meta.json
E:\training\cache\data-{corpus_conf_hash}-{vocab_hash}-{corpus_h}.npy
E:\training\cache\data-{corpus_conf_hash}-{vocab_hash}-{corpus_h}.meta.json
```

**Why the `{conf_hash}` prefix?** On resume, Python first globs the cache
directory for files matching the config-only hashes (computed in <1ms).
If a match exists, the expensive data-dir scans (`vocab_hash`, `corpus_h`)
are skipped entirely. This reduces cold-start time from ~4s to <10ms.

The `--cache-renew` CLI flag forces a full rebuild, ignoring any existing
cache files regardless of hash match.

### `vocab.meta.json` Schema

| Field | Type | Description |
|-------|------|-------------|
| `vocab_size` | int | Final vocabulary size (reserved + chars + tier words) |
| `max_vocab_size` | int | Configured maximum |
| `capped` | bool | `true` if vocab reached `max_vocab_size` |
| `char_slots` | int | Number of pre-populated character tokens (~71) |
| `sources` | array | Per-source contribution (see below) |
| `tier_ratios` | array | Tier allocation ratios used |

**Template:** `templates/cache/vocab.meta.json.tmpl`

**`sources` array entries:**

| Field | Type | Description |
|-------|------|-------------|
| `dir` | string | Source directory path |
| `file` | string | Corpus filename |
| `tier` | int | Allocation tier (1-3) |
| `language` | string | ISO language code (or `null`) |
| `words_in_vocab` | int | How many words from this source made it into the vocab |

**Example:**
```json
{
  "vocab_size": 25600,
  "max_vocab_size": 32768,
  "capped": false,
  "char_slots": 71,
  "sources": [
    { "dir": "E:\\training\\data", "file": "tinystories.txt", "tier": 1, "language": "en", "words_in_vocab": 21627 },
    { "dir": "E:\\training\\data2\\bulgarian-corpus-33b", "file": "chitanka.txt", "tier": 3, "language": "bg", "words_in_vocab": 3902 }
  ],
  "tier_ratios": [0.66, 0.22, 0.12]
}
```

### `data.meta.json` Schema

| Field | Type | Description |
|-------|------|-------------|
| `tokens` | int | Total token count in the `.npy` cache |
| `vocab_size` | int | Vocabulary size used during tokenization |

**Template:** `templates/cache/data.meta.json.tmpl`

---

## Tokenizer Details (BPE via SentencePiece)

### Training Pipeline

1. `SentenceIterator` yields sentences lazily from corpus files
2. `BPETokenizer.train()` estimates sentence counts per source, allocates a
   proportional sample from each (total capped at `vocab_sample_cap`), and
   writes a single sampled training file
3. SentencePiece trains a BPE model on the sampled file with tier-aware
   vocabulary allocation
4. The trained model is saved as `vocab-{hash}.json`

### Sentence Sampling

`vocab_sample_cap` (default: 25M sentences) limits the number of sentences
fed to SentencePiece. For corpora larger than the cap, sentences are sampled
proportionally per source using file-size-based rasterization. This keeps
vocab training fast and memory-efficient.

`pre_sample_per_source` (default: 500) controls how many sentences are kept
in memory for tier allocation estimation.

### Source Filtering

Config arrays `tokenizer.sources` and `training.sources` specify which corpus
files to use. `SentenceIterator` filters `.txt` files by name prefix:

```json
"tokenizer": { "sources": ["tinystories", "wikipedia_en_corpus", "chitanka_epub_corpus"] }
"training": { "sources": ["tinystories", "wikipedia_en_corpus", "chitanka_test_corpus", "chitanka_epub_corpus"] }
```

This allows different corpus subsets for vocabulary building vs. training.

---

## Checkpoint Structure

```
E:\training\checkpoints\{ckpt_hash}\model.pth
E:\training\checkpoints\{ckpt_hash}\resume.json
E:\training\checkpoints\{ckpt_hash}\config.json
E:\training\checkpoints\{ckpt_hash}\checkpoint_status.txt
E:\training\checkpoints\{ckpt_hash}\{tier}\...     # tiered checkpoints
```

Tiered sub-directories preserve checkpoints at epoch milestones:

| Tier | Epoch Interval |
|------|---------------|
| 1    | Every 10th epoch |
| 2    | Every 100th epoch |
| 3    | Every 1,000th epoch |
| 4    | Every 10,000th epoch |
| ...  | ×10 each level |

### `resume.json` Schema

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | int | Last completed epoch |
| `loss` | float | Average loss at checkpoint |
| `global_batch` | int | Cumulative batch index |
| `batch_size` | int | Training batch size |
| `seq_length` | int | Sequence length |
| `training_samples` | int | Cumulative samples processed |
| `training_start_time` | float | Epoch timestamp of training start |
| `vocab_size` | int | Vocabulary size |
| `dataset_tokens` | int | Total tokens in dataset |
| `dataset_samples` | int | Total samples in dataset |
| `config_hash` | string | Hash of model + tokenizer config |

**Template:** `templates/checkpoint/resume.json.tmpl`

### `checkpoint_status.txt` Format

Append-only, tab-separated log with header row:

```
time	epoch	batch	loss	tok/s	batch/s	total_samples
12:34:56	1	1000	4.5234	125000	15.3	64000
```

### `cache_lock.json` Schema

Stored inside each checkpoint directory. Records the cache file basenames used
by that checkpoint. On resume, the trainer reads these basenames and prepends
the configured `cache_dir` to locate the actual files.

| Field | Type | Description |
|-------|------|-------------|
| `vocab_cache` | string | Basename of vocab cache (e.g. `vocab-a1b2-11b6.json`) |
| `data_cache` | string | Basename of data cache (e.g. `data-566e-11b6-f69e.npy`) |

**Example:**
```json
{"vocab_cache":"vocab-a9d7833c141af1b2-11b6add98fad06f1.json","data_cache":"data-566e43acfb5c5879-11b6add98fad06f1-f69edacea6f75f0e.npy"}
```

### `current_checkpoint.json` Schema

Stored in the checkpoint directory root. Records the active checkpoint hash,
allowing resume even when data files are missing (path 2).

| Field | Type | Description |
|-------|------|-------------|
| `ckpt_hash` | string | 16-char hex checkpoint hash |
| `epoch` | int | Epoch at time of last save |
| `loss` | float | Loss at time of last save |

**Example:**
```json
{"ckpt_hash":"d4e5f6a1b2c34567","epoch":15,"loss":3.452100}
```

### Startup Resolution (Two Paths)

**Path 1 — Data files present (can compute `vocab_hash`):**
1. Compute `ckpt_hash` from model dims + `vocab_hash`
2. Read/write `current_checkpoint.json` (update if hash changed)
3. Read `cache_lock.json` from checkpoint dir; if missing, rebuild from hashes
4. Verify cache files exist; if not, build them

**Path 2 — Data files missing (cannot compute `vocab_hash`):**
1. Read `ckpt_hash` from `current_checkpoint.json`
2. If missing or checkpoint dir doesn't exist → exit with error
3. Read `cache_lock.json` from checkpoint dir
4. If missing or cache files don't exist → exit with error
5. Resume training from cached data

---

## File Naming Summary

```
# Data sources
E:\training\data\tinystories.txt
E:\training\data\tinystories.txt.meta.json
E:\training\data2\bulgarian-corpus-33b\chitanka.txt
E:\training\data2\bulgarian-corpus-33b\chitanka.txt.meta.json

# Cache
E:\training\cache\vocab-{vocab_conf_hash}-{vocab_hash}.json
E:\training\cache\vocab-{vocab_conf_hash}-{vocab_hash}.meta.json
E:\training\cache\data-{corpus_conf_hash}-{vocab_hash}-{corpus_h}.npy
E:\training\cache\data-{corpus_conf_hash}-{vocab_hash}-{corpus_h}.meta.json

# Checkpoints
E:\training\checkpoints\current_checkpoint.json
E:\training\checkpoints\{ckpt_hash}\cache_lock.json
E:\training\checkpoints\{ckpt_hash}\model.pth
E:\training\checkpoints\{ckpt_hash}\resume.json
E:\training\checkpoints\{ckpt_hash}\config.json
E:\training\checkpoints\{ckpt_hash}\checkpoint_status.txt
E:\training\checkpoints\{ckpt_hash}\{tier}\...
```

---

## Dependency Chain

```
data files (*.txt)
      │
      ├─→ .meta.json (tier, language)
      │
      ├─→ vocab_conf_hash ──→ vocab-{conf}-{hash}.json
      │       │
      │       └──→ vocab_hash (file metadata)
      │
      ├─→ corpus_conf_hash ──→ data-{conf}-{vhash}-{chash}.npy
      │       │
      │       └──→ corpus_h (content sampling)
      │
      └──→ ckpt_hash (= hash(model_dims, vocab_hash))  ───→ checkpoints/{hash}/
              │
              ├─→ cache_lock.json (vocab_cache, data_cache basenames)
              └──→ checkpoints/current_checkpoint.json → {ckpt_hash}
```

**Propagation rule:**  Changing any upstream input (data content, tier assignment,
tokenizer params, model dims) invalidates all downstream artifacts automatically
via the new hash.

---

## What Is NOT Included (and Why)

| Parameter | Excluded from | Reason |
|-----------|---------------|--------|
| `seq_length` | `vocab_hash`, `corpus_h` | The `.npy` cache is a flat token array; `seq_length` only controls runtime window slicing in `__getitem__`. |
| `batch_size` | everything | A DataLoader parameter, has zero effect on cached data or model weights. |
| `lr`, `epochs`, `gradient_accumulation_steps` | `ckpt_hash` | Training hyperparameters don't change model architecture or vocabulary. Different hyperparameters should resume from the same checkpoint. |
| `n_layer`, `n_head`, `head_dim` | `vocab_hash` | Model architecture doesn't affect tokenization. |
| `st_mtime` | `vocab_hash`, `corpus_h` | Modification time varies across copies and transfers. File content changes are caught by size (`vocab_hash`) or content sampling (`corpus_h`). |

---

## Functions (all in `train_gpt.py`)

| Function | Signature | Public? |
|----------|-----------|---------|
| `get_vocab_hash()` | `(vocab_cfg: dict, data_dirs: list) -> str` | Yes — imported by DDP scripts |
| `get_model_hash()` | `(model, vocab_hash: str = None) -> str` | Yes — imported by DDP scripts |
| `compute_corpus_hash()` | `(data_dirs: list) -> str` | Yes — imported by DDP scripts |
| `get_vocab_conf_hash()` | `(vocab_cfg: dict, sources: list) -> str` | Yes — config-only vocab hash |
| `get_corpus_conf_hash()` | `(sources: list) -> str` | Yes — config-only corpus hash |
| `ensure_cache_ready()` | `(model_cfg, vocab_cfg, paths, force, tokenizer_sources, training_sources) -> tuple[str, str]` | Yes — DDP pre-flight |
| `write_cache_lock()` | `(ckpt_hash_dir, vocab_cache_name, data_cache_name)` | Yes — writes cache_lock.json |
| `read_cache_lock()` | `(ckpt_hash_dir) -> dict \| None` | Yes — reads cache_lock.json |
| `write_current_checkpoint()` | `(ckpt_dir, ckpt_hash, epoch, loss)` | Yes — writes pointer |
| `read_current_checkpoint()` | `(ckpt_dir) -> dict \| None` | Yes — reads pointer |
| `resolve_checkpoint_and_cache()` | `(ckpt_dir, cache_dir, vocab_conf_h, corpus_conf_h, vocab_hash, corpus_h, ckpt_hash) -> dict` | Yes — two-path resolver |
| `SentenceIterator` | `(data_dir, extra_dirs, sources) -> Iterator[str]` | Yes — lazy sentence iterator |
| `BPETokenizer.train()` | `(sources, tier_ratios, sentence_sample_cap, pre_sample_per_source)` | Yes — streaming BPE training |
| `BPETokenizer.encode()` | `(text) -> list[int]` | Yes — full text encoding |
| `WordDataset` | `(sentences_or_iter, tokenizer, seq_length, cache_file)` | Yes — streaming dataset with npy cache |

---

## Design Principles

1. **Single source of truth** — Each hash function is defined once in `train_gpt.py`. DDP scripts import it; they do not compute their own hash.
2. **Model-first** — `ckpt_hash` reads from the instantiated model object, not from a config dict, guaranteeing the hash matches the actual tensor shapes.
3. **Vocabulary-bound checkpoints** — `vocab_hash` is embedded in `ckpt_hash`; different vocabularies can never share or collide on checkpoints.
4. **Content-based invalidation** — `vocab_hash` uses cheap file metadata (name, size, tier, sample cap) without mtime. `corpus_h` samples actual content at 16 evenly-spaced locations for stability across copies while catching real changes.
5. **No training config in names** — Hyperparameters (`lr`, `batch_size`, `epochs`, gradient accumulation) don't affect file names. Only structural attributes (tokenizer, data files, model dims) do.
6. **Streaming pipeline** — `SentenceIterator` yields sentences lazily. No full corpus is loaded into memory. `BPETokenizer.train()` consumes the iterator and writes a sampled training file for SentencePiece. `WordDataset` streams from the iterator or from cached `.npy`.
7. **Source filtering** — `tokenizer.sources` and `training.sources` config arrays allow different corpus subsets for vocab building vs. training. `SentenceIterator` filters files by name prefix.
8. **Mtime-free hashes** — Neither hash function includes file modification time, ensuring stable cache names across file copies, transfers, and re-downloads.
9. **Cache lock per checkpoint** — `cache_lock.json` lives inside each checkpoint dir, recording only cache file basenames. Combined with `current_checkpoint.json` at the root, this enables two-path resume: path 1 computes hashes from data files; path 2 reads the pointer and lock when data is missing.
