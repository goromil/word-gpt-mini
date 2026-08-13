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

When `build_vocab` runs, it allocates vocabulary slots proportionally per tier:

| Tier | Default Ratio | Slots (32768 vocab) |
|------|---------------|---------------------|
| T1   | 66%           | 21,627              |
| T2   | 22%           | 7,209               |
| T3   | 12%           | 3,931               |

Ratio configurable via `tokenizer.tier_ratios` in config.

Within each tier, words are ranked by frequency. Only the top-N words per tier
are included. This guarantees that a tier-3 Bulgarian corpus gets its allocated
slots even though English (T1) is thousands of times larger.

---

## Hash Definitions

### `vocab_hash`

```python
get_vocab_hash(vocab_cfg, data_dirs) -> str    # 16-char hex
```

**Inputs**

| Component | Details |
|-----------|---------|
| Tokenizer config | `max_vocab_size`, `max_word_len` |
| Data file metadata | For every `*.txt` in `data_dirs`: filename, file size (`st_size`), modification time (`st_mtime`) |
| **File tier** | From adjacent `.txt.meta.json` (or default `1`) |

**Used for**

- `vocab-{vocab_hash}.json` — cached word vocabulary
- `data-{vocab_hash}-{corpus_h}.npy` — cached tokenized data array

**Why it works:**  Changing any source file (new content → new size/mtime),
adding a file, altering tokenizer params, or changing a file's tier all produce
a new hash.  The model architecture is intentionally excluded — the vocabulary
and tokenizer are independent of `n_layer`, `n_head`, etc.

---

### `corpus_h`

```python
corpus_hash(data_dirs) -> str    # 16-char hex (local function)
```

**Inputs** — Full file contents of every file under `data_dirs` (hashed in 8 KB chunks).

**Used for** — Second segment of data cache: `data-{vocab_hash}-{corpus_h}.npy`.

**Why separate from `vocab_hash`:**  `vocab_hash` uses lightweight file metadata
(name, size, mtime) to detect whether the vocabulary might have changed.
`corpus_h` hashes the full content, ensuring the `.npy` cache is invalidated
even if file content changes without a size or mtime change (e.g., in-place
editing that preserves size).

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
E:\training\cache\vocab-{vocab_hash}.json
E:\training\cache\vocab-{vocab_hash}.meta.json
E:\training\cache\data-{vocab_hash}-{corpus_h}.npy
E:\training\cache\data-{vocab_hash}-{corpus_h}.meta.json
```

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

## Tokenizer Details

### Reserved Tokens

| Token | Index | Purpose |
|-------|-------|---------|
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown character fallback |
| `<eos>` | 2 | End-of-sequence marker |
| `<sep>` | 3 | Separator between split characters |

### Pre-populated Characters

Before tier allocation, these character tokens are always present in the vocab
(to enable character-level fallback for words not found in the tier vocabulary):

| Set | Count | Characters |
|-----|-------|------------|
| Latin lowercase | 26 | `a-z` |
| Cyrillic lowercase | 33 | `абвгдежзийклмнопрстуфхцчшщъыьэюяё` |
| Digits | 10 | `0-9` |
| Punctuation | 2 | `'`, `-` |
| **Total** | **71** | |

### Character Fallback

When `encode()` encounters a word not in the vocabulary, it falls through to
character-level encoding using `<sep>` between characters (no trailing `<sep>`):

```
"hello" (in vocab)       →  [idx_hello]
"неизвестен" (not found)  →  н<sep>е<sep>и<sep>з<sep>в<sep>е<sep>с<sep>т<sep>е<sep>н
```

This ensures no text is lost to `<unk>` — any word in any alphabet can be
represented as a sequence of character tokens.

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

---

## File Naming Summary

```
# Data sources
E:\training\data\tinystories.txt
E:\training\data\tinystories.txt.meta.json
E:\training\data2\bulgarian-corpus-33b\chitanka.txt
E:\training\data2\bulgarian-corpus-33b\chitanka.txt.meta.json

# Cache
E:\training\cache\vocab-{vocab_hash}.json
E:\training\cache\vocab-{vocab_hash}.meta.json
E:\training\cache\data-{vocab_hash}-{corpus_h}.npy
E:\training\cache\data-{vocab_hash}-{corpus_h}.meta.json

# Checkpoints
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
     ├─→ vocab_hash  ───→ vocab-{hash}.json  +  vocab-{hash}.meta.json
     │       │
     │       └──→ data-{hash}-{corpus_h}.npy  +  data-{hash}-{corpus_h}.meta.json
     │               │
     │               └──→ corpus_h (full content hash)
     │
     └──→ ckpt_hash (= hash(model_dims, vocab_hash))  ───→ checkpoints/{hash}/
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

---

## Functions (all in `gpt_mini3.py`)

| Function | Signature | Public? |
|----------|-----------|---------|
| `get_vocab_hash()` | `(vocab_cfg: dict, data_dirs: list) -> str` | Yes — imported by DDP scripts |
| `get_model_hash()` | `(model, vocab_hash: str = None) -> str` | Yes — imported by DDP scripts |
| `ensure_corpus()` | `(data_dir, extra_dirs) -> {"sentences": [], "sources": []}` | Yes — imported by DDP scripts |
| `corpus_hash()` | defined locally inside `train()` | No — local helper |
| `WordTokenizer.build_vocab()` | `(texts, sources, tier_ratios)` | Yes — accepts tier-tagged sources |
| `WordTokenizer.tokenize_word()` | `(word) -> list[int]` | Yes — word or char-fallback encoding |
| `WordTokenizer.encode()` | `(text) -> list[int]` | Yes — full text encoding |

---

## Design Principles

1. **Single source of truth** — Each hash function is defined once in `gpt_mini3.py`. DDP scripts import it; they do not compute their own hash.
2. **Model-first** — `ckpt_hash` reads from the instantiated model object, not from a config dict, guaranteeing the hash matches the actual tensor shapes.
3. **Vocabulary-bound checkpoints** — `vocab_hash` is embedded in `ckpt_hash`; different vocabularies can never share or collide on checkpoints.
4. **Metadata-first invalidation** — `vocab_hash` uses cheap file metadata (name, size, mtime, tier) to detect changes. `corpus_h` provides a full-content safety net.
5. **No training config in names** — Hyperparameters (`lr`, `batch_size`, `epochs`, gradient accumulation) don't affect file names. Only structural attributes (tokenizer, data files, model dims) do.
6. **Tier-guaranteed representation** — Each tier gets a proportional share of vocab slots, preventing large corpora from dominating and starving smaller-language sources.
7. **Character fallback safety net** — No word is ever truly unknown. Words missing from the tier vocabulary decompose into character tokens, preserving signal for the model.
