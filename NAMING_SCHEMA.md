# Deterministic Naming Schema

## Overview

All cache and checkpoint files are named by **deterministic hashes** that form a
causal dependency chain.  Identical inputs always produce identical names; any
change invalidates only the affected downstream artifacts.

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

**Used for**

- `vocab-{vocab_hash}.json` — cached word vocabulary
- `data-{vocab_hash}-{corpus_h}.npy` — cached tokenized data array (first segment)

**Why it works:**  Changing any source file (new content → new size/mtime), adding a file, or altering tokenizer params all produce a new hash.  The model architecture is intentionally excluded — the vocabulary and tokenizer are independent of `n_layer`, `n_head`, etc.

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

## What Is NOT Included (and Why)

| Parameter | Excluded from | Reason |
|-----------|---------------|--------|
| `seq_length` | `vocab_hash`, `corpus_h` | The `.npy` cache is a flat token array; `seq_length` only controls runtime window slicing in `__getitem__`. |
| `batch_size` | everything | A DataLoader parameter, has zero effect on cached data or model weights. |
| `lr`, `epochs`, `gradient_accumulation_steps` | `ckpt_hash` | Training hyperparameters don't change model architecture or vocabulary. Different hyperparameters should resume from the same checkpoint. |
| `n_layer`, `n_head`, `head_dim` | `vocab_hash` | Model architecture doesn't affect tokenization. |

---

## File Naming Summary

```
E:\training\cache\vocab-{vocab_hash}.json
E:\training\cache\data-{vocab_hash}-{corpus_h}.npy
E:\training\checkpoints\{ckpt_hash}\model.pth
E:\training\checkpoints\{ckpt_hash}\resume.json
E:\training\checkpoints\{ckpt_hash}\config.json
E:\training\checkpoints\{ckpt_hash}\checkpoint_status.txt
E:\training\checkpoints\{ckpt_hash}\{tier}\...     # tiered checkpoints
```

---

## Dependency Chain

```
data files (*.txt)
    │
    ├─→ vocab_hash  ───→ vocab-{vocab_hash}.json
    │       │
    │       └──→ data-{vocab_hash}-{corpus_h}.npy
    │               │
    │               └──→ corpus_h (full content hash)
    │
    └──→ ckpt_hash (= hash(model_dims, vocab_hash))  ───→ checkpoints/{ckpt_hash}/
```

**Propagation rule:**  Changing any upstream input invalidates all downstream
artifacts automatically via the new hash.

---

## Functions (all in `gpt_mini3.py`)

| Function | Signature | Public? |
|----------|-----------|---------|
| `get_vocab_hash()` | `(vocab_cfg: dict, data_dirs: list) -> str` | Yes — imported by DDP scripts |
| `get_model_hash()` | `(model, vocab_hash: str = None) -> str` | Yes — imported by DDP scripts |
| `corpus_hash()` | defined locally inside `train()` | No — local helper |

---

## Design Principles

1. **Single source of truth** — Each hash function is defined once in `gpt_mini3.py`. DDP scripts import it; they do not compute their own hash.
2. **Model-first** — `ckpt_hash` reads from the instantiated model object, not from a config dict, guaranteeing the hash matches the actual tensor shapes.
3. **Vocabulary-bound checkpoints** — `vocab_hash` is embedded in `ckpt_hash`; different vocabularies can never share or collide on checkpoints.
4. **Metadata-first invalidation** — `vocab_hash` uses cheap file metadata (name, size, mtime) to detect changes. `corpus_h` provides a full-content safety net for the data cache.
5. **No training config in names** — Hyperparameters (`lr`, `batch_size`, `epochs`, gradient accumulation) don't affect file names. Only structural attributes (tokenizer, data files, model dims) do.
