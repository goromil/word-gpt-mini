# Code Development

## Description
Guides code changes: architecture modifications, new features, bug fixes, refactoring, and test updates.

## Triggers
"change code", "implement feature", "fix bug", "refactor", "add test", "code change", "modify model"

## Workflow

### Step 1 — Understand the architecture

#### Core modules in `train_gpt.py`
| Section | Line Range | Classes/Functions |
|---------|-----------|-------------------|
| Config | 20-30 | `load_config()`, `config_hash()` |
| Vocab estimation | 37-68 | `_estimate_vocab_size_streaming()` |
| Data streaming | 69-167 | `SentenceIterator` |
| BPE Tokenizer | 169-430 | `BPETokenizer`, `train()`, `encode()`, `decode()` |
| Dataset | 432-550 | `WordDataset`, `__getitem__`, memmap caching |
| Model | 552-680 | `GPTMini`, `CausalSelfAttention`, `Block`, `MLP` |
| Hash functions | 682-719 | `get_vocab_hash()`, `compute_corpus_hash()` |
| Cache building | 722-775 | `ensure_cache_ready()` |
| Checkpointing | 778-900 | `save_checkpoint()`, `_write_tier()`, `find_latest_checkpoint()` |
| Generation | 902-935 | `generate_text()` |
| Training loop | 937-end | `train()`, data loading, DDP setup, epoch loop |

#### Trainer scripts
| Script | GPU Sync | Key Differences |
|--------|----------|-----------------|
| `train_noipc_ddp.py` | CPU all_reduce (gloo) | Gradients moved to CPU for sync, back to GPU |
| `train_ipc_ddp.py` | GPU all_reduce (NCCL) | Standard DDP, requires P2P/NVLink |
| `experimental/train_rpc.py` | Raw TCP sockets | No DDP, manual gradient averaging |

### Step 2 — Rules for modifications

#### Adding a new model component
1. Define class in `train_gpt.py` model section
2. Wire into `GPTMini.__init__()` and `forward()`
3. Verify parameter count with `calc_params.py`
4. Add test to `test_train_gpt.py`

#### Changing config schema
1. Update `train_gpt.json` with new key
2. Update `train_gpt_draft.json` for designer defaults
3. Update `train_designer.py` if it needs to generate the key
4. Update hash functions if the key affects vocab/corpus/model identity

#### Modifying hash functions
1. `get_vocab_hash()` — anything that changes what the vocabulary would be
2. `compute_corpus_hash()` — anything that changes the tokenized output
3. `get_model_hash()` — anything that changes tensor shapes

**Rule:** If the change means old cache would produce wrong results, the hash must change.

#### Adding a new source filter
1. Add prefix to `tokenizer.sources` and/or `training.sources` in `train_gpt.json`
2. Ensure corpus file name starts with that prefix (e.g., `"mydata"` matches `mydata_v1.txt`)
3. Create `.meta.json` with tier assignment

### Step 3 — Test coverage

#### Unit tests (`test_train_gpt.py`)
| Category | Tests | What they verify |
|----------|-------|-----------------|
| Tokenizer | 11 | BPE train, encode, decode, save/load |
| Dataset | 5 | Length, getitem, EOS, empty, boundary |
| Attention | 6 | Init, forward, causal mask, batch, seq |
| Model | 10 | Init, forward, weight tying, layers |
| Checkpoint | 11 | Save, resume, tiers, find, overwrite |
| Config | 6 | Hash stability, exclusion of training/paths |
| Generation | 4 | Text gen, temperature, EOS stop, long prompt |
| Edge cases | 4 | Seq length limits, multiple models |

#### Running tests
```bash
# Full suite (requires torch + sentencepiece)
wsl /home/george/miniconda3/envs/ai/bin/python /mnt/c/Users/gorom/source/ai/word-gpt-mini/test_train_gpt.py

# Syntax check only (no deps needed)
Get-ChildItem -Recurse -Filter "*.py" | ForEach-Object { python -m py_compile $_.FullName }
```

### Step 4 — Validation checklist before commit
1. [ ] All `.py` files compile (`python -m py_compile`)
2. [ ] `test_train_gpt.py` passes (71/71)
3. [ ] No legacy references (`ensure_corpus`, `BPETokenizerLegacy`, inline `_corpus_hash`)
4. [ ] Hash functions produce stable results
5. [ ] Config validates: `python -c "import json; json.load(open('train_gpt.json'))"`

### Step 5 — Documentation updates
When changing public APIs or config schema, update:
- `README.md` — user-facing changes
- `CACHE_CHECKPOINT_SCHEMA.md` — hash/cache/checkpoint changes
- Relevant `.agents/*.md` — workflow changes
