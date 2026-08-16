# Corpus Download & Build

## Description
Manages the full data pipeline: downloading corpus from chitanka.info/Wikipedia, processing, tier assignment, and cache invalidation.

## Triggers
"download corpus", "build dataset", "update cache", "corpus status", "data pipeline", "dataset status", "download data"

## Workflow

### Step 1 — Check current data status
```bash
# List corpus files in data dir
Get-ChildItem "E:\training\data" -Filter "*.txt" | Select-Object Name, @{N="SizeMB";E={[math]::Round($_.Length/1MB,1)}}, LastWriteTime | Format-Table

# Check for .meta.json tier assignments
Get-ChildItem "E:\training\data" -Filter "*.meta.json" | ForEach-Object {
    Write-Host "$($_.Name) -> $((Get-Content $_.FullName) -join '')"
}

# Check download registry
if (Test-Path "dataset_dl_registry.json") {
    python -c "import json; r=json.load(open('dataset_dl_registry.json')); print(f'Queries: {len(r.get(\"queries\",{}))}')"
} else {
    Write-Host "No download registry found — no downloads yet"
}
```

### Step 2 — Download corpus (if needed)

#### Method A: Cumulative downloads (new pipeline)
```bash
# Run cumulative downloader (handles ZIP, HuggingFace, torrents)
python dataset_cumulative_dl.py
```
This processes entries from `dataset_dl.json` under `"cumulative"` and `"torrents"`:
- `chitanka-epub` — full epub archive from pechkov.chitanka.info
- `wiki-en` — English Wikipedia from HuggingFace (41 parquet shards)
- Torrent entries — Bulgarian literature archive

#### Method B: API scraping (chitanka.info)
```bash
# Single pass download
python dataset_dl.py --api chitanka-xml

# Auto-retry mode (recommended for Cloudflare-blocked sites)
python dataset_dl.py --api chitanka-xml --max-cycles 100 --cycle-interval 1800

# Retry only blocked queries
python dataset_dl.py --retry --api chitanka-xml

# Check status
python dataset_dl.py --status --api chitanka-xml
```

### Step 3 — Build corpus files
```bash
# Process downloaded texts into training format
python dataset_builder.py

# Or with explicit input/output
python dataset_builder.py --input "E:\training\dataset_dl\extracted" --combine "E:\training\data\chitanka_combined.txt"
```

### Step 4 — Assign tiers and languages
For each corpus file, create `.meta.json`:
```json
// E:\training\data\tinystories.txt.meta.json
{ "tier": 1, "language": "en" }

// E:\training\data\chitanka_epub_corpus.txt.meta.json
{ "tier": 2, "language": "bg" }
```

Tier allocation (from `CACHE_CHECKPOINT_SCHEMA.md`):
| Tier | Ratio | Purpose |
|------|-------|---------|
| 1 | 66% | Primary language (English) |
| 2 | 22% | Secondary language |
| 3 | 12% | Tertiary language |

### Step 5 — Update source filtering
Edit `train_gpt.json` to control which corpora are used:
```json
"tokenizer": {
    "sources": ["tinystories", "wikipedia_en_corpus", "chitanka_epub_corpus"]
},
"training": {
    "sources": ["tinystories", "wikipedia_en_corpus", "chitanka_test_corpus", "chitanka_epub_corpus"]
}
```
`sources` filters `.txt` files by **name prefix**. File `tinystories.txt` matches `"tinystories"`.

### Step 6 — Invalidate and rebuild cache
```bash
# Check current hashes
python -c "
import json, hashlib
from train_gpt import get_vocab_hash, compute_corpus_hash
cfg = json.load(open('train_gpt.json'))
dirs = [cfg['paths']['data_dir']]
vh = get_vocab_hash(cfg['tokenizer'], dirs)
ch = compute_corpus_hash(dirs)
print(f'vocab_hash: {vh}')
print(f'corpus_hash: {ch}')
"

# List cached files to compare
Get-ChildItem "E:\training\cache" | Select-Object Name

# Force rebuild (delete old cache or pass --force-cache to trainer)
Remove-Item "E:\training\cache\data-*.npy" -Force
```

### Step 7 — Verify pipeline
```bash
# Benchmark the streaming pipeline
python benchmark_streaming.py --hash --iterator
```

Expected output:
- `vocab_hash` — stable, < 1ms
- `corpus_hash` — stable, < 1ms
- `total_sentences` — matches expected corpus size
- `peak_mb` — < 100MB (iterator is lazy)

## Key Files
| File | Purpose |
|------|---------|
| `dataset_dl.json` | Download sources: cumulative, torrents, API |
| `dataset_dl.py` | Playwright-based scraper for chitanka.info |
| `dataset_cumulative_dl.py` | New pipeline: ZIP, HuggingFace, torrents |
| `dataset_builder.py` | Language filter, dedup, combine |
| `benchmark_streaming.py` | Hash/iterator benchmarks |
| `E:\training\data\` | Corpus files + `.meta.json` tier assignments |
| `E:\training\cache\` | Cached vocab and dataset |
| `E:\training\dataset_dl\downloads\` | Raw downloaded files |
| `E:\training\dataset_dl\extracted\` | Extracted text files |

## Common Issues
| Symptom | Fix |
|---------|-----|
| "No module named 'playwright'" | `pip install playwright` then `playwright install chromium` |
| Cloudflare block (1015) | Use `--retry` with `--cycle-interval 1800` (30 min) |
| HuggingFace download fails | Check `.hf_token.json` exists with valid token |
| Cache not updating after data change | Delete old `data-*.npy`, `compute_corpus_hash` will change |
| Tier assignment ignored | Check `.meta.json` filename matches exactly: `file.txt.meta.json` |
| Source filter not matching | `sources` prefix must match start of filename (e.g., `"tinystories"` matches `tinystories.txt`) |
