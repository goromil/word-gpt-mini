import json, math, sys, os


# ---------------------------------------------------------------------------
# CLI: python init_training.py [draft.json] [--download] [--scan-vocab] [--force]
#   draft      = path to draft config (default: gpt_mini3_draft.json)
#   --download = download corpus automatically
#   --scan-vocab = scan corpus to determine max_vocab_size (round up to pow 2)
#   --force    = overwrite existing gpt_mini3.json
# ---------------------------------------------------------------------------
FLAGS = {"--download", "--scan-vocab", "--force"}
args = [a for a in sys.argv[1:] if a not in FLAGS]
download = "--download" in sys.argv
scan_vocab = "--scan-vocab" in sys.argv
force = "--force" in sys.argv
draft_path = args[0] if len(args) > 0 else "gpt_mini3_draft.json"


def next_pow2(x):
    """Round x up to next power of 2."""
    return 1 << (x - 1).bit_length()


def estimate_vocab(filepath, sample_mb=200, max_word_len=20):
    """Sample first sample_mb of file, count unique words (same logic as WordTokenizer)."""
    print(f"  Scanning vocab from {filepath} (sampling {sample_mb} MB)...")
    import string
    translator = str.maketrans("", "", string.punctuation)
    sample_bytes = sample_mb * 1024 * 1024
    words = set()
    total_words = 0
    total_read = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        while total_read < sample_bytes:
            chunk = f.read(1024*1024)  # 1MB chunks
            if not chunk:
                break
            total_read += len(chunk)
            for line in chunk.split("\n"):
                for word in line.lower().split():
                    w = word.translate(translator)
                    if 1 <= len(w) <= max_word_len:
                        words.add(w)
                        total_words += 1
            if total_read % (50*1024*1024) == 0:
                print(f"\r    Read {total_read//(1024*1024)} MB, {len(words):,} unique words", end="", flush=True)
    print()
    estimated = int(len(words) * 1.05)
    rounded = next_pow2(estimated)
    print(f"  Sampled words: {len(words):,} unique, {total_words:,} total")
    print(f"  Estimated (x1.05): {estimated:,}")
    print(f"  Rounded to pow2: {rounded:,}")
    return rounded


def calc_params(m, t):
    """Calculate total model parameters."""
    n_layer = m["n_layer"]
    n_head = m["n_head"]
    head_dim = m["head_dim"]
    seq_length = m["seq_length"]
    nvocab = t["max_vocab_size"]
    n_embd = n_head * head_dim

    wte = nvocab * n_embd
    wpe = seq_length * n_embd
    c_attn = n_embd * (3*n_embd) + 3*n_embd
    c_proj = (3*n_embd) * n_embd + n_embd
    mlp = (n_embd*(4*n_embd)+4*n_embd) + ((4*n_embd)*n_embd+n_embd)
    ln = 4 * n_embd
    per_layer = c_attn + c_proj + ln + mlp
    ln_f = 2 * n_embd
    lm_head = 0
    total = wte + wpe + per_layer * n_layer + ln_f + lm_head
    return total, n_embd


def main():
    _scan = scan_vocab
    _download = download
    _force = force

    with open(draft_path, "r") as f:
        draft = json.load(f)

    m = draft["model"]
    vocab_cfg = m.get("vocab", {})
    td = draft.get("training-defaults", {})
    paths = draft.get("paths", {})

    # Model params
    total_params, n_embd = calc_params(m, vocab_cfg)
    n_layer = m["n_layer"]
    n_head = m["n_head"]
    head_dim = m["head_dim"]
    block = m["seq_length"]
    bs = td.get("batch_size", 256)
    nvocab = vocab_cfg.get("max_vocab_size", 32768)

    # -----------------------------------------------------------------------
    # EPOCH CALCULATION
    # -----------------------------------------------------------------------
    corpora = {
        "TinyStories-train (full)": {
            "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
            "size_mb": 2300,
            "est_tokens": 2_200_000_000,
            "desc": "Full training set, ~2.2B tokens",
        },
        "TinyStories-v2 GPT4": {
            "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
            "size_mb": 1800,
            "est_tokens": 1_800_000_000,
            "desc": "GPT-4 only, higher quality",
        },
    }

    print(f"{'='*65}")
    print(f"  EPOCH CALCULATOR")
    print(f"  Formula: epochs = ceil( (params x N) / corpus_tokens )")
    print(f"  Chinchilla-optimal: N = 200")
    print(f"{'='*65}")
    print(f"  Model: {n_layer}L / {n_head}H / hd={head_dim} / emb={n_embd} / vocab={nvocab}")
    print(f"  Params: {total_params:,}  (~{total_params/1e6:.1f}M)")
    print(f"{'='*65}")
    print(f"")

    for N in [20, 50, 100, 200]:
        target = total_params * N
        for name, info in corpora.items():
            ratio = info["est_tokens"] / total_params
            epochs = math.ceil(target / info["est_tokens"])
            batches_per_epoch = (info["est_tokens"] // block) // bs
            print(f"  N={N:>3} | {name:<25} -> {epochs:>4} epochs  ({batches_per_epoch:>10,} batches/epoch)")
        print(f"")

    # Pick best corpus: smallest download where epochs <= 50
    corpus_order = sorted(corpora.keys(), key=lambda n: corpora[n]["size_mb"])
    recommended = None
    for name in corpus_order:
        rec = corpora[name]
        ep = math.ceil((total_params * 50) / rec["est_tokens"])
        if ep <= 50:
            recommended = name
            break
    if recommended is None:
        recommended = corpus_order[-1]

    rec = corpora[recommended]
    epochs_rec = math.ceil((total_params * 50) / rec["est_tokens"])

    print(f"{'='*65}")
    print(f"  RECOMMENDATION  (N=50)")
    print(f"{'='*65}")
    print(f"  Corpus:    {recommended}  ({rec['size_mb']} MB)")
    print(f"  Epochs:    {epochs_rec}")
    print(f"  Total tok: {epochs_rec * rec['est_tokens']:>14,}  ({(epochs_rec*rec['est_tokens'])/total_params:.1f}x params)")
    print(f"")

    # Download if requested
    data_dir = paths.get("data_dir", "E:\\training\\data")
    outfile = os.path.join(data_dir, "tinystories.txt")

    if _download:
        print(f"  Downloading to {outfile} ...")
        import time
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        import requests
        sess = requests.Session()
        retry = Retry(total=5, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504])
        sess.mount("https://", HTTPAdapter(max_retries=retry))
        state = {"pct": -1, "t": time.time(), "total": 0, "done": 0}
        resume_from = 0
        if os.path.exists(outfile):
            resume_from = os.path.getsize(outfile)
            if resume_from > 0:
                print(f"  Resuming from {resume_from / (1024*1024):.1f} MB ...")
        mode = "ab" if resume_from > 0 else "wb"
        headers = {}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
        try:
            resp = sess.get(rec['url'], headers=headers, stream=True, timeout=120)
        except Exception as e:
            print(f"\n  Error: {e}")
            print(f"  Use: aria2c -x 16 -s 16 -j 1 --continue -d \"{data_dir}\" -o tinystories.txt \"{rec['url']}\"")
            sys.exit(1)
        total = int(resp.headers.get("Content-Length", 0))
        total_expected = resume_from + total
        state["total"] = total_expected
        state["done"] = resume_from
        with open(outfile, mode) as f:
            for chunk in resp.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                state["done"] += len(chunk)
                now = time.time()
                pct = int(state["done"] * 100 / state["total"]) if state["total"] > 0 else 0
                if pct != state["pct"] or (now - state["t"]) > 5:
                    dl_mb = state["done"] / (1024*1024)
                    tot_mb = state["total"] / (1024*1024)
                    print(f"\r  [{pct:3d}%] {dl_mb:8.1f} / {tot_mb:8.1f} MB", end="", flush=True)
                    state["pct"] = pct
                    state["t"] = now
        size_mb = os.path.getsize(outfile) / (1024*1024)
        print(f"\n  Done! File size: {size_mb:.1f} MB")
        _scan = True

    # Scan vocab if requested and file exists
    new_vocab = vocab_cfg.get("max_vocab_size", 32768)
    if _scan and os.path.exists(outfile):
        new_vocab = estimate_vocab(outfile, sample_mb=200, max_word_len=vocab_cfg.get("max_word_len", 20))
        print(f"  Using vocab_size = {new_vocab:,}")

    # LR selection based on epochs
    if epochs_rec >= 50:
        new_lr = 0.00015
    elif epochs_rec >= 10:
        new_lr = 0.0002
    else:
        new_lr = 0.0003

    # Build training section from defaults + calculated values
    training = {
        "epochs": epochs_rec,
        "batch_size": td.get("batch_size", 256),
        "lr": td.get("lr", new_lr),
        "checkpoint_every": max(1, epochs_rec // 5),
        "checkpoint_interval": td.get("checkpoint_interval", 10000),
        "checkpoint_every_min": td.get("checkpoint_every_min", 30)
    }

    # Remove model-level keys that belong in model
    model_out = {
        "n_layer": m["n_layer"],
        "n_head": m["n_head"],
        "head_dim": m["head_dim"],
        "seq_length": m["seq_length"],
    }

    # Build final config
    new_cfg = {
        "model": model_out,
        "training": training,
        "tokenizer": {
            "max_vocab_size": new_vocab,
            "max_word_len": vocab_cfg.get("max_word_len", 20)
        },
        "paths": paths
    }

    out_file = "gpt_mini3.json"

    # Force check
    if os.path.exists(out_file) and not _force:
        print(f"  ERROR: {out_file} already exists. Use --force to overwrite.")
        sys.exit(1)

    with open(out_file, "w") as f:
        json.dump(new_cfg, f, indent=2)

    print(f"  New config -> {out_file}")
    print(f"    epochs:    {epochs_rec}")
    print(f"    lr:        {new_lr}")
    print(f"    batch_size:{training['batch_size']}")
    print(f"    vocab:     {new_vocab}")
    print(f"    ckpt:      every {training['checkpoint_every']} epochs / {training['checkpoint_interval']} batches / {training['checkpoint_every_min']} min")
    print(f"")

    # Embedding table info
    emb_params = new_vocab * n_embd
    emb_mb = emb_params * 2 / (1024 * 1024)  # FP16
    print(f"  Embedding table: {new_vocab:,} x {n_embd} = {emb_params:,} params ({emb_mb:.1f} MB FP16)")
    print(f"{'='*65}")

    # Manual download commands if file not found
    if not os.path.exists(outfile):
        print(f"  DOWNLOAD:")
        print(f"  aria2c -x 16 -s 16 -j 1 -d \"{data_dir}\" -o tinystories.txt \"{rec['url']}\"")
        print(f"")

    # Cleanup draft if force used and file was overwritten
    if _force and os.path.exists(out_file):
        print(f"  Overwrote {out_file} (--force)")


if __name__ == "__main__":
    main()
