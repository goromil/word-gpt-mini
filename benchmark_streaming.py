"""Benchmark suite for streaming corpus/vocab/dataset pipeline.

Measures:
  1. Hash computation speed (vocab_hash, corpus_hash)
  2. SentenceIterator iteration speed (sentences/sec, memory footprint)
  3. Tokenizer training speed (tokens/sec via SentencePiece)
  4. WordDataset caching speed (tokens/sec written to .npy)
  5. End-to-end pipeline throughput

Usage:
  python benchmark_streaming.py                     # run all benchmarks
  python benchmark_streaming.py --hash              # hash benchmarks only
  python benchmark_streaming.py --iterator          # iterator benchmarks only
  python benchmark_streaming.py --tokenizer         # tokenizer benchmarks only
  python benchmark_streaming.py --dataset           # dataset benchmarks only
  python benchmark_streaming.py --end-to-end        # full pipeline (slow)
"""

import sys
import time
import json
import os
from pathlib import Path
from dataclasses import dataclass, field

import train_gpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    metrics: dict = field(default_factory=dict)


results: list[BenchResult] = []


def run_bench(name, fn):
    """Run a benchmark function, capture wall-time, store result."""
    print(f"\n{'='*60}")
    print(f"BENCH: {name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    metrics = fn()
    elapsed = time.perf_counter() - t0
    metrics["elapsed_s"] = round(elapsed, 3)
    results.append(BenchResult(name, metrics))
    print(f"  elapsed: {elapsed:.3f}s")
    for k, v in metrics.items():
        if k != "elapsed_s":
            print(f"  {k}: {v}")


def load_config():
    with open("train_gpt.json", "r") as f:
        return json.load(f)


def get_paths(config):
    return config.get("paths", {})


def get_vocab_cfg(config):
    """Extract tokenizer/vocab config matching train()'s logic."""
    if "tokenizer" in config:
        return dict(config["tokenizer"])
    if "vocab" in config:
        return dict(config["vocab"])
    model_cfg = dict(config.get("model", {}))
    val = model_cfg.pop("tokenizer", None) or model_cfg.pop("vocab", None)
    return val if isinstance(val, dict) else {}


def get_model_cfg(config):
    """Extract model config without tokenizer sub-keys."""
    cfg = dict(config.get("model", {}))
    cfg.pop("tokenizer", None)
    cfg.pop("vocab", None)
    return cfg


# ---------------------------------------------------------------------------
# 1. Hash benchmarks
# ---------------------------------------------------------------------------

def bench_vocab_hash():
    """Measure vocab_hash computation speed."""
    config = load_config()
    paths = get_paths(config)
    vocab_cfg = get_vocab_cfg(config)

    data_dirs = [paths["data_dir"]]
    if "extra_data_dirs" in paths:
        data_dirs.extend(paths["extra_data_dirs"])

    # Warm-up
    h = train_gpt.get_vocab_hash(vocab_cfg, data_dirs)
    print(f"  vocab_hash: {h}")

    # Timed runs
    N = 10
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        h2 = train_gpt.get_vocab_hash(vocab_cfg, data_dirs)
        times.append(time.perf_counter() - t0)
        assert h == h2, "Hash not stable!"

    avg_ms = sum(times) / len(times) * 1000
    print(f"  {N} runs, avg: {avg_ms:.2f}ms")
    return {"hash_value": h, "avg_ms": round(avg_ms, 2), "runs": N}


def bench_corpus_hash():
    """Measure corpus_hash computation speed (16-location sampling)."""
    config = load_config()
    paths = get_paths(config)

    data_dirs = [paths["data_dir"]]
    if "extra_data_dirs" in paths:
        data_dirs.extend(paths["extra_data_dirs"])

    # Warm-up
    h = train_gpt.compute_corpus_hash(data_dirs)
    print(f"  corpus_hash: {h}")

    # Timed runs
    N = 5
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        h2 = train_gpt.compute_corpus_hash(data_dirs)
        times.append(time.perf_counter() - t0)
        assert h == h2, "Hash not stable!"

    avg_ms = sum(times) / len(times) * 1000
    print(f"  {N} runs, avg: {avg_ms:.2f}ms")

    # Estimate total corpus size
    total_size = 0
    file_count = 0
    for d in data_dirs:
        for root, _, files in os.walk(d):
            for fn in files:
                fp = Path(root) / fn
                total_size += fp.stat().st_size
                file_count += 1

    gb = total_size / (1024**3)
    return {
        "hash_value": h,
        "avg_ms": round(avg_ms, 2),
        "runs": N,
        "total_corpus_gb": round(gb, 2),
        "file_count": file_count,
    }


def bench_conf_hashes():
    """Measure config-only hash computation speed (no file I/O)."""
    config = load_config()
    vocab_cfg = get_vocab_cfg(config)
    tokenizer_sources = vocab_cfg.get("sources")
    training_sources = config.get("training", {}).get("sources")

    # Warm-up
    vc_h = train_gpt.get_vocab_conf_hash(vocab_cfg, tokenizer_sources)
    cp_h = train_gpt.get_corpus_conf_hash(training_sources)
    print(f"  vocab_conf_hash: {vc_h}")
    print(f"  corpus_conf_hash: {cp_h}")

    # Timed runs
    N = 100
    vc_times = []
    cp_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        train_gpt.get_vocab_conf_hash(vocab_cfg, tokenizer_sources)
        vc_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        train_gpt.get_corpus_conf_hash(training_sources)
        cp_times.append(time.perf_counter() - t0)

    vc_avg = sum(vc_times) / len(vc_times) * 1000
    cp_avg = sum(cp_times) / len(cp_times) * 1000
    print(f"  {N} runs: vocab_conf avg {vc_avg:.3f}ms, corpus_conf avg {cp_avg:.3f}ms")
    return {
        "vocab_conf_hash": vc_h,
        "corpus_conf_hash": cp_h,
        "vocab_conf_avg_ms": round(vc_avg, 3),
        "corpus_conf_avg_ms": round(cp_avg, 3),
        "runs": N,
    }


# ---------------------------------------------------------------------------
# 2. SentenceIterator benchmarks
# ---------------------------------------------------------------------------

def bench_iterator_count():
    """Measure SentenceIterator sentence counting speed."""
    config = load_config()
    paths = get_paths(config)
    vocab_cfg = get_vocab_cfg(config)

    sources = vocab_cfg.get("sources")
    data_dir = paths["data_dir"]
    extra_dirs = paths.get("extra_data_dirs")

    iter_ = train_gpt.SentenceIterator(data_dir, extra_dirs, sources=sources)
    total, tier_counts = iter_.count_sentences()

    return {
        "total_sentences": f"{total:,}",
        "tiers": {f"T{t}": f"{c:,}" for t, c in sorted(tier_counts.items())},
    }


def bench_iterator_memory():
    """Verify SentenceIterator doesn't load full corpus into memory."""
    import tracemalloc

    config = load_config()
    paths = get_paths(config)

    data_dir = paths["data_dir"]
    extra_dirs = paths.get("extra_data_dirs")

    tracemalloc.start()
    iter_ = train_gpt.SentenceIterator(data_dir, extra_dirs)

    # Snapshot after constructor (should be low — iterators are lazy)
    curr, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    curr_mb = curr / (1024**2)
    peak_mb = peak / (1024**2)

    ok = peak_mb < 100  # Should be well under 100 MB (just file handles + metadata)
    status = "PASS" if ok else "WARN"
    return {
        "current_mb": round(curr_mb, 2),
        "peak_mb": round(peak_mb, 2),
        "status": status,
    }


# ---------------------------------------------------------------------------
# 3. Tokenizer benchmarks (dry-run, no SentencePiece training)
# ---------------------------------------------------------------------------

def bench_tokenizer_load():
    """Measure tokenizer load/save speed from cached vocab."""
    config = load_config()
    paths = get_paths(config)
    vocab_cfg = get_vocab_cfg(config)

    cache_dir = Path(paths.get("cache_dir", "E:\\training\\cache"))
    vocab_files = sorted(cache_dir.glob("vocab-*.json"))

    if not vocab_files:
        print("  SKIP: no cached vocab found")
        return {"status": "skipped"}

    vocab_file = str(vocab_files[-1])
    print(f"  Using: {vocab_file}")

    # Benchmark load
    N = 5
    times = []
    for _ in range(N):
        tok = train_gpt.BPETokenizer(
            max_vocab_size=vocab_cfg.get("max_vocab_size", 65536),
            max_word_len=vocab_cfg.get("max_word_len", 20),
        )
        t0 = time.perf_counter()
        tok.load(vocab_file)
        times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    return {
        "vocab_file": vocab_file,
        "vocab_size": tok.vocab_size,
        "load_avg_ms": round(avg_ms, 2),
        "runs": N,
    }


def bench_tokenizer_encode():
    """Measure encode speed on sample sentences."""
    config = load_config()
    paths = get_paths(config)
    vocab_cfg = get_vocab_cfg(config)

    cache_dir = Path(paths.get("cache_dir", "E:\\training\\cache"))
    vocab_files = sorted(cache_dir.glob("vocab-*.json"))

    if not vocab_files:
        print("  SKIP: no cached vocab found")
        return {"status": "skipped"}

    tok = train_gpt.BPETokenizer(
        max_vocab_size=vocab_cfg.get("max_vocab_size", 65536),
        max_word_len=vocab_cfg.get("max_word_len", 20),
    )
    tok.load(str(vocab_files[-1]))

    # Collect ~1000 sample sentences
    sentences = []
    data_dir = paths["data_dir"]
    for fn in sorted(Path(data_dir).glob("*.txt")):
        if fn.name.endswith(".meta.json"):
            continue
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 2:
                    sentences.append(line)
                    if len(sentences) >= 1000:
                        break
        if len(sentences) >= 1000:
            break

    # Benchmark encode
    N = 3
    total_tokens = 0
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        for s in sentences:
            ids = tok.encode(s)
            total_tokens += len(ids)
        times.append(time.perf_counter() - t0)

    avg_s = sum(times) / len(times)
    sent_per_s = len(sentences) / avg_s
    tok_per_s = total_tokens / (avg_s * N)

    return {
        "sentences": len(sentences),
        "tokens_per_sentence_avg": round(total_tokens / (len(sentences) * N), 1),
        "sentences_per_sec": round(sent_per_s, 0),
        "tokens_per_sec": f"{tok_per_s:,.0f}",
    }


# ---------------------------------------------------------------------------
# 4. Dataset benchmarks
# ---------------------------------------------------------------------------

def bench_dataset_cache_hit():
    """Measure dataset loading from cache (should be near-instant)."""
    config = load_config()
    paths = get_paths(config)
    model_cfg = get_model_cfg(config)
    vocab_cfg = get_vocab_cfg(config)
    train_cfg = config.get("training", {})

    cache_dir = Path(paths.get("cache_dir", "E:\\training\\cache"))
    data_files = sorted(cache_dir.glob("data-*.npy"))
    vocab_files = sorted(cache_dir.glob("vocab-*.json"))

    if not data_files or not vocab_files:
        print("  SKIP: no cached dataset found")
        return {"status": "skipped"}

    data_file = data_files[-1]
    print(f"  Using: {data_file.name}")

    tok = train_gpt.BPETokenizer(
        max_vocab_size=vocab_cfg.get("max_vocab_size", 65536),
        max_word_len=vocab_cfg.get("max_word_len", 20),
    )
    tok.load(str(vocab_files[-1]))

    # Load from cache (empty sentences list = cache mode)
    t0 = time.perf_counter()
    dataset = train_gpt.WordDataset(
        [], tok, model_cfg["seq_length"], cache_file=str(data_file)
    )
    elapsed = time.perf_counter() - t0

    npy_size = data_file.stat().st_size / (1024**3)
    return {
        "npy_size_gb": round(npy_size, 2),
        "samples": f"{len(dataset):,}",
        "load_sec": round(elapsed, 3),
        "seq_length": model_cfg["seq_length"],
    }


# ---------------------------------------------------------------------------
# 5. End-to-end pipeline (optional, slow)
# ---------------------------------------------------------------------------

def bench_end_to_end():
    """Full pipeline: hash -> SentenceIterator -> vocab -> dataset.

    This is a SLOW benchmark — only run when you want full numbers.
    Uses a small subset of data for speed.
    """
    config = load_config()
    paths = get_paths(config)
    model_cfg = get_model_cfg(config)
    vocab_cfg = get_vocab_cfg(config)
    train_cfg = config.get("training", {})

    data_dirs = [paths["data_dir"]]
    if "extra_data_dirs" in paths:
        data_dirs.extend(paths["extra_data_dirs"])

    # Phase 1: Hashes
    t0 = time.perf_counter()
    vh = train_gpt.get_vocab_hash(vocab_cfg, data_dirs)
    ch = train_gpt.compute_corpus_hash(data_dirs)
    hash_time = time.perf_counter() - t0

    # Phase 2: Iterator
    t0 = time.perf_counter()
    sources = vocab_cfg.get("sources")
    si = train_gpt.SentenceIterator(paths["data_dir"], paths.get("extra_data_dirs"),
                                     sources=sources)
    total, tier_counts = si.count_sentences()
    iter_time = time.perf_counter() - t0

    return {
        "vocab_hash": vh,
        "corpus_hash": ch,
        "hash_total_ms": round(hash_time * 1000, 2),
        "iterator_sentences": f"{total:,}",
        "iterator_tiers": {f"T{t}": f"{c:,}" for t, c in sorted(tier_counts.items())},
        "iterator_sec": round(iter_time, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary():
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        tag = f"{r.name}"
        metrics_str = ", ".join(f"{k}={v}" for k, v in r.metrics.items())
        print(f"  {tag}: {metrics_str}")


def main():
    args = sys.argv[1:]

    # Determine which benchmarks to run
    run_all = not args or "--all" in args
    targets = {
        "hash": [
            lambda: run_bench("vocab_hash", bench_vocab_hash),
            lambda: run_bench("corpus_hash", bench_corpus_hash),
            lambda: run_bench("conf_hashes", bench_conf_hashes),
        ],
        "iterator": [
            lambda: run_bench("iterator_count", bench_iterator_count),
            lambda: run_bench("iterator_memory", bench_iterator_memory),
        ],
        "tokenizer": [
            lambda: run_bench("tokenizer_load", bench_tokenizer_load),
            lambda: run_bench("tokenizer_encode", bench_tokenizer_encode),
        ],
        "dataset": [
            lambda: run_bench("dataset_cache_hit", bench_dataset_cache_hit),
        ],
        "end-to-end": [
            lambda: run_bench("end_to_end", bench_end_to_end),
        ],
    }

    to_run = []
    if run_all:
        for group in targets.values():
            to_run.extend(group)
    else:
        for arg in args:
            key = arg.lstrip("-")
            if key in targets:
                to_run.extend(targets[key])
            else:
                print(f"Unknown target: {arg}")
                print(f"Available: {', '.join(targets.keys())}")
                sys.exit(1)

    if not to_run:
        print("No benchmarks selected.")
        print(f"Available: {', '.join(targets.keys())}")
        sys.exit(0)

    for fn in to_run:
        fn()

    print_summary()


if __name__ == "__main__":
    main()
