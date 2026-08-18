#!/usr/bin/env python3
"""
gpt_info.py — List checkpoints, vocab/cache, and training data status.

Usage:
    gpt_info                              # show everything from default config
    gpt_info --checkpoints                # checkpoints only
    gpt_info --cache                      # vocab/data cache only
    gpt_info --data                       # training data files only
    gpt_info --config                     # config summary only
    gpt_info --config /path/to/config.json
"""

import json
import sys
import os
import time
from pathlib import Path
from wordgpt.config import get_default_config_path


def fmt_size(nbytes):
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def fmt_ts(epoch_time):
    """Format unix timestamp to readable date."""
    if epoch_time is None:
        return "N/A"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch_time))


def load_config(config_path):
    """Load config from path or default."""
    if not config_path:
        config_path = get_default_config_path()
    cp = Path(config_path)
    if not cp.exists():
        print(f"  Config not found: {cp}")
        sys.exit(1)
    return json.loads(cp.read_text()), str(cp)


def show_config(cfg, config_path):
    """Show config summary."""
    print(f"\n{'='*70}")
    print(f"  CONFIG: {config_path}")
    print(f"{'='*70}")

    model = cfg.get("model", {})
    n_layer = model.get("n_layer", "?")
    n_head = model.get("n_head", "?")
    head_dim = model.get("head_dim", "?")
    seq_length = model.get("seq_length", "?")
    n_embd = int(n_head) * int(head_dim) if n_head != "?" and head_dim != "?" else "?"
    total_params = 0
    if n_embd != "?":
        n_embd_i = n_embd
        total_params = (
            int(cfg.get("tokenizer", {}).get("max_vocab_size", 0)) * n_embd_i +
            int(seq_length) * n_embd_i +
            n_embd_i * (3 * n_embd_i) * 4 * int(n_layer) +  # simplified
            2 * n_embd_i * int(n_layer)
        )

    print(f"  Model:  {n_layer}L / {n_head}H / hd={head_dim} / emb={n_embd}")
    print(f"  SeqLen: {seq_length}")

    train = cfg.get("training", {})
    epochs = train.get("epochs", "?")
    bs = train.get("batch_size", "?")
    lr = train.get("lr", "?")
    ckpt = train.get("checkpoint", {})
    sync = train.get("sync", {})
    grad_accum = sync.get("gradient_accumulation_steps", 1)

    print(f"\n  Training:")
    print(f"    epochs:          {epochs}")
    print(f"    batch_size:      {bs}  (grad_accum: {grad_accum}x)")
    print(f"    lr:              {lr}")
    print(f"    ckpt/batch:      {ckpt.get('every_batch', '?')}")
    print(f"    ckpt/min:        {ckpt.get('every_min', '?')}")
    print(f"    ckpt/epoch:      {ckpt.get('every_epoch', '?')}")

    tok = cfg.get("tokenizer", {})
    print(f"\n  Tokenizer:")
    print(f"    max_vocab_size:  {tok.get('max_vocab_size', '?')}")
    print(f"    max_word_len:    {tok.get('max_word_len', '?')}")
    sources_tok = tok.get("sources", [])
    if sources_tok:
        print(f"    sources:         {', '.join(sources_tok)}")

    train_sources = train.get("sources", [])
    if train_sources:
        print(f"    train sources:   {', '.join(train_sources)}")

    paths = cfg.get("paths", {})
    print(f"\n  Paths:")
    for k, v in paths.items():
        print(f"    {k}: {v}")

    # Current checkpoint pointer
    cp = cfg.get("checkpoint", {})
    if cp:
        print(f"\n  Checkpoint pointer:")
        print(f"    hash:  {cp.get('ckpt_hash', 'N/A')}")
        print(f"    epoch: {cp.get('epoch', '?')}")
        print(f"    loss:  {cp.get('loss', '?')}")


def show_checkpoints(cfg):
    """List all checkpoints with their info."""
    paths = cfg.get("paths", {})
    ckpt_dir = paths.get("checkpoint_dir")
    if not ckpt_dir:
        print("  No checkpoint_dir configured.")
        return

    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        print(f"  Checkpoint dir not found: {ckpt_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  CHECKPOINTS: {ckpt_dir}")
    print(f"{'='*70}")

    ckpt_dirs = sorted([d for d in ckpt_path.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not ckpt_dirs:
        print("  No checkpoints found.")
        return

    print(f"\n  {'Hash':<20} {'Epoch':>5} {'Loss':>8} {'Batch':>10} {'Samples':>12} {'Tier':>5} {'Age':>10}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*10} {'-'*12} {'-'*5} {'-'*10}")

    now = time.time()
    for ckpt_hash_dir in ckpt_dirs:
        hash_name = ckpt_hash_dir.name
        # Check base tier (0)
        resume = ckpt_hash_dir / "resume.json"
        if not resume.exists():
            continue

        info = json.loads(resume.read_text())
        epoch = info.get("epoch", "?")
        loss = info.get("loss", "?")
        global_batch = info.get("global_batch", "?")
        samples = info.get("training_samples", "?")
        mtime = resume.stat().st_mtime
        age_sec = now - mtime
        if age_sec < 60:
            age = f"{int(age_sec)}s"
        elif age_sec < 3600:
            age = f"{int(age_sec/60)}m"
        elif age_sec < 86400:
            age = f"{int(age_sec/3600)}h"
        else:
            age = f"{int(age_sec/86400)}d"

        print(f"  {hash_name:<20} {epoch:>5} {loss:>8.4f} {global_batch:>10,} {samples:>12,} {'0':>5} {age:>10}")

        # Check higher tiers
        for tier_dir in sorted(ckpt_hash_dir.iterdir(), key=lambda d: d.name):
            if not tier_dir.is_dir():
                continue
            tier_resume = tier_dir / "resume.json"
            if not tier_resume.exists():
                continue
            tier_info = json.loads(tier_resume.read_text())
            t_epoch = tier_info.get("epoch", "?")
            t_loss = tier_info.get("loss", "?")
            t_batch = tier_info.get("global_batch", "?")
            t_samples = tier_info.get("training_samples", "?")
            t_mtime = tier_resume.stat().st_mtime
            t_age_sec = now - t_mtime
            if t_age_sec < 60:
                t_age = f"{int(t_age_sec)}s"
            elif t_age_sec < 3600:
                t_age = f"{int(t_age_sec/60)}m"
            elif t_age_sec < 86400:
                t_age = f"{int(t_age_sec/3600)}h"
            else:
                t_age = f"{int(t_age_sec/86400)}d"
            tier = tier_dir.name
            print(f"  {'':20} {t_epoch:>5} {t_loss:>8.4f} {t_batch:>10,} {t_samples:>12,} {tier:>5} {t_age:>10}")

    # Status log
    print(f"\n  Recent status entries (last 5):")
    print(f"  {'Time':<20} {'Epoch':>5} {'Batch':>10} {'Loss':>8} {'Tok/s':>10}")
    print(f"  {'-'*20} {'-'*5} {'-'*10} {'-'*8} {'-'*10}")
    for ckpt_hash_dir in reversed(ckpt_dirs):
        status_file = ckpt_hash_dir / "checkpoint_status.txt"
        if not status_file.exists():
            continue
        lines = status_file.read_text().strip().split("\n")
        if not lines:
            continue
        for line in lines[-5:]:
            parts = line.split("\t")
            if len(parts) >= 5:
                ts_str = parts[0][:16]
                ep = parts[1]
                batch_str = parts[2].replace(",", "") if "," in parts[2] else parts[2]
                loss_val = parts[3]
                tok_s_str = parts[4].replace(",", "") if "," in parts[4] else parts[4]
                try:
                    batch_num = int(batch_str)
                    tok_s_num = int(tok_s_str)
                    print(f"  {ts_str:<20} {ep:>5} {batch_num:>10,} {loss_val:>8} {tok_s_num:>10,}")
                except ValueError:
                    print(f"  {ts_str:<20} {ep:>5} {batch_str:>10} {loss_val:>8} {tok_s_str:>10}")
        break  # Only show latest checkpoint's status


def show_cache(cfg):
    """List vocab and data cache files."""
    paths = cfg.get("paths", {})
    cache_dir = paths.get("cache_dir")
    if not cache_dir:
        print("  No cache_dir configured.")
        return

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"  Cache dir not found: {cache_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  CACHE: {cache_dir}")
    print(f"{'='*70}")

    now = time.time()
    # Vocab files
    vocab_files = sorted(cache_path.glob("vocab-*.json"))
    print(f"\n  Vocab cache ({len(vocab_files)}):")
    if vocab_files:
        print(f"  {'File':<60} {'Size':>8} {'Age':>8}")
        print(f"  {'-'*60} {'-'*8} {'-'*8}")
        for vf in vocab_files:
            sz = fmt_size(vf.stat().st_size)
            mtime = vf.stat().st_mtime
            age_sec = now - mtime
            if age_sec < 3600:
                age = f"{int(age_sec/60)}m"
            elif age_sec < 86400:
                age = f"{int(age_sec/3600)}h"
            else:
                age = f"{int(age_sec/86400)}d"
            print(f"  {vf.name:<60} {sz:>8} {age:>8}")
    else:
        print("    (none)")

    # Data files
    data_files = sorted(cache_path.glob("data-*.npy"))
    print(f"\n  Data cache ({len(data_files)}):")
    if data_files:
        print(f"  {'File':<60} {'Size':>10} {'Age':>8}")
        print(f"  {'-'*60} {'-'*10} {'-'*8}")
        for df in data_files:
            sz = fmt_size(df.stat().st_size)
            mtime = df.stat().st_mtime
            age_sec = now - mtime
            if age_sec < 3600:
                age = f"{int(age_sec/60)}m"
            elif age_sec < 86400:
                age = f"{int(age_sec/3600)}h"
            else:
                age = f"{int(age_sec/86400)}d"
            print(f"  {df.name:<60} {sz:>10} {age:>8}")

        # Meta files
        meta_files = sorted(cache_path.glob("*.npy.meta.json"))
        if meta_files:
            print(f"\n  Data metadata ({len(meta_files)}):")
            for mf in meta_files:
                try:
                    meta_info = json.loads(mf.read_text())
                    tokens = meta_info.get("tokens", meta_info.get("token_count", "?"))
                    samples = meta_info.get("samples", meta_info.get("sample_count", "?"))
                    print(f"    {mf.name}: {tokens:,} tokens, {samples} samples")
                except Exception:
                    print(f"    {mf.name}: (unreadable)")
    else:
        print("    (none)")

    # Check cache locks in checkpoints
    ckpt_dir = paths.get("checkpoint_dir")
    if ckpt_dir:
        ckpt_path = Path(ckpt_dir)
        if ckpt_path.exists():
            locks = list(ckpt_path.glob("*/cache_lock.json"))
            if locks:
                print(f"\n  Active cache locks ({len(locks)}):")
                for lock in locks:
                    try:
                        lock_info = json.loads(lock.read_text())
                        print(f"    {lock.parent.name}:")
                        print(f"      vocab: {lock_info.get('vocab_cache', '?')}")
                        print(f"      data:  {lock_info.get('data_cache', '?')}")
                    except Exception:
                        print(f"    {lock.parent.name}: (unreadable)")


def show_data(cfg):
    """List training data files."""
    paths = cfg.get("paths", {})
    data_dir = paths.get("data_dir")
    if not data_dir:
        print("  No data_dir configured.")
        return

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"  Data dir not found: {data_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  TRAINING DATA: {data_dir}")
    print(f"{'='*70}")

    txt_files = sorted(data_path.glob("*.txt"))
    if not txt_files:
        print("  No .txt files found.")
        return

    print(f"\n  {'File':<45} {'Size':>10}")
    print(f"  {'-'*45} {'-'*10}")
    total_size = 0
    for tf in txt_files:
        sz = tf.stat().st_size
        total_size += sz
        print(f"  {tf.name:<45} {fmt_size(sz):>10}")

    print(f"\n  Total: {len(txt_files)} files, {fmt_size(total_size)}")

    # Check extra_data_dirs
    extra_dirs = paths.get("extra_data_dirs", [])
    if extra_dirs:
        print(f"\n  Extra data dirs:")
        for ed in extra_dirs:
            ep = Path(ed)
            if not ep.exists():
                print(f"    {ed} — NOT FOUND")
                continue
            ext_txt = sorted(ep.glob("*.txt"))
            dir_size = sum(f.stat().st_size for f in ext_txt)
            print(f"    {ed}: {len(ext_txt)} files, {fmt_size(dir_size)}")

    # Sources filter info
    train = cfg.get("training", {})
    train_sources = train.get("sources", [])
    tok = cfg.get("tokenizer", {})
    tok_sources = tok.get("sources", [])

    if train_sources:
        print(f"\n  Training sources filter: {', '.join(train_sources)}")
    if tok_sources:
        print(f"  Vocab sources filter:    {', '.join(tok_sources)}")


def parse_args():
    args = sys.argv[1:]
    flags = {"--checkpoints", "--cache", "--data", "--config"}
    config_path = None
    sections = set()

    i = 0
    while i < len(args):
        if args[i] in flags:
            sections.add(args[i])
        elif args[i] == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            sections.add(args[i])
        else:
            # Bare path arg
            if Path(args[i]).exists():
                config_path = args[i]
        i += 1

    if not sections:
        sections = {"--config", "--checkpoints", "--cache", "--data"}

    return config_path, sections


def main():
    config_path, sections = parse_args()
    cfg, resolved_path = load_config(config_path)

    if "--config" in sections:
        show_config(cfg, resolved_path)

    if "--checkpoints" in sections:
        show_checkpoints(cfg)

    if "--cache" in sections:
        show_cache(cfg)

    if "--data" in sections:
        show_data(cfg)


if __name__ == "__main__":
    main()
