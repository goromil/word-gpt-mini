#!/usr/bin/env python3
"""
dataset_builder.py — Process chitanka downloaded corpus into training format.

Reads downloaded text files, filters by language, deduplicates, outputs TinyStories-style format.

Usage:
    python dataset_builder.py                           # auto-detect from dataset_dl.json
    python dataset_builder.py --input E:\\training\\data2
    python dataset_builder.py --output tinystories_bg.txt
    python dataset_builder.py --combine combined_corpus.txt
"""
import json, sys, os, re
from pathlib import Path


def parse_args():
    args = sys.argv[1:]
    input_dir = None
    output_file = None
    combine = None
    i = 0
    while i < len(args):
        if args[i] == "--input" and i+1 < len(args):
            input_dir = args[i+1]; i += 2
        elif args[i] == "--output" and i+1 < len(args):
            output_file = args[i+1]; i += 2
        elif args[i] == "--combine" and i+1 < len(args):
            combine = args[i+1]; i += 2
        elif args[i].endswith(".json"):
            i += 1
        else:
            i += 1
    return input_dir, output_file, combine


def load_config():
    try:
        with open("dataset_dl.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_text_files(input_dir):
    """Find all text files in the input directory."""
    files = []
    input_path = Path(input_dir)
    if not input_path.exists():
        return files
    for root, _, filenames in os.walk(input_path):
        for fn in filenames:
            if fn.endswith((".txt", ".text")):
                fp = Path(root) / fn
                if fp.stat().st_size > 1000:
                    files.append(fp)
    return sorted(files)


def detect_language(text):
    """Detect language by character ranges."""
    bg = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
    total = max(len(text), 1)
    ratio = bg / total
    if ratio > 0.3:
        return "bg"
    if ratio > 0.1:
        return "cyrillic"
    return "other"


def clean_text(line):
    """Clean a single text line."""
    # Remove control characters
    line = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', line)
    # Normalize whitespace
    line = re.sub(r'[ \t]+', ' ', line)
    return line.strip()


def compute_hash(text, length=200):
    """MD5 hash for deduplication."""
    return hashlib.md5(text[:length].encode('utf-8')).hexdigest()


def build_dataset(input_dir, output_file=None, combine_files=None):
    config = load_config()
    settings = config.get("settings", {}) if config else {}

    input_path = input_dir or settings.get("output_dir", "E:\\training\\data2")
    min_len = settings.get("min_text_length", 50)
    max_len = settings.get("max_text_length", 4096)
    languages = settings.get("languages", ["bg", "ru", "mk"])

    text_files = get_text_files(input_path)
    if not text_files:
        print(f"[ERROR] No text files found in {input_path}")
        return False

    print(f"Found {len(text_files)} text files")

    if combine_files:
        # Combine mode: merge all text files
        print(f"Combining all text files...")
        seen = set()
        line_count = 0
        dup_count = 0
        lang_counts = {}

        with open(combine_files, "w", encoding="utf-8") as f:
            for i, fp in enumerate(text_files):
                print(f"\r  Processing {i+1}/{len(text_files)}: {fp.name}", end="", flush=True)
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            line = clean_text(line)
                            if not line or len(line) < min_len or len(line) > max_len:
                                continue

                            lang = detect_language(line)
                            if languages and lang not in languages and lang != "other":
                                continue

                            h = compute_hash(line)
                            if h in seen:
                                dup_count += 1
                                continue
                            seen.add(h)

                            f.write(line + "\n")
                            line_count += 1
                            lang_counts[lang] = lang_counts.get(lang, 0) + 1
                except Exception as e:
                    print(f"\n  [WARN] {fp}: {e}")

        lang_pct = {k: f"{v/max(line_count,1)*100:.0f}%" for k, v in lang_counts.items()}
        print(f"\n  [DONE] {combine_files}")
        print(f"    {line_count:,} lines, {dup_count:,} duplicates removed")
        print(f"    Languages: {lang_pct}")
        print(f"    Size: {Path(combine_files).stat().st_size / (1024*1024):.0f} MB")

    elif output_file:
        # Single output mode
        print(f"Building {output_file}...")
        seen = set()
        line_count = 0
        dup_count = 0
        lang_counts = {}

        with open(output_file, "w", encoding="utf-8") as f:
            for fp in text_files:
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            line = clean_text(line)
                            if not line or len(line) < min_len or len(line) > max_len:
                                continue

                            lang = detect_language(line)
                            if languages and lang not in languages and lang != "other":
                                continue

                            h = compute_hash(line)
                            if h in seen:
                                dup_count += 1
                                continue
                            seen.add(h)

                            f.write(line + "\n")
                            line_count += 1
                            lang_counts[lang] = lang_counts.get(lang, 0) + 1
                except Exception as e:
                    pass

        print(f"  [DONE] {output_file}")
        print(f"    {line_count:,} lines, {dup_count:,} duplicates")
        print(f"    Languages: {lang_counts}")

    else:
        # Auto mode: per-source output
        print(f"Auto-building per-source datasets...")
        sources = {}
        for fp in text_files:
            src = fp.parent.name
            sources.setdefault(src, []).append(fp)

        for src, files in sources.items():
            out = Path(input_path) / f"{src}_combined.txt"
            seen = set()
            count = 0
            dups = 0
            with open(out, "w", encoding="utf-8") as f:
                for fp in files:
                    try:
                        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                            for line in fh:
                                line = clean_text(line)
                                if not line or len(line) < min_len or len(line) > max_len:
                                    continue
                                h = compute_hash(line)
                                if h in seen:
                                    dups += 1
                                    continue
                                seen.add(h)
                                f.write(line + "\n")
                                count += 1
                    except:
                        pass
            print(f"  {out}: {count:,} lines, {dups:,} dups, {out.stat().st_size / (1024*1024):.0f} MB")


if __name__ == "__main__":
    cli()


def cli():
    import hashlib
    input_dir, output_file, combine = parse_args()
    if not build_dataset(input_dir, output_file, combine):
        sys.exit(1)
