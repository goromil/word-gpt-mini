#!/usr/bin/env python3
"""
ensure_utf8.py — Detect encoding of corpus files, convert to UTF-8 if needed,
strip HTML entities, and record original encoding in metadata.

Usage:
    python ensure_utf8.py E:\\training\\data              # scan and convert all .txt
    python ensure_utf8.py --dry-run E:\\training\\data     # show what would change
    python ensure_utf8.py file1.txt file2.txt              # specific files
"""
import sys, os, json, re, html
from pathlib import Path
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Common encodings to try (order matters: most likely first)
ENCODINGS = ["utf-8", "utf-8-sig", "cp1251", "iso-8859-5", "windows-1251", "cp866", "latin-1"]

# Regex to detect encoding issues: sequences of replacement chars or mojibake
MOJIBAKE_PATTERN = re.compile(r"[\ufffd]{2,}|[\xc0-\xff]{4,}")


def detect_encoding(file_path: str) -> tuple[str, str]:
    """Detect file encoding by trying common encodings.

    Returns (encoding, quality): quality is 'clean' if utf-8 works,
    'converted' if we had to transcode, 'uncertain' if ambiguous.
    """
    raw = Path(file_path).read_bytes()

    # Try UTF-8 first (most common for modern web content)
    try:
        text = raw.decode("utf-8")
        if not text.strip():
            return "utf-8", "empty"
        # Check for BOM
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig", "converted"
        # Heuristic: if all chars are printable or common whitespace, it's clean
        non_printable = sum(1 for c in text if ord(c) > 127 and not c.isprintable())
        if non_printable == 0:
            return "utf-8", "clean"
        return "utf-8", "uncertain"
    except UnicodeDecodeError:
        pass

    # Try other encodings
    for enc in ENCODINGS[1:]:
        try:
            text = raw.decode(enc)
            if not text.strip():
                return enc, "empty"
            # Validate: check if decoded text makes sense
            if len(text) > 10:
                return enc, "converted"
        except (UnicodeDecodeError, LookupError):
            continue

    # Last resort: latin-1 (never fails, maps bytes 1:1)
    return "latin-1", "uncertain"


def clean_html_entities(text: str) -> str:
    """Decode HTML entities and clean up common artifacts."""
    # Decode standard HTML entities
    text = html.unescape(text)

    # Clean up common HTML artifacts
    text = re.sub(r"&\w+;", " ", text)  # leftover entities
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)  # strip remaining tags

    return text


def ensure_utf8(file_path: str, dry_run: bool = False) -> dict:
    """Convert a file to UTF-8 if needed. Returns status dict."""
    fp = Path(file_path)
    encoding, quality = detect_encoding(str(fp))

    result = {
        "file": str(fp),
        "original_encoding": encoding,
        "quality": quality,
        "converted": False,
    }

    if quality == "empty":
        return result

    if encoding == "utf-8" and quality == "clean":
        # Already clean UTF-8
        # Still clean HTML entities
        text = fp.read_text(encoding="utf-8")
        cleaned = clean_html_entities(text)
        if cleaned != text and not dry_run:
            fp.write_text(cleaned, encoding="utf-8")
            result["html_cleaned"] = True
        return result

    # Need conversion
    if dry_run:
        result["dry_run"] = True
        return result

    # Read in original encoding, clean, write as UTF-8
    text = fp.read_text(encoding=encoding)
    text = clean_html_entities(text)
    fp.write_text(text, encoding="utf-8")
    result["converted"] = True
    result["html_cleaned"] = True

    return result


def update_metadata(file_path: str, original_encoding: str):
    """Update or create .meta.json with original_encoding field."""
    meta_path = Path(str(file_path) + ".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            meta = {}

    old_enc = meta.get("original_encoding", "utf-8")
    meta["original_encoding"] = original_encoding
    meta["utf8_ensured_at"] = datetime.now().isoformat()

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if old_enc != original_encoding:
        return True  # changed
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ensure corpus files are UTF-8")
    parser.add_argument("paths", nargs="*", default=["E:\\training\\data"],
                        help="Files or directories to scan")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying files")
    args = parser.parse_args()

    files = []
    for p in args.paths:
        p = Path(p)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("*.txt")))
        else:
            print(f"Warning: path not found: {p}")

    if not files:
        print("No .txt files found.")
        return

    total = 0
    converted = 0
    cleaned = 0
    for fp in files:
        result = ensure_utf8(str(fp), dry_run=args.dry_run)
        total += 1

        if result["quality"] == "empty":
            status = "EMPTY"
        elif result.get("dry_run"):
            status = f"WOULD CONVERT ({result['original_encoding']} -> utf-8)"
        elif result["converted"]:
            status = f"CONVERTED ({result['original_encoding']} -> utf-8)"
            converted += 1
        elif result.get("html_cleaned"):
            status = "HTML entities cleaned"
            cleaned += 1
        else:
            status = "OK (already UTF-8)"

        print(f"  [{status}] {fp.name} ({fp.stat().st_size / 1024:.0f} KB)")

        if not args.dry_run and result["quality"] != "empty":
            update_metadata(str(fp), result["original_encoding"])

    print(f"\n  Summary: {total} files scanned, {converted} converted, {cleaned} cleaned")


if __name__ == "__main__":
    main()
