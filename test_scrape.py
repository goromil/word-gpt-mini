from pathlib import Path
from dataset_dl import scrape_api, load_config
import sys

config = load_config("dataset_dl.json")
settings = config["settings"]
settings["max_workers"] = 2

source = config["api"][0]
# Only first query for testing
source["queries"] = [config["api"][0]["queries"][0]]

print("Starting scrape...", file=sys.stderr)
result = scrape_api(source, settings)
print(f"Result: {result}", file=sys.stderr)

# Check output
out = Path("E:\\training\\data2\\chitanka-xml_combined.txt")
if out.exists():
    print(f"Output file: {out.stat().st_size} bytes", file=sys.stderr)
    with open(out, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Lines: {len(lines)}", file=sys.stderr)
    print(f"First 200 chars:", file=sys.stderr)
    print("".join(lines[:5]), file=sys.stderr)
