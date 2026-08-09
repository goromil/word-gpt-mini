#!/usr/bin/env python3
"""
dataset_dl.py — Download TinyStories-compatible corpus from chitanka.info.

Uses Playwright (headless Chrome) to bypass Cloudflare 1015 protection.
Uses curl for metadata XML (no Cloudflare challenge needed).

Features:
- Automatic retry cycles for blocked/rate-limited queries
- Checkpoint-based resume (restart and continue where you left off)
- Tracks successes and failures in a JSON registry
- Resilient to crashes/interrupts — safe to Ctrl+C and resume

Usage:
    python dataset_dl.py                              # use dataset_dl.json (auto-retry mode)
    python dataset_dl.py --api                        # scrape only (single pass)
    python dataset_dl.py --api chitanka-xml           # specific source
    python dataset_dl.py --retry                      # retry blocked queries only
    python dataset_dl.py --status                     # show download status/registry
    python dataset_dl.py --list                       # list sources
    python dataset_dl.py --max-cycles 100             # limit retry cycles
    python dataset_dl.py --cycle-interval 1800        # seconds between retry cycles (default: 1800 = 30 min)
    python dataset_dl.py --cycles-before-wait 5       # start waiting after N cycles (default: 5)
    python dataset_dl.py --help                       # show this help

Auto-retry mode (--retry or default):
    Runs download cycles repeatedly. Blocked queries are skipped in each cycle,
    then retried in the next cycle after a configurable delay. Continues until
    all queries succeed OR max-cycles is reached.

Single-pass mode (--api):
    Runs one cycle. Blocked queries are recorded in the registry but not retried.
    Use --retry to retry blocked queries later.

Registry (dataset_dl_registry.json):
    Stores per-query status: success/failure counts, blocked text IDs,
    downloaded text IDs, and timestamps. Used for resume and status reporting.

Examples:
    # Download with auto-retry (default, waits 30 min between cycles)
    python dataset_dl.py --api chitanka-xml

    # Download with faster retries (10 min between cycles, max 50 cycles)
    python dataset_dl.py --api chitanka-xml --cycle-interval 600 --max-cycles 50

    # Check current status
    python dataset_dl.py --status --api chitanka-xml

    # Retry only previously blocked queries
    python dataset_dl.py --retry --api chitanka-xml

    # Continue after interrupt (automatic, just run again)
    python dataset_dl.py --api chitanka-xml
"""
import json, sys, os, re, hashlib, time, subprocess, urllib.parse, signal
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

CURL = "curl"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    shutdown_requested = True
    print("\n\n[INTERRUPT] Saving checkpoint and shutting down...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_args():
    args = sys.argv[1:]
    config_path = "dataset_dl.json"
    mode = "all"
    source_name = None
    max_cycles = 100
    cycle_interval = 1800  # 30 minutes
    cycles_before_wait = 5
    i = 0
    while i < len(args):
        if args[i] == "--help" or args[i] == "-h":
            print(__doc__)
            sys.exit(0)
        if args[i] == "--list":
            return "list", None, config_path, 0, 0, 0
        if args[i] == "--api":
            mode = "api"
            if i+1 < len(args) and not args[i+1].startswith("--"):
                source_name = args[i+1]
                i += 1
            i += 1
        elif args[i] == "--retry":
            mode = "retry"
            i += 1
        elif args[i] == "--status":
            mode = "status"
            i += 1
        elif args[i] == "--max-cycles" and i+1 < len(args):
            max_cycles = int(args[i+1])
            i += 2
        elif args[i] == "--cycle-interval" and i+1 < len(args):
            cycle_interval = int(args[i+1])
            i += 2
        elif args[i] == "--cycles-before-wait" and i+1 < len(args):
            cycles_before_wait = int(args[i+1])
            i += 2
        elif args[i].endswith(".json"):
            config_path = args[i]
            i += 1
        else:
            i += 1
    return mode, source_name, config_path, max_cycles, cycle_interval, cycles_before_wait


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_registry(output_dir):
    """Load or create download registry."""
    reg_file = Path(output_dir) / "dataset_dl_registry.json"
    if reg_file.exists():
        try:
            return json.loads(reg_file.read_text(encoding='utf-8'))
        except:
            pass
    return {
        "queries": {},
        "last_run": None,
        "total_cycles": 0,
        "completed_at": None,
        "blocked_queries": [],
        "unblocked_queries": []
    }


def save_registry(registry, output_dir):
    """Save download registry."""
    reg_file = Path(output_dir) / "dataset_dl_registry.json"
    registry["last_run"] = datetime.now().isoformat()
    reg_file.write_text(json.dumps(registry, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def fetch_url(url, output_file=None):
    """Fetch URL using curl."""
    cmd = [CURL, "-s", "-L", "-A", UA,
           "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
           "-H", "Accept-Language: bg-BG,bg;q=0.9,en;q=0.8",
           "--connect-timeout", "10", "--max-time", "30", url]
    if output_file:
        cmd.extend(["-o", output_file])
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    else:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            return result.stdout.decode('utf-8')
        return None


def strip_cloudflare(content):
    """Strip Cloudflare challenge script from XML response."""
    idx = content.find('</results>')
    if idx != -1:
        return content[:idx+10]
    return content


def get_metadata_xml(query, base_url="http://chitanka.info/texts/search.xml",
                     by="title", match="prefix", out_file=None, retries=3):
    """Get metadata from XML search API using curl with retry logic."""
    query_enc = urllib.parse.quote(query.encode('utf-8'))
    params = f"q={query_enc}&by={by}&match={match}"
    url = f"{base_url}?{params}"
    
    for attempt in range(retries):
        time.sleep(2 * attempt)
        success = fetch_url(url, out_file)
        if success and out_file:
            content = out_file.read_text(encoding='utf-8')
            content = strip_cloudflare(content)
            out_file.write_text(content, encoding='utf-8')
            if out_file.stat().st_size > 200:
                return True
        if attempt < retries - 1 and not shutdown_requested:
            print(f"\r  [RETRY] Metadata for '{query}' (attempt {attempt+2}/{retries})", end="", flush=True)
    
    return success


def clean_text(html_text):
    """Extract clean story text from HTML."""
    start = html_text.find('Към текста')
    if start == -1:
        start = html_text.find('id="text-content"')
    end = html_text.rfind('Към началото')
    if end == -1:
        end = html_text.find('</body>')
    if start == -1 or end <= start:
        return None

    raw = html_text[start:end]
    raw = re.sub(r'<br\s*/?>', '\n', raw)
    raw = re.sub(r'<p[^>]*>', '\n', raw)
    raw = re.sub(r'</p>', '', raw)
    raw = re.sub(r'<[^>]+>', ' ', raw)
    raw = re.sub(r'\n\s*\n\s*\n+', '\n\n', raw)

    lines = []
    found = False
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        if re.match(r'^(Сваляне|Включено|Традиционна|Характеристика|Вашата|'
                    r'Отбелязване|Сканиране|В други|Коментари|Метаданни|Данни|'
                    r'Информация|История|Добавяне|Добави|Вход|Начална)', line):
            continue
        if any(kw in line for kw in ['Към навигацията', 'Към текста', 'Моята библиотека',
                                      'Читалня', 'Работилница', 'Колекции', 'Проекти',
                                      'Читателите', 'са прочели и']):
            continue
        if re.match(r'^[1-6]\s*[—–-]', line) and len(line) < 40:
            continue
        if re.match(r'^\d{4}\s*$', line):
            continue
        if not found and len(line) < 15:
            continue
        if not found and len(line) >= 15:
            found = True
        if found and 15 <= len(line) <= 2048:
            lines.append(line)
        if line in ('Край', 'Конец', 'КРАЙ'):
            break
    return '\n'.join(lines)


def _get_playwright():
    """Lazy import and init playwright (single browser instance per process)."""
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    return pw, browser


def _fetch_with_playwright(pw, browser, url, max_wait=15, retries=3):
    """Fetch a page using Playwright with retry logic for Cloudflare."""
    for attempt in range(retries):
        page = browser.new_page()
        try:
            page.goto(url, timeout=30000, wait_until='domcontentloaded')
            time.sleep(max_wait)
            
            content = page.content()
            if len(content) > 5000:
                return page
            else:
                page.close()
                if attempt < retries - 1 and not shutdown_requested:
                    time.sleep(5)
        except Exception as e:
            try:
                page.close()
            except:
                pass
            if attempt < retries - 1 and not shutdown_requested:
                time.sleep(5)
    
    return None


def download_batch_playwright(text_ids, out_dir, registry, query_name, cycle):
    """Download multiple text pages using a single Playwright browser instance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = out_dir / ".download_checkpoint.txt"
    downloaded_ids = set()
    if checkpoint.exists():
        downloaded_ids = set(checkpoint.read_text(encoding='utf-8').strip().split('\n'))

    # Load previously saved texts
    prev_output = out_dir / ".prev_texts.json"
    all_texts = []
    if prev_output.exists():
        try:
            all_texts = json.loads(prev_output.read_text(encoding='utf-8'))
        except:
            all_texts = []

    # Load registry for this query
    if "queries" not in registry:
        registry["queries"] = {}
    if query_name not in registry["queries"]:
        registry["queries"][query_name] = {
            "downloaded_ids": [],
            "failed_ids": [],
            "blocked_ids": [],
            "cycles_attempted": 0,
            "last_success": None,
            "total_downloads": 0,
            "last_query": None,
            "query_success": False
        }
    query_reg = registry["queries"][query_name]
    
    # Remove IDs that were in previous failed/blocked lists but succeeded now
    # We keep them in failed_ids but allow retry
    failed_ids = set(query_reg.get("failed_ids", []))
    blocked_ids = set(query_reg.get("blocked_ids", []))
    
    # Download remaining + failed (retry)
    retry_ids = failed_ids - downloaded_ids
    remaining = [tid for tid in text_ids if tid not in downloaded_ids]
    to_download = remaining + sorted(retry_ids)
    to_download = list(dict.fromkeys(to_download))  # unique, preserve order
    
    print(f"  [CYCLE {cycle}] Downloading {len(to_download)} texts "
          f"({len(remaining)} new, {len(retry_ids)} failed retry, "
          f"{len(downloaded_ids)} previously done, "
          f"{len(blocked_ids)} blocked)")

    print("  [PW] Initializing Playwright...")
    pw, browser = _get_playwright()

    new_texts = []
    new_success = 0
    new_fail = 0
    newly_downloaded = set()

    for idx, text_id in enumerate(to_download):
        if shutdown_requested:
            break

        out_file = out_dir / f"{text_id}.html"

        if out_file.exists() and out_file.stat().st_size > 1000:
            html = out_file.read_text(encoding='utf-8')
        else:
            url = f"http://chitanka.info/text/{text_id}"
            page = _fetch_with_playwright(pw, browser, url, max_wait=5, retries=2)
            if page is None:
                new_fail += 1
                time.sleep(3)
                continue
            html = page.content()
            page.close()
            out_file.write_text(html, encoding='utf-8')

        text = clean_text(html)
        if text and len(text) > 100:
            body_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if body_hash not in [hashlib.md5(t.encode('utf-8')).hexdigest() for t in all_texts]:
                all_texts.append(text)
                new_texts.append(text)
            new_success += 1
            newly_downloaded.add(text_id)
        else:
            new_fail += 1

        downloaded_ids.add(text_id)
        newly_downloaded.add(text_id)

        # Save checkpoint periodically
        if len(newly_downloaded) % 10 == 0:
            checkpoint.write_text('\n'.join(sorted(downloaded_ids)), encoding='utf-8')
            prev_output.write_text(json.dumps(all_texts), encoding='utf-8')

        if (idx + 1) % 20 == 0 or idx + 1 == len(to_download):
            total_done = len(downloaded_ids)
            print(f"\r  [CYCLE {cycle}] {idx+1}/{len(to_download)} processed "
                  f"({total_done}/{len(text_ids)} total done, "
                  f"{len(all_texts):,} unique texts)", end="", flush=True)

        time.sleep(2)

    browser.close()
    pw.stop()

    # Update registry
    if shutdown_requested:
        checkpoint.write_text('\n'.join(sorted(downloaded_ids)), encoding='utf-8')
        prev_output.write_text(json.dumps(all_texts), encoding='utf-8')
        # Mark unprocessed as failed
        for tid in to_download[idx+1:]:
            if tid not in downloaded_ids:
                failed_ids.add(tid)
    
    # Update query registry
    query_reg["downloaded_ids"] = sorted(downloaded_ids)
    query_reg["failed_ids"] = sorted(failed_ids - newly_downloaded)
    query_reg["total_downloads"] = len(downloaded_ids)
    query_reg["cycles_attempted"] = cycle
    query_reg["query_success"] = True
    
    if new_success > 0 and new_fail == 0:
        query_reg["last_success"] = datetime.now().isoformat()
    if new_fail > 0 and new_success == 0:
        for tid in to_download:
            if tid not in downloaded_ids:
                query_reg["blocked_ids"].append(tid)

    total_done = len(downloaded_ids)
    unique_texts = len(all_texts)
    return all_texts, total_done, unique_texts, new_fail, shutdown_requested


def scrape_api_cycle(source, settings, registry, source_name, cycle, output_dir, extract_dir):
    """Run one cycle of metadata collection + text download."""
    min_size = settings.get("min_size_kb", 1)
    max_size = settings.get("max_size_kb", 100)
    queries = source.get("queries", [])
    all_query_names = [q["q"] for q in queries]
    
    # Initialize registry entries for all queries
    for qname in all_query_names:
        if qname not in registry["queries"]:
            registry["queries"][qname] = {
                "downloaded_ids": [],
                "failed_ids": [],
                "blocked_ids": [],
                "cycles_attempted": 0,
                "last_success": None,
                "total_downloads": 0,
                "last_query": None,
                "query_success": False
            }
    
    # Phase 1: Collect metadata
    all_meta = []
    seen_ids = set()
    
    for i, query in enumerate(queries):
        if shutdown_requested:
            break
        q = query["q"]
        by = query.get("by", "title")
        match = query.get("match", "prefix")
        qmin = query.get("min_size", min_size)
        qmax = query.get("max_size", max_size)

        out_file = output_dir / f"meta_{hashlib.md5(q.encode()).hexdigest()}.xml"
        query_reg = registry["queries"][q]
        
        # Check if this query was previously blocked
        was_blocked = q in registry.get("blocked_queries", [])
        
        # Mark query as failed if XML is too small
        if out_file.exists() and out_file.stat().st_size < 100:
            if q not in registry.get("blocked_queries", []):
                if "blocked_queries" not in registry:
                    registry["blocked_queries"] = []
                registry["blocked_queries"].append(q)
            query_reg["query_success"] = False
            print(f"\r  [BLOCKED] Query '{q}' -> too small", end="", flush=True)
            continue
        
        # Check if this query was previously blocked
        was_blocked = q in registry.get("blocked_queries", [])
        
        # Only retry if it succeeded before OR we're forcing retry
        if was_blocked and query_reg.get("query_success", False):
            # Query was unblocked, remove from blocked list
            if q in registry.get("blocked_queries", []):
                registry["blocked_queries"].remove(q)
                if "unblocked_queries" not in registry:
                    registry["unblocked_queries"] = []
                registry["unblocked_queries"].append(q)
            print(f"  [UNBLOCKED] Query '{q}' is now accessible!")
        
        # Skip blocked queries that still haven't succeeded
        if was_blocked and not query_reg.get("query_success", False):
            print(f"  [SKIP] Query '{q}' blocked (cycle {query_reg['cycles_attempted']})", end="", flush=True)
            continue
        if get_metadata_xml(q, by=by, match=match, out_file=out_file):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(out_file)
                root = tree.getroot()
                meta_count = 0
                for text_el in root.findall('.//text'):
                    size_el = text_el.find('size')
                    if size_el is None or not size_el.text:
                        continue
                    try:
                        size_kb = int(size_el.text)
                    except (ValueError, TypeError):
                        continue
                    if size_kb < qmin or size_kb > qmax:
                        continue
                    id_el = text_el.find('id')
                    if id_el is None or not id_el.text:
                        continue
                    if id_el.text not in seen_ids:
                        seen_ids.add(id_el.text)
                        all_meta.append({"id": id_el.text, "size": size_kb})
                        meta_count += 1
                if meta_count > 0:
                    query_reg["query_success"] = True
                    query_reg["last_query"] = datetime.now().isoformat()
            except Exception as e:
                pass

        print(f"\r  [META] Query {i+1}/{len(queries)}: {q} -> {len(all_meta)} total", end="", flush=True)

    if not all_meta:
        # Check which queries are still failing
        for qname in all_query_names:
            qr = registry["queries"][qname]
            if not qr.get("query_success", False):
                if qname not in registry.get("blocked_queries", []):
                    if "blocked_queries" not in registry:
                        registry["blocked_queries"] = []
                    registry["blocked_queries"].append(qname)
        return 0, 0, False  # no texts found, queries blocked

    # Phase 2: Download texts
    ids_to_download = [m["id"] for m in all_meta]
    all_texts, total_done, unique_texts, failures, interrupted = \
        download_batch_playwright(ids_to_download, extract_dir, registry, source_name, cycle)

    # Write combined output
    if all_texts:
        output_file = output_dir / f"{source_name}_combined.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for text in all_texts:
                f.write(text + "\n\n")
        file_size = output_file.stat().st_size
        print(f"\n  [CYCLE {cycle}] Saved {unique_texts:,} texts, {file_size / (1024*1024):.1f} MB")

    return total_done, unique_texts, False


def run_retry_loop(source, settings, registry, source_name, max_cycles, cycle_interval, cycles_before_wait, output_dir, extract_dir):
    """Run download in retry cycles until complete or max_cycles reached."""
    output_file = output_dir / f"{source_name}_combined.txt"
    
    # Check if already complete
    if output_file.exists() and output_file.stat().st_size > 10_000_000:
        print(f"  [SKIP] {source_name} already complete ({output_file.stat().st_size / (1024*1024):.0f} MB)")
        return True

    print(f"\n{'='*60}")
    print(f"  [RETRY MODE] {source_name}")
    print(f"  Max cycles: {max_cycles}")
    print(f"  Interval between cycles: {cycle_interval}s ({cycle_interval/60:.0f} min)")
    print(f"{'='*60}\n")

    # Show current blocked queries
    blocked = registry.get("blocked_queries", [])
    if blocked:
        print(f"  Currently blocked queries: {', '.join(blocked)}")
    else:
        print(f"  No blocked queries")
    
    cycle = 0
    while cycle < max_cycles:
        cycle += 1
        registry["total_cycles"] = cycle
        
        print(f"\n{'#'*60}")
        print(f"  CYCLE {cycle}/{max_cycles}")
        print(f"{'#'*60}")
        
        total_done, unique_texts, complete = scrape_api_cycle(
            source, settings, registry, source_name, cycle, output_dir, extract_dir
        )
        
        if shutdown_requested:
            print("\n\n[SHUTDOWN] Interrupted. Checkpoint saved.")
            save_registry(registry, output_dir)
            return False
        
        # Check if all queries succeeded in this cycle
        if unique_texts > 0:
            # Count queries that are still blocked or failing
            all_queries = [q["q"] for q in source.get("queries", [])]
            active_blocked = [q for q in all_queries if q in registry.get("blocked_queries", []) and not registry["queries"][q].get("query_success", False)]
            
            if len(active_blocked) == 0 and total_done > 0:
                print(f"\n[SUCCESS] All queries completed! {unique_texts:,} texts total")
                registry["completed_at"] = datetime.now().isoformat()
                save_registry(registry, output_dir)
                return True
            else:
                print(f"\n[PARTIAL] {unique_texts:,} texts so far, {len(active_blocked)} queries still blocked: {', '.join(active_blocked)}")

        # Wait between cycles (skip wait after last cycle or if complete)
        if cycle < max_cycles:
            # After initial cycles, wait longer to avoid rate limits
            wait_time = cycle_interval
            if cycle < cycles_before_wait:
                wait_time = min(60, cycle_interval // 3)  # Shorter waits initially
            
            print(f"\n[WAIT] Next cycle in {wait_time}s ({wait_time/60:.1f} min)...")
            print(f"       Ctrl+C to interrupt (checkpoint will be saved)")
            
            # Interruptible sleep
            remaining = wait_time
            while remaining > 0 and not shutdown_requested:
                sleep_step = min(remaining, 5)
                time.sleep(sleep_step)
                remaining -= sleep_step
                
                if remaining % 30 == 0 and remaining > 0:
                    print(f"\r[WAIT] {remaining}s remaining...", end="", flush=True)
            
            if shutdown_requested:
                print("\n\n[SHUTDOWN] Interrupted. Checkpoint saved.")
                save_registry(registry, output_dir)
                return False
            
            print(f"\n[RESUME] Starting next cycle...")

    print(f"\n[COMPLETE] Reached max cycles ({max_cycles})")
    save_registry(registry, output_dir)
    return False


def show_status(source_name, output_dir):
    """Show download status from registry."""
    reg_file = Path(output_dir) / "dataset_dl_registry.json"
    if not reg_file.exists():
        print(f"No registry found at {reg_file}")
        return
    
    registry = json.loads(reg_file.read_text(encoding='utf-8'))
    print(f"\n{'='*60}")
    print(f"  Download Registry Status")
    print(f"{'='*60}\n")
    
    print(f"Last run: {registry.get('last_run', 'never')}")
    print(f"Total cycles: {registry.get('total_cycles', 0)}")
    print(f"Completed at: {registry.get('completed_at', 'no')}")
    
    blocked = registry.get("blocked_queries", [])
    unblocked = registry.get("unblocked_queries", [])
    if blocked:
        print(f"Currently blocked: {', '.join(blocked)}")
    if unblocked:
        print(f"Previously blocked (now unblocked): {', '.join(unblocked)}")
    
    if source_name and source_name in registry.get("queries", {}):
        qr = registry["queries"][source_name]
        print(f"\nSource: {source_name}")
        print(f"  Cycles attempted: {qr.get('cycles_attempted', 0)}")
        print(f"  Total downloads: {qr.get('total_downloads', 0)}")
        print(f"  Last success: {qr.get('last_success', 'never')}")
        print(f"  Downloaded: {len(qr.get('downloaded_ids', []))}")
        print(f"  Failed: {len(qr.get('failed_ids', []))}")
        print(f"  Blocked: {len(qr.get('blocked_ids', []))}")
    else:
        print(f"\nAll queries:")
        for qn, qr in registry.get("queries", {}).items():
            status = "OK" if not qr.get("failed_ids") and not qr.get("blocked_ids") else "FAILED"
            print(f"  [{status}] {qn}: {len(qr.get('downloaded_ids', []))} downloaded, "
                  f"{len(qr.get('failed_ids', []))} failed, "
                  f"{len(qr.get('blocked_ids', []))} blocked")
    
    # Check output file
    output_file = Path(output_dir) / f"{source_name or 'chitanka-xml'}_combined.txt"
    if output_file.exists():
        print(f"\nOutput file: {output_file.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"\nNo output file yet")
    
    print(f"\n{'='*60}")


def scrape_api(source, settings):
    """Single-pass scrape (no retry loop)."""
    name = source.get("name", "chitanka-api")
    output_dir = Path(settings["output_dir"])
    extract_dir = Path(settings.get("extract_dir", "E:\\training\\data2\\extracted"))

    output_file = output_dir / f"{name}_combined.txt"
    if output_file.exists() and output_file.stat().st_size > 1_000_000:
        print(f"  [SKIP] {name} already scraped ({output_file.stat().st_size / (1024*1024):.0f} MB)")
        return True

    print(f"  [API] Scraping {name} (size {settings.get('min_size_kb', 1)}-{settings.get('max_size_kb', 100)} KB)...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Initialize registry
    registry = {"queries": {}, "last_run": datetime.now().isoformat(), "total_cycles": 1}
    for q in source.get("queries", []):
        registry["queries"][q["q"]] = {
            "downloaded_ids": [],
            "failed_ids": [],
            "blocked_ids": [],
            "cycles_attempted": 0,
            "last_success": None,
            "total_downloads": 0,
            "last_query": None,
            "query_success": False
        }

    # Collect metadata
    all_meta = []
    seen_ids = set()
    for i, query in enumerate(source.get("queries", [])):
        q = query["q"]
        by = query.get("by", "title")
        match = query.get("match", "prefix")
        qmin = query.get("min_size", settings.get("min_size_kb", 1))
        qmax = query.get("max_size", settings.get("max_size_kb", 100))

        out_file = output_dir / f"meta_{hashlib.md5(q.encode()).hexdigest()}.xml"
        if out_file.exists() and out_file.stat().st_size < 100:
            print(f"\r  [WARN] Query {i+1}/{len(source['queries'])}: {q} -> too small, skipping", end="", flush=True)
            continue
        if get_metadata_xml(q, by=by, match=match, out_file=out_file):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(out_file)
                root = tree.getroot()
                for text_el in root.findall('.//text'):
                    size_el = text_el.find('size')
                    if size_el is None or not size_el.text:
                        continue
                    try:
                        size_kb = int(size_el.text)
                    except (ValueError, TypeError):
                        continue
                    if size_kb < qmin or size_kb > qmax:
                        continue
                    id_el = text_el.find('id')
                    if id_el is None or not id_el.text:
                        continue
                    if id_el.text not in seen_ids:
                        seen_ids.add(id_el.text)
                        all_meta.append({"id": id_el.text, "size": size_kb})
                registry["queries"][q]["query_success"] = True
            except Exception as e:
                print(f"  [WARN] Parse error for '{q}': {e}")

        print(f"\r  [META] Query {i+1}/{len(source['queries'])}: {q} -> {len(all_meta)} total", end="", flush=True)

    print(f"\n  [META] Total unique texts: {len(all_meta):,}")

    if not all_meta:
        print(f"  [WARN] No texts found")
        save_registry(registry, output_dir)
        return True

    ids_to_download = [m["id"] for m in all_meta]
    all_texts, total_done, unique_texts, failures, interrupted = \
        download_batch_playwright(ids_to_download, extract_dir, registry, name, 1)

    if all_texts:
        with open(output_file, "w", encoding="utf-8") as f:
            for text in all_texts:
                f.write(text + "\n\n")
        file_size = output_file.stat().st_size
        print(f"\n  [DONE] {name}: {unique_texts:,} texts, {file_size / (1024*1024):.0f} MB")
        save_registry(registry, output_dir)
    else:
        print(f"\n  [WARN] {name}: no texts found")

    return True


def main():
    mode, source_name, config_path, max_cycles, cycle_interval, cycles_before_wait = parse_args()

    if mode == "list":
        config = load_config("dataset_dl.json")
        print("Available sources:")
        for s in config.get("api", []):
            print(f"  [API] {s['name']}: {len(s.get('queries', []))} queries")
        print(f"\nSettings: size filter {config.get('settings', {}).get('min_size_kb', 1)}-{config.get('settings', {}).get('max_size_kb', 100)} KB")
        return

    config = load_config(config_path)
    settings = config.get("settings", {})

    if mode == "status":
        output_dir = Path(settings["output_dir"])
        show_status(source_name, output_dir)
        return

    if mode == "api":
        sources = config.get("api", [])
        if source_name:
            sources = [s for s in sources if s.get("name") == source_name]
        if not sources:
            print("Error: no API sources found")
            sys.exit(1)
        
        for s in sources:
            scrape_api(s, settings)
        print(f"\nAPI complete: {len(sources)}/{len(sources)}")
        return

    if mode == "retry":
        # Retry mode: load registry and run retry loop
        sources = config.get("api", [])
        if source_name:
            sources = [s for s in sources if s.get("name") == source_name]
        
        output_dir = Path(settings["output_dir"])
        extract_dir = Path(settings.get("extract_dir", "E:\\training\\data2\\extracted"))
        registry = load_registry(output_dir)
        
        for s in sources:
            run_retry_loop(s, settings, registry, s["name"],
                         max_cycles, cycle_interval, cycles_before_wait,
                         output_dir, extract_dir)
        return

    # Mode == "all" (default: retry loop)
    output_dir = Path(settings["output_dir"])
    extract_dir = Path(settings.get("extract_dir", "E:\\training\\data2\\extracted"))
    registry = load_registry(output_dir)
    
    success = 0
    for s in config.get("api", []):
        print(f"\n{'='*60}")
        print(f"  Processing: {s['name']}")
        if run_retry_loop(s, settings, registry, s["name"],
                         max_cycles, cycle_interval, cycles_before_wait,
                         output_dir, extract_dir):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"  Complete: {success}/{len(config.get('api', []))} sources processed")


if __name__ == "__main__":
    main()
