"""Download + extract cumulative archives from dataset_dl.json config.

Supports:
  - URL archives (zip with epubs)
  - HuggingFace datasets (parquet shards)

Usage:
    python dataset_cumulative_dl.py                    # all entries
    python dataset_cumulative_dl.py chitanka-epub      # specific entry
    python dataset_cumulative_dl.py wiki-en             # HF entry
    python dataset_cumulative_dl.py --skip-dl          # extract only
    python dataset_cumulative_dl.py --max 10000        # limit
"""
import sys, os, io, re, html, json, argparse, time, subprocess
from pathlib import Path
import zipfile
from lxml import etree

CFG = "dataset_dl.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def load_hf_token(cfg):
    """Load HF token from settings.hf_token_file if configured."""
    tf = cfg.get("settings", {}).get("hf_token_file")
    if not tf or not Path(tf).exists():
        return None
    try:
        t = json.loads(Path(tf).read_text(encoding="utf-8"))
        return t.get("hf_token")
    except Exception:
        return None


def load_cfg():
    p = Path(CFG)
    if not p.exists():
        print(f"Error: {CFG} not found"); sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


def download(url, dl_dir):
    dl_dir = Path(dl_dir); dl_dir.mkdir(parents=True, exist_ok=True)
    local = dl_dir / url.rsplit("/", 1)[-1]
    if local.exists():
        print(f"  Exists: {local.name} ({local.stat().st_size/(1024**2):.0f} MB)")
        if input("  Overwrite? (y/N): ").strip().lower() != "y":
            return str(local)
        local.unlink()
    print(f"  -> {local}")
    cmd = ["curl", "-L", "-A", UA, "--connect-timeout", "30", "--max-time", "0",
           "-f", "--retry", "3", "--retry-delay", "10", "-C", "-", "-o", str(local), url]
    if subprocess.run(cmd).returncode != 0:
        print(f"  Curl failed. Partial: {local.stat().st_size/(1024**2):.0f} MB" if local.exists() else "  Curl failed.")
        sys.exit(1)
    print(f"  OK: {local.stat().st_size/(1024**2):.0f} MB")
    return str(local)


# ---------- EPUB archive helpers ----------

def epub_text(buf):
    try:
        z = zipfile.ZipFile(io.BytesIO(buf))
    except zipfile.BadZipFile:
        return ""
    opf = None
    try:
        with z.open("META-INF/container.xml") as f:
            r = etree.fromstring(f.read())
        x = r.find("container:rootfiles/container:rootfile/@full-path",
                    {"container": "urn:oasis:names:tc:opendocument:xmlns:container"})
        if x is None: x = r.find(".//{*}rootfile/@full-path")
        if x is not None: opf = x
    except Exception:
        pass
    if not opf:
        for c in ("content.opf", "OEBPS/content.opf", "OPS/content.opf"):
            try:
                z.open(c).read(100); opf = c; break
            except Exception:
                pass
    if not opf: return ""
    try:
        with z.open(opf) as f:
            opr = etree.fromstring(f.read())
        od = os.path.dirname(opf) + "/"
    except Exception:
        od = ""
    xh = [n for n in z.namelist() if n.endswith((".xhtml",".html",".xml")) and not n.startswith("META-INF")]
    ord, seen = [], set()
    try:
        sp = opr.find(".//{*}spine")
        if sp is not None:
            mf = opr.find(".//{*}manifest")
            if mf is not None:
                for ir in sp.findall("{*}itemref"):
                    d = ir.get("idref")
                    if d:
                        for it in mf.findall("{*}item"):
                            if it.get("id") == d:
                                h = it.get("href","")
                                if h.startswith("/"): h = h[1:]
                                f = (od+h).replace("//","/")
                                if f in xh and f not in seen: ord.append(f); seen.add(f)
                                break
    except Exception:
        pass
    for n in xh:
        if n not in seen: ord.append(n)
    out = []
    for nm in ord:
        try:
            with z.open(nm) as f:
                raw = f.read()
            if raw[:3] == b'\xef\xbb\xbf': raw = raw[3:]
            try: t = raw.decode("utf-8")
            except UnicodeDecodeError:
                try: t = raw.decode("windows-1251")
                except UnicodeDecodeError: t = raw.decode("utf-8", errors="replace")
            t = re.sub(r'<[^>]+>', ' ', html.unescape(t))
            t = re.sub(r'\s+', ' ', t).strip()
            if t: out.append(t)
        except Exception:
            pass
    z.close()
    return "\n\n".join(out)


def extract_archive(zip_path, out_path, lang, tier, maxf=0, skip=0):
    zp, op = Path(zip_path), Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Archive: {zp.name} ({zp.stat().st_size//1024//1024} MB)")
    with zipfile.ZipFile(str(zp)) as z:
        eps = [n for n in z.namelist() if n.lower().endswith(".epub") and not n.endswith("/")]
        tot = len(eps)
        print(f"  {tot:,} epubs")
        if skip:
            eps = eps[skip:]
            print(f"  Skipped {skip}, left {len(eps):,}")
        cap = min(tot, maxf) if maxf else tot
        print(f"  Processing {cap:,}\n")
        ln = ch = dn = er = 0; t0 = time.time()
        with open(op, "a", encoding="utf-8") as o:
            for nm in eps:
                if maxf and dn >= maxf: break
                try:
                    with z.open(nm) as f: b = f.read()
                except Exception:
                    er += 1; continue
                t = epub_text(b)
                if t:
                    o.write(t + "\n\n"); ln += t.count("\n")+1; ch += len(t)
                dn += 1
                if dn % 1000 == 0:
                    e = time.time()-t0
                    print(f"  [{dn}/{cap}] {ch/1024/1024:.1f} MB, {dn/e:.0f}/s, err={er}", flush=True)
        e = time.time()-t0
        print(f"\n  Done: {dn:,} in {e:.0f}s -> {ch/1024/1024:.1f} MB, {ln:,} lines, err={er}")
        meta = {"source":"archive","language":lang,"epub_count":dn,"total_epubs":len(eps),
                "original_encoding":"utf-8","tier":tier,"chars":ch,"lines":ln}
        mp = Path(str(op)+".meta.json")
        mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        print(f"  Meta: {mp}")


# ---------- HuggingFace helpers ----------

def hf_download_shards(repo_id, split, shard_dir, max_shards=0, skip_shards=0, token=None):
    """Download parquet shards from HF repo to shard_dir. Returns list of local parquet paths."""
    from huggingface_hub import list_repo_files, hf_hub_download
    shard_dir = Path(shard_dir); shard_dir.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(repo_id, repo_type="dataset")
    # Find parquet files matching split (e.g. "20231101.en/train-*.parquet")
    split_dir = repo_id.rsplit("/", 1)[-1] + "/" + split  # e.g. 20231101.en/
    # Actually, split is like "20231101.en" — the files are under that dir
    pqs = sorted([f for f in files if f.startswith(split + "/") and f.endswith(".parquet")], key=lambda f: f.split("/")[-1])
    if not pqs:
        print(f"  No parquet files found under '{split}/'"); return []
    print(f"  Found {len(pqs)} parquet shards")
    if skip_shards:
        pqs = pqs[skip_shards:]
        print(f"  Skipped {skip_shards}, left {len(pqs):,}")
    if max_shards:
        pqs = pqs[:max_shards]
        print(f"  Limited to {max_shards}")
    locals = []
    split_prefix = split + "/"
    for f in pqs:
        # Strip split prefix to avoid double nesting
        local_name = f[len(split_prefix):] if f.startswith(split_prefix) else f
        local = shard_dir / local_name
        if local.exists():
            locals.append(str(local))
            continue
        print(f"  DL: {f}")
        try:
            p = hf_hub_download(repo_id, f, repo_type="dataset", cache_dir=str(shard_dir / ".cache"), token=token)
            import shutil
            shutil.copy2(p, local)
            locals.append(str(local))
        except Exception as e:
            print(f"  Error: {f}: {e}")
    return locals


def extract_hf(parquet_files, out_path, lang, tier, text_col="text", min_len=50, max_len=100000):
    """Extract text column from parquet files to corpus."""
    import pyarrow.parquet as pq
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    ln = ch = dn = er = 0; t0 = time.time()
    # Truncate existing file for fresh run
    if op.exists():
        op.unlink()
    with open(op, "a", encoding="utf-8") as o:
        for pf in parquet_files:
            try:
                t = pq.read_table(pf)
                col = t.column(text_col)
                for val in col:
                    s = val.as_py()
                    if not s: continue
                    s = s.strip()
                    if len(s) < min_len or len(s) > max_len: continue
                    o.write(s + "\n\n")
                    ln += s.count("\n") + 1
                    ch += len(s)
            except Exception as e:
                print(f"  Error {pf}: {e}"); er += 1; continue
            dn += 1
            e = time.time() - t0
            print(f"  [{dn}/{len(parquet_files)}] {ch/1024/1024:.1f} MB text, {dn/e:.1f} files/s", flush=True)
    e = time.time() - t0
    print(f"\n  Done: {dn} parquet files in {e:.0f}s -> {ch/1024/1024:.1f} MB, {ln:,} lines, err={er}")
    meta = {"source":"huggingface","language":lang,"parquet_files":dn,
            "original_encoding":"utf-8","tier":tier,"chars":ch,"lines":ln}
    mp = Path(str(op)+".meta.json")
    mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"  Meta: {mp}")


# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="DL+extract cumulative from dataset_dl.json")
    p.add_argument("name", nargs="?", default=None, help="Entry name (default: all)")
    p.add_argument("--skip-dl", action="store_true", help="Extract only")
    p.add_argument("--max", type=int, default=0, help="Max items/shards (0=all)")
    p.add_argument("--skip", type=int, default=0, help="Skip first N")
    a = p.parse_args()
    cfg = load_cfg()
    s = cfg["settings"]
    out_dir = s["output_dir"]
    dl_dir = s["download_dir"]
    hf_token = load_hf_token(cfg)
    entries = cfg.get("cumulative", [])
    if not entries:
        print("No cumulative entries in", CFG); sys.exit(1)
    if a.name:
        entries = [e for e in entries if e.get("name") == a.name]
        if not entries:
            print(f"Not found: {a.name}"); print(f"Available: {[e['name'] for e in cfg['cumulative']]}"); sys.exit(1)

    for ent in entries:
        nm = ent["name"]
        ofile = ent.get("output_file", f"{nm}.txt")
        lang = ent.get("language", "en")
        tier = ent.get("tier", 1)
        out_path = os.path.join(out_dir, ofile)
        print("="*60)
        print(f"  {nm} — {ent.get('description','')}")
        print(f"  Output: {out_path}")
        print("="*60)

        # HF dataset entry
        if "hf_repo_id" in ent:
            repo = ent["hf_repo_id"]
            split = ent.get("split", "train")
            text_col = ent.get("text_col", "text")
            min_len = ent.get("min_len", 50)
            max_len = ent.get("max_len", 100000)
            shard_dir = Path(dl_dir) / "hf" / repo.replace("/","--") / split
            print(f"\n  HF repo: {repo}/{split}")
            print(f"  Shards: {shard_dir}")
            if not a.skip_dl:
                pqs = hf_download_shards(repo, split, shard_dir, a.max, a.skip, hf_token)
            else:
                # Find existing parquet files
                pqs = sorted([str(x) for x in shard_dir.rglob("*.parquet")])
            if not pqs:
                print("  No parquet files found"); continue
            print(f"\n  Extracting text...")
            extract_hf(pqs, out_path, lang, tier, text_col, min_len, max_len)

        # URL archive entry
        elif "url" in ent:
            ap = None
            if not a.skip_dl:
                ap = download(ent["url"], dl_dir)
            if not ap:
                fn = ent.get("archive_file", ent["url"].rsplit("/",1)[-1])
                ap = str(Path(dl_dir)/fn)
            if not Path(ap).exists():
                print(f"  Not found: {ap}"); continue
            print(f"  Archive: {ap}")
            extract_archive(ap, out_path, lang, tier, a.max, a.skip)
        else:
            print("  Entry has no 'url' or 'hf_repo_id'"); continue
        print()


if __name__ == "__main__":
    main()
