import sys
import os
import json
import math
import hashlib
import time
import numpy as np
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# =============================================================================
# 1. CONFIG
# =============================================================================
def load_config(path=None):
    if path is None:
        import sys
        path = sys.argv[1] if len(sys.argv) > 1 else "gpt_mini3.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def config_hash(cfg):
    hashable = {k: v for k, v in cfg.items() if k in ("model", "tokenizer")}
    canonical = json.dumps(hashable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# =============================================================================
# 2. DATA: Wikipedia download + word-level tokenization
# =============================================================================
class BPETokenizer:
    """BPE tokenizer using SentencePiece with tier tracking."""

    def __init__(self, max_vocab_size: int = 32768, max_word_len: int = None):
        self.max_vocab_size = max_vocab_size
        self._sources = None
        self._tier_ratios = None
        self.sp = None
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.vocab_size = 3

    def train(self, sources: list[dict], tier_ratios: list[float] = None):
        import tempfile
        import sentencepiece as spm_module

        tier_ratios = tier_ratios or [0.66, 0.22, 0.12]

        # Group by tier for logging
        tier_counts = {}
        for src in sources:
            t = src.get("tier", 1)
            tier_counts[t] = tier_counts.get(t, 0) + len(src.get("sentences", []))

        total = sum(tier_counts.values())
        print(f"  Building vocab from {total} texts across {len(tier_counts)} tiers", flush=True)
        for t in sorted(tier_counts):
            print(f"    Tier {t}: {tier_counts[t]} sentences")

        # Write all text to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as f:
            for src in sources:
                for s in src.get("sentences", []):
                    f.write(s + '\n')
            train_file = f.name

        model_prefix = train_file.replace('.txt', '')

        # Estimate upper bound: count unique characters + whitespace-split words
        all_text = Path(train_file).read_text(encoding='utf-8')
        unique_chars = set(all_text)
        unique_words = set(all_text.lower().split())
        effective_vocab = min(self.max_vocab_size, len(unique_chars) + len(unique_words) + 10)
        # SentencePiece needs at least 4 pieces (pad, unk, eos + 1 data piece)
        effective_vocab = max(4, effective_vocab)

        # Retry with smaller vocab if SentencePiece rejects (tiny corpus constraint)
        attempt_vocab = effective_vocab
        while attempt_vocab >= 4:
            try:
                # Suppress SentencePiece C++ logs (absl writes directly to FD 2)
                devnull = os.open(os.devnull, os.O_WRONLY)
                old_stderr = os.dup(2)
                os.dup2(devnull, 2)
                os.close(devnull)
                try:
                    spm_module.SentencePieceTrainer.train(
                        input=train_file,
                        model_prefix=model_prefix,
                        vocab_size=attempt_vocab,
                        character_coverage=1.0,
                        model_type='bpe',
                        pad_id=0,
                        unk_id=1,
                        eos_id=2,
                        bos_id=-1,
                    )
                finally:
                    os.dup2(old_stderr, 2)
                    os.close(old_stderr)
                break
            except RuntimeError as e:
                err = str(e)
                if "Vocabulary size too high" in err:
                    # Extract suggested max from error: "value <= N"
                    import re
                    m = re.search(r'<=\s*(\d+)', err)
                    if m:
                        attempt_vocab = int(m.group(1))
                    else:
                        attempt_vocab = max(4, attempt_vocab // 2)
                else:
                    raise

        self.sp = spm_module.SentencePieceProcessor()
        self.sp.Load(model_prefix + '.model')
        self.vocab_size = self.sp.GetPieceSize()

        # Build word2idx
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        for i in range(3, self.sp.GetPieceSize()):
            self.word2idx[self.sp.IdToPiece(i)] = i

        # Tier metadata
        source_meta = []
        for src in sources:
            tier = src.get("tier", 1)
            sample = src.get("sentences", [])[:500]
            tokens = sum(len(self.sp.encode_as_ids(s)) for s in sample) if sample else 0
            words = sum(len(s.split()) for s in sample) if sample else 0
            source_meta.append({
                "dir": src.get("dir", ""),
                "file": src.get("file", ""),
                "tier": tier,
                "language": src.get("language"),
                "sample_tokens": tokens,
                "sample_words": words,
            })
        self._sources = source_meta
        self._tier_ratios = tier_ratios

        # Cleanup
        for ext in ['.txt', '.model', '.vocab']:
            p = model_prefix + ext
            if os.path.exists(p):
                os.unlink(p)

        capped = "(CAP REACHED)" if self.vocab_size >= self.max_vocab_size else ""
        print(f"Vocabulary size: {self.vocab_size} (max: {self.max_vocab_size}) {capped}", flush=True)

    def build_vocab(self, texts=None, sources=None, tier_ratios=None):
        """Backward compat alias for train()."""
        if sources is None and texts is not None:
            sources = [{"sentences": texts, "tier": 1, "dir": "", "file": "", "language": None}]
        self.train(sources, tier_ratios)

    def encode(self, text: str) -> list[int]:
        return self.sp.encode_as_ids(text)

    def decode(self, indices: list[int]) -> str:
        return self.sp.decode_ids(indices)

    def save(self, path):
        proto = self.sp.serialized_model_proto()
        Path(path).write_bytes(proto)
        meta_path = Path(str(path) + ".meta.json")
        meta = {
            "vocab_size": self.vocab_size,
            "max_vocab_size": self.max_vocab_size,
        }
        if self._sources:
            meta["sources"] = self._sources
        if self._tier_ratios:
            meta["tier_ratios"] = self._tier_ratios
        meta_path.write_text(json.dumps(meta, separators=(",", ":")))

    def load(self, path):
        import sentencepiece as spm_module
        self.sp = spm_module.SentencePieceProcessor()
        self.sp.LoadFromSerializedProto(Path(path).read_bytes())
        self.vocab_size = self.sp.GetPieceSize()
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        for i in range(3, self.sp.GetPieceSize()):
            self.word2idx[self.sp.IdToPiece(i)] = i
        meta_path = Path(str(path) + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._sources = meta.get("sources")
            self._tier_ratios = meta.get("tier_ratios")
            self.max_vocab_size = meta.get("max_vocab_size", self.max_vocab_size)


class BPEDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, seq_length: int, cache_file=None, device=None):
        eos = 2

        if cache_file is None:
            cache_file = Path("data_cache.npy")
        else:
            cache_file = Path(cache_file)
        meta_file = Path(str(cache_file) + ".meta.json")

        if cache_file.exists() and cache_file.stat().st_size > 1_000_000_000:
            print(f"  Loading cached dataset ({cache_file.stat().st_size // 1_000_000_000}GB)...", flush=True)
            arr = np.load(str(cache_file))
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                self.token_count = meta["tokens"]
            else:
                self.token_count = len(arr)
            print(f"  Cache hit: {self.token_count:,} tokens loaded", flush=True)
        else:
            print(f"  Creating dataset from {len(texts)} texts...", flush=True)
            total = len(texts)
            arr_list = []
            for i, text in enumerate(texts):
                tokens = tokenizer.encode(text)
                arr_list.extend(tokens)
                arr_list.append(eos)
                if (i + 1) % 2000000 == 0 or i == total - 1:
                    print(f"    {i+1}/{total} texts, {len(arr_list)//1_000_000}M tokens", flush=True)

            arr = np.array(arr_list, dtype=np.int32)
            self.token_count = len(arr)
            print(f"  Saving cache ({self.token_count//1_000_000}M tokens)...", flush=True)
            np.save(str(cache_file), arr)
            meta_file.write_text(json.dumps({"tokens": self.token_count}, separators=(",", ":")))

        print(f"  Keeping {self.token_count//1_000_000}M tokens on CPU (transfers per batch)...", flush=True)
        self.data = arr
        self.seq_length = seq_length
        print(f"  Dataset ready: {len(self):,} samples (from {self.token_count:,} tokens, seq_len={self.seq_length})", flush=True)

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


# Backward compatibility
WordTokenizer = BPETokenizer
WordDataset = BPEDataset


def ensure_corpus(data_dir: str, extra_dirs: list = None) -> dict:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Gather all .txt from primary data_dir
    text_files = []
    for fn in sorted(data_path.glob("*.txt")):
        if fn.name.endswith(".meta.json"):
            continue
        if fn not in text_files:
            text_files.append(fn)
            sz = fn.stat().st_size / (1024*1024)
            label = "primary" if fn.name == "tinystories.txt" else "corpus"
            print(f"  Found {label}: {fn.name} ({sz:.1f} MB)")

    if extra_dirs:
        for extra in extra_dirs:
            extra_path = Path(extra)
            if extra_path.exists():
                for fn in sorted(extra_path.glob("*.txt")):
                    if fn not in text_files:
                        text_files.append(fn)
                        print(f"  Found extra corpus: {fn} ({fn.stat().st_size / (1024*1024):.0f} MB)")

    if not text_files:
        raise RuntimeError(f"No .txt corpus files found in {data_dir}{', '.join(extra_dirs or []) if extra_dirs else ''}")

    import re as _re
    import html as _html

    # Split on sentence boundaries: .!? followed by space+uppercase or end of line
    _sent_split = _re.compile(r'(?<=[.!?…])\s+(?=[A-ZА-ЯАa-я\u0100-\u024F\u0400-\u04FF\u00C0-\u024F])')

    all_sentences = []
    sources = []
    for tf in text_files:
        meta_file = Path(str(tf) + ".meta.json")
        meta = {}
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
        tier = meta.get("tier", 1)
        language = meta.get("language", None)
        original_encoding = meta.get("original_encoding", "utf-8")
        print(f"  Reading {tf} ... (tier={tier}" + (f", lang={language}" if language else "") + f", enc={original_encoding})")
        with open(tf, "r", encoding="utf-8") as f:
            raw = f.read()
        # Decode HTML entities
        raw = _html.unescape(raw)
        # Split on sentence boundaries
        raw = ' '.join(raw.split())  # normalize whitespace
        raw = _sent_split.split(raw)
        sentences = [s.strip() for s in raw if s.strip() and len(s.strip()) > 2]
        print(f"    -> {len(sentences)} sentences (from {tf.name})")
        sources.append({
            "dir": str(tf.parent),
            "file": tf.name,
            "tier": tier,
            "language": language,
            "original_encoding": original_encoding,
            "sentences": sentences
        })
        all_sentences.extend(sentences)

    print(f"Loaded {len(all_sentences)} total sentences from {len(text_files)} files")
    return {"sentences": all_sentences, "sources": sources}


# =============================================================================
# 3. MODEL: Causal GPT (GPT-mini2 architecture)
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        n_embd = config["n_head"] * config["head_dim"]
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(0.0)
        self.resid_drop = nn.Dropout(0.0)
        self.n_head = config["n_head"]
        self.head_dim = config["head_dim"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        n_embd = config["n_head"] * config["head_dim"]
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTMini(nn.Module):
    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        self.n_embd = config["n_head"] * config["head_dim"]
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, self.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config["n_layer"])]),
                ln_f=nn.LayerNorm(self.n_embd),
            )
        )
        self.register_buffer(
            "wpe", torch.zeros(1, config["seq_length"], self.n_embd)
        )
        self.lm_head = nn.Linear(self.n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.numel()
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.size()
        assert t <= self.wpe.size(1), f"Cannot forward, seq_length exhausted ({t} > {self.wpe.size(1)})"
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.wpe[:, :t, :]
        x = tok_emb + pos_emb
        if self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.transformer.h:
                x = checkpoint(block, x, use_reentrant=True)
        else:
            for block in self.transformer.h:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1).long())
        return logits, loss


# =============================================================================
# 4. GENERATION
# =============================================================================
def generate_text(model: GPTMini, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, device: str = "cpu") -> str:
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = tokens[-model.wpe.size(1):]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)[0]
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == tokenizer.word2idx.get("<eos>", 2):
                break
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

    generated_tokens = input_tensor.squeeze().tolist()
    return tokenizer.decode(generated_tokens)


# =============================================================================
# 5. HASHES
# =============================================================================
def get_vocab_hash(vocab_cfg: dict, data_dirs: list) -> str:
    """Hash tokenizer config + data source file metadata (name, size, mtime, tier).
    The result is the single source of truth for vocab/data cache naming."""
    h = hashlib.sha256()
    # Tokenizer params
    h.update(json.dumps({
        "max_vocab_size": vocab_cfg.get("max_vocab_size", 32768),
        "model_type": "bpe"
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    # Data source file metadata + tier from .meta.json
    file_meta = []
    for d in data_dirs:
        dp = Path(d)
        if dp.exists():
            for fp in sorted(dp.glob("*.txt")):
                st = fp.stat()
                meta_file = Path(str(fp) + ".meta.json")
                meta = {}
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                file_meta.append({
                    "n": fp.name,
                    "s": st.st_size,
                    "t": st.st_mtime,
                    "tier": meta.get("tier", 1)
                })
    file_meta.sort(key=lambda x: x["n"])
    h.update(json.dumps(file_meta, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()[:16]


def get_model_hash(model, vocab_hash: str = None) -> str:
    """Derive a deterministic hash from the model's actual tensor-defining
    attributes + vocabulary identity.  Including vocab_hash ensures
    checkpoints are never shared between different vocabularies, which
    would make embedding weights meaningless."""
    m = model.module if hasattr(model, "module") else model
    h = hashlib.sha256()
    h.update(json.dumps({
        "vocab_size": int(m.transformer.wte.num_embeddings),
        "n_embd": int(m.n_embd),
        "n_layer": len(m.transformer.h),
        "n_head": int(m.transformer.h[0].attn.n_head),
        "head_dim": int(m.transformer.h[0].attn.head_dim),
        "seq_length": int(m.wpe.size(1)),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if vocab_hash:
        h.update(vocab_hash.encode("utf-8"))
    return h.hexdigest()[:16]


# =============================================================================
# 6. CHECKPOINTING
# =============================================================================
# Layout: <ckpt_dir>/<model_hash>/          ← latest checkpoint + config (once)
#         <ckpt_dir>/<model_hash>/1         ← every 10th epoch
#         <ckpt_dir>/<model_hash>/2         ← every 100th epoch
#         <ckpt_dir>/<model_hash>/3         ← every 1000th epoch
#         <ckpt_dir>/<model_hash>/4         ← every 10000th epoch
#         ...
#         <ckpt_dir>/<cfg_hash>/15         ← every 1000000000000000th epoch


def _write_tier(ckpt_dir: str, cfg_hash: str, tier: int, epoch: int, loss: float, config: dict | None, model: GPTMini, optimizer=None, extra: dict | None = None):
    if tier == 0:
        d = Path(ckpt_dir) / cfg_hash
    else:
        d = Path(ckpt_dir) / cfg_hash / str(tier)
    d.mkdir(parents=True, exist_ok=True)

    sd = model.state_dict()
    sd_bf16 = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in sd.items()}

    # Tier 0 alternates between model.0.pth and model.1.pth (slot = epoch & 1)
    if tier == 0:
        slot = epoch & 1
        torch.save(sd_bf16, d / f"model.{slot}.pth")
        # Save optimizer state (only in base tier, same slot)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), d / f"optimizer.{slot}.pt")
    else:
        torch.save(sd_bf16, d / "model.pth")

    # Resume metadata as compact JSON
    meta = {"epoch": epoch, "loss": round(loss, 6), "config_hash": cfg_hash}
    if extra:
        meta.update(extra)
    with open(d / "resume.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))

    # Config only in base tier, only once
    if tier == 0 and config is not None:
        cfg_out = dict(config, _hash=cfg_hash)
        with open(d / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg_out, f, indent=2)


def _tiers_for_epoch(epoch: int) -> list[int]:
    tiers = []
    t = 1
    threshold = 10
    while threshold <= 10**15:
        if epoch % threshold == 0:
            tiers.append(t)
        t += 1
        threshold *= 10
    return tiers


# =============================================================================
# 5b. CHECKPOINT SLOT CLEANUP
# =============================================================================
def _cleanup_corrupt_checkpoint(ckpt_dir: Path):
    """Delete all slot-based checkpoint files."""
    for fn in ["model.0.pth", "model.1.pth", "optimizer.0.pt", "optimizer.1.pt",
               "model.pth", "optimizer.pt"]:
        p = ckpt_dir / fn
        if p.exists():
            p.unlink()


def save_checkpoint(epoch: int, loss: float, config: dict, cfg_hash: str, model: GPTMini, ckpt_dir: str, optimizer=None, extra: dict | None = None):
    base = Path(ckpt_dir) / cfg_hash
    needs_config = not (base / "config.json").exists()
    _write_tier(ckpt_dir, cfg_hash, 0, epoch, loss, config if needs_config else None, model, optimizer, extra)
    tiers = _tiers_for_epoch(epoch)
    if tiers:
        for t in tiers:
            _write_tier(ckpt_dir, cfg_hash, t, epoch, loss, None, model, None, extra)
        print(f"  -> Saved checkpoint at epoch {epoch} (tiers {', '.join(map(str, tiers))})")


def find_latest_checkpoint(ckpt_dir: str, expected_hash: str):
    base = Path(ckpt_dir) / expected_hash
    meta = base / "resume.json"
    if not meta.exists():
        return None
    info = json.loads(meta.read_text())
    ep = int(info.get("epoch", 0))
    loss = float(info.get("loss", 0))
    return (ep, info, base)


# =============================================================================
# 6. TRAINING
# =============================================================================
def setup_ddp():
    if dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Force IPv4 on Windows (hostname resolves to IPv6 -> error 10049)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"  DDP initialized: rank={dist.get_rank()}, local_rank={local_rank}, world_size={dist.get_world_size()}")
        return local_rank
    return 0  # Single process


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _log_error(err_file, msg):
    if err_file is None:
        return
    import traceback
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    err_file.write(f"{ts}\t{msg}\n")
    err_file.flush()


def _write_status(status_file, epoch, global_batch, loss, training_samples, seq_length, training_start_time):
    if status_file is None:
        return
    elapsed = time.time() - training_start_time
    tok_per_sec = training_samples * seq_length / elapsed
    batch_per_sec = global_batch / elapsed
    line = f"{time.strftime('%H:%M:%S')}\t{epoch}\t{global_batch}\t{loss:.4f}\t{tok_per_sec:.0f}\t{batch_per_sec:.1f}\t{training_samples}\n"
    status_file.write(line)
    status_file.flush()


def train():
    print(f"Python: {sys.executable}")

    # Check if running in DDP mode
    in_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    local_rank = setup_ddp() if in_ddp else 0

    # Load config - supports both combined and split formats
    config_path = "gpt_mini3.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Extract model_config and training_config from combined format
    if "model" in full_config and "training" in full_config:
        model_cfg = dict(full_config["model"])
        train_cfg = full_config["training"]
        paths = full_config.get("paths", {})
    else:
        # Legacy format: model/tok/tr directly
        model_cfg = dict(full_config.get("model", full_config))
        train_cfg = full_config.get("training", {})
        paths = full_config.get("paths", {})

    vocab_cfg = model_cfg.pop("vocab") if "vocab" in model_cfg else model_cfg.pop("tokenizer", {})

    DEVICE = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (local_rank == 0)
    print(f"Device: {DEVICE} (rank={local_rank})")
    if is_main:
        print(f"CUDA: {torch.cuda.is_available()} (GPU: {torch.cuda.get_device_name(DEVICE.index)})")

    cache_dir = Path(paths.get("cache_dir", "E:\\training\\cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all data directories for vocab hash
    data_dirs = [paths["data_dir"]]
    if "extra_data_dirs" in paths:
        data_dirs.extend(paths["extra_data_dirs"])

    vocab_hash = get_vocab_hash(vocab_cfg, data_dirs)
    if is_main:
        print(f"Vocab hash: {vocab_hash}")

    vocab_cache = cache_dir / f"vocab-{vocab_hash}.json"

    # Tokenizer
    tokenizer = WordTokenizer(max_vocab_size=vocab_cfg.get("max_vocab_size", 32768), max_word_len=vocab_cfg.get("max_word_len", 20))

    # Corpus hash for data cache invalidation
    def corpus_hash(data_dirs):
        """Hash all files in data_dirs for data cache invalidation."""
        h = hashlib.sha256()
        for data_dir in data_dirs:
            for root, _, files in os.walk(data_dir):
                for fn in sorted(files):
                    fp = Path(root) / fn
                    with open(fp, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
        return h.hexdigest()[:16]

    corpus_h = corpus_hash(data_dirs)
    data_cache = cache_dir / f"data-{vocab_hash}-{corpus_h}.npy"

    sentences = []
    corpus = None
    if vocab_cache.exists():
        if is_main:
            print(f"Loading cached vocab from {vocab_cache}", flush=True)
        tokenizer.load(str(vocab_cache))
    else:
        if is_main:
            corpus = ensure_corpus(paths["data_dir"], paths.get("extra_data_dirs", []))
            sentences = corpus["sentences"]
            tokenizer.build_vocab(sentences, sources=corpus["sources"])
            tokenizer.save(str(vocab_cache))
            print(f"Vocab cached to {vocab_cache}", flush=True)

    if data_cache.exists() and data_cache.stat().st_size > 1_000_000_000:
        if is_main:
            print(f"Loading cached dataset ({data_cache.stat().st_size // 1_000_000_000}GB)...", flush=True)
        sentences = []
    elif sentences:
        pass  # sentences already loaded from vocab build
    else:
        if is_main:
            corpus = ensure_corpus(paths["data_dir"], paths.get("extra_data_dirs", []))
            sentences = corpus["sentences"]

    # Dataset
    dataset = WordDataset(sentences, tokenizer, model_cfg["seq_length"], cache_file=str(data_cache))
    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], sampler=sampler, drop_last=True)
    if is_main:
        print(f"Dataset: {len(dataset)} samples", flush=True)

    # Model
    model = GPTMini(model_cfg, tokenizer.vocab_size).to(DEVICE)

    # DDP wrap
    if dist.is_initialized():
        find_unused_parameters = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                find_unused_parameters = True
                break
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)

    # Get unwrapped model for parameter access
    unwrapped_model = model.module if hasattr(model, "module") else model

    # Canonical checkpoint hash — derived from actual model tensor dims
    ckpt_hash = get_model_hash(model, vocab_hash)
    if is_main:
        print(f"Checkpoint hash: {ckpt_hash}", flush=True)

    # Resume check
    start_epoch = 0
    global_batch = 0
    ckpt_state = None
    optim_state = None
    ckpt = None
    resume_info = None

    # Find latest checkpoint (rank 0 only, to avoid contention)
    if is_main:
        ckpt = find_latest_checkpoint(paths["checkpoint_dir"], ckpt_hash)
        if ckpt:
            ep, info, ckpt_path = ckpt
            global_batch = int(info.get("global_batch", 0))
            resume_info = info
            print(f"Resuming from {ckpt_path} (epoch {ep}, loss {info['loss']:.6f}, global_batch {global_batch})", flush=True)

    if dist.is_initialized():
        # Broadcast resume info from rank 0 to all ranks
        resume_data = [start_epoch, global_batch]
        dist.broadcast_object_list(resume_data, src=0)
        start_epoch, global_batch = resume_data
        # Broadcast whether there's a checkpoint to load
        has_ckpt = [ckpt is not None]
        dist.broadcast_object_list(has_ckpt, src=0)
        has_ckpt = has_ckpt[0]
    else:
        has_ckpt = ckpt is not None

    # ALL ranks load checkpoint (DDP requires all ranks to have same weights)
    if has_ckpt:
        ckpt_path = Path(paths["checkpoint_dir"]) / ckpt_hash
        # Determine slot from saved epoch (slot = epoch & 1)
        slot = start_epoch & 1
        model_path = ckpt_path / f"model.{slot}.pth"
        # Fallback: try other slot, then legacy model.pth
        if not model_path.exists():
            model_path = ckpt_path / f"model.{1 - slot}.pth"
        if not model_path.exists():
            model_path = ckpt_path / "model.pth"
        if model_path.exists():
            ckpt_state = torch.load(str(model_path), map_location=DEVICE)
            unwrapped_model.load_state_dict(ckpt_state)
        # Only main rank loads optimizer state (each rank has its own optimizer)
        if is_main:
            optim_path = ckpt_path / f"optimizer.{slot}.pt"
            if not optim_path.exists():
                optim_path = ckpt_path / f"optimizer.{1 - slot}.pt"
            if not optim_path.exists():
                optim_path = ckpt_path / "optimizer.pt"
            if optim_path.exists():
                optim_state = torch.load(str(optim_path), map_location=DEVICE)
                print(f"  Loaded optimizer state ({optim_path.stat().st_size // 1_000_000}MB)", flush=True)

    optimizer = torch.optim.Adam(unwrapped_model.parameters(), lr=train_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # Restore optimizer state and advance scheduler to correct LR
    if has_ckpt and optim_state is not None:
        optimizer.load_state_dict(optim_state)
        # Advance scheduler to match the epoch we're resuming from
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  Scheduler LR after resume: {optimizer.param_groups[0]['lr']:.6e}", flush=True)

    # Checkpoint config (from 'training.checkpoint' block or legacy 'training' flat keys)
    ckpt_cfg = train_cfg.get("checkpoint", {})
    ckpt_every_batch = ckpt_cfg.get("every_batch", train_cfg.get("checkpoint_interval", 0))
    ckpt_every_min = ckpt_cfg.get("every_min", train_cfg.get("checkpoint_every_min", 0))
    ckpt_every_epoch = ckpt_cfg.get("every_epoch", train_cfg.get("checkpoint_every", 1))
    sync_cfg = train_cfg.get("sync", {})
    grad_accum = sync_cfg.get("gradient_accumulation_steps",
                  train_cfg.get("gradient_accumulation_steps", 1))
    use_bf16 = torch.cuda.is_bf16_supported()
    log_interval = train_cfg.get("log_interval", 100)

    debug_one_step = os.environ.get("DEBUG_ONE_STEP", "0") == "1"

    # Combined config for checkpoints (model + training)
    combined_config = {"model": model_cfg, "training": train_cfg, "paths": paths}
    combined_config["model"]["vocab"] = vocab_cfg

    log_file = None
    err_file = None
    training_start_time = None
    if resume_info and is_main:
        training_start_time = resume_info.get("training_start_time", None)
    if is_main:
        ckpt_dir = Path(paths["checkpoint_dir"])
        ckpt_base = ckpt_dir / ckpt_hash
        ckpt_base.mkdir(parents=True, exist_ok=True)
        status_path = ckpt_base / "checkpoint_status.txt"
        log_file = open(status_path, "a", encoding="utf-8")
        if not status_path.exists() or status_path.stat().st_size == 0:
            log_file.write("time\tepoch\tbatch\tloss\ttok/s\tbatch/s\ttotal_samples\n")
        err_file = open(ckpt_base / "errors.log", "w", encoding="utf-8")
        training_start_time = time.time()
        _ts = time.strftime("%Y-%m-%d %H:%M:%S")
        precision = "bf16" if use_bf16 else "fp32"
        print(f"Precision: {precision}, Grad accumulation: {grad_accum}x", flush=True)
        print(f"Start time: {_ts}", flush=True)
        print(f"Training: vocab={tokenizer.vocab_size} | tokens={dataset.token_count:,} | samples={len(dataset):,} | params={sum(p.numel() for p in unwrapped_model.parameters())/1e6:.2f}M", flush=True)
        print("Starting training..." + (" [DEBUG_ONE_STEP]" if debug_one_step else ""), flush=True)

    last_ckpt_time = time.time()
    epoch_start_time = None
    num_batches = 0
    total_loss = 0.0
    training_samples = 0
    for epoch in range(start_epoch + 1, train_cfg["epochs"] + 1):
        if dist.is_initialized():
            try:
                sampler.set_epoch(epoch)
            except Exception as e:
                _log_error(err_file, f"sampler.set_epoch({epoch}): {e}")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start_time = time.time()
        for batch_idx, (x, y) in enumerate(dataloader):
            if debug_one_step and (batch_idx > 0 or global_batch > 0):
                if is_main:
                    print(f"  [DEBUG] breaking after 1 batch", flush=True)
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits, loss = model(x, y)
            loss = loss / grad_accum
            loss.backward()
            if (global_batch + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            actual_loss = loss.item() * grad_accum
            total_loss += actual_loss
            num_batches += 1
            global_batch += 1
            training_samples += x.size(0)
            # Progress logging
            if is_main and global_batch % log_interval == 0:
                avg = total_loss / max(1, num_batches)
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [{time.strftime('%H:%M:%S')}] Epoch {epoch} | "
                      f"batch {global_batch} | loss {actual_loss:.4f} | "
                      f"avg {avg:.4f} | lr {lr:.6e}", flush=True)
            should_ckpt = False
            if ckpt_every_batch > 0 and global_batch % ckpt_every_batch == 0:
                should_ckpt = True
            if ckpt_every_min > 0 and (time.time() - last_ckpt_time) >= ckpt_every_min * 60:
                should_ckpt = True
            if should_ckpt:
                avg = total_loss / max(1, num_batches)
                try:
                    if is_main:
                        _write_status(log_file, epoch, global_batch, avg, training_samples, model_cfg["seq_length"], training_start_time)
                        save_checkpoint(epoch, avg, combined_config, ckpt_hash, unwrapped_model, paths["checkpoint_dir"],
                                          optimizer=optimizer,
                                          extra={"global_batch": global_batch, "batch_size": train_cfg["batch_size"],
                                                 "seq_length": model_cfg["seq_length"], "training_samples": training_samples,
                                                 "training_start_time": training_start_time,
                                                 "vocab_size": tokenizer.vocab_size,
                                                 "dataset_tokens": dataset.token_count,
                                                  "dataset_samples": len(dataset)})
                    if dist.is_initialized():
                        dist.barrier()
                except Exception as e:
                    _log_error(err_file, f"checkpoint batch {global_batch}: {e}")
                last_ckpt_time = time.time()
                num_batches = 0
                total_loss = 0.0
        avg_loss = total_loss / max(1, num_batches)
        try:
            scheduler.step()
        except Exception as e:
            _log_error(err_file, f"scheduler.step epoch {epoch}: {e}")

        if ckpt_every_epoch > 0 and epoch % ckpt_every_epoch == 0:
            try:
                if is_main:
                    _write_status(log_file, epoch, global_batch, avg_loss, training_samples, model_cfg["seq_length"], training_start_time)
                    save_checkpoint(epoch, avg_loss, combined_config, ckpt_hash, unwrapped_model, paths["checkpoint_dir"],
                                    optimizer=optimizer,
                                    extra={"global_batch": global_batch, "batch_size": train_cfg["batch_size"],
                                           "seq_length": model_cfg["seq_length"], "training_samples": training_samples,
                                           "training_start_time": training_start_time,
                                           "vocab_size": tokenizer.vocab_size,
                                           "dataset_tokens": dataset.token_count,
                                           "dataset_samples": len(dataset)})
                if dist.is_initialized():
                    dist.barrier()
            except Exception as e:
                _log_error(err_file, f"epoch checkpoint {epoch}: {e}")

    if is_main:
        if log_file:
            log_file.close()
        if err_file:
            err_file.close()

    if dist.is_initialized():
        cleanup_ddp()


if __name__ == "__main__":
    train()
