"""
Multi-GPU trainer — manual gradient sync over TCP, no DDP, no gloo.

Each process runs independently. Gradients are synced via raw TCP sockets
between ranks. Avoids all PyTorch distributed ops that crash on Windows.

Usage (launcher — recommended):
    python launch_rpc.py -g 0,1

Or manually in separate terminals:
  Terminal 1: python train_rpc.py --rank 0 --world_size 2 --device 0
  Terminal 2: python train_rpc.py --rank 1 --world_size 2 --device 1
"""
import os
import sys
import json
import time
import struct
import hashlib
import tempfile
import argparse
from pathlib import Path

# Allow imports from project root when run from experimental/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gpt_train import (
    GPTMini, WordTokenizer, WordDataset, SentenceIterator,
    save_checkpoint, find_latest_checkpoint,
    compute_corpus_hash, get_vocab_hash,
    get_vocab_conf_hash, get_corpus_conf_hash, _try_conf_cache,
    _write_status, _log_error,
)


# =============================================================================
# 1. FILE-BASED BARRIER
# =============================================================================
class FileBarrier:
    """Simple file-based barrier for independent processes."""
    def __init__(self, world_size, barrier_id="default"):
        self.barrier_dir = Path(tempfile.gettempdir()) / f"rpc_barrier_{barrier_id}_{os.getppid()}"
        self.barrier_dir.mkdir(exist_ok=True)
        self.my_file = self.barrier_dir / f"rank_{os.getpid()}"
        self.world_size = world_size

    def wait(self):
        self.my_file.touch()
        while len(list(self.barrier_dir.glob("rank_*"))) < self.world_size:
            time.sleep(0.1)
        time.sleep(0.3)
        self._cleanup()

    def _cleanup(self):
        for f in self.barrier_dir.glob("rank_*"):
            try:
                f.unlink()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._cleanup()


# =============================================================================
# 2. TCP GRADIENT SYNC (2 GPU only)
# =============================================================================
def _serialize_grads(grads):
    """Pack gradient list into bytes: [n_grads][n_dims, dim0, dim1..., data]..."""
    parts = []
    parts.append(struct.pack("!Q", len(grads)))
    for g in grads:
        dims = g.shape
        parts.append(struct.pack("!Q", len(dims)))
        for d in dims:
            parts.append(struct.pack("!Q", d))
        parts.append(struct.pack("!Q", g.nbytes))
        parts.append(g.tobytes())
    return b"".join(parts)


def _deserialize_grads(data, dtype):
    """Unpack bytes back into list of numpy arrays."""
    pos = 0
    n_grads = struct.unpack("!Q", data[pos:pos+8])[0]
    pos += 8
    result = []
    for _ in range(n_grads):
        n_dims = struct.unpack("!Q", data[pos:pos+8])[0]
        pos += 8
        shape = []
        for _ in range(n_dims):
            d = struct.unpack("!Q", data[pos:pos+8])[0]
            pos += 8
            shape.append(d)
        nbytes = struct.unpack("!Q", data[pos:pos+8])[0]
        pos += 8
        flat = np.frombuffer(data[pos:pos+nbytes], dtype=dtype)
        pos += nbytes
        result.append(flat.reshape(shape))
    return result


class GradSync:
    """TCP-based gradient exchange between 2 ranks.
    Rank 0 = server (listens), Rank 1 = client (connects)."""

    def __init__(self, rank, world_size, port):
        self.rank = rank
        self.world_size = world_size
        if world_size <= 1:
            self.server = None
            self.client = None
            return

        import socket
        host = "127.0.0.1"

        if rank == 0:
            # Server: listen for rank 1
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((host, port))
            self.server.listen(1)
            self.server.settimeout(60)
            print(f"  [Rank 0] Listening on {host}:{port} for rank 1...", flush=True)
            conn, addr = self.server.accept()
            self.client = conn
            print(f"  [Rank 0] Connected to rank 1 ({addr})", flush=True)
        else:
            # Client: connect to rank 0
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(60)
            # Wait for server to be ready
            connected = False
            for attempt in range(30):
                try:
                    self.client.connect((host, port))
                    connected = True
                    break
                except ConnectionRefusedError:
                    time.sleep(1)
            if not connected:
                raise RuntimeError(f"Rank {rank} couldn't connect to rank 0 on {host}:{port}")
            print(f"  [Rank {rank}] Connected to rank 0", flush=True)

    def sync(self, model, device):
        """Average gradients between rank 0 and rank 1.
        Rank 0 sends grads, receives grads, averages, applies.
        Rank 1 sends grads, receives grads, averages, applies."""
        if self.world_size <= 1:
            return

        # Collect local gradients as CPU numpy arrays
        local_grads = []
        shapes = []
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach().cpu().numpy()
                local_grads.append(g)
                shapes.append(g.shape)
            else:
                local_grads.append(None)
                shapes.append(None)

        # Serialize
        packed = _serialize_grads(local_grads)
        msg = struct.pack("!Q", len(packed)) + packed

        if self.rank == 0:
            # Send our grads to rank 1
            self.client.sendall(msg)
            # Receive grads from rank 1
            header = self._recv_exact(8)
            remote_len = struct.unpack("!Q", header)[0]
            remote_data = self._recv_exact(remote_len)
            remote_grads = _deserialize_grads(remote_data, np.float32)
        else:
            # Receive header + our grads from rank 0
            header = self._recv_exact(8)
            remote_len = struct.unpack("!Q", header)[0]
            remote_data = self._recv_exact(remote_len)
            remote_grads = _deserialize_grads(remote_data, np.float32)
            # Send our grads to rank 0
            self.client.sendall(msg)

        # Average: (local + remote) / 2
        for i, (lg, rg) in enumerate(zip(local_grads, remote_grads)):
            if lg is not None and rg is not None:
                avg = (lg + rg) / 2.0
                param = list(model.parameters())[i]
                param.grad = torch.from_numpy(avg).to(device)

    def _recv_exact(self, n):
        """Receive exactly n bytes."""
        data = bytearray()
        while len(data) < n:
            chunk = self.client.recv(n - len(data))
            if not chunk:
                raise RuntimeError("Connection closed during recv")
            data.extend(chunk)
        return bytes(data)

    def close(self):
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
        except OSError:
            pass
        try:
            if hasattr(self, 'server') and self.server:
                self.server.close()
        except OSError:
            pass


# =============================================================================
# 3. CONFIG + TOKENIZER + DATASET
# =============================================================================
def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model_cfg = dict(cfg.get("model", {}))
    vocab_cfg = cfg.get("tokenizer", cfg.get("vocab", {}))
    train_cfg = cfg.get("training", {})
    paths = cfg.get("paths", {})
    return model_cfg, vocab_cfg, train_cfg, paths


def build_tokenizer_and_dataset(rank, world_size, model_cfg, vocab_cfg, paths, barrier, force=False):
    cache_dir = Path(paths.get("cache_dir", "E:\\training\\cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_dirs = [paths["data_dir"]]
    if "extra_data_dirs" in paths:
        data_dirs.extend(paths["extra_data_dirs"])

    tokenizer_sources = vocab_cfg.get("sources")
    training_sources = vocab_cfg.get("sources")  # RPC doesn't distinguish

    # Step 1: config-only hashes (no file I/O)
    vocab_conf_h = get_vocab_conf_hash(vocab_cfg, tokenizer_sources)
    corpus_conf_h = get_corpus_conf_hash(training_sources)

    # Step 2: try config-only cache lookup
    vocab_cache = None
    data_cache = None
    if not force:
        hit = _try_conf_cache(cache_dir, vocab_conf_h, corpus_conf_h)
        if hit:
            vocab_cache, data_cache = Path(hit[0]), Path(hit[1])

    # Step 3: content hashes
    if not vocab_cache:
        vocab_hash = get_vocab_hash(vocab_cfg, data_dirs)
        corpus_h = compute_corpus_hash(data_dirs)
        hit = _try_conf_cache(cache_dir, vocab_conf_h, corpus_conf_h, vocab_hash, corpus_h)
        if hit:
            vocab_cache, data_cache = Path(hit[0]), Path(hit[1])

    # Step 4: fallback paths for build
    if not vocab_cache:
        vocab_cache = cache_dir / f"vocab-{vocab_conf_h}-{vocab_hash}.json"
        data_cache = cache_dir / f"data-{corpus_conf_h}-{vocab_hash}-{corpus_h}.npy"

    is_main = (rank == 0)

    tokenizer = WordTokenizer(
        max_vocab_size=vocab_cfg.get("max_vocab_size", 32768),
        max_word_len=vocab_cfg.get("max_word_len", 20),
    )

    sentence_sample_cap = vocab_cfg.get("sentence_sample_cap", vocab_cfg.get("vocab_sample_cap", 25000000))
    pre_sample = vocab_cfg.get("pre_sample_per_source", 500)

    vocab_ok = vocab_cache.exists() and vocab_cache.stat().st_size > 0
    data_ok = data_cache.exists() and data_cache.stat().st_size > 1_000_000_000

    if vocab_ok and data_ok:
        if is_main:
            print("Loading cached vocab + dataset...", flush=True)
        tokenizer.load(vocab_cache)
    else:
        if is_main:
            corpus_iter = SentenceIterator(paths["data_dir"], paths.get("extra_data_dirs"))
            tokenizer.train(corpus_iter._sources_meta,
                            sentence_sample_cap=sentence_sample_cap,
                            pre_sample_per_source=pre_sample)
            tokenizer.save(vocab_cache)
            print(f"Vocab built: {tokenizer.vocab_size} tokens", flush=True)
        barrier.wait()
        if not is_main:
            tokenizer.load(vocab_cache)

    if is_main:
        corpus_iter2 = SentenceIterator(paths["data_dir"], paths.get("extra_data_dirs"))
        dataset = WordDataset(corpus_iter2, tokenizer, model_cfg["seq_length"],
                              cache_file=str(data_cache))
        print(f"Dataset: {len(dataset)} samples", flush=True)
    else:
        dataset = WordDataset([], tokenizer, model_cfg["seq_length"],
                              cache_file=str(data_cache))

    barrier.wait()
    return tokenizer, dataset


# =============================================================================
# 4. TRAINER
# =============================================================================
class Trainer:
    def __init__(self, rank, world_size, device, model_cfg, train_cfg, paths,
                 ckpt_hash, combined_config, tokenizer, dataset, grad_sync, barrier):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.paths = paths
        self.ckpt_hash = ckpt_hash
        self.combined_config = combined_config
        self.tokenizer = tokenizer
        self.grad_sync = grad_sync
        self.barrier = barrier

        self.ckpt_every_batch = train_cfg.get("checkpoint", {}).get("every_batch", train_cfg.get("checkpoint_interval", 0))
        self.ckpt_every_min = train_cfg.get("checkpoint", {}).get("every_min", train_cfg.get("checkpoint_every_min", 0))
        self.ckpt_every = train_cfg.get("checkpoint", {}).get("every_epoch", train_cfg.get("checkpoint_every", 1))
        self.grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
        self.use_bf16 = torch.cuda.is_bf16_supported()
        self.log_interval = train_cfg.get("log_interval", 100)

        # Build model — NO DDP wrapper
        self.model = GPTMini(model_cfg, tokenizer.vocab_size).to(f"cuda:{device}")

        # Manual data shard: rank i takes every world_size-th sample
        batch_size = train_cfg.get("batch_size", 16)
        total_samples = len(dataset)
        shard_indices = list(range(rank, total_samples, world_size))
        shard_dataset = Subset(dataset, shard_indices)
        self.dataloader = DataLoader(
            shard_dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=0, pin_memory=False,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=train_cfg.get("lr", 0.0002)
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.1
        )

        # Logging (rank 0 only)
        self.log_file = None
        self.err_file = None
        self.training_start_time = None
        if rank == 0:
            ckpt_base = Path(paths["checkpoint_dir"]) / ckpt_hash
            ckpt_base.mkdir(parents=True, exist_ok=True)
            self.log_file = open(ckpt_base / "checkpoint_status.txt", "w", encoding="utf-8")
            self.err_file = open(ckpt_base / "errors.log", "w", encoding="utf-8")
            self.training_start_time = time.time()
            precision = "bf16" if self.use_bf16 else "fp32"
            print(f"Rank {rank}: Precision={precision}, GradAccum={self.grad_accum}x, "
                  f"Device=cuda:{device}", flush=True)
            print(f"Rank {rank}: Checkpoint hash={ckpt_hash}", flush=True)
            print("Starting training...", flush=True)

        # Counters
        self.global_batch = 0
        self.num_batches = 0
        self.total_loss = 0.0
        self.training_samples = 0
        self.last_ckpt_time = time.time()

        self._maybe_resume()

    def _maybe_resume(self):
        ckpt = find_latest_checkpoint(self.paths["checkpoint_dir"], self.ckpt_hash)
        if ckpt:
            ep, info, ckpt_dir = ckpt
            self.global_batch = int(info.get("global_batch", 0))
            if self.rank == 0:
                print(f"Resuming from {ckpt_dir} (epoch {ep}, loss {info['loss']:.6f}, "
                      f"global_batch {self.global_batch})", flush=True)
            # Determine slot from saved epoch (slot = epoch & 1)
            slot = ep & 1
            model_path = ckpt_dir / f"model.{slot}.pth"
            if not model_path.exists():
                model_path = ckpt_dir / f"model.{1 - slot}.pth"
            if not model_path.exists():
                model_path = ckpt_dir / "model.pth"
            ckpt_state = torch.load(model_path,
                                    map_location=f"cuda:{self.device}")
            self.model.load_state_dict(ckpt_state)
            del ckpt_state
            torch.cuda.empty_cache()
            # Restore optimizer state (rank 0 only)
            opt_path = ckpt_dir / f"optimizer.{slot}.pt"
            if not opt_path.exists():
                opt_path = ckpt_dir / f"optimizer.{1 - slot}.pt"
            if not opt_path.exists():
                opt_path = ckpt_dir / "optimizer.pt"
            if self.rank == 0 and opt_path.exists():
                self.optimizer.load_state_dict(torch.load(opt_path, map_location=f"cuda:{self.device}"))
                print(f"  Optimizer state restored (momentum buffers preserved)", flush=True)
            # Advance scheduler to correct LR
            lr = float(info.get("lr", self.train_cfg.get("lr", 0.0002)))
            self.optimizer.param_groups[0]["lr"] = lr
            for _ in range(ep):
                self.scheduler.step()

    def _run_epoch(self, epoch: int):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for x, y in self.dataloader:
            x, y = x.to(f"cuda:{self.device}"), y.to(f"cuda:{self.device}")

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                _, loss = self.model(x, y)

            loss = loss / self.grad_accum
            loss.backward()

            if (self.global_batch + 1) % self.grad_accum == 0:
                # Manual gradient sync via TCP
                self.grad_sync.sync(self.model, f"cuda:{self.device}")
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            self.total_loss += loss.item() * self.grad_accum
            self.num_batches += 1
            self.global_batch += 1
            self.training_samples += x.size(0)

            if self.rank == 0 and self.global_batch % self.log_interval == 0:
                avg = self.total_loss / max(1, self.num_batches)
                print(f"  [{time.strftime('%H:%M:%S')}] Epoch {epoch} | "
                      f"batch {self.global_batch} | loss {avg:.4f} | "
                      f"lr {self.optimizer.param_groups[0]['lr']:.6e}",
                      flush=True)

            should_ckpt = False
            if self.ckpt_every_batch > 0 and self.global_batch % self.ckpt_every_batch == 0:
                should_ckpt = True
            if self.ckpt_every_min > 0 and (time.time() - self.last_ckpt_time) >= self.ckpt_every_min * 60:
                should_ckpt = True

            if should_ckpt:
                avg = self.total_loss / max(1, self.num_batches)
                try:
                    self._save_checkpoint(epoch, avg)
                except Exception as e:
                    _log_error(self.err_file, f"checkpoint batch {self.global_batch}: {e}")
                self.last_ckpt_time = time.time()
                self.num_batches = 0
                self.total_loss = 0.0

    def _save_checkpoint(self, epoch: int, loss: float):
        ckpt_dir = Path(self.paths["checkpoint_dir"]) / self.ckpt_hash
        if self.rank == 0:
            _write_status(self.log_file, epoch, self.global_batch, loss,
                          self.training_samples, self.model_cfg["seq_length"],
                          self.training_start_time)
            save_checkpoint(epoch, loss, self.combined_config, self.ckpt_hash,
                            self.model, self.paths["checkpoint_dir"],
                            optimizer=self.optimizer,
                            extra={"global_batch": self.global_batch,
                                   "batch_size": self.train_cfg["batch_size"],
                                   "seq_length": self.model_cfg["seq_length"],
                                   "training_samples": self.training_samples})
        self.barrier.wait()

    def train(self, total_epochs: int, start_epoch: int = 0):
        for epoch in range(start_epoch + 1, total_epochs + 1):
            self._run_epoch(epoch)

            avg_loss = self.total_loss / max(1, self.num_batches)
            if self.rank == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] Epoch {epoch}/{total_epochs} "
                      f"| avg_loss {avg_loss:.4f} | global_batch {self.global_batch}",
                      flush=True)

            try:
                self.scheduler.step()
            except Exception as e:
                _log_error(self.err_file, f"scheduler.step epoch {epoch}: {e}")

            if epoch % self.ckpt_every == 0:
                try:
                    self._save_checkpoint(epoch, avg_loss)
                except Exception as e:
                    _log_error(self.err_file, f"epoch checkpoint {epoch}: {e}")

    def close(self):
        if self.rank == 0:
            if self.log_file:
                self.log_file.close()
            if self.err_file:
                self.err_file.close()


# =============================================================================
# 5. ENTRY POINT
# =============================================================================
def run():
    parser = argparse.ArgumentParser(
        description="Multi-GPU trainer (TCP grad sync, no DDP, no gloo)",
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("config", nargs="?", default="gpt_train.json")
    parser.add_argument("--cache-renew", action="store_true",
                        help="Force rebuild vocab + data cache")
    args = parser.parse_args()

    if args.rank < 0 or args.rank >= args.world_size:
        print(f"Error: rank {args.rank} out of range [0, {args.world_size})")
        sys.exit(1)

    # Set CUDA device early (fresh context, no mp.spawn)
    torch.cuda.set_device(args.device)
    print(f"[Rank {args.rank}] Initializing... cuda:{args.device} "
          f"| {torch.cuda.get_device_name(args.device)}", flush=True)

    # Barrier (file-based, no dist.barrier)
    barrier = FileBarrier(args.world_size, "init")

    # TCP gradient sync setup
    grad_sync = GradSync(args.rank, args.world_size, args.port)

    # Load config
    model_cfg, vocab_cfg, train_cfg, paths = load_config(args.config)

    if args.epochs > 0:
        train_cfg["epochs"] = args.epochs
    if args.save_every > 0:
        train_cfg["checkpoint_every"] = args.save_every

    total_epochs = train_cfg.get("epochs", 10)

    # Tokenizer + dataset
    tokenizer, dataset = build_tokenizer_and_dataset(
        args.rank, args.world_size, model_cfg, vocab_cfg, paths, barrier,
        force=args.cache_renew
    )

    # Checkpoint hash = model params only (matches gpt_train.py's cfg_hash)
    model_param_dict = {
        "n_layer": model_cfg["n_layer"], "n_head": model_cfg["n_head"],
        "head_dim": model_cfg["head_dim"], "seq_length": model_cfg["seq_length"],
        "max_vocab_size": vocab_cfg.get("max_vocab_size", 32768),
        "max_word_len": vocab_cfg.get("max_word_len", 20),
    }
    ckpt_hash = hashlib.sha256(
        json.dumps(model_param_dict, sort_keys=True).encode()
    ).hexdigest()[:16]

    combined_config = {"model": model_cfg, "training": train_cfg, "paths": paths}
    combined_config["model"]["vocab"] = vocab_cfg

    trainer = Trainer(
        args.rank, args.world_size, args.device,
        model_cfg, train_cfg, paths, ckpt_hash,
        combined_config, tokenizer, dataset, grad_sync, barrier,
    )

    try:
        trainer.train(total_epochs)
    except Exception as e:
        print(f"Rank {args.rank} training error: {e}", flush=True)
        raise
    finally:
        trainer.close()
        grad_sync.close()

    print(f"[Rank {args.rank}] Done.", flush=True)


if __name__ == "__main__":
    run()
