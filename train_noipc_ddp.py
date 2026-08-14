"""
Multi-GPU trainer with manual CPU gradient sync.

Root cause of DDP crash on Windows: gloo's all_reduce on CUDA tensors
requires CUDA IPC, which requires P2P between GPUs. Our RTX 3090s have
PXB topology (no P2P). Fix: sync gradients via CPU where gloo works.

Usage:
    python train_noipc_ddp.py                          # all CUDA GPUs, default config
    python train_noipc_ddp.py -d 0,1                   # GPUs 0 and 1
    python train_noipc_ddp.py -d 0 gpt_mini3.json      # GPU 0 only, custom config
    python train_noipc_ddp.py --epochs 10 --save_every 3
"""
import os, sys, json, time, hashlib, signal, subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Sampler as DataSampler

from gpt_mini3 import (
    GPTMini, WordTokenizer, WordDataset, ensure_corpus,
    save_checkpoint, find_latest_checkpoint, generate_text, get_model_hash, get_vocab_hash,
    _write_status, _log_error,
    _cleanup_corrupt_checkpoint,
)


# =============================================================================
# 1. DISTRIBUTED SETUP  (gloo for CPU all_reduce only)
# =============================================================================
def dist_setup(rank: int, world_size: int, device: int, master_port: str):
    """Init gloo for CPU all_reduce. No DDP wrapper, no CUDA IPC needed."""
    torch.cuda.set_device(device)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port

    backend = "gloo"
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except RuntimeError as e:
        if "built in" in str(e):
            raise RuntimeError(f"Backend '{backend}' not compiled.") from e
        raise

    has_p2p = torch.cuda.can_device_access_peer(device, (device + 1) % 2)
    print(f"Rank {rank} -> cuda:{device} | {torch.cuda.get_device_name(device)}  "
          f"(world_size={world_size}, P2P={has_p2p})")


# =============================================================================
# 2. GRADIENT SYNC METHODS  (select via sync.method in config)
# =============================================================================

def _allreduce_gpu(model):
    """Direct GPU all_reduce — zero CPU memory.
    NOTE: crashes with gloo on Windows (gloo lacks CUDA tensor support).
    Use only with NCCL backend."""
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.AVG)


def _build_chunked_buffers(model, num_chunks):
    """Split params into num_chunks groups, each with own pinned CPU buffer."""
    params = [p for p in model.parameters() if p.requires_grad]
    dtype = next(model.parameters()).dtype

    total = sum(p.numel() for p in params)
    chunk_size = total // num_chunks
    chunks = []
    idx = 0
    for i in range(num_chunks):
        chunk_params = []
        chunk_elems = 0
        target = chunk_size if i < num_chunks - 1 else total
        while idx < len(params) and chunk_elems < target:
            chunk_params.append(params[idx])
            chunk_elems += params[idx].numel()
            idx += 1
        if chunk_params:
            chunks.append(chunk_params)

    model._chunk_params = chunks
    model._chunk_bufs = []
    for cp in chunks:
        numel = sum(p.numel() for p in cp)
        buf = torch.empty(numel, dtype=dtype, device='cpu', pin_memory=True)
        model._chunk_bufs.append(buf)


def _allreduce_chunked(model):
    """Chunked CPU all_reduce — fully synchronous, one chunk at a time.
    No pipelining: each chunk completes (copy→reduce→copy-back) before next.
    Guarantees no race with next batch's backward."""
    chunks = model._chunk_params
    bufs = model._chunk_bufs

    for cp, buf in zip(chunks, bufs):
        grads = [p.grad.data for p in cp]
        flat = torch._utils._flatten_dense_tensors(grads)
        buf[:flat.numel()].copy_(flat)  # blocking GPU→CPU
        dist.all_reduce(buf[:buf.numel()], op=dist.ReduceOp.AVG)
        split = torch._utils._unflatten_dense_tensors(buf[:buf.numel()], grads)
        for p, g in zip(cp, split):
            p.grad.data.copy_(g)  # blocking CPU→GPU

    torch.cuda.synchronize()


def all_reduce_grads(model, sync_method="cpu"):
    """Dispatch to sync method. Default: cpu (gloo on Windows)."""
    world_size = dist.get_world_size()
    if world_size <= 1:
        return
    if sync_method == "cpu":
        _allreduce_chunked(model)
    else:
        _allreduce_gpu(model)


# =============================================================================
# 2. LOAD TRAINING OBJECTS
# =============================================================================
def load_config(config_path: str) -> tuple[dict, dict, dict, dict]:
    """Load and split config into model_cfg, vocab_cfg, train_cfg, paths."""
    with open(config_path, "r") as f:
        cfg = json.load(f)
    model_cfg = dict(cfg.get("model", {}))
    vocab_cfg = model_cfg.pop("tokenizer", model_cfg.pop("vocab", {}))
    train_cfg = cfg.get("training", {})
    paths = cfg.get("paths", {})
    return model_cfg, vocab_cfg, train_cfg, paths


def build_tokenizer_and_dataset(rank, world_size, model_cfg, vocab_cfg, train_cfg, paths):
    """Build vocab + dataset. Rank 0 builds, others load from cache."""
    # Collect all data directories for vocab hash
    data_dirs = [paths["data_dir"]]
    if "extra_data_dirs" in paths:
        data_dirs.extend(paths["extra_data_dirs"])

    vocab_hash = get_vocab_hash(vocab_cfg, data_dirs)

    cache_dir = Path(paths.get("cache_dir", "E:\\training\\cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    vocab_cache = cache_dir / f"vocab-{vocab_hash}.json"

    # Corpus hash for data cache
    def _corpus_hash(directories):
        h = hashlib.sha256()
        for d in directories:
            for root, _, files in os.walk(d):
                for fn in sorted(files):
                    fp = Path(root) / fn
                    with open(fp, "rb") as fobj:
                        for chunk in iter(lambda: fobj.read(8192), b""):
                            h.update(chunk)
        return h.hexdigest()[:16]

    corpus_h = _corpus_hash(data_dirs)
    data_cache = cache_dir / f"data-{vocab_hash}-{corpus_h}.npy"
    is_main = (rank == 0)

    tokenizer = WordTokenizer(
        max_vocab_size=vocab_cfg.get("max_vocab_size", 32768),
        max_word_len=vocab_cfg.get("max_word_len", 20),
    )

    sentences = []
    corpus = None
    if vocab_cache.exists() and data_cache.exists() and data_cache.stat().st_size > 1_000_000_000:
        if is_main:
            print("Loading cached vocab + dataset...", flush=True)
        tokenizer.load(vocab_cache)
    else:
        if is_main:
            corpus = ensure_corpus(paths["data_dir"], paths.get("extra_data_dirs", []))
            tokenizer.build_vocab(corpus["sentences"], sources=corpus["sources"])
            tokenizer.save(vocab_cache)
            print(f"Vocab built: {tokenizer.vocab_size} tokens, cached to {vocab_cache}", flush=True)
        dist.barrier()
        if not is_main:
            tokenizer.load(vocab_cache)
    if corpus:
        sentences = corpus["sentences"]

    # Build / load dataset
    if is_main:
        dataset = WordDataset(sentences, tokenizer, model_cfg["seq_length"],
                              cache_file=str(data_cache))
        print(f"Dataset: {len(dataset)} samples", flush=True)
        del sentences
    else:
        dataset = WordDataset([], tokenizer, model_cfg["seq_length"],
                              cache_file=str(data_cache))

    dist.barrier()
    if is_main:
        print(f"Dataset (all ranks ready): {len(dataset)} samples", flush=True)

    return tokenizer, dataset, vocab_hash


def load_train_objs(tokenizer, dataset, model_cfg, device):
    """Returns model. Per tutorial: called AFTER ddp_setup()."""
    return GPTMini(model_cfg, tokenizer.vocab_size).to(f"cuda:{device}")


# =============================================================================
# 3. LAZY DISTRIBUTED SAMPLER  (no randperm, no MemoryError)
# =============================================================================
class LazyDistributedSampler(DataSampler):
    """Yield indices for one rank without materializing a full permutation array."""
    def __init__(self, dataset_len, rank=0, world_size=1, batch_size=1):
        self.dataset_len = dataset_len
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.total = (dataset_len // (batch_size * world_size)) * (batch_size * world_size)

    def __iter__(self):
        for i in range(self.rank, self.total, self.world_size):
            yield i

    def __len__(self):
        return (self.total + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch):
        pass


def prepare_dataloader(dataset, batch_size: int, rank: int, world_size: int):
    """
    LazyDistributedSampler avoids torch.randperm(453M) MemoryError.
    Each process gets batch_size samples; effective batch = batch_size * nprocs.
    """
    from torch.utils.data import DataLoader
    sampler = LazyDistributedSampler(len(dataset), rank=rank, world_size=world_size,
                                     batch_size=batch_size)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True,
                      num_workers=0, pin_memory=False), sampler


# =============================================================================
# 4. TRAINER CLASS  (per tutorial pattern)
# =============================================================================
class Trainer:
    def __init__(self, model, train_data, sampler, optimizer, rank, world_size,
                  device, model_cfg, train_cfg, paths, ckpt_hash, combined_config,
                  tokenizer, dataset_tokens=0, dataset_samples=0):
        self.model = model
        self.train_data = train_data
        self.sampler = sampler
        self.optimizer = optimizer
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.gpu_id = rank  # rank == 0 is the main GPU
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.paths = paths
        self.ckpt_hash = ckpt_hash
        self.combined_config = combined_config
        self.tokenizer = tokenizer
        self.dataset_tokens = dataset_tokens
        self.dataset_samples = dataset_samples

        # Checkpoint config (from 'training.checkpoint' block or legacy flat keys)
        ckpt_cfg = train_cfg.get("checkpoint", {})
        self.ckpt_every_batch = ckpt_cfg.get("every_batch", train_cfg.get("checkpoint_interval", 0))
        self.ckpt_every_min = ckpt_cfg.get("every_min", train_cfg.get("checkpoint_every_min", 0))
        self.ckpt_every = ckpt_cfg.get("every_epoch", train_cfg.get("checkpoint_every", 1))
        sync_cfg = train_cfg.get("sync", {})
        self.grad_accum = sync_cfg.get("gradient_accumulation_steps",
                      train_cfg.get("gradient_accumulation_steps", 1))
        self.sync_method = sync_cfg.get("method", "cpu")  # "cpu" (default) or "gpu" (NCCL only)
        self.sync_chunks = sync_cfg.get("chunks", 4)
        self.use_bf16 = torch.cuda.is_bf16_supported()
        self.log_interval = train_cfg.get("log_interval", 100)

        # Logging
        self.log_file = None
        self.err_file = None
        self.training_start_time = None
        if self.gpu_id == 0:
            ckpt_dir = Path(paths["checkpoint_dir"])
            ckpt_base = ckpt_dir / ckpt_hash
            ckpt_base.mkdir(parents=True, exist_ok=True)
            status_path = ckpt_base / "checkpoint_status.txt"
            self.log_file = open(status_path, "a", encoding="utf-8")
            # Write header only for new file
            if not status_path.exists() or status_path.stat().st_size == 0:
                self.log_file.write("time\tepoch\tbatch\tloss\ttok/s\tbatch/s\ttotal_samples\n")
            self.err_file = open(ckpt_base / "errors.log", "w", encoding="utf-8")
            self.training_start_time = time.time()
            precision = "bf16" if self.use_bf16 else "fp32"
            print(f"Precision: {precision}, Grad accumulation: {self.grad_accum}x, Sync: {self.sync_method}", flush=True)
            print(f"Checkpoint hash: {ckpt_hash}", flush=True)
            print(f"Training: vocab={self.tokenizer.vocab_size} | tokens={self.dataset_tokens:,} | samples={self.dataset_samples:,} | params={sum(p.numel() for p in self.model.parameters())/1e6:.2f}M", flush=True)
            print("Starting training...", flush=True)

        # Counters
        self.global_batch = 0
        self.num_batches = 0          # resets on checkpoint — for checkpoint metadata
        self.total_loss = 0.0         # resets on checkpoint — for checkpoint metadata
        self.session_total_loss = 0.0 # never resets — for display
        self.session_num_batches = 0  # never resets — for display
        self.training_samples = 0
        self.last_ckpt_time = time.time()

        # Timing (rank 0 only, logged every log_interval)
        self.elapsed_fwd_bwd = 0.0
        self.elapsed_sync = 0.0
        self.elapsed_data = 0.0

        # Resume check (ONLY rank 0)
        self._maybe_resume()

    def _maybe_resume(self):
        ckpt = find_latest_checkpoint(self.paths["checkpoint_dir"], self.ckpt_hash)
        if not ckpt:
            return
        ep, info, ckpt_dir = ckpt
        self.global_batch = int(info.get("global_batch", 0))
        device_str = f"cuda:{self.device}"
        # Determine slot from saved epoch (slot = epoch & 1)
        slot = ep & 1
        model_path = ckpt_dir / f"model.{slot}.pth"
        if not model_path.exists():
            model_path = ckpt_dir / f"model.{1 - slot}.pth"
        if not model_path.exists():
            model_path = ckpt_dir / "model.pth"
        try:
            ckpt_state = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            if self.gpu_id == 0:
                print(f"Checkpoint corrupt: {model_path}, starting fresh", flush=True)
            return
        if self.gpu_id == 0:
            print(f"Resuming from {ckpt_dir} (epoch {ep}, loss {info['loss']:.6f}, "
                  f"global_batch {self.global_batch})", flush=True)
        self.model.load_state_dict(ckpt_state)
        del ckpt_state
        # Restore session counters so avg doesn't reset on resume
        self.session_num_batches = self.global_batch
        self.session_total_loss = info["loss"] * self.global_batch
        self.num_batches = 0
        self.total_loss = 0.0
        self.training_samples = int(info.get("training_samples", 0))
        self.training_start_time = info.get("training_start_time", self.training_start_time)

    def _run_epoch(self, epoch: int):
        """Run one training epoch. Per tutorial: call sampler.set_epoch() every epoch."""
        try:
            self.sampler.set_epoch(epoch)
        except Exception as e:
            _log_error(self.err_file, f"sampler.set_epoch({epoch}): {e}")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for x, y in self.train_data:
            t0 = time.time()
            try:
                x, y = x.to(f"cuda:{self.device}"), y.to(f"cuda:{self.device}")
                t_data = time.time() - t0

                t1 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    _, loss = self.model(x, y)
                loss = loss / self.grad_accum
                loss.backward()
                t_fwd = time.time() - t1

                t_sync = 0.0
                if (self.global_batch + 1) % self.grad_accum == 0:
                    t2 = time.time()
                    all_reduce_grads(self.model, self.sync_method)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    t_sync = time.time() - t2

                actual_loss = loss.item() * self.grad_accum
                self.total_loss += actual_loss
                self.num_batches += 1
                self.session_total_loss += actual_loss
                self.session_num_batches += 1
                self.global_batch += 1
                self.training_samples += x.size(0)

                # Accumulate timing (rank 0 only)
                if self.gpu_id == 0:
                    self.elapsed_data += t_data
                    self.elapsed_fwd_bwd += t_fwd
                    self.elapsed_sync += t_sync

                # Progress logging (rank 0 only)
                if self.gpu_id == 0 and self.global_batch % self.log_interval == 0:
                    sess_avg = self.session_total_loss / max(1, self.session_num_batches)
                    total_t = self.elapsed_data + self.elapsed_fwd_bwd + self.elapsed_sync
                    p_data = self.elapsed_data / total_t * 100 if total_t else 0
                    p_fwd = self.elapsed_fwd_bwd / total_t * 100 if total_t else 0
                    p_sync = self.elapsed_sync / total_t * 100 if total_t else 0
                    print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch} | "
                          f"batch {self.global_batch} | loss {actual_loss:.4f} | "
                          f"avg {sess_avg:.4f} | "
                          f"lr {self.optimizer.param_groups[0]['lr']:.6e} | "
                          f"time: data={p_data:.0f}% fwd/bwd={p_fwd:.0f}% sync={p_sync:.0f}%",
                          flush=True)
                    self.elapsed_data = 0.0
                    self.elapsed_fwd_bwd = 0.0
                    self.elapsed_sync = 0.0
            except Exception as e:
                _log_error(self.err_file, f"batch {self.global_batch}: {e}")
                raise

            # Checkpoint at interval / time
            should_ckpt = False
            if self.ckpt_every_batch > 0 and self.global_batch % self.ckpt_every_batch == 0:
                should_ckpt = True
            if self.ckpt_every_min > 0 and (time.time() - self.last_ckpt_time) >= self.ckpt_every_min * 60:
                should_ckpt = True

            if should_ckpt:
                avg = self.total_loss / max(1, self.num_batches)
                try:
                    # Collective calls BEFORE the gpu_id check
                    dist.barrier()
                    # Save from gpu_id == 0 only
                    ckpt_dir = Path(self.paths["checkpoint_dir"]) / self.ckpt_hash
                    if self.gpu_id == 0:
                        _write_status(self.log_file, epoch, self.global_batch, avg,
                                      self.training_samples, self.model_cfg["seq_length"],
                                      self.training_start_time)
                        save_checkpoint(epoch, avg, self.combined_config, self.ckpt_hash,
                                        self.model, self.paths["checkpoint_dir"],
                                        optimizer=self.optimizer,
                                        extra={"global_batch": self.global_batch,
                                                "batch_size": self.train_cfg["batch_size"],
                                                "seq_length": self.model_cfg["seq_length"],
                                                "training_samples": self.training_samples,
                                                "training_start_time": self.training_start_time,
                                                "vocab_size": self.tokenizer.vocab_size,
                                                "dataset_tokens": self.dataset_tokens,
                                                "dataset_samples": self.dataset_samples})
                    dist.barrier()
                    # No need to reload — all_reduce already keeps weights in sync
                except Exception as e:
                    _log_error(self.err_file, f"checkpoint batch {self.global_batch}: {e}")
                self.last_ckpt_time = time.time()
                self.num_batches = 0
                self.total_loss = 0.0

    def _save_checkpoint(self, epoch: int, loss: float):
        """Save from gpu_id == 0. No reload needed — all_reduce keeps weights synced."""
        ckpt_dir = Path(self.paths["checkpoint_dir"]) / self.ckpt_hash
        if self.gpu_id == 0:
            _write_status(self.log_file, epoch, self.global_batch, loss,
                          self.training_samples, self.model_cfg["seq_length"],
                          self.training_start_time)
            save_checkpoint(epoch, loss, self.combined_config, self.ckpt_hash,
                            self.model, self.paths["checkpoint_dir"],
                            optimizer=self.optimizer,
                            extra={"global_batch": self.global_batch,
                                    "batch_size": self.train_cfg["batch_size"],
                                    "seq_length": self.model_cfg["seq_length"],
                                    "training_samples": self.training_samples,
                                    "training_start_time": self.training_start_time,
                                    "vocab_size": self.tokenizer.vocab_size,
                                    "dataset_tokens": self.dataset_tokens,
                                    "dataset_samples": self.dataset_samples})
        dist.barrier()

    def train(self, total_epochs: int, start_epoch: int = 0):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1, last_epoch=-1)
        for epoch in range(start_epoch + 1, total_epochs + 1):
            self._run_epoch(epoch)

            avg_loss = self.total_loss / max(1, self.num_batches)
            if self.gpu_id == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch}/{total_epochs} "
                      f"completed | avg_loss {avg_loss:.4f} | "
                      f"global_batch {self.global_batch}",
                      flush=True)

            # End-of-epoch checkpoint
            if epoch % self.ckpt_every == 0:
                try:
                    self._save_checkpoint(epoch, avg_loss)
                except Exception as e:
                    _log_error(self.err_file, f"epoch checkpoint {epoch}: {e}")

    def close(self):
        if self.gpu_id == 0:
            if self.log_file:
                self.log_file.close()
            if self.err_file:
                self.err_file.close()


# =============================================================================
# 5. DDP WORKER  (per tutorial: rank, world_size, mp.spawn)
# =============================================================================
def run_ddp(rank: int, world_size: int, devices: tuple, master_port: str, config_path: str,
            total_epochs: int, checkpoint_every: int):
    """
    Args:
        rank:           auto-allocated by mp.spawn
        world_size:     number of GPUs in use
        devices:        tuple of actual CUDA device indices (e.g. (0, 1))
        master_port:    TCP port for rendezvous
        config_path:    path to config JSON
        total_epochs:   override for training epochs
        checkpoint_every: save checkpoint every N epochs
    """
    device = devices[rank]
    dist_initialized = False
    cuda_initialized = False

    # Register this child's PID for cleanup
    pid_file = os.environ.get("DDP_PID_FILE")
    if pid_file:
        try:
            with open(pid_file, 'a') as f:
                f.write(f"{os.getpid()}\n")
        except Exception:
            pass

    try:
        # Use multiple CPU threads for gloo all_reduce (default is 1 — too slow)
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 8) // 2)))

        # Reset CUDA context inherited from parent (fixes Windows 0xC0000005 crash)
        try:
            torch.cuda.set_per_process_memory_fraction(1.0, device)
            cuda_initialized = True
        except (RuntimeError, AttributeError):
            pass

        dist_setup(rank, world_size, device, master_port)
        dist_initialized = True

        # --- Load config ---
        model_cfg, vocab_cfg, train_cfg, paths = load_config(config_path)

        # Override from CLI
        if total_epochs > 0:
            train_cfg["epochs"] = total_epochs
        if checkpoint_every > 0:
            train_cfg["checkpoint_every"] = checkpoint_every

        total_epochs = train_cfg.get("epochs", 10)
        batch_size = train_cfg.get("batch_size", 16)

        # --- Build tokenizer + dataset ---
        tokenizer, dataset, vocab_hash = build_tokenizer_and_dataset(
            rank, world_size, model_cfg, vocab_cfg, train_cfg, paths
        )

        # --- Prepare dataloader ---
        train_data, sampler = prepare_dataloader(dataset, batch_size, rank, world_size)

        # --- Build model (NO DDP wrapper — manual grad sync) ---
        model = GPTMini(model_cfg, tokenizer.vocab_size).to(f"cuda:{device}")

        # --- Init sync buffers (cpu method only) ---
        sync_cfg = train_cfg.get("sync", {})
        sync_method = sync_cfg.get("method", "gpu")
        if sync_method == "cpu" and world_size > 1:
            num_chunks = sync_cfg.get("chunks", 4)
            _build_chunked_buffers(model, num_chunks)
            if rank == 0:
                chunk_mem = sum(b.element_size() * b.numel() for b in model._chunk_bufs) / (1024**3)
                print(f"Sync buffers: {num_chunks} chunks, {chunk_mem:.2f} GB total (pinned CPU)", flush=True)

        # --- Optimizer ---
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get("lr", 0.0002))

        # --- Combined config ---
        combined_config = {"model": model_cfg, "training": train_cfg, "paths": paths}
        combined_config["model"]["vocab"] = vocab_cfg

        # --- Trainer (per tutorial) ---
        ckpt_hash = get_model_hash(model, vocab_hash)
        trainer = Trainer(model, train_data, sampler, optimizer, rank, world_size,
                            device, model_cfg, train_cfg, paths, ckpt_hash,
                            combined_config, tokenizer,
                            dataset_tokens=dataset.token_count,
                            dataset_samples=len(dataset))
        try:
            trainer.train(total_epochs)
        finally:
            trainer.close()

    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] Rank {rank}: {e}", flush=True)
        print(f"[OOM] Rank {rank}: Aborting — reduce n_layer, batch_size, or seq_length.", flush=True)
        raise
    except KeyboardInterrupt:
        print(f"[Rank {rank}] Interrupted.", flush=True)
        raise
    except Exception as e:
        print(f"Rank {rank} training error: {e}", flush=True)
        raise
    finally:
        if cuda_initialized:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if dist_initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                pass


# =============================================================================
# 6. GPU COUNT (without touching CUDA)
# =============================================================================
def _get_cuda_device_count() -> int:
    """Get GPU count without initializing CUDA context.
    Uses nvidia-smi to avoid corrupting CUDA context for mp.spawn children."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=10
        )
        return int(out.strip())
    except Exception:
        try:
            return torch.cuda.device_count()
        except Exception:
            return 0


_pid_file = None


def _get_pid_file():
    """Return a temp file path to track child PIDs."""
    global _pid_file
    if _pid_file is None:
        import tempfile
        _pid_file = tempfile.NamedTemporaryFile(prefix="ddp_pids_", suffix=".txt",
                                                 delete=False, mode='w').name
    return _pid_file


def _get_child_pids():
    """Read tracked child PIDs from file."""
    pf = _get_pid_file()
    try:
        with open(pf, 'r') as f:
            return [int(line.strip()) for line in f if line.strip().isdigit()]
    except (FileNotFoundError, ValueError):
        return []


def _kill_orphans():
    """Kill only our spawned child processes on Windows."""
    pids = _get_child_pids()
    if not pids:
        print("[Cleanup] No child processes to terminate.", flush=True)
        return
    print(f"[Cleanup] Force-killing {len(pids)} child process(es): {pids}", flush=True)
    for pid in pids:
        try:
            res = subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                capture_output=True, text=True, timeout=5)
            if res.returncode != 0:
                print(f"  taskkill PID {pid}: {res.stderr.strip()}", flush=True)
        except Exception as e:
            print(f"  Failed to kill PID {pid}: {e}", flush=True)
    time.sleep(0.5)
    try:
        os.unlink(_get_pid_file())
    except OSError:
        pass


# =============================================================================
# 7. ENTRY POINT
# =============================================================================
def run():
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-GPU DDP trainer (PyTorch tutorial pattern)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="Comma-separated GPU indices (e.g. 0,1). Default: all CUDA devices.")
    parser.add_argument("--port", type=str, default="29500",
                        help="TCP port for DDP rendezvous.")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Override training epochs from config.")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Override checkpoint_every from config.")
    parser.add_argument("config", nargs="?", default="gpt_mini3.json",
                        help="Path to config JSON.")
    args = parser.parse_args()

    # Resolve GPU count WITHOUT touching CUDA (avoids corrupted context for mp.spawn on Windows)
    cuda_count = _get_cuda_device_count()
    if cuda_count == 0:
        print("Error: No CUDA devices available.")
        sys.exit(1)

    if args.device:
        devices = tuple(int(x.strip()) for x in args.device.split(","))
        for d in devices:
            if d < 0 or d >= cuda_count:
                print(f"Error: GPU {d} not available. Valid range: 0..{cuda_count - 1}")
                sys.exit(1)
    else:
        devices = tuple(range(cuda_count))

    world_size = len(devices)

    # Setup PID tracking for children
    pid_file = _get_pid_file()
    try:
        os.unlink(pid_file)
    except FileNotFoundError:
        pass
    os.environ["DDP_PID_FILE"] = pid_file

    try:
        if world_size == 1:
            # TODO: single-GPU DDP silently exits after "Starting training..."
            # Root cause: DDP + dist.init_process_group + gloo for world_size=1 hangs on Windows
            # Fix: skip DDP wrapper for world_size=1, use regular DataLoader + model (no dist calls)
            # For now: use gpt_mini3.py directly for single-GPU training
            run_ddp(0, 1, devices, args.port, args.config, args.epochs, args.save_every)
        else:
            mp.set_sharing_strategy("file_system")
            mp.spawn(run_ddp,
                      args=(world_size, devices, args.port, args.config,
                            args.epochs, args.save_every),
                      nprocs=world_size,
                      start_method="spawn",
                      join=True)
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Shutting down...")
        _kill_orphans()
        time.sleep(1)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}", flush=True)
        _kill_orphans()
        time.sleep(1)
        sys.exit(1)


if __name__ == "__main__":
    run()
