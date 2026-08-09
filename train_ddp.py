"""
Multi-GPU DDP trainer following PyTorch tutorial pattern:
  https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html

Usage:
    python train_ddp.py                          # all CUDA GPUs, default config
    python train_ddp.py -d 0,1                   # GPUs 0 and 1
    python train_ddp.py -d 0 gpt_mini3.json      # GPU 0 only, custom config
    python train_ddp.py --epochs 10 --save_every 3
"""
import os, sys, json, time, hashlib
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt_mini3 import (
    GPTMini, WordTokenizer, WordDataset, ensure_corpus,
    save_checkpoint, find_latest_checkpoint, generate_text,
    _write_status, _log_error,
)


# =============================================================================
# 1. DDP SETUP  (exactly per PyTorch tutorial)
# =============================================================================
def ddp_setup(rank: int, world_size: int, device: int, master_port: str):
    """
    Args:
        rank:       Unique identifier of each process (0..world_size-1)
        world_size: Total number of processes
        device:     Actual CUDA device index this rank maps to
        master_port: TCP port for rendezvous
    """
    # Set device BEFORE init_process_group (prevents hangs / OOM on GPU:0)
    torch.cuda.set_device(device)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port

    # NCCL not compiled on Windows PyTorch wheel -> use gloo
    backend = "gloo"
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except RuntimeError as e:
        if "built in" in str(e):
            raise RuntimeError(f"Backend '{backend}' not compiled. Install torch with NCCL support or use gloo.") from e
        raise

    print(f"Rank {rank} -> cuda:{device} | {torch.cuda.get_device_name(device)}  "
          f"(world_size={world_size})")


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
    model_param_dict = {
        "n_layer": model_cfg["n_layer"], "n_head": model_cfg["n_head"],
        "head_dim": model_cfg["head_dim"], "seq_length": model_cfg["seq_length"],
        "max_vocab_size": vocab_cfg.get("max_vocab_size", 32768),
        "max_word_len": vocab_cfg.get("max_word_len", 20),
    }
    model_param_hash = hashlib.sha256(
        json.dumps(model_param_dict, sort_keys=True).encode()
    ).hexdigest()[:16]

    cache_dir = Path(paths.get("cache_dir", "E:\\training\\cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    vocab_cache = cache_dir / f"vocab-{model_param_hash}.json"

    # Corpus hash for data cache
    data_dirs = [paths["data_dir"]]
    if "extra_data_dirs" in paths:
        data_dirs.extend(paths["extra_data_dirs"])

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
    data_cache = cache_dir / f"data-{model_param_hash}-{corpus_h}.npy"
    is_main = (rank == 0)

    tokenizer = WordTokenizer(
        max_vocab_size=vocab_cfg.get("max_vocab_size", 32768),
        max_word_len=vocab_cfg.get("max_word_len", 20),
    )

    sentences = []
    if vocab_cache.exists() and data_cache.exists() and data_cache.stat().st_size > 1_000_000_000:
        if is_main:
            print("Loading cached vocab + dataset...", flush=True)
        tokenizer.load(vocab_cache)
    else:
        if is_main:
            sentences = ensure_corpus(paths["data_dir"], paths.get("extra_data_dirs", []))
            tokenizer.build_vocab(sentences)
            tokenizer.save(vocab_cache)
            print(f"Vocab built: {tokenizer.vocab_size} tokens, cached to {vocab_cache}", flush=True)
        dist.barrier()
        if not is_main:
            tokenizer.load(vocab_cache)

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

    return tokenizer, dataset, model_param_hash


def load_train_objs(tokenizer, dataset, model_cfg, device):
    """Returns model. Per tutorial: called AFTER ddp_setup()."""
    return GPTMini(model_cfg, tokenizer.vocab_size).to(f"cuda:{device}")


# =============================================================================
# 3. PREPARE DATALOADER  (per tutorial)
# =============================================================================
def prepare_dataloader(dataset, batch_size: int):
    """
    DistributedSampler chunks input data across all processes.
    Each process gets batch_size samples; effective batch = batch_size * nprocs.
    """
    from torch.utils.data import DataLoader
    sampler = DistributedSampler(dataset, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True,
                      num_workers=0, pin_memory=False), sampler


# =============================================================================
# 4. TRAINER CLASS  (per tutorial pattern)
# =============================================================================
class Trainer:
    def __init__(self, model, train_data, sampler, optimizer, rank, world_size,
                 device, model_cfg, train_cfg, paths, ckpt_hash, combined_config,
                 tokenizer, unwrapped_model):
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
        self.unwrapped_model = unwrapped_model

        # Checkpoint config
        self.ckpt_interval = train_cfg.get("checkpoint_interval", 0)
        self.ckpt_every_min = train_cfg.get("checkpoint_every_min", 0)
        self.ckpt_every = train_cfg.get("checkpoint_every", 1)
        self.grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
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
            self.log_file = open(ckpt_base / "checkpoint_status.txt", "w", encoding="utf-8")
            self.err_file = open(ckpt_base / "errors.log", "w", encoding="utf-8")
            self.training_start_time = time.time()
            precision = "bf16" if self.use_bf16 else "fp32"
            print(f"Precision: {precision}, Grad accumulation: {self.grad_accum}x", flush=True)
            print(f"Checkpoint hash: {ckpt_hash}", flush=True)
            print("Starting training...", flush=True)

        # Counters
        self.global_batch = 0
        self.num_batches = 0
        self.total_loss = 0.0
        self.training_samples = 0
        self.last_ckpt_time = time.time()

        # Resume check (ONLY rank 0)
        self._maybe_resume()

    def _maybe_resume(self):
        ckpt = find_latest_checkpoint(self.paths["checkpoint_dir"], self.ckpt_hash)
        if ckpt:
            ep, info, ckpt_dir = ckpt
            self.global_batch = int(info.get("global_batch", 0))
            if self.gpu_id == 0:
                print(f"Resuming from {ckpt_dir} (epoch {ep}, loss {info['loss']:.6f}, "
                      f"global_batch {self.global_batch})", flush=True)
            # ALL ranks load to sync weights
            ckpt_state = torch.load(ckpt_dir / "model.pth", map_location=f"cuda:{self.device}")
            self.unwrapped_model.load_state_dict(ckpt_state)
            del ckpt_state

    def _run_epoch(self, epoch: int):
        """Run one training epoch. Per tutorial: call sampler.set_epoch() every epoch."""
        try:
            self.sampler.set_epoch(epoch)
        except Exception as e:
            _log_error(self.err_file, f"sampler.set_epoch({epoch}): {e}")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for x, y in self.train_data:
            try:
                x, y = x.to(f"cuda:{self.device}"), y.to(f"cuda:{self.device}")

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    _, loss = self.model(x, y)

                loss = loss / self.grad_accum
                loss.backward()

                if (self.global_batch + 1) % self.grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                self.total_loss += loss.item() * self.grad_accum
                self.num_batches += 1
                self.global_batch += 1
                self.training_samples += x.size(0)

                # Progress logging (rank 0 only)
                if self.gpu_id == 0 and self.global_batch % self.log_interval == 0:
                    avg = self.total_loss / max(1, self.num_batches)
                    print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch} | "
                          f"batch {self.global_batch} | loss {avg:.4f} | "
                          f"lr {self.optimizer.param_groups[0]['lr']:.6e}",
                          flush=True)
            except Exception as e:
                _log_error(self.err_file, f"batch {self.global_batch}: {e}")
                raise

            # Checkpoint at interval / time
            should_ckpt = False
            if self.ckpt_interval > 0 and self.global_batch % self.ckpt_interval == 0:
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
                                        self.unwrapped_model, self.paths["checkpoint_dir"],
                                        extra={"global_batch": self.global_batch,
                                               "batch_size": self.train_cfg["batch_size"],
                                               "seq_length": self.model_cfg["seq_length"],
                                               "training_samples": self.training_samples})
                    dist.barrier()
                    # ALL ranks load the checkpoint to sync weights
                    ckpt_state = torch.load(ckpt_dir / "model.pth", map_location=f"cuda:{self.device}")
                    self.unwrapped_model.load_state_dict(ckpt_state)
                    del ckpt_state
                except Exception as e:
                    _log_error(self.err_file, f"checkpoint batch {self.global_batch}: {e}")
                self.last_ckpt_time = time.time()
                self.num_batches = 0
                self.total_loss = 0.0

    def _save_checkpoint(self, epoch: int, loss: float):
        """Save from gpu_id == 0, then ALL ranks load to sync weights."""
        ckpt_dir = Path(self.paths["checkpoint_dir"]) / self.ckpt_hash
        if self.gpu_id == 0:
            _write_status(self.log_file, epoch, self.global_batch, loss,
                          self.training_samples, self.model_cfg["seq_length"],
                          self.training_start_time)
            save_checkpoint(epoch, loss, self.combined_config, self.ckpt_hash,
                            self.unwrapped_model, self.paths["checkpoint_dir"],
                            extra={"global_batch": self.global_batch,
                                   "batch_size": self.train_cfg["batch_size"],
                                   "seq_length": self.model_cfg["seq_length"],
                                   "training_samples": self.training_samples})
        dist.barrier()
        # ALL ranks load to sync weights
        ckpt_state = torch.load(ckpt_dir / "model.pth", map_location=f"cuda:{self.device}")
        self.unwrapped_model.load_state_dict(ckpt_state)
        del ckpt_state

    def train(self, total_epochs: int, start_epoch: int = 0):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        for epoch in range(start_epoch + 1, total_epochs + 1):
            self._run_epoch(epoch)

            avg_loss = self.total_loss / max(1, self.num_batches)
            if self.gpu_id == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch}/{total_epochs} "
                      f"completed | avg_loss {avg_loss:.4f} | "
                      f"global_batch {self.global_batch}",
                      flush=True)

            try:
                scheduler.step()
            except Exception as e:
                _log_error(self.err_file, f"scheduler.step epoch {epoch}: {e}")

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

    # Reset CUDA context inherited from parent (fixes Windows 0xC0000005 crash)
    try:
        torch.cuda.set_per_process_memory_fraction(1.0, device)
    except (RuntimeError, AttributeError):
        pass

    ddp_setup(rank, world_size, device, master_port)

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
    tokenizer, dataset, model_param_hash = build_tokenizer_and_dataset(
        rank, world_size, model_cfg, vocab_cfg, train_cfg, paths
    )

    # --- Prepare dataloader (per tutorial) ---
    train_data, sampler = prepare_dataloader(dataset, batch_size)

    # --- Build model ---
    model = GPTMini(model_cfg, tokenizer.vocab_size).to(f"cuda:{device}")

    # --- DDP wrap (per tutorial) ---
    model = DDP(model, device_ids=[device], output_device=device)

    # --- Optimizer ---
    unwrapped_model = model.module
    optimizer = torch.optim.Adam(unwrapped_model.parameters(), lr=train_cfg.get("lr", 0.0002))

    # --- Combined config ---
    combined_config = {"model": model_cfg, "training": train_cfg, "paths": paths}
    combined_config["model"]["vocab"] = vocab_cfg

    # --- Trainer (per tutorial) ---
    trainer = Trainer(model, train_data, sampler, optimizer, rank, world_size,
                      device, model_cfg, train_cfg, paths, model_param_hash,
                      combined_config, tokenizer, unwrapped_model)
    try:
        trainer.train(total_epochs)
    except Exception as e:
        print(f"Rank {rank} training error: {e}", flush=True)
        raise
    finally:
        trainer.close()

    # --- Cleanup ---
    dist.destroy_process_group()


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
        os._exit(1)
    except Exception as e:
        print(f"Training failed: {e}", flush=True)
        raise


if __name__ == "__main__":
    run()
