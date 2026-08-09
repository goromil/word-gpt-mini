# RPC Gradient Sync — Experimental (Does Not Work on Windows)

## Objective

Multi-GPU training on Windows without `mp.spawn`. Goal: each process launches independently with a fresh CUDA context, avoiding the `0xC0000005` access violation from PyTorch's `mp.spawn` context inheritance bug.

## What Was Tried

### 1. PyTorch Distributed RPC (`torch.distributed.rpc`)

**Idea:** Use `rpc.init_rpc()` + `rpc.rpc_async()` to exchange gradients between ranks.

**Result:** `AttributeError: module 'torch.distributed.rpc' has no attribute 'init_rpc'`

**Why:** The TensorPipe backend (required for RPC) is not compiled in the Windows PyTorch wheel. RPC is Linux-only.

---

### 2. DDP via `dist.init_process_group()` — independent process launch

**Idea:** Skip `mp.spawn` entirely. Launch 2 independent Python processes with `subprocess.Popen()`, each calling `dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:PORT")`. Use DDP wrapper for `all_reduce`.

**Result:** `0xC0000005` (exit code 3221225477) — both ranks crash immediately after "Starting training...".

**Why:** `gloo` backend cannot do `all_reduce` on CUDA tensors on Windows. DDP's backward pass calls `all_reduce` on GPU-resident gradients. `gloo` tries to copy CUDA → CPU → all_reduce → CUDA, and the CUDA tensor handoff triggers an access violation. This is a fundamental Windows limitation in the `gloo` backend.

The `[MACUBE]:29500 error 10049` warning is cosmetic — hostname resolves to IPv6 first, fails, falls back to IPv4. DDP init succeeds; the crash happens during backward.

---

### 3. Manual gradient sync over raw TCP sockets

**Idea:** No DDP, no gloo, no collectives. Each rank computes its own gradients, serializes them as CPU numpy arrays, sends over TCP to the other rank, both average `(g0 + g1) / 2`, then apply.

**Result:** Serialization round-trip works correctly. But **impractical bandwidth**:

| Model param | Gradient size (fp32) | Per-batch transfer (2-way) | Time on 1Gbps LAN |
|-------------|---------------------|---------------------------|-------------------|
| 900M params | 3.6 GB per rank     | 7.2 GB per batch          | ~58 seconds       |

DDP's `all_reduce` is ~1000x faster because gradients sync over PCIe/NVLink (~100 GB/s) without CPU round-trip. TCP gradient sync adds ~58 seconds per batch on top of compute time, making training unusable.

**Why it fails:** Two problems:

1. **Bandwidth (impractical):** You cannot match PCIe/NVLink bandwidth over 1Gbps Ethernet. Even with fp16 quantization, it's ~29 seconds per batch.

2. **`0xC0000005` during first batch:** Rank 0 crashed with access violation during the first `loss.backward()` — before any TCP sync happened. This proves the crash is from PyTorch CUDA kernel execution itself on Windows, not from DDP/gloo. Even a bare model + backward + autocast crashes on Windows for this model size.

---

## Why Multi-GPU on Windows Is Broken

| Layer | Problem |
|-------|---------|
| `mp.spawn` | Corrupts inherited CUDA context → `0xC0000005` |
| NCCL | Not compiled in Windows PyTorch wheels |
| `gloo` + CUDA tensors | Access violation on `all_reduce` with GPU tensors |
| `gloo` + CPU tensors | Works but requires CPU copy overhead |
| RPC (TensorPipe) | Not compiled in Windows wheel |
| Independent processes + DDP | Same `gloo` crash as above |
| Manual TCP sync | Bandwidth too slow for large models |

**Bottom line:** PyTorch's multi-GPU stack was designed for Linux. Windows support is best-effort and breaks at the boundary between CUDA and `gloo`.

## The Working Path: WSL2

`train_ddp.py` works as-is on Linux because:

- `fork()` (default `start_method`) gives each child a fresh CUDA context
- NCCL is compiled → GPU↔GPU sync over NVLink/PCIe, no `gloo` needed
- `mp.spawn` works without context corruption

**Migration is 0 lines of code.** Copy `train_ddp.py` + `gpt_mini3.py` + config to WSL2 and run:
```bash
python train_ddp.py -d 0,1
```

## Files

| File | Status |
|------|--------|
| `experimental/train_rpc.py` | TCP grad sync — serializes correctly, bandwidth impractical |
| `experimental/launch_rpc.py` | Launcher for independent process training |
| `train_ddp.py` | Works on Linux (WSL2), broken on Windows |
| `gpt_mini3.py` | Single-GPU training, works everywhere |

## Decision

Multi-GPU training stays on **WSL2 + `train_ddp.py`**. Windows multi-GPU via PyTorch is not viable — not a code problem, a platform limitation.
