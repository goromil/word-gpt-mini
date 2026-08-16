#!/usr/bin/env python3
"""
train_designer.py — Interactive training config designer

Replaces init_training.py with GPU-memory-aware interactive mode.
Loads draft config, detects GPUs, proposes valid (seq_length, batch_size)
combinations that fit the hardware, and generates the final config.

Usage:
    python train_designer.py                  # interactive
    python train_designer.py --no-interact     # auto-select best config
    python train_designer.py --force           # overwrite without prompt
    python train_designer.py --scan-vocab      # scan for vocab size
"""

import json, math, sys, os

FLAGS = {"--download", "--scan-vocab", "--force", "--no-interact"}
args = [a for a in sys.argv[1:] if a not in FLAGS]
scan_vocab_flag = "--scan-vocab" in sys.argv
force = "--force" in sys.argv
no_interact = "--no-interact" in sys.argv
draft_path = args[0] if len(args) > 0 else "gpt_train_draft.json"


def next_pow2(x):
    return 1 << (x - 1).bit_length()


# --- GPU detection -------------------------------------------------------

def detect_gpus():
    """Detect available GPUs and their VRAM."""
    try:
        import torch
        count = torch.cuda.device_count()
        gpus = []
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpus.append({"idx": i, "name": name, "gb": round(mem, 1)})
        return gpus
    except Exception:
        return []


def prompt_gpus(gpus, interactive=True):
    """Let user select GPU(s) to use."""
    print()
    print(f"  Detected {len(gpus)} GPU(s):")
    for g in gpus:
        print(f"    [{g['idx']}] {g['name']}  ({g['gb']} GB)")

    if len(gpus) == 0:
        print("  ERROR: No GPU found. Training requires CUDA.")
        sys.exit(1)

    if len(gpus) == 1:
        print(f"\n  Using GPU 0 ({gpus[0]['name']}, {gpus[0]['gb']} GB)")
        return gpus

    if not interactive:
        print(f"\n  Non-interactive mode: using all {len(gpus)} GPUs")
        total_gb = sum(g['gb'] for g in gpus)
        print(f"  Using {len(gpus)}x {gpus[0]['name']} ({total_gb} GB total)")
        return gpus

    print(f"\n  Enter GPU indices to use, comma-separated (e.g. '0,1'):")
    while True:
        raw = input(f"  GPUs [{', '.join(str(g['idx']) for g in gpus)}]: ").strip()
        if not raw:
            idxs = [gpus[0]['idx']]  # default: first GPU
            break
        try:
            idxs = [int(x.strip()) for x in raw.split(',')]
            if all(i in [g['idx'] for g in gpus] for i in idxs):
                break
            print("  ERROR: Invalid GPU index(s)")
        except ValueError:
            print("  ERROR: Enter numbers separated by commas")

    selected = [g for g in gpus if g['idx'] in idxs]
    total_gb = sum(g['gb'] for g in selected)
    print(f"\n  Using {len(selected)}x {selected[0]['name']} ({total_gb} GB total)")
    return selected


# --- Memory estimation ---------------------------------------------------

def calc_params(m):
    """Calculate total model parameters."""
    n_layer = m["n_layer"]
    n_head = m["n_head"]
    head_dim = m["head_dim"]
    seq_length = m["seq_length"]
    vocab_dict = m.get("vocab") or m.get("tokenizer", {})
    nvocab = vocab_dict.get("max_vocab_size", 32768) if isinstance(vocab_dict, dict) else 32768
    n_embd = n_head * head_dim

    wte = nvocab * n_embd
    wpe = seq_length * n_embd
    c_attn = n_embd * (3*n_embd) + 3*n_embd
    c_proj = (3*n_embd) * n_embd + n_embd
    mlp = (n_embd*(4*n_embd)+4*n_embd) + ((4*n_embd)*n_embd+n_embd)
    ln = 4 * n_embd
    per_layer = c_attn + c_proj + ln + mlp
    ln_f = 2 * n_embd
    lm_head = 0
    total = wte + wpe + per_layer * n_layer + ln_f + lm_head
    return total, n_embd


def estimate_memory_per_gpu(params, n_heads, n_embd, n_layer, seq_length, bs_per_gpu, world_size, overhead_factor=1.15):
    """
    Full memory estimation per GPU.

    Returns dict with breakdown.
    """
    # Fixed costs (per-GPU) — each GPU holds FULL model + optimizer copy in DDP
    # FP16 working weights: 2B, FP32 master weights: 4B, Adam m: 4B, Adam v: 4B = 14B/param
    # NOTE: NOT divided by world_size — DDP replicates model+optimizer on each GPU
    fixed_gb = params * 14 / (1024**3)

    # Variable costs (activations, depend on bs x seq_length)
    #
    # Per-token per-layer memory (empirical, validated against actual PyTorch):
    #   QKV (FP32):     12 x n_embd bytes     (3 projections x n_embd x 4B)
    #   Attention:      4 x n_heads x seq     (n_heads x seq^2 x 4B, stored for backward)
    #   c_proj+residual: 4 x n_embd bytes     (FP16 output + FP16 hidden, 2 x n_embd x 2B)
    #   MLP fc1:        8 x n_embd bytes      (FP16 output, 4 x n_embd x 2B)
    #   MLP fc2:        2 x n_embd bytes      (FP16 output, n_embd x 2B)
    #   ln_1+ln_2:      4 x n_embd bytes      (FP16 outputs, 2 x n_embd x 2B)
    #   Input (FP16):   2 x n_embd bytes      (saved for backward)
    #
    # Total per token per layer:
    #   = (12+4+8+2+4+2) x n_embd + 4 x n_heads x seq_length
    #   = 32 x n_embd + 4 x n_heads x seq_length
    #
    # Actual PyTorch uses mixed precision (FP16 for most intermediates, FP32 for QKV/attn).
    # Per-token per-layer forward activations (FP16 intermediates + FP32 for LN):
    #   LN1 input+output:    6 x n_embd  (fp16 input + fp32 mean/var)
    #   QKV projections:    12 x n_embd  (fp16, 3 x n_embd)
    #   c_proj output:       2 x n_embd  (fp16)
    #   LN2 input+output:    6 x n_embd  (fp16 + fp32)
    #   MLP fc1 output:      8 x n_embd  (fp16, 4 x n_embd)
    #   MLP fc2 output:      2 x n_embd  (fp16)
    #   Residual inputs:     4 x n_embd  (fp16, 2 residuals)
    #   Attention scores:    4 x n_heads x seq_length (fp16, T x T per head, per-token)
    #
    # Total: (38 x n_embd + 4 x n_heads x seq_length) bytes/token/layer (forward only)
    #
    # With gradient checkpointing (checkpoint_every=1):
    #   - Forward: only store block input (2 x n_embd per token) + LN1/LN2 fp32 state
    #   - Backward: recompute forward per block, store gradients (0.5-0.7x forward)
    #   - Effective: ~20 x n_embd + 2 x n_heads x seq_length per token (checkpointed)

    use_checkpointing = True  # gpt_train.py uses checkpoint_sequential in training
    if use_checkpointing:
        bytes_per_token = 20 * n_embd + 2 * n_heads * seq_length
        fwd_mult = 1.0   # store block inputs across all layers
        bwd_mult = 0.6   # recompute per-block, store gradients
    else:
        bytes_per_token = 38 * n_embd + 4 * n_heads * seq_length
        fwd_mult = 1.0
        bwd_mult = 1.0   # backward stores gradients for all stored activations

    act_per_layer_bytes = bs_per_gpu * seq_length * bytes_per_token
    act_per_layer_gb = act_per_layer_bytes / (1024**3)

    # Forward pass activations (stored for backward)
    var_fwd_gb = act_per_layer_gb * n_layer * fwd_mult

    # Backward pass: gradient storage (recomputed activations don't add extra)
    var_bwd_gb = var_fwd_gb * bwd_mult

    # DDP all-reduce buffer (~params / world_size x 2B for gradient communication)
    ddp_gb = params * 2 / world_size / (1024**3)

    total_gb = (fixed_gb + var_fwd_gb + var_bwd_gb + ddp_gb) * 1.15

    return {
        "fixed_gb": fixed_gb,
        "var_fwd_gb": var_fwd_gb,
        "var_bwd_gb": var_bwd_gb,
        "ddp_gb": ddp_gb,
        "total_gb": total_gb,
        "per_layer_act_gb": act_per_layer_gb,
        "fits": total_gb <= 23.0,  # leave 1 GB headroom
    }


# --- Propose solutions ---------------------------------------------------

def propose_solutions(params, n_heads, n_embd, n_layer, draft_bs, gpu_gb, world_size, nvocab):
    """
    Find valid (seq_length, bs_per_gpu) combos that fit in GPU memory.
    Return sorted by training speed (batches per second).
    """
    candidates = []

    # Search space
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    bs_per_gpu_values = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]

    for bs_pg in bs_per_gpu_values:
        for bs in seq_lengths:
            mem = estimate_memory_per_gpu(params, n_heads, n_embd, n_layer, bs, bs_pg, world_size)
            if mem["fits"]:
                total_bs = bs_pg * world_size
                if total_bs >= 2:  # minimum global batch
                    # Batches per epoch with TinyStories-v2 GPT4 (1.8B tokens)
                    tokens = 1_800_000_000
                    bpe = tokens // (total_bs * bs)
                    candidates.append({
                        "seq_length": bs,
                        "bs_per_gpu": bs_pg,
                        "global_bs": total_bs,
                        **mem,
                        "batches_per_epoch": bpe,
                    })

    # Sort by global batch (bigger = faster training), then by seq_length
    candidates.sort(key=lambda c: (c["global_bs"], c["seq_length"]), reverse=True)
    return candidates[:20]


# --- Epoch calculation ---------------------------------------------------

def calc_epochs(params, tokens, N):
    return math.ceil(params * N / tokens)


# --- Interactive flow ----------------------------------------------------

def run(draft_path):
    # Load draft
    with open(draft_path, "r") as f:
        draft = json.load(f)

    m = draft["model"]
    vocab_cfg = draft.get("tokenizer", draft.get("vocab", m.get("vocab", {})))
    td = draft.get("training-defaults", {})
    paths = draft.get("paths", {})

    nvocab = vocab_cfg.get("max_vocab_size", 32768)

    # Calculate params
    total_params, n_embd = calc_params(m)
    n_layer = m["n_layer"]
    n_head = m["n_head"]
    head_dim = m["head_dim"]
    draft_bs = td.get("batch_size", 64)
    draft_block = m["seq_length"]

    # Detect GPUs
    gpus = detect_gpus()
    selected_gpus = prompt_gpus(gpus, interactive=not no_interact)
    world_size = len(selected_gpus)
    gpu_gb = selected_gpus[0]["gb"]
    total_gb = sum(g["gb"] for g in selected_gpus)

    print(f"\n  Model: {n_layer}L / {n_head}H / hd={head_dim} / emb={n_embd} / vocab={nvocab}")
    print(f"  Params: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Hardware: {world_size}x RTX 3090 ({gpu_gb} GB each, {total_gb} GB total)")

    # Epoch table
    corpora = {
        "TinyStories-v2 GPT4": {"tokens": 1_800_000_000, "size_mb": 1800, "desc": "GPT-4 only, ~1.8B tokens"},
        "TinyStories-train":   {"tokens": 2_200_000_000, "size_mb": 2300, "desc": "Full set, ~2.2B tokens"},
    }

    print(f"\n  {'='*60}")
    print(f"  EPOCH CALCULATOR (N=200)")
    print(f"  {'='*60}")
    N = 200
    for name, info in corpora.items():
        epochs = calc_epochs(total_params, info["tokens"], N)
        bpe = info["tokens"] // (draft_bs * draft_block)
        print(f"  {name}:")
        print(f"    {epochs:>4} epochs, {bpe:>8,} batches/epoch")

    # Memory analysis
    print(f"\n  {'='*60}")
    print(f"  GPU MEMORY ANALYSIS")
    print(f"  {'='*60}")

    bs_per_gpu = draft_bs // world_size
    mem = estimate_memory_per_gpu(total_params, n_head, n_embd, n_layer, draft_block, bs_per_gpu, world_size)

    print(f"\n  Draft config: seq_length={draft_block}, batch_size={draft_bs}")
    print(f"    bs_per_gpu  = {draft_bs} / {world_size} = {bs_per_gpu}")
    print(f"\n    Fixed (params+Adam):     {mem['fixed_gb']:6.2f} GB")
    print(f"    Activation (forward):   {mem['var_fwd_gb']:6.2f} GB")
    print(f"    Activation (backward):  {mem['var_bwd_gb']:6.2f} GB")
    print(f"    DDP communication:      {mem['ddp_gb']:6.2f} GB")
    print(f"    {'-'*56}")
    print(f"    Total estimated:          {mem['total_gb']:6.2f} GB")
    print(f"    GPU capacity:             {gpu_gb:6.1f} GB")
    status = 'OK' if mem['fits'] else '[OOM]'
    print(f"    Status: {status}")

    if not mem["fits"]:
        print(f"\n  [OOM] DRAFT EXCEEDS GPU MEMORY")
        print(f"\n  The breaking term:")
        print(f"    Attention matrix per layer:")
        print(f"      bs_per_gpu * n_heads * seq_length2 * 4B")
        print(f"      {bs_per_gpu} * {n_head} * {draft_block} * {draft_block} * 4 = "
              f"{bs_per_gpu * n_head * draft_block * draft_block * 4 / (1024**2):.0f} MB/layer")
        print(f"    MLP feedforward (dominant):")
        print(f"      bs_per_gpu * seq_length * 16 * n_embd * 4B")
        print(f"      {bs_per_gpu} * {draft_block} * 16 * {n_embd} * 4 = "
              f"{bs_per_gpu * draft_block * 16 * n_embd * 4 / (1024**2):.0f} MB/layer")
        print(f"\n  * {n_layer} layers * 1.6 (backward) = {mem['var_fwd_gb']:.1f} GB activations")

    # Propose solutions
    print(f"\n  {'='*60}")
    print(f"  PROPOSED CONFIGURATIONS (fit in {gpu_gb} GB GPU)")
    print(f"  {'='*60}")
    print()
    print(f"  {'seq_length':>10} {'global_bs':>9} {'bs/gpu':>7} {'mem_gb':>7} {'batches/epoch':>15} {'status':>6}")
    print(f"  {'-'*65}")

    candidates = propose_solutions(total_params, n_head, n_embd, n_layer, draft_bs, gpu_gb, world_size, nvocab)

    if not candidates:
        print("  ERROR: No valid configuration found. Reduce model size (n_layer/n_heads/head_dim).")
        sys.exit(1)

    for i, c in enumerate(candidates[:8]):
        status = "OK" if c["fits"] else "OOM"
        marker = "  RECOMMENDED" if i < 2 and not c["fits"] == False else ""
        print(f"  {c['seq_length']:>10} {c['global_bs']:>9} {c['bs_per_gpu']:>7} "
              f"{c['total_gb']:>6.1f} GB {c['batches_per_epoch']:>14,}  [{status}]{marker}")

    # Training speed estimate (rough: 2x3090  90 TFLOPS FP16)
    # sec/batch  (bs * seq_length * params * 6) / (TFLOPS * world_size * 1e12)
    tflops_per_gpu = 90  # RTX 3090 tensor cores FP16
    print(f"\n  Training speed (2xRTX 3090, ~{tflops_per_gpu} TFLOPS):")
    for i, c in enumerate(candidates[:4]):
        if not c["fits"]:
            continue
        batches = c["batches_per_epoch"]
        # Rough: each step = 2 * params * tokens * FLOPs/token
        # tokens_per_batch = bs * seq_length
        # FLOPs  6 * params * tokens (forward + backward)
        flops_per_batch = 2 * total_params * (c["global_bs"] * c["seq_length"])
        sec_per_batch = flops_per_batch / (tflops_per_gpu * 1e12 * world_size)
        hours_per_epoch = batches * sec_per_batch / 3600
        total_epochs = calc_epochs(total_params, 1_800_000_000, 50)
        total_hours = hours_per_epoch * total_epochs
        total_days = total_hours / 24

        arrow = " <--" if i < 2 else ""
        print(f"    seq_length={c['seq_length']:>5d}, bs={c['global_bs']:>3d}: "
              f"{sec_per_batch:.3f}s/batch, {hours_per_epoch:.1f}h/epoch, "
              f"{total_days:.0f} days total ({total_epochs} epochs){arrow}")

    # Select solution
    print(f"\n  {'='*60}")
    print(f"  SELECT CONFIGURATION")
    print(f"  {'='*60}")

    if no_interact:
        # Prefer balanced configs: high seq_length (better model quality) with reasonable batch
        # Sort by seq_length desc, then by global_bs desc
        balanced = sorted(candidates, key=lambda c: (c["seq_length"], c["global_bs"]), reverse=True)
        # Pick first with seq_length >= 128 if available
        selected = next((c for c in balanced if c["seq_length"] >= 128), balanced[0])
        print(f"  Auto-selecting: seq_length={selected['seq_length']}, batch_size={selected['global_bs']}")
    else:
        # Show numbered options
        print(f"\n  {'#':>3}  {'seq_length':>10} {'bs':>4} {'bs/gpu':>7} {'mem':>8}  {'batches/epoch':>15}")
        print(f"  {'-'*60}")
        for i, c in enumerate(candidates[:20]):
            arrow = " <--" if i == 0 else ""
            print(f"  {i:>3}  {c['seq_length']:>10} {c['global_bs']:>4} {c['bs_per_gpu']:>7} "
                  f"{c['total_gb']:.1f} GB {c['batches_per_epoch']:>14,}{arrow}")

        print(f"\n  0  = fastest training  (highest batch)")
        print(f"  1  = better context  (higher seq_length)")
        print(f"  Enter number, 'custom' for manual input, or press Enter for option 0:")

        while True:
            raw = input(f"  Choice [0]: ").strip()
            if not raw:
                selected = candidates[0]
                print(f"  Using: seq_length={selected['seq_length']}, batch_size={selected['global_bs']}")
                break
            try:
                idx = int(raw)
                if 0 <= idx < len(candidates):
                    selected = candidates[idx]
                    print(f"  Using: seq_length={selected['seq_length']}, batch_size={selected['global_bs']}")
                    break
                print(f"  Enter 0-{len(candidates)-1}")
            except ValueError:
                if raw.lower() == 'custom':
                    break
                print("  Enter a number from the list")

        if raw.lower() == 'custom':
            print(f"\n  Enter custom values:")
            while True:
                bs_raw = input(f"  seq_length (default {draft_block}): ").strip()
                bs = int(bs_raw) if bs_raw else draft_block
                bs_pg_raw = input(f"  batch_size per GPU (default 1): ").strip()
                bs_pg = int(bs_pg_raw) if bs_pg_raw else 1
                total_bs = bs_pg * world_size
                mem = estimate_memory_per_gpu(total_params, n_head, n_embd, n_layer, bs, bs_pg, world_size)
                status = 'OK' if mem['fits'] else 'OOM'
                print(f"    Estimated: {mem['total_gb']:.1f} GB [{status}]")
                if mem["fits"]:
                    break
                print(f"    OOM! Reduce seq_length or batch_size")

            selected = {"seq_length": bs, "bs_per_gpu": bs_pg, "global_bs": total_bs,
                       "total_gb": mem["total_gb"], "batches_per_epoch": 1_800_000_000 // (total_bs * bs)}

    # Epoch selection
    N_options = [20, 50, 100, 200]
    print(f"\n  Chinchilla multiplier (N):")
    for N in N_options:
        epochs = calc_epochs(total_params, 1_800_000_000, N)
        bpe = selected["batches_per_epoch"]
        print(f"    N={N:>3}: {epochs:>4} epochs, {bpe:>8,} batches/epoch")

    if no_interact:
        N = 50
    else:
        while True:
            N_raw = input(f"  Choose N (default 50): ").strip()
            if not N_raw:
                N = 50
                break
            try:
                N = int(N_raw)
                if N in N_options or (1 <= N <= 200):
                    break
            except ValueError:
                pass
            print("  Enter a number")

    epochs = calc_epochs(total_params, 1_800_000_000, N)

    # LR selection
    if epochs >= 50:
        lr = 0.00015
    elif epochs >= 10:
        lr = 0.0002
    else:
        lr = 0.0003

    # Checkpoint settings
    ckpt_every = max(1, epochs // 5)
    ckpt_interval = td.get("checkpoint_interval", 10000)
    ckpt_every_min = td.get("checkpoint_every_min", 30)

    # Vocab scan
    new_vocab = nvocab
    data_dir = paths.get("data_dir", "E:\\training\\data")
    outfile = os.path.join(data_dir, "tinystories.txt")

    if scan_vocab_flag and os.path.exists(outfile):
        def estimate_vocab(fp, sample_mb=200, max_word_len=20):
            import string
            translator = str.maketrans("", "", string.punctuation)
            sample_bytes = sample_mb * 1024 * 1024
            words = set()
            total_words = 0
            total_read = 0
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                while total_read < sample_bytes:
                    chunk = f.read(1024*1024)
                    if not chunk:
                        break
                    total_read += len(chunk)
                    for line in chunk.split("\n"):
                        for word in line.lower().split():
                            w = word.translate(translator)
                            if 1 <= len(w) <= max_word_len:
                                words.add(w)
                                total_words += 1
                    if total_read % (50*1024*1024) == 0:
                        print(f"\r    Read {total_read//(1024*1024)} MB, {len(words):,} unique", end="", flush=True)
            print()
            return next_pow2(int(len(words) * 1.05))

        new_vocab = estimate_vocab(outfile)
        print(f"  Vocab: {nvocab}  {new_vocab:,}")

    # Build final config
    config = {
        "model": {
            "n_layer": n_layer,
            "n_head": n_head,
            "head_dim": head_dim,
            "seq_length": selected["seq_length"],
        },
        "training": {
            "epochs": epochs,
            "batch_size": selected["global_bs"],
            "lr": lr,
            "checkpoint_every": ckpt_every,
            "checkpoint_interval": ckpt_interval,
            "checkpoint_every_min": ckpt_every_min,
        },
        "tokenizer": {
            "max_vocab_size": new_vocab,
            "max_word_len": vocab_cfg.get("max_word_len", 20),
        },
        "paths": paths,
    }

    out_file = "gpt_train.json"
    if os.path.exists(out_file) and not force and not no_interact:
        print(f"\n  ERROR: {out_file} already exists. Use --force to overwrite.")
        sys.exit(1)

    with open(out_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Generated {out_file}")
    print(f"    n_layer={n_layer}, n_head={n_head}, head_dim={head_dim}, seq_length={selected['seq_length']}")
    print(f"    batch_size={selected['global_bs']} (per GPU: {selected['bs_per_gpu']})")
    print(f"    epochs={epochs}, lr={lr}")
    print(f"    vocab={new_vocab}")
    print(f"    GPUs: {world_size}x ({gpu_gb} GB each)")

    # Embedding info
    emb_params = new_vocab * n_embd
    print(f"\n    Embedding: {new_vocab:,} * {n_embd} = {emb_params:,} params ({emb_params*2/(1024*1024):.1f} MB)")

    # Next steps
    print(f"\n  {'='*60}")
    print(f"  NEXT STEPS")
    print(f"  {'='*60}")
    if not os.path.exists(outfile):
        print(f"  1. Download corpus:")
        print(f"     aria2c -x 16 -s 16 -j 1 -d \"{data_dir}\" -o tinystories.txt \"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt\"")
    print(f"  2. Train on {world_size} GPU(s):")
    print(f"     python train_ddp.py")
    print(f"  3. Or single GPU:")
    print(f"     python gpt_train.py")
    print()


if __name__ == "__main__":
    run(draft_path)
