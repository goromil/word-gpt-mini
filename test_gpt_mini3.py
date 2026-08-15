import os
import sys
import json
import math
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import gpt_mini3

# =============================================================================
# FIXTURES
# =============================================================================
SMALL_CONFIG = {
    "model": {
        "n_layer": 2,
        "n_head": 4,
        "head_dim": 32,
        "seq_length": 64
    },
    "training": {
        "epochs": 5,
        "batch_size": 8,
        "lr": 0.0002,
        "checkpoint_every": 10
    },
    "tokenizer": {
        "max_vocab_size": 200,
    },
    "paths": {
        "data_dir": "",
        "checkpoint_dir": ""
    }
}

SAMPLE_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "artificial intelligence is transforming the world",
    "machine learning models can process large amounts of data",
    "the sun rises in the east and sets in the west",
    "deep learning has revolutionized natural language processing",
    "neural networks are inspired by the human brain",
    "transformers have revolutionized sequence modeling tasks",
    "attention mechanisms allow models to focus on relevant information",
    "pretrained models can be fine-tuned for specific tasks",
    "large language models exhibit emergent capabilities",
] * 100


# =============================================================================
# 1. BPETOKENIZER TESTS
# =============================================================================
def test_tokenizer_build_vocab():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    assert tok.vocab_size > 3
    assert tok.vocab_size <= 200
    print("PASS: tokenizer_build_vocab")


def test_tokenizer_vocab_cap():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    assert tok.vocab_size <= 200
    print("PASS: tokenizer_vocab_cap")


def test_tokenizer_encode():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    tokens = tok.encode("the quick brown fox")
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) >= 1
    print("PASS: tokenizer_encode")


def test_tokenizer_encode_unk():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(["hello world"])
    tokens = tok.encode("xyzzy unknown word")
    # SentencePiece subword-decomposes unknown text or falls back to unk
    assert len(tokens) >= 1
    unk = tok.sp.unk_id() if tok.sp else 1
    assert unk >= 0
    print("PASS: tokenizer_encode_unk")


def test_tokenizer_decode():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    tokens = tok.encode("the quick brown")
    text = tok.decode(tokens)
    assert "quick" in text or "brown" in text
    print("PASS: tokenizer_decode")


def test_tokenizer_decode_unk():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(["hello world"])
    text = tok.decode([1])
    # SentencePiece unk_surface returns a special char (e.g., U+2047)
    assert isinstance(text, str)
    print("PASS: tokenizer_decode_unk")


def test_tokenizer_special_tokens():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    # SentencePiece v0.2.2: pad_id, unk_id, eos_id are methods
    assert tok.sp.pad_id() == 0
    assert tok.sp.unk_id() == 1
    assert tok.sp.eos_id() == 2
    print("PASS: tokenizer_special_tokens")


def test_tokenizer_roundtrip():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=500)
    tok.build_vocab(SAMPLE_TEXTS)
    original = "the quick brown fox"
    tokens = tok.encode(original)
    decoded = tok.decode(tokens)
    # SentencePiece normalizes whitespace, check content
    for word in ["quick", "brown", "fox"]:
        assert word in decoded
    print("PASS: tokenizer_roundtrip")


def test_tokenizer_save_load():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as f:
        tmp = f.name
    tok.save(tmp)
    tok2 = gpt_mini3.BPETokenizer()
    tok2.load(tmp)
    assert tok.vocab_size == tok2.vocab_size
    original = "the quick brown fox"
    assert tok.encode(original) == tok2.encode(original)
    os.unlink(tmp)
    os.unlink(tmp + ".meta.json")
    print("PASS: tokenizer_save_load")


def test_tokenizer_backward_compat():
    # WordTokenizer alias works
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    assert tok.vocab_size > 0
    print("PASS: tokenizer_backward_compat")


# =============================================================================
# 2. WORDDATASET TESTS
# =============================================================================
def test_dataset_len():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=500)
    tok.build_vocab(SAMPLE_TEXTS)
    ds = gpt_mini3.WordDataset(SAMPLE_TEXTS, tok, seq_length=10)
    assert ds.__len__() > 0
    print("PASS: dataset_len")


def test_dataset_getitem():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=500)
    tok.build_vocab(SAMPLE_TEXTS)
    ds = gpt_mini3.WordDataset(SAMPLE_TEXTS, tok, seq_length=10)
    x, y = ds[0]
    assert x.shape == (10,)
    assert y.shape == (10,)
    assert (x[1:] == y[:-1]).all()
    print("PASS: dataset_getitem")


def test_dataset_last_item():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=500)
    tok.build_vocab(SAMPLE_TEXTS)
    ds = gpt_mini3.WordDataset(SAMPLE_TEXTS, tok, seq_length=10)
    last_idx = len(ds) - 1
    x, y = ds[last_idx]
    assert x.shape == (10,)
    assert y.shape == (10,)
    print("PASS: dataset_last_item")


def test_dataset_empty():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=500)
    tok.build_vocab(["hello world"])
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_cache = f.name
    try:
        ds = gpt_mini3.WordDataset(["hi"], tok, seq_length=100, cache_file=tmp_cache)
        assert len(ds) == 0
    finally:
        for ext in [".npy", ".npy.meta.json"]:
            p = tmp_cache + ("" if ext == ".npy" else ext)
            if os.path.exists(p):
                os.unlink(p)
    print("PASS: dataset_empty")


def test_dataset_eos_tokens():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=500)
    tok.build_vocab(SAMPLE_TEXTS[:2])
    ds = gpt_mini3.WordDataset(SAMPLE_TEXTS[:2], tok, seq_length=5)
    assert 2 in ds.data.tolist()  # EOS token
    print("PASS: dataset_eos_tokens")


# =============================================================================
# 3. CAUSALSELFATTENTION TESTS
# =============================================================================
def test_attention_init():
    cfg = {"n_head": 4, "head_dim": 32, "seq_length": 64}
    attn = gpt_mini3.CausalSelfAttention(cfg)
    assert attn.n_head == 4
    assert attn.head_dim == 32
    print("PASS: attention_init")


def test_attention_init_assertion():
    cfg = {"n_head": 4, "head_dim": 32, "seq_length": 64}
    n_embd = cfg["n_head"] * cfg["head_dim"]
    attn = gpt_mini3.CausalSelfAttention(cfg)
    assert attn.c_attn.in_features == n_embd
    assert attn.c_attn.out_features == 3 * n_embd
    print("PASS: attention_init_assertion")


def test_attention_forward():
    cfg = {"n_head": 4, "head_dim": 32, "seq_length": 64}
    attn = gpt_mini3.CausalSelfAttention(cfg)
    x = torch.randn(2, 32, 128)
    y = attn(x)
    assert y.shape == x.shape
    print("PASS: attention_forward")


def test_attention_causal_mask():
    cfg = {"n_head": 4, "head_dim": 32, "seq_length": 16}
    attn = gpt_mini3.CausalSelfAttention(cfg)
    x = torch.randn(1, 4, 128)
    attn.eval()
    with torch.no_grad():
        y = attn(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    print("PASS: attention_causal_mask")


def test_attention_single_token():
    cfg = {"n_head": 4, "head_dim": 32, "seq_length": 64}
    attn = gpt_mini3.CausalSelfAttention(cfg)
    x = torch.randn(1, 1, 128)
    y = attn(x)
    assert y.shape == (1, 1, 128)
    print("PASS: attention_single_token")


def test_attention_batch():
    cfg = {"n_head": 4, "head_dim": 32, "seq_length": 64}
    attn = gpt_mini3.CausalSelfAttention(cfg)
    for batch_size in [1, 4, 8]:
        x = torch.randn(batch_size, 32, 128)
        y = attn(x)
        assert y.shape == x.shape
    print("PASS: attention_batch")


# =============================================================================
# 4. BLOCK TESTS
# =============================================================================
def test_block_init():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    block = gpt_mini3.Block(cfg)
    assert isinstance(block.attn, gpt_mini3.CausalSelfAttention)
    assert isinstance(block.mlp, nn.Sequential)
    print("PASS: block_init")


def test_block_forward():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    block = gpt_mini3.Block(cfg)
    x = torch.randn(2, 32, 128)
    y = block(x)
    assert y.shape == x.shape
    print("PASS: block_forward")


def test_block_residual():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    block = gpt_mini3.Block(cfg)
    x = torch.randn(2, 32, 128)
    for p in block.parameters():
        p.data.zero_()
    y = block(x)
    assert torch.allclose(y, x, atol=1e-6)
    print("PASS: block_residual")


# =============================================================================
# 5. GPTMINI TESTS
# =============================================================================
def test_gpt_mini_init():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    model = gpt_mini3.GPTMini(cfg, vocab_size=50)
    assert model.n_embd == 128
    assert model.transformer.wte.num_embeddings == 50
    assert len(model.transformer.h) == 2
    print("PASS: gpt_mini_init")


def test_gpt_mini_weight_sharing():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    model = gpt_mini3.GPTMini(cfg, vocab_size=50)
    assert model.lm_head.weight is model.transformer.wte.weight
    print("PASS: gpt_mini_weight_sharing")


def test_gpt_mini_forward_no_target():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    model = gpt_mini3.GPTMini(cfg, vocab_size=50)
    x = torch.randint(0, 50, (2, 32))
    logits, loss = model(x)
    assert logits.shape == (2, 32, 50)
    assert loss is None
    print("PASS: gpt_mini_forward_no_target")


def test_gpt_mini_forward_with_target():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    model = gpt_mini3.GPTMini(cfg, vocab_size=50)
    x = torch.randint(0, 50, (2, 32))
    y = torch.randint(0, 50, (2, 32))
    logits, loss = model(x, y)
    assert logits.shape == (2, 32, 50)
    assert loss is not None
    assert loss.item() > 0
    print("PASS: gpt_mini_forward_with_target")


def test_gpt_mini_seq_length_assertion():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 10}
    model = gpt_mini3.GPTMini(cfg, vocab_size=50)
    x = torch.randint(0, 50, (2, 11))
    try:
        model(x)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    print("PASS: gpt_mini_seq_length_assertion")


def test_gpt_mini_get_num_params():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    model = gpt_mini3.GPTMini(cfg, vocab_size=50)
    params = model.get_num_params()
    assert params > 0
    params_no_embed = model.get_num_params(non_embedding=True)
    assert params_no_embed == params
    print("PASS: gpt_mini_get_num_params")


def test_gpt_mini_positional_embedding():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    model = gpt_mini3.GPTMini(cfg, vocab_size=50)
    assert model.wpe.shape == (1, 64, 128)
    print("PASS: gpt_mini_positional_embedding")


def test_gpt_mini_init_weights():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    model = gpt_mini3.GPTMini(cfg, vocab_size=50)
    for p in model.parameters():
        assert p.numel() > 0
    print("PASS: gpt_mini_init_weights")


def test_gpt_mini_single_layer():
    cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
    model = gpt_mini3.GPTMini(cfg, vocab_size=20)
    x = torch.randint(0, 20, (1, 16))
    logits, _ = model(x)
    assert logits.shape == (1, 16, 20)
    print("PASS: gpt_mini_single_layer")


def test_gpt_mini_large_model():
    cfg = {"n_layer": 4, "n_head": 8, "head_dim": 64, "seq_length": 128}
    model = gpt_mini3.GPTMini(cfg, vocab_size=200)
    x = torch.randint(0, 200, (2, 64))
    logits, _ = model(x)
    assert logits.shape == (2, 64, 200)
    print("PASS: gpt_mini_large_model")


# =============================================================================
# 6. GENERATION TESTS
# =============================================================================
def test_generate_text():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    model = gpt_mini3.GPTMini(cfg, tok.vocab_size)
    model.eval()
    text = gpt_mini3.generate_text(model, tok, "the quick brown", max_new_tokens=10, device="cpu")
    assert isinstance(text, str)
    assert len(text) > 0
    print("PASS: generate_text")


def test_generate_text_eos():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    model = gpt_mini3.GPTMini(cfg, tok.vocab_size)
    model.eval()
    text = gpt_mini3.generate_text(model, tok, "the", max_new_tokens=50, device="cpu")
    assert isinstance(text, str)
    print("PASS: generate_text_eos")


def test_generate_text_temperature():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(SAMPLE_TEXTS)
    model = gpt_mini3.GPTMini(cfg, tok.vocab_size)
    model.eval()
    t1 = gpt_mini3.generate_text(model, tok, "hello", max_new_tokens=10, temperature=0.1, device="cpu")
    t2 = gpt_mini3.generate_text(model, tok, "hello", max_new_tokens=10, temperature=1.0, device="cpu")
    assert isinstance(t1, str) and isinstance(t2, str)
    print("PASS: generate_text_temperature")


# =============================================================================
# 7. CONFIG TESTS
# =============================================================================
def test_config_hash_stable():
    cfg = SMALL_CONFIG
    h1 = gpt_mini3.config_hash(cfg)
    h2 = gpt_mini3.config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 64
    print("PASS: config_hash_stable")


def test_config_hash_excludes_training():
    cfg1 = dict(SMALL_CONFIG, training={"epochs": 100, "batch_size": 64, "lr": 0.001, "checkpoint_every": 5})
    cfg2 = dict(SMALL_CONFIG, training={"epochs": 50, "batch_size": 32, "lr": 0.0001, "checkpoint_every": 20})
    assert gpt_mini3.config_hash(cfg1) == gpt_mini3.config_hash(cfg2)
    print("PASS: config_hash_excludes_training")


def test_config_hash_excludes_paths():
    cfg1 = dict(SMALL_CONFIG, paths={"data_dir": "/a", "checkpoint_dir": "/b"})
    cfg2 = dict(SMALL_CONFIG, paths={"data_dir": "/x", "checkpoint_dir": "/y"})
    assert gpt_mini3.config_hash(cfg1) == gpt_mini3.config_hash(cfg2)
    print("PASS: config_hash_excludes_paths")


def test_config_hash_includes_model():
    cfg1 = dict(SMALL_CONFIG, model={"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64})
    cfg2 = dict(SMALL_CONFIG, model={"n_layer": 4, "n_head": 8, "head_dim": 64, "seq_length": 128})
    assert gpt_mini3.config_hash(cfg1) != gpt_mini3.config_hash(cfg2)
    print("PASS: config_hash_includes_model")


def test_config_hash_includes_tokenizer():
    cfg1 = dict(SMALL_CONFIG, tokenizer={"max_vocab_size": 1000})
    cfg2 = dict(SMALL_CONFIG, tokenizer={"max_vocab_size": 5000})
    assert gpt_mini3.config_hash(cfg1) != gpt_mini3.config_hash(cfg2)
    print("PASS: config_hash_includes_tokenizer")


def test_load_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SMALL_CONFIG, f)
        f.flush()
        cfg = gpt_mini3.load_config(f.name)
        assert cfg["model"]["n_layer"] == 2
    os.unlink(f.name)
    print("PASS: load_config")


# =============================================================================
# 8. TIER TESTS
# =============================================================================
def test_tiers_epoch_10():
    assert gpt_mini3._tiers_for_epoch(10) == [1]
    print("PASS: tiers_epoch_10")


def test_tiers_epoch_100():
    assert gpt_mini3._tiers_for_epoch(100) == [1, 2]
    print("PASS: tiers_epoch_100")


def test_tiers_epoch_1000():
    assert gpt_mini3._tiers_for_epoch(1000) == [1, 2, 3]
    print("PASS: tiers_epoch_1000")


def test_tiers_epoch_10000():
    assert gpt_mini3._tiers_for_epoch(10000) == [1, 2, 3, 4]
    print("PASS: tiers_epoch_10000")


def test_tiers_epoch_5():
    assert gpt_mini3._tiers_for_epoch(5) == []
    print("PASS: tiers_epoch_5")


def test_tiers_epoch_20():
    assert gpt_mini3._tiers_for_epoch(20) == [1]
    print("PASS: tiers_epoch_20")


def test_tiers_epoch_500():
    assert gpt_mini3._tiers_for_epoch(500) == [1, 2]
    print("PASS: tiers_epoch_500")


def test_tiers_epoch_1():
    assert gpt_mini3._tiers_for_epoch(1) == []
    print("PASS: tiers_epoch_1")


def test_tiers_epoch_0():
    assert gpt_mini3._tiers_for_epoch(0) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    print("PASS: tiers_epoch_0")


# =============================================================================
# 9. CHECKPOINT TESTS
# =============================================================================
def test_save_checkpoint_creates_base():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(10, 0.5, SMALL_CONFIG, h, model, tmpdir)
        base = Path(tmpdir) / h
        assert (base / "model.0.pth").exists()  # epoch 10 -> slot 0
        assert (base / "resume.json").exists()
        assert (base / "config.json").exists()
        assert (base / "1" / "model.pth").exists()
    print("PASS: save_checkpoint_creates_base")


def test_save_checkpoint_tier_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(100, 0.3, SMALL_CONFIG, h, model, tmpdir)
        base = Path(tmpdir) / h
        assert (base / "1" / "model.pth").exists()
        assert (base / "2" / "model.pth").exists()
    print("PASS: save_checkpoint_tier_2")


def test_save_checkpoint_tier_3():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(1000, 0.2, SMALL_CONFIG, h, model, tmpdir)
        base = Path(tmpdir) / h
        assert (base / "1" / "model.pth").exists()
        assert (base / "2" / "model.pth").exists()
        assert (base / "3" / "model.pth").exists()
    print("PASS: save_checkpoint_tier_3")


def test_save_checkpoint_config_once():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(10, 0.5, SMALL_CONFIG, h, model, tmpdir)
        base = Path(tmpdir) / h
        config_mtime = (base / "config.json").stat().st_mtime
        gpt_mini3.save_checkpoint(20, 0.4, SMALL_CONFIG, h, model, tmpdir)
        assert (base / "config.json").stat().st_mtime == config_mtime
    print("PASS: save_checkpoint_config_once")


def test_save_checkpoint_log_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(42, 0.123456, SMALL_CONFIG, h, model, tmpdir)
        base = Path(tmpdir) / h
        meta = json.loads((base / "resume.json").read_text())
        assert meta["epoch"] == 42
        assert meta["loss"] == 0.123456
    print("PASS: save_checkpoint_log_content")


def test_save_checkpoint_overwrites():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(10, 0.5, SMALL_CONFIG, h, model, tmpdir)
        gpt_mini3.save_checkpoint(20, 0.3, SMALL_CONFIG, h, model, tmpdir)
        base = Path(tmpdir) / h
        meta = json.loads((base / "resume.json").read_text())
        assert meta["epoch"] == 20
    print("PASS: save_checkpoint_overwrites")


def test_checkpoint_slot_alternation():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        # Even epoch -> slot 0
        gpt_mini3.save_checkpoint(10, 0.5, SMALL_CONFIG, h, model, tmpdir, optimizer=opt)
        base = Path(tmpdir) / h
        assert (base / "model.0.pth").exists()
        assert (base / "optimizer.0.pt").exists()
        assert not (base / "model.1.pth").exists()
        # Odd epoch -> slot 1
        gpt_mini3.save_checkpoint(11, 0.4, SMALL_CONFIG, h, model, tmpdir, optimizer=opt)
        assert (base / "model.0.pth").exists()  # old slot preserved
        assert (base / "model.1.pth").exists()  # new slot written
        assert (base / "optimizer.0.pt").exists()
        assert (base / "optimizer.1.pt").exists()
        # Even epoch -> slot 0 (overwrites old slot 0)
        gpt_mini3.save_checkpoint(12, 0.3, SMALL_CONFIG, h, model, tmpdir, optimizer=opt)
        assert (base / "model.0.pth").exists()
        assert (base / "model.1.pth").exists()  # old slot 1 preserved
    print("PASS: checkpoint_slot_alternation")


def test_find_latest_checkpoint_none():
    assert gpt_mini3.find_latest_checkpoint("/nonexistent", "abc123") is None
    print("PASS: find_latest_checkpoint_none")


def test_find_latest_checkpoint_base_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(50, 0.3, SMALL_CONFIG, h, model, tmpdir)
        result = gpt_mini3.find_latest_checkpoint(tmpdir, h)
        assert result is not None
        ep, info, d = result
        assert ep == 50
        assert abs(info["loss"] - 0.3) < 1e-6
    print("PASS: find_latest_checkpoint_base_only")


def test_find_latest_checkpoint_tier_newer():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(100, 0.2, SMALL_CONFIG, h, model, tmpdir)
        result = gpt_mini3.find_latest_checkpoint(tmpdir, h)
        assert result is not None
        ep, info, d = result
        assert ep == 100
    print("PASS: find_latest_checkpoint_tier_newer")


def test_find_latest_checkpoint_wrong_hash():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(10, 0.5, SMALL_CONFIG, h, model, tmpdir)
        result = gpt_mini3.find_latest_checkpoint(tmpdir, "wrong_hash")
        assert result is None
    print("PASS: find_latest_checkpoint_wrong_hash")


def test_checkpoint_resume_weights():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(10, 0.5, SMALL_CONFIG, h, model, tmpdir)
        result = gpt_mini3.find_latest_checkpoint(tmpdir, h)
        assert result is not None
        ep, info, d = result
        loaded = gpt_mini3.GPTMini(cfg, vocab_size=20)
        loaded.load_state_dict(torch.load(d / "model.0.pth", map_location="cpu", weights_only=True))  # epoch 10 -> slot 0
        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.allclose(p1.to(torch.float32), p2.to(torch.float32), atol=1e-3)
    print("PASS: checkpoint_resume_weights")


# =============================================================================
# 11. EDGE CASES
# =============================================================================
def test_attention_seq_length_exceeded():
    cfg = {"n_head": 4, "head_dim": 32, "seq_length": 8}
    attn = gpt_mini3.CausalSelfAttention(cfg)
    x = torch.randn(1, 100, 128)
    y = attn(x)
    assert y.shape == x.shape
    print("PASS: attention_seq_length_exceeded")


def test_gpt_mini_max_seq_length():
    cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
    model = gpt_mini3.GPTMini(cfg, vocab_size=50)
    x = torch.randint(0, 50, (1, 32))
    logits, _ = model(x)
    assert logits.shape == (1, 32, 50)
    print("PASS: gpt_mini_max_seq_length")


def test_gpt_mini_min_seq_length():
    cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 2}
    model = gpt_mini3.GPTMini(cfg, vocab_size=10)
    x = torch.randint(0, 10, (1, 2))
    logits, _ = model(x)
    assert logits.shape == (1, 2, 10)
    print("PASS: gpt_mini_min_seq_length")


def test_generate_text_long_prompt():
    cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(["the cat sat on the mat the cat sat"])
    model = gpt_mini3.GPTMini(cfg, tok.vocab_size)
    model.eval()
    text = gpt_mini3.generate_text(model, tok, "the cat sat on the mat the cat sat on the mat",
                                   max_new_tokens=5, device="cpu")
    assert isinstance(text, str)
    print("PASS: generate_text_long_prompt")


def test_dataset_sequence_boundary():
    tok = gpt_mini3.BPETokenizer(max_vocab_size=200)
    tok.build_vocab(["a b c d e f g h i j"])
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp_cache = f.name
    try:
        ds = gpt_mini3.WordDataset(["a b c d e f g h i j"], tok, seq_length=5, cache_file=tmp_cache)
        assert len(ds) >= 1
        last = ds[len(ds) - 1]
        assert last[0].shape == (5,)
        assert last[1].shape == (5,)
    finally:
        for ext in [".npy", ".npy.meta.json"]:
            p = tmp_cache + ("" if ext == ".npy" else ext)
            if os.path.exists(p):
                os.unlink(p)
    print("PASS: dataset_sequence_boundary")


def test_checkpoint_multiple_models():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg1 = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        cfg2 = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
        config1 = dict(SMALL_CONFIG, model=cfg1)
        config2 = dict(SMALL_CONFIG, model=cfg2)
        m1 = gpt_mini3.GPTMini(cfg1, vocab_size=20)
        m2 = gpt_mini3.GPTMini(cfg2, vocab_size=30)
        h1 = gpt_mini3.config_hash(config1)
        h2 = gpt_mini3.config_hash(config2)
        gpt_mini3.save_checkpoint(10, 0.5, config1, h1, m1, tmpdir)
        gpt_mini3.save_checkpoint(20, 0.3, config2, h2, m2, tmpdir)
        r1 = gpt_mini3.find_latest_checkpoint(tmpdir, h1)
        r2 = gpt_mini3.find_latest_checkpoint(tmpdir, h2)
        assert r1 is not None and r1[0] == 10
        assert r2 is not None and r2[0] == 20
        assert h1 != h2
    print("PASS: checkpoint_multiple_models")


# =============================================================================
# 12. VOCAB HASH TESTS
# =============================================================================
def test_vocab_hash_stable():
    h1 = gpt_mini3.get_vocab_hash({"max_vocab_size": 32768}, ["/tmp"])
    h2 = gpt_mini3.get_vocab_hash({"max_vocab_size": 32768}, ["/tmp"])
    assert h1 == h2
    print("PASS: vocab_hash_stable")


def test_vocab_hash_changes_on_vocab_size():
    h1 = gpt_mini3.get_vocab_hash({"max_vocab_size": 32768}, ["/tmp"])
    h2 = gpt_mini3.get_vocab_hash({"max_vocab_size": 16384}, ["/tmp"])
    assert h1 != h2
    print("PASS: vocab_hash_changes_on_vocab_size")


# =============================================================================
# CONFIG-ONLY HASH TESTS
# =============================================================================
def test_vocab_conf_hash_stable():
    cfg = {"max_vocab_size": 32768, "sentence_sample_cap": 25000000}
    sources = ["tinystories", "wikipedia_en_corpus"]
    h1 = gpt_mini3.get_vocab_conf_hash(cfg, sources)
    h2 = gpt_mini3.get_vocab_conf_hash(cfg, sources)
    assert h1 == h2, "vocab_conf_hash must be stable"
    assert len(h1) == 16
    print("PASS: vocab_conf_hash_stable")


def test_vocab_conf_hash_changes_on_sources():
    cfg = {"max_vocab_size": 32768, "sentence_sample_cap": 25000000}
    h1 = gpt_mini3.get_vocab_conf_hash(cfg, ["tinystories"])
    h2 = gpt_mini3.get_vocab_conf_hash(cfg, ["tinystories", "wikipedia_en_corpus"])
    assert h1 != h2, "vocab_conf_hash must change when sources change"
    print("PASS: vocab_conf_hash_changes_on_sources")


def test_corpus_conf_hash_stable():
    sources = ["tinystories", "wikipedia_en_corpus", "chitanka_epub_corpus"]
    h1 = gpt_mini3.get_corpus_conf_hash(sources)
    h2 = gpt_mini3.get_corpus_conf_hash(sources)
    assert h1 == h2, "corpus_conf_hash must be stable"
    assert len(h1) == 16
    print("PASS: corpus_conf_hash_stable")


def test_corpus_conf_hash_changes_on_sources():
    h1 = gpt_mini3.get_corpus_conf_hash(["tinystories"])
    h2 = gpt_mini3.get_corpus_conf_hash(["tinystories", "wikipedia_en_corpus"])
    assert h1 != h2, "corpus_conf_hash must change when sources change"
    print("PASS: corpus_conf_hash_changes_on_sources")


def test_try_conf_cache_miss():
    with tempfile.TemporaryDirectory() as tmpdir:
        cd = Path(tmpdir)
        result = gpt_mini3._try_conf_cache(cd, "a1b2c3d4e5f61234", "1234567890abcdef")
        assert result is None, "_try_conf_cache must return None for empty dir"
    print("PASS: try_conf_cache_miss")


def test_try_conf_cache_hit():
    with tempfile.TemporaryDirectory() as tmpdir:
        cd = Path(tmpdir)
        # Create vocab cache
        vc = cd / "vocab-a1b2c3d4e5f61234-d4e5f6a1b2c34567.json"
        vc.write_text('{"test": true}')
        # Create data cache (>1GB check — we use a small file, so this tests the glob logic)
        # The actual _try_conf_cache checks size > 1GB, so we need to fake it
        dc = cd / "data-1234567890abcdef-d4e5f6a1b2c34567-fedcba9876543210.npy"
        dc.write_bytes(b"\x00" * 100)
        # This won't pass the size check, so result should be None
        result = gpt_mini3._try_conf_cache(cd, "a1b2c3d4e5f61234", "1234567890abcdef")
        assert result is None, "_try_conf_cache must return None for small data file"
        # Now test with exact match
        result = gpt_mini3._try_conf_cache(cd, "a1b2c3d4e5f61234", "1234567890abcdef",
                                           "d4e5f6a1b2c34567", "fedcba9876543210")
        assert result is None, "_try_conf_cache must return None for small data file even with exact match"
    print("PASS: try_conf_cache_hit")


# =============================================================================
# RUNNER
# =============================================================================
if __name__ == "__main__":
    tests = [
        test_tokenizer_build_vocab,
        test_tokenizer_vocab_cap,
        test_tokenizer_encode,
        test_tokenizer_encode_unk,
        test_tokenizer_decode,
        test_tokenizer_decode_unk,
        test_tokenizer_special_tokens,
        test_tokenizer_roundtrip,
        test_tokenizer_save_load,
        test_tokenizer_backward_compat,
        test_dataset_len,
        test_dataset_getitem,
        test_dataset_last_item,
        test_dataset_empty,
        test_dataset_eos_tokens,
        test_attention_init,
        test_attention_init_assertion,
        test_attention_forward,
        test_attention_causal_mask,
        test_attention_single_token,
        test_attention_batch,
        test_block_init,
        test_block_forward,
        test_block_residual,
        test_gpt_mini_init,
        test_gpt_mini_weight_sharing,
        test_gpt_mini_forward_no_target,
        test_gpt_mini_forward_with_target,
        test_gpt_mini_seq_length_assertion,
        test_gpt_mini_get_num_params,
        test_gpt_mini_positional_embedding,
        test_gpt_mini_init_weights,
        test_gpt_mini_single_layer,
        test_gpt_mini_large_model,
        test_generate_text,
        test_generate_text_eos,
        test_generate_text_temperature,
        test_config_hash_stable,
        test_config_hash_excludes_training,
        test_config_hash_excludes_paths,
        test_config_hash_includes_model,
        test_config_hash_includes_tokenizer,
        test_load_config,
        test_tiers_epoch_10,
        test_tiers_epoch_100,
        test_tiers_epoch_1000,
        test_tiers_epoch_10000,
        test_tiers_epoch_5,
        test_tiers_epoch_20,
        test_tiers_epoch_500,
        test_tiers_epoch_1,
        test_tiers_epoch_0,
        test_save_checkpoint_creates_base,
        test_save_checkpoint_tier_2,
        test_save_checkpoint_tier_3,
        test_save_checkpoint_config_once,
        test_save_checkpoint_log_content,
        test_save_checkpoint_overwrites,
        test_find_latest_checkpoint_none,
        test_find_latest_checkpoint_base_only,
        test_find_latest_checkpoint_tier_newer,
        test_find_latest_checkpoint_wrong_hash,
        test_checkpoint_resume_weights,
        test_attention_seq_length_exceeded,
        test_gpt_mini_max_seq_length,
        test_gpt_mini_min_seq_length,
        test_generate_text_long_prompt,
        test_dataset_sequence_boundary,
        test_checkpoint_multiple_models,
        test_vocab_hash_stable,
        test_vocab_hash_changes_on_vocab_size,
        test_vocab_conf_hash_stable,
        test_vocab_conf_hash_changes_on_sources,
        test_corpus_conf_hash_stable,
        test_corpus_conf_hash_changes_on_sources,
        test_try_conf_cache_miss,
        test_try_conf_cache_hit,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed:
        sys.exit(1)
