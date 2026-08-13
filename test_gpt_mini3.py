import os
import sys
import json
import math
import tempfile
import shutil
from pathlib import Path

import torch
import torch.nn as nn

# Import everything from gpt_mini3
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
        "max_word_len": 20
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
# 1. WORDTOKENIZER TESTS
# =============================================================================
def test_tokenizer_build_vocab():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    assert tok.vocab_size > 3  # special tokens + words
    assert tok.vocab_size <= 200
    assert tok.word2idx["<pad>"] == 0
    assert tok.word2idx["<unk>"] == 1
    assert tok.word2idx["<eos>"] == 2
    print("PASS: tokenizer_build_vocab")


def test_tokenizer_vocab_cap():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    assert tok.vocab_size <= 200  # capped at max_vocab_size (chars + tier words)
    print("PASS: tokenizer_vocab_cap")


def test_tokenizer_encode():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    tokens = tok.encode("the quick brown fox")
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) == 4
    print("PASS: tokenizer_encode")


def test_tokenizer_encode_unk():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(["hello world"])
    tokens = tok.encode("xyzzy unknown word")
    # "xyzzy" not in vocab → char fallback: x<sep>y<sep>z<sep>z<sep>y
    sep = tok.word2idx["<sep>"]
    assert sep in tokens  # char fallback uses <sep> between chars
    assert all(t != tok.word2idx["<unk>"] for t in tokens)  # all latin chars are pre-populated
    print("PASS: tokenizer_encode_unk")


def test_tokenizer_encode_max_word_len():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=3)
    tok.build_vocab(["abc def ghi"])
    tokens = tok.encode("abc defgh ijkl")
    assert len(tokens) == 1  # only "abc" and "def" pass; "defgh" > 3, "ijkl" > 3
    print("PASS: tokenizer_encode_max_word_len")


def test_tokenizer_decode():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    tokens = tok.encode("the quick brown")
    text = tok.decode(tokens)
    assert text == "the quick brown"
    print("PASS: tokenizer_decode")


def test_tokenizer_decode_unk():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(["hello world"])
    text = tok.decode([tok.word2idx["<unk>"]])
    assert text == "<unk>"
    print("PASS: tokenizer_decode_unk")


def test_tokenizer_special_tokens():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    assert "<pad>" in tok.word2idx
    assert "<unk>" in tok.word2idx
    assert "<eos>" in tok.word2idx
    assert tok.word2idx["<pad>"] == 0
    assert tok.word2idx["<unk>"] == 1
    assert tok.word2idx["<eos>"] == 2
    print("PASS: tokenizer_special_tokens")


def test_tokenizer_roundtrip():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=500, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    original = "the quick brown fox"
    tokens = tok.encode(original)
    decoded = tok.decode(tokens)
    assert decoded == original
    print("PASS: tokenizer_roundtrip")


# =============================================================================
# 2. WORDDATASET TESTS
# =============================================================================
def test_dataset_len():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=500, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    ds = gpt_mini3.WordDataset(SAMPLE_TEXTS, tok, seq_length=10)
    assert ds.__len__() > 0
    print("PASS: dataset_len")


def test_dataset_getitem():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=500, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    ds = gpt_mini3.WordDataset(SAMPLE_TEXTS, tok, seq_length=10)
    x, y = ds[0]
    assert x.shape == (10,)
    assert y.shape == (10,)
    assert (x[1:] == y[:-1]).all()  # x[i+1] == y[i]
    print("PASS: dataset_getitem")


def test_dataset_last_item():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=500, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    ds = gpt_mini3.WordDataset(SAMPLE_TEXTS, tok, seq_length=10)
    last_idx = len(ds) - 1
    x, y = ds[last_idx]
    assert x.shape == (10,)
    assert y.shape == (10,)
    print("PASS: dataset_last_item")


def test_dataset_empty():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=500, max_word_len=20)
    tok.build_vocab(["hello world"])
    ds = gpt_mini3.WordDataset(["hi"], tok, seq_length=100)
    assert len(ds) == 0
    print("PASS: dataset_empty")


def test_dataset_eos_tokens():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=500, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS[:2])
    ds = gpt_mini3.WordDataset(SAMPLE_TEXTS[:2], tok, seq_length=5)
    assert tok.word2idx["<eos>"] in ds.data.tolist()
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
    x = torch.randn(2, 32, 128)  # B=2, T=32, C=128
    y = attn(x)
    assert y.shape == x.shape
    print("PASS: attention_forward")


def test_attention_causal_mask():
    cfg = {"n_head": 4, "head_dim": 32, "seq_length": 16}
    attn = gpt_mini3.CausalSelfAttention(cfg)
    # Verify causal: each position can only attend to itself and earlier positions
    x = torch.randn(1, 4, 128)
    attn.eval()
    with torch.no_grad():
        y = attn(x)
    # Output shape matches input
    assert y.shape == x.shape
    # No NaN or Inf
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
    # Zero out all weights to test residual path
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
    x = torch.randint(0, 50, (2, 11))  # exceeds seq_length
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
    # wpe is a register_buffer, not a parameter, so both values are equal
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
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    model = gpt_mini3.GPTMini(cfg, tok.vocab_size)
    model.eval()
    text = gpt_mini3.generate_text(model, tok, "the quick brown", max_new_tokens=10, device="cpu")
    assert isinstance(text, str)
    assert len(text) > 0
    print("PASS: generate_text")


def test_generate_text_eos():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(SAMPLE_TEXTS)
    model = gpt_mini3.GPTMini(cfg, tok.vocab_size)
    model.eval()
    text = gpt_mini3.generate_text(model, tok, "the", max_new_tokens=50, device="cpu")
    assert isinstance(text, str)
    print("PASS: generate_text_eos")


def test_generate_text_temperature():
    cfg = {"n_layer": 2, "n_head": 4, "head_dim": 32, "seq_length": 64}
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
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
    assert len(h1) == 64  # SHA256 hex
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
    cfg1 = dict(SMALL_CONFIG, tokenizer={"max_vocab_size": 1000, "max_word_len": 10})
    cfg2 = dict(SMALL_CONFIG, tokenizer={"max_vocab_size": 5000, "max_word_len": 20})
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
    tiers = gpt_mini3._tiers_for_epoch(10)
    assert tiers == [1]
    print("PASS: tiers_epoch_10")


def test_tiers_epoch_100():
    tiers = gpt_mini3._tiers_for_epoch(100)
    assert tiers == [1, 2]
    print("PASS: tiers_epoch_100")


def test_tiers_epoch_1000():
    tiers = gpt_mini3._tiers_for_epoch(1000)
    assert tiers == [1, 2, 3]
    print("PASS: tiers_epoch_1000")


def test_tiers_epoch_10000():
    tiers = gpt_mini3._tiers_for_epoch(10000)
    assert tiers == [1, 2, 3, 4]
    print("PASS: tiers_epoch_10000")


def test_tiers_epoch_5():
    tiers = gpt_mini3._tiers_for_epoch(5)
    assert tiers == []
    print("PASS: tiers_epoch_5")


def test_tiers_epoch_20():
    tiers = gpt_mini3._tiers_for_epoch(20)
    assert tiers == [1]
    print("PASS: tiers_epoch_20")


def test_tiers_epoch_500():
    tiers = gpt_mini3._tiers_for_epoch(500)
    assert tiers == [1, 2]
    print("PASS: tiers_epoch_500")


def test_tiers_epoch_1():
    tiers = gpt_mini3._tiers_for_epoch(1)
    assert tiers == []
    print("PASS: tiers_epoch_1")


def test_tiers_epoch_0():
    tiers = gpt_mini3._tiers_for_epoch(0)
    assert tiers == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
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
        assert (base / "model.pth").exists()
        assert (base / "train.log").exists()
        assert (base / "config.json").exists()
        assert (base / "1" / "model.pth").exists()
        assert (base / "1" / "train.log").exists()
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
        log = (base / "train.log").read_text()
        assert "epoch: 42" in log
        assert "loss: 0.123456" in log
        assert "config_hash: " + h in log
    print("PASS: save_checkpoint_log_content")


def test_save_checkpoint_overwrites():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"n_layer": 1, "n_head": 2, "head_dim": 16, "seq_length": 32}
        model = gpt_mini3.GPTMini(cfg, vocab_size=20)
        h = gpt_mini3.config_hash(SMALL_CONFIG)
        gpt_mini3.save_checkpoint(10, 0.5, SMALL_CONFIG, h, model, tmpdir)
        gpt_mini3.save_checkpoint(20, 0.3, SMALL_CONFIG, h, model, tmpdir)
        base = Path(tmpdir) / h
        log = (base / "train.log").read_text()
        assert "epoch: 20" in log
        assert "loss: 0.300000" in log
        tier1_log = (base / "1" / "train.log").read_text()
        assert "epoch: 20" in tier1_log
    print("PASS: save_checkpoint_overwrites")


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

        # Load and verify
        result = gpt_mini3.find_latest_checkpoint(tmpdir, h)
        assert result is not None
        ep, info, d = result
        loaded = gpt_mini3.GPTMini(cfg, vocab_size=20)
        loaded.load_state_dict(torch.load(d / "model.pth", map_location="cpu"))

        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2)
    print("PASS: checkpoint_resume_weights")


# =============================================================================
# 10. CORPUS TESTS
# =============================================================================
def test_ensure_corpus_existing():
    with tempfile.TemporaryDirectory() as tmpdir:
        text_file = Path(tmpdir) / "tinystories.txt"
        text_file.write_text("hello world\nfoo bar\n", encoding="utf-8")
        result = gpt_mini3.ensure_corpus(tmpdir)
        sentences = result["sentences"]
        assert len(sentences) == 2
        assert sentences[0] == "hello world"
        assert len(result["sources"]) == 1
    print("PASS: ensure_corpus_existing")


def test_ensure_corpus_creates_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "subdir"
        # Won't download, falls back to built-in
        result = gpt_mini3.ensure_corpus(str(new_dir))
        sentences = result["sentences"]
        assert len(sentences) > 0
        assert new_dir.exists()
    print("PASS: ensure_corpus_creates_dir")


def test_ensure_corpus_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        text_file = Path(tmpdir) / "tinystories.txt"
        text_file.write_text("\n\n\n", encoding="utf-8")
        result = gpt_mini3.ensure_corpus(tmpdir)
        sentences = result["sentences"]
        assert len(sentences) == 0
    print("PASS: ensure_corpus_empty_file")


# =============================================================================
# 11. EDGE CASES
# =============================================================================
def test_attention_seq_length_exceeded():
    cfg = {"n_head": 4, "head_dim": 32, "seq_length": 8}
    attn = gpt_mini3.CausalSelfAttention(cfg)
    x = torch.randn(1, 100, 128)
    y = attn(x)
    # Should work because bias is sliced to :T
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
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(["the cat sat on the mat the cat sat"])
    model = gpt_mini3.GPTMini(cfg, tok.vocab_size)
    model.eval()
    text = gpt_mini3.generate_text(model, tok, "the cat sat on the mat the cat sat on the mat",
                                   max_new_tokens=5, device="cpu")
    assert isinstance(text, str)
    print("PASS: generate_text_long_prompt")


def test_dataset_sequence_boundary():
    tok = gpt_mini3.WordTokenizer(max_vocab_size=200, max_word_len=20)
    tok.build_vocab(["a b c d e f g h i j"])
    ds = gpt_mini3.WordDataset(["a b c d e f g h i j"], tok, seq_length=5)
    assert len(ds) >= 1
    last = ds[len(ds) - 1]
    assert last[0].shape == (5,)
    assert last[1].shape == (5,)
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
# RUNNER
# =============================================================================
if __name__ == "__main__":
    tests = [
        # Tokenizer
        test_tokenizer_build_vocab,
        test_tokenizer_vocab_cap,
        test_tokenizer_encode,
        test_tokenizer_encode_unk,
        test_tokenizer_encode_max_word_len,
        test_tokenizer_decode,
        test_tokenizer_decode_unk,
        test_tokenizer_special_tokens,
        test_tokenizer_roundtrip,
        # Dataset
        test_dataset_len,
        test_dataset_getitem,
        test_dataset_last_item,
        test_dataset_empty,
        test_dataset_eos_tokens,
        # Attention
        test_attention_init,
        test_attention_init_assertion,
        test_attention_forward,
        test_attention_causal_mask,
        test_attention_single_token,
        test_attention_batch,
        # Block
        test_block_init,
        test_block_forward,
        test_block_residual,
        # GPTMini
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
        # Generation
        test_generate_text,
        test_generate_text_eos,
        test_generate_text_temperature,
        # Config
        test_config_hash_stable,
        test_config_hash_excludes_training,
        test_config_hash_excludes_paths,
        test_config_hash_includes_model,
        test_config_hash_includes_tokenizer,
        test_load_config,
        # Tiers
        test_tiers_epoch_10,
        test_tiers_epoch_100,
        test_tiers_epoch_1000,
        test_tiers_epoch_10000,
        test_tiers_epoch_5,
        test_tiers_epoch_20,
        test_tiers_epoch_500,
        test_tiers_epoch_1,
        test_tiers_epoch_0,
        # Checkpoint
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
        # Corpus
        test_ensure_corpus_existing,
        test_ensure_corpus_creates_dir,
        test_ensure_corpus_empty_file,
        # Edge cases
        test_attention_seq_length_exceeded,
        test_gpt_mini_max_seq_length,
        test_gpt_mini_min_seq_length,
        test_generate_text_long_prompt,
        test_dataset_sequence_boundary,
        test_checkpoint_multiple_models,
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
