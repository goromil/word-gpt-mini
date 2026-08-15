import json
import torch
from gpt_mini3 import GPTMini, WordTokenizer, WordDataset

# Load config
with open("gpt_mini3.json", "r") as f:
    config = json.load(f)

model_cfg = dict(config.get("model", {}))
vocab_cfg = config.get("tokenizer", config.get("vocab", {}))
train_cfg = config.get("training", {})

# Tokenizer
tokenizer = WordTokenizer(max_vocab_size=vocab_cfg.get("max_vocab_size", 32768), max_word_len=vocab_cfg.get("max_word_len", 20))
vocab_cache = "E:\\training\\cache\\vocab-291b73fcb9d2d904.json"
tokenizer.load(vocab_cache)

# Dataset
dataset = WordDataset([], tokenizer, model_cfg["seq_length"])
import numpy as np
data = np.load("E:\\training\\cache\\data-291b73fcb9d2d904-4f2b1acc60576d3d.npy")
dataset.data = data

# Model
device = torch.device("cuda:0")
model = GPTMini(model_cfg, tokenizer.vocab_size).to(device)
model.train()

# One batch
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=24, shuffle=True, drop_last=True)
x, y = next(iter(dataloader))
x, y = x.to(device), y.to(device)

logits, loss = model(x, y)
loss.backward()

print(f"Batch test - Loss: {loss.item():.4f}")
print(f"Logits shape: {logits.shape}")
print("SUCCESS: One training step completed")
