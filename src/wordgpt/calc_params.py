import json, math, sys

def calc_params(config_path="gpt_train.json"):
    with open(config_path) as f:
        cfg = json.load(f)

    m = cfg["model"]
    t = cfg["tokenizer"]

    n_layer = m["n_layer"]
    n_head  = m["n_head"]
    head_dim = m["head_dim"]
    seq_length = m["seq_length"]
    nvocab = t["max_vocab_size"]

    n_embd = n_head * head_dim

    wte = nvocab * n_embd
    wpe = seq_length * n_embd
    c_attn = n_embd * (3*n_embd) + 3*n_embd
    c_proj = (3*n_embd) * n_embd + n_embd
    ln12 = 4 * n_embd
    mlp = (n_embd*(4*n_embd)+4*n_embd) + ((4*n_embd)*n_embd+n_embd)
    per_layer = c_attn + c_proj + ln12 + mlp
    ln_f = 2 * n_embd
    lm_head = 0
    total = wte + wpe + per_layer * n_layer + ln_f + lm_head

    print(f"{'='*55}")
    print(f"  Parameter Calculator")
    print(f"  Config: {config_path}")
    print(f"{'='*55}")
    print(f"  nvocab      = {nvocab}")
    print(f"  n_layer     = {n_layer}")
    print(f"  n_head      = {n_head}")
    print(f"  head_dim    = {head_dim}")
    print(f"  n_embd      = {n_head} x {head_dim} = {n_embd}")
    print(f"  seq_length  = {seq_length}")
    print(f"{'='*55}")

    print(f"")
    print(f"  {'Component':<30} {'Params':>14}")
    print(f"  {'-'*44}")
    print(f"  {'wte (embedding)':<30} {wte:>14,}")
    print(f"  {'wpe (positional)':<30} {wpe:>14,}")
    print(f"  {'-'*44}")
    print(f"  {'Per transformer layer:'}")
    print(f"  {'  c_attn (QKV)':<32} {c_attn:>13,}")
    print(f"  {'  c_proj (out)':<32} {c_proj:>13,}")
    print(f"  {'  ln_1 + ln_2':<32} {ln12:>13,}")
    print(f"  {'  mlp (fc1+fc2)':<32} {mlp:>13,}")
    print(f"  {'  = sub-total':<32} {per_layer:>13,}")
    print(f"  {'x' + str(n_layer) + ' layers':<30} {per_layer*n_layer:>14,}")
    print(f"  {'-'*44}")
    print(f"  {'ln_f (final norm)':<30} {ln_f:>14,}")
    print(f"  {'lm_head (tied, +0)':<30} {lm_head:>14,}")
    print(f"  {'='*44}")
    print(f"  {'TOTAL':<30} {total:>14,}")
    print(f"")
    print(f"  ~{total/1e6:.2f}M parameters")
    print(f"")

    # Epoch formula
    print(f"  {'='*55}")
    print(f"  EPOCH FORMULA")
    print(f"  epochs = ceil( (params x N) / corpus_tokens )")
    print(f"  N = Chinchilla multiplier (20, 50, 100, 200)")
    print(f"  {'='*55}")
    print(f"")

    # Show for common N values
    example_corpora = {
        "40M tokens":   40_000_000,
        "250M tokens":  250_000_000,
        "2.5B tokens":  2_500_000_000,
    }
    for N in [20, 50, 100, 200]:
        for label, tokens in example_corpora.items():
            target = total * N
            epochs = math.ceil(target / tokens)
            print(f"  N={N:>3} | {label:>12} -> {epochs:>4} epochs")
        print(f"")

    return total

if __name__ == "__main__":
    cli()


def cli():
    from wordgpt.config import get_default_config_path
    default_cfg = str(get_default_config_path())
    path = sys.argv[1] if len(sys.argv) > 1 else default_cfg
    calc_params(path)
