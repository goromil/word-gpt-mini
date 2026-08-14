import os, sys, torch, json
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(__file__))
from gpt_mini3 import GPTMini, WordTokenizer

def test_ddp(rank, world_size):
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12369"

    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        print(f"Rank {rank}: gloo init OK", flush=True)

        # Test with a tiny model first
        print(f"Rank {rank}: testing tiny model...", flush=True)
        tiny = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)).cuda(rank)
        tiny_ddp = DDP(tiny, device_ids=[rank])
        tiny_ddp.train()
        for i in range(3):
            x = torch.randn(4, 10).cuda(rank)
            y = tiny_ddp(x).sum()
            y.backward()
            tiny_ddp.zero_grad()
        print(f"Rank {rank}: tiny model OK", flush=True)
        del tiny_ddp

        # Now test GPTMini
        with open("gpt_mini3.json") as f:
            cfg = json.load(f)
        model_cfg = dict(cfg["model"])
        vocab_cfg = model_cfg.pop("tokenizer", {})

        tok = WordTokenizer()
        tok.load(f"E:\\training\\cache\\vocab-b216b5d27286a3c1.json")

        print(f"Rank {rank}: building GPTMini...", flush=True)
        model = GPTMini(model_cfg, tok.vocab_size).cuda(rank)
        model = DDP(model, device_ids=[rank])
        model.train()
        print(f"Rank {rank}: DDP wrap OK, params={sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)

        opt = torch.optim.Adam(model.parameters(), lr=0.0002)

        # Run a few batches
        for i in range(5):
            x = torch.randint(0, tok.vocab_size, (4, 64)).cuda(rank)
            y = torch.randint(0, tok.vocab_size, (4, 64)).cuda(rank)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
                _, loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if rank == 0:
                print(f"Rank {rank}: batch {i} OK, loss={loss.item():.4f}", flush=True)

        print(f"Rank {rank}: ALL OK", flush=True)
    except Exception as e:
        print(f"Rank {rank} ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        dist.destroy_process_group()

def run():
    mp.set_sharing_strategy("file_system")
    mp.spawn(test_ddp, args=(2,), nprocs=2, start_method="spawn", join=True)
    print("DDP test passed", flush=True)

if __name__ == "__main__":
    run()
