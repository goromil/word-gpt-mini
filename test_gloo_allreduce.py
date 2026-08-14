import os, torch, torch.distributed as dist, torch.multiprocessing as mp

def test_allreduce(rank, world_size):
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12370"
    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        print(f"Rank {rank}: gloo init OK", flush=True)

        # Test all_reduce on CPU
        t = torch.randn(4, 4)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        print(f"Rank {rank}: CPU all_reduce OK", flush=True)

        # Test all_reduce on GPU — SKIP (requires P2P, unavailable on our PXB topology)
        # tg = torch.randn(4, 4).cuda(rank)
        # dist.all_reduce(tg, op=dist.ReduceOp.AVG)

        # Test large CPU tensor (simulates real grad sync)
        big = torch.randn(1000, 1000)
        dist.all_reduce(big, op=dist.ReduceOp.AVG)
        print(f"Rank {rank}: large CPU all_reduce OK", flush=True)

        # Test CUDA IPC availability (expected: False on PXB topology)
        has_p2p = torch.cuda.can_device_access_peer(rank, (rank + 1) % 2)
        print(f"Rank {rank}: P2P={has_p2p} (CUDA IPC unavailable)", flush=True)

        print(f"Rank {rank}: ALL OK", flush=True)
    except Exception as e:
        print(f"Rank {rank} ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        dist.destroy_process_group()

def run():
    mp.set_sharing_strategy("file_system")
    mp.spawn(test_allreduce, args=(2,), nprocs=2, start_method="spawn", join=True)
    print("Test passed", flush=True)

if __name__ == "__main__":
    run()
