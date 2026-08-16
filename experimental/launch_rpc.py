"""
Launcher for train_rpc.py — starts N independent processes (no mp.spawn).

Each process gets a fresh CUDA context, avoiding the Windows
0xC0000005 crash from mp.spawn context inheritance.

Usage:
    python launch_rpc.py -g 0,1                    # GPUs 0 and 1
    python launch_rpc.py -g 0,1,2,3                # GPUs 0-3
    python launch_rpc.py -g 0,1 --port 29500        # custom port
    python launch_rpc.py -g 0,1 --epochs 20         # override epochs
    python launch_rpc.py -g 0,1 gpt_train.json      # custom config
"""
import subprocess
import sys
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Launch DDP trainer on multiple GPUs (independent processes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-g", "--gpus", type=str, required=True,
                        help="Comma-separated GPU indices (e.g. 0,1)")
    parser.add_argument("--port", type=str, default="29500",
                        help="TCP port for gradient sync (rank 0 listens, rank 1 connects)")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Override epochs from config")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Override checkpoint_every from config")
    parser.add_argument("config", nargs="?", default="gpt_train.json",
                        help="Path to config JSON")
    args = parser.parse_args()

    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    world_size = len(gpus)

    print(f"Launching {world_size} DDP workers (independent processes):")
    for rank, gpu in enumerate(gpus):
        print(f"  Rank {rank} -> GPU {gpu}")
    print()

    # Resolve train_rpc.py relative to this launcher's directory
    script_dir = Path(__file__).parent
    train_rpc = script_dir / "train_rpc.py"

    # Launch all processes — stagger to let rank 0 set up TCP store first
    processes = []
    for rank, gpu in enumerate(gpus):
        cmd = [
            sys.executable, str(train_rpc),
            "--rank", str(rank),
            "--world_size", str(world_size),
            "--device", str(gpu),
            "--port", args.port,
        ]
        if args.epochs > 0:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.save_every > 0:
            cmd.extend(["--save_every", str(args.save_every)])
        cmd.append(args.config)

        proc = subprocess.Popen(cmd)
        processes.append((rank, gpu, proc))
        print(f"  Started rank {rank} (PID {proc.pid}) -> GPU {gpu}")

        if rank < world_size - 1:
            # Stagger: rank 0 needs time to set up the TCP store
            time.sleep(3)

    print(f"\nAll {world_size} processes launched. Waiting for completion...")
    print("Press Ctrl+C to kill all workers.\n")

    try:
        # Wait for all to finish
        for rank, gpu, proc in processes:
            code = proc.wait()
            print(f"  Rank {rank} (GPU {gpu}, PID {proc.pid}) exited with code {code}")
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Killing all workers...")
        for rank, gpu, proc in processes:
            print(f"  Killing rank {rank} (PID {proc.pid})...")
            proc.kill()
        for _, _, proc in processes:
            proc.wait()
        print("All workers terminated.")

    print("\nDone.")


if __name__ == "__main__":
    main()
