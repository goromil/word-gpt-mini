"""Demo 1: Named pipes (multiprocessing.Pipe) — bidirectional, cross-platform."""
import os, time
from multiprocessing import Process, Pipe


def child_func(conn):
    msg = conn.recv()
    pid = os.getpid()
    print(f"  [child  PID {pid:>5}] recv: {msg}", flush=True)
    reply = f"hello from child (PID {pid})"
    print(f"  [child  PID {pid:>5}] send: {reply}", flush=True)
    conn.send(reply)
    conn.close()


def run():
    parent_conn, child_conn = Pipe()
    child = Process(target=child_func, args=(child_conn,))
    child.start()

    pid = os.getpid()
    msg = f"hello from parent (PID {pid})"
    print(f"  [parent PID {pid:>5}] send: {msg}", flush=True)
    parent_conn.send(msg)

    reply = parent_conn.recv()
    print(f"  [parent PID {pid:>5}] recv: {reply}", flush=True)
    child.join()
    print(f"  child exited (code={child.exitcode}).\n", flush=True)


if __name__ == "__main__":
    print("=== Named Pipe (multiprocessing.Pipe) ===", flush=True)
    run()
