"""Demo 2: IPv6 socket — parent listens on [::1], child connects and exchanges messages."""
import os, sys, time, socket, subprocess, tempfile


def run():
    python = sys.executable

    # Parent: bind to random IPv6 port on localhost
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("::1", 0))
        port = srv.getsockname()[1]
        srv.listen(1)
        print(f"  [parent] listening on [::1]:{port}", flush=True)

        # Create child script inline with the discovered port
        child_path = tempfile.NamedTemporaryFile(
            mode="w", suffix="_child.py", delete=False
        )
        child_path.write(f'''
import os, socket, time
time.sleep(0.3)
s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
s.connect(("::1", {port}))
data = s.recv(256).decode()
pid = os.getpid()
print(f"  [child  PID {{pid:>5}}] recv: {{data}}", flush=True)
reply = f"hello from child (PID {{pid}})"
s.sendall(reply.encode())
print(f"  [child  PID {{pid:>5}}] send: {{reply}}", flush=True)
s.close()
''')
        child_path.close()

        child = subprocess.Popen([python, child_path.name])
        conn, addr = srv.accept()
        pid = os.getpid()
        msg = f"hello from parent (PID {pid})"
        print(f"  [parent PID {pid:>5}] send: {msg}", flush=True)
        conn.sendall(msg.encode())

        reply = conn.recv(256).decode()
        print(f"  [parent PID {pid:>5}] recv: {reply}", flush=True)
        conn.close()

        child.wait()
        print(f"  child exited (code={child.returncode}).\n", flush=True)

    os.unlink(child_path.name)


if __name__ == "__main__":
    print("=== IPv6 Socket Demo ===", flush=True)
    run()
