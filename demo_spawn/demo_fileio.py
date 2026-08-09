"""Demo 3: File I/O — parent writes to temp file, child reads; child writes reply, parent reads."""
import os, sys, time, subprocess, tempfile


def run():
    python = sys.executable

    # Create temp files for communication
    input_fd, input_path = tempfile.mkstemp(suffix=".in")
    os.close(input_fd)
    os.unlink(input_path)  # child waits for this to appear

    output_fd, output_path = tempfile.mkstemp(suffix=".out")
    os.close(output_fd)
    os.unlink(output_path)

    # Create child script
    child_path = tempfile.NamedTemporaryFile(
        mode="w", suffix="_child.py", delete=False
    )
    child_path.write(f'''
import os, time, sys
input_path = r"{input_path}"
output_path = r"{output_path}"
while not os.path.exists(input_path):
    time.sleep(0.05)
with open(input_path) as f:
    msg = f.read().strip()
pid = os.getpid()
print(f"  [child  PID {{pid:>5}}] recv: {{msg}}", flush=True)
reply = f"hello from child (PID {{pid}})"
print(f"  [child  PID {{pid:>5}}] send: {{reply}}", flush=True)
with open(output_path, "w") as f:
    f.write(reply)
''')
    child_path.close()

    child = subprocess.Popen([python, child_path.name],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    time.sleep(0.3)  # let child start polling
    pid = os.getpid()
    msg = f"hello from parent (PID {pid})"
    print(f"  [parent PID {pid:>5}] send: {msg}", flush=True)
    with open(input_path, "w") as f:
        f.write(msg)

    # Wait for child reply
    deadline = time.time() + 5
    while time.time() < deadline:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            break
        time.sleep(0.05)

    with open(output_path) as f:
        reply = f.read().strip()
    print(f"  [parent PID {pid:>5}] recv: {reply}", flush=True)

    child.wait()
    # Print child stdout
    out = child.stdout.read().strip() if child.stdout else ""
    if out:
        for line in out.split("\n"):
            print(line)
    print(f"  child exited (code={child.returncode}).\n", flush=True)

    for p in (child_path.name, input_path, output_path):
        if os.path.exists(p):
            os.unlink(p)


if __name__ == "__main__":
    print("=== File I/O Demo ===", flush=True)
    run()
