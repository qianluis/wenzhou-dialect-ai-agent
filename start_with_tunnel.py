"""
Launch Gradio + public tunnel via localhost.run (no auth needed).
"""
import os, sys, time, subprocess, threading, urllib.request, signal

os.environ.setdefault("FUSION_MODE", "single")
os.environ.setdefault("ASR_BACKEND", "firered")
os.environ.setdefault("ASR_WENZHOU_BACKEND", "firered")
os.environ.setdefault("ASR_MANDARIN_BACKEND", "sensevoice")
os.environ.setdefault("GRADIO_PORT", "7860")
port = int(os.environ["GRADIO_PORT"])

root = os.path.dirname(os.path.abspath(__file__))

# 1. Start Gradio
gradio_proc = subprocess.Popen(
    [sys.executable, os.path.join(root, "frontend/gradio_app.py")],
    cwd=root,
    env={**os.environ, "PYTHONPATH": root},
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(f"✅ Gradio started (PID {gradio_proc.pid}) on port {port}")

# 2. Wait for server
for i in range(30):
    try:
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=3)
        print("✅ Gradio server ready")
        break
    except Exception:
        time.sleep(1)
else:
    print("❌ Gradio failed to start")
    gradio_proc.kill()
    sys.exit(1)

# 3. SSH tunnel via localhost.run
public_url_file = "/tmp/tunnel_url.txt"
# Clear any previous
if os.path.exists(public_url_file):
    os.remove(public_url_file)

tunnel_proc = subprocess.Popen(
    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
     "-R", f"80:localhost:{port}",
     "nokey@localhost.run"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

def capture_url(proc, outfile):
    """Read tunnel output until we get the URL."""
    for line in proc.stdout:
        print(f"[tunnel] {line}", end="")
        import re
        m = re.search(r'(https://\S+\.lhr\.life)', line)
        if m:
            with open(outfile, "w") as f:
                f.write(m.group(1))
            break

t = threading.Thread(target=capture_url, args=(tunnel_proc, public_url_file), daemon=True)
t.start()

# 4. Wait for URL
url = None
for i in range(30):
    if os.path.exists(public_url_file):
        with open(public_url_file) as f:
            url = f.read().strip()
        break
    time.sleep(1)

if url:
    print(f"\n{'='*55}")
    print(f"  🌐 公网可访问网址 (Public URL):")
    print(f"  {url}")
    print(f"{'='*55}")
    print(f"  📱 手机浏览器直接打开即可测评")
    print(f"  ⏳ 按 Ctrl+C 停止服务")
    print(f"{'='*55}\n")
else:
    print("⚠️ Tunnel URL not yet available, checking...")
    time.sleep(5)
    try:
        with open(public_url_file) as f:
            url = f.read().strip()
        print(f"URL: {url}")
    except:
        print("Tunnel failed. Trying alternative...")

# Keep alive
try:
    while True:
        time.sleep(10)
        if gradio_proc.poll() is not None:
            print("❌ Gradio died")
            break
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    tunnel_proc.kill()
    gradio_proc.kill()
    print("Done.")
