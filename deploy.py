"""
One-shot deploy: start FastAPI server + pyngrok tunnel.
"""
import os, sys, time, signal, subprocess, threading

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8000

# 1. Kill any leftover
os.system(f"fuser -k {PORT}/tcp 2>/dev/null")
time.sleep(1)

# 2. Start FastAPI server
env = {**os.environ, "PYTHONPATH": ROOT, "FUSION_MODE": "single",
       "ASR_BACKEND": "firered", "ASR_WENZHOU_BACKEND": "firered",
       "ASR_MANDARIN_BACKEND": "sensevoice", "PORT": str(PORT)}

server_proc = subprocess.Popen(
    [sys.executable, os.path.join(ROOT, "serve.py")],
    env=env, cwd=ROOT,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print(f"✅ Server started (PID {server_proc.pid})")

# 3. Wait for server
import urllib.request
for i in range(30):
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/", timeout=2)
        print("✅ Server ready on port", PORT)
        break
    except Exception:
        time.sleep(1)
else:
    print("❌ Server failed to start")
    server_proc.kill()
    sys.exit(1)

# 4. Start ngrok tunnel via pyngrok
try:
    from pyngrok import ngrok
    # Kill any existing ngrok
    ngrok.kill()
    tunnel = ngrok.connect(PORT, "http", bind_tls=True)
    public_url = tunnel.public_url
    print(f"\n{'='*50}")
    print(f"  🌐 公网可访问网址:")
    print(f"  {public_url}")
    print(f"  📱 手机浏览器直接打开")
    print(f"{'='*50}\n")
    
    # Save URL
    with open("/tmp/ngrok_url.txt", "w") as f:
        f.write(public_url)
except Exception as e:
    print(f"⚠️ pyngrok failed: {e}")
    print("Falling back to localhost.run...")
    
    tunnel_proc = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
         "-R", f"80:localhost:{PORT}", "nokey@localhost.run"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    
    def capture_url():
        import re
        for line in tunnel_proc.stdout:
            m = re.search(r'(https://\S+\.lhr\.life)', line)
            if m:
                url = m.group(1)
                print(f"\n{'='*50}")
                print(f"  🌐 公网可访问网址:")
                print(f"  {url}")
                print(f"{'='*50}\n")
                with open("/tmp/ngrok_url.txt", "w") as f:
                    f.write(url)
                break
    
    threading.Thread(target=capture_url, daemon=True).start()

# 5. Keep alive
print("⏳ 按 Ctrl+C 停止...")
try:
    while True:
        time.sleep(10)
        if server_proc.poll() is not None:
            print("❌ Server died")
            break
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    server_proc.kill()
    try:
        ngrok.disconnect(public_url)
    except:
        pass
    print("Done.")
