#!/usr/bin/env python3
"""
Tunnel Watchdog — keeps the public tunnel alive.
Checks every 30s: if the public URL returns 503 or connection fails,
kills the dead SSH and spawns a fresh one.

Usage:
  python3 tunnel_watchdog.py

Requires: the FastAPI server already running on PORT.
"""
import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
import re
import signal
import atexit
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(os.getenv("PORT", "8000"))
CHECK_INTERVAL = 30  # seconds
TUNNEL_URL_FILE = "/tmp/wenzhou_tunnel_url.txt"
SSH_LOG = "/tmp/wenzhou_ssh_watchdog.log"

_tunnel_proc: subprocess.Popen | None = None
_current_url: str | None = None


def get_local_url() -> str:
    return f"http://localhost:{PORT}"


def check_local() -> bool:
    """Verify local server is alive."""
    try:
        urllib.request.urlopen(get_local_url() + "/api/lexicon/count", timeout=5)
        return True
    except Exception:
        return False


def check_tunnel(url: str | None) -> bool:
    """Check if the tunnel URL is accessible (returns 200)."""
    if not url:
        return False
    try:
        req = urllib.request.Request(url + "/api/lexicon/count", method="GET")
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 503:
            return False  # tunnel registered but routing dead
        return e.code == 200
    except Exception:
        return False


def read_url() -> str | None:
    try:
        with open(TUNNEL_URL_FILE) as f:
            u = f.read().strip()
            return u if u else None
    except Exception:
        return None


def kill_tunnel():
    global _tunnel_proc
    if _tunnel_proc:
        try:
            _tunnel_proc.terminate()
            _tunnel_proc.wait(timeout=3)
        except Exception:
            try:
                _tunnel_proc.kill()
            except Exception:
                pass
        _tunnel_proc = None
    # Also kill any other localhost.run processes
    subprocess.run(
        ["pkill", "-f", "nokey@localhost.run"],
        capture_output=True, timeout=5
    )
    # Clear URL file
    try:
        os.remove(TUNNEL_URL_FILE)
    except Exception:
        pass


def start_tunnel() -> str | None:
    global _tunnel_proc
    kill_tunnel()
    time.sleep(2)

    # Clear old log
    try:
        os.remove(SSH_LOG)
    except Exception:
        pass

    # Start SSH tunnel
    log_fd = open(SSH_LOG, "w")
    _tunnel_proc = subprocess.Popen(
        [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=2",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"80:localhost:{PORT}",
            "nokey@localhost.run",
        ],
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for URL (up to 20s)
    url = None
    for i in range(20):
        time.sleep(1)
        # Read log for URL
        try:
            with open(SSH_LOG) as f:
                content = f.read()
                m = re.search(r'https://[a-zA-Z0-9-]+\.lhr\.life', content)
                if m:
                    url = m.group(0)
                    break
        except Exception:
            pass

    if url:
        with open(TUNNEL_URL_FILE, "w") as f:
            f.write(url)
        return url
    return None


def cleanup():
    kill_tunnel()


atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))


def main():
    global _current_url

    print(f"{'='*50}")
    print(f"  🚀 温州话隧道守护进程 v2")
    print(f"  本地端口: {PORT}")
    print(f"  检测间隔: {CHECK_INTERVAL}s")
    print(f"  日志: {SSH_LOG}")
    print(f"{'='*50}")

    # Initial check: is local server alive?
    if not check_local():
        print("❌ 本地服务不可用！请先启动 uvicorn")
        sys.exit(1)
    print("✅ 本地服务正常")

    # Initial tunnel setup
    url = read_url()
    if url and check_tunnel(url):
        _current_url = url
        print(f"✅ 已有可用隧道: {url}")
    else:
        print("🔄 需要新建隧道...")
        url = start_tunnel()
        if url:
            _current_url = url
            print(f"✅ 隧道已建立: {url}")
        else:
            print("❌ 隧道建立失败，5秒后重试...")
            time.sleep(5)
            url = start_tunnel()
            if url:
                _current_url = url
                print(f"✅ 隧道已建立: {url}")
            else:
                print("❌ 隧道彻底失败！")
                sys.exit(1)

    # Watchdog loop
    while True:
        time.sleep(CHECK_INTERVAL)

        # 1. Check local
        if not check_local():
            print("⚠️ 本地服务挂了！等待恢复...")
            time.sleep(5)
            continue

        # 2. Check tunnel
        if not check_tunnel(_current_url):
            print(f"⚠️ 隧道失效 ({_current_url})，重建中...")
            url = start_tunnel()
            if url:
                _current_url = url
                print(f"✅ 隧道已重建: {url}")
            else:
                print("❌ 隧道重建失败，30秒后重试")
        else:
            # Periodically print heartbeat
            print(f"💚 隧道正常: {_current_url}")


if __name__ == "__main__":
    main()
