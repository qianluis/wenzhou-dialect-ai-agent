#!/usr/bin/env python3
"""
Quick test: start SSH tunnel, wait for URL, run 10 quick tests.
All in one script to avoid process management issues.
"""
import subprocess
import re
import urllib.request
import time
import sys
import os

LOCAL_PORT = 8000
SSH_SERVER = "nokey@localhost.run"

print("=== Starting SSH tunnel to localhost.run ===")
proc = subprocess.Popen(
    ["ssh",
     "-o", "StrictHostKeyChecking=no",
     "-o", "ServerAliveInterval=10",
     "-o", "ServerAliveCountMax=2",
     "-o", "ExitOnForwardFailure=yes",
     "-R", f"80:localhost:{LOCAL_PORT}",
     SSH_SERVER],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

url = None
url_pattern = re.compile(r'(https?://[a-zA-Z0-9.-]+\.lhr\.life)')

print("Waiting for tunnel URL...")
for line in iter(proc.stdout.readline, ""):
    line = line.strip()
    if line:
        print(f"  SSH: {line[:120]}")
    m = url_pattern.search(line)
    if m:
        url = m.group(1)
        print(f"\n✅ Tunnel URL: {url}")
        with open("/tmp/tunnel_url.txt", "w") as f:
            f.write(url)
        break
    if proc.poll() is not None:
        print(f"❌ SSH process died (code {proc.returncode})")
        sys.exit(1)

if not url:
    print("❌ Failed to get URL")
    sys.exit(1)

# Give tunnel a moment
time.sleep(3)

# Run 10 tests as fast as possible
print(f"\n{'='*50}")
print("Running 10 consecutive tests...")
print(f"{'='*50}")

ok = 0
fail = 0
for i in range(1, 11):
    try:
        req = urllib.request.Request(url + "/api/lexicon/count")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            if resp.status == 200:
                print(f"  TEST[{i:2d}] ✅ 200 - {body}")
                ok += 1
            else:
                print(f"  TEST[{i:2d}] ❌ {resp.status}")
                fail += 1
    except Exception as e:
        print(f"  TEST[{i:2d}] ❌ ERROR: {type(e).__name__}")
        fail += 1
    time.sleep(2)

# Also test the root page once
try:
    req = urllib.request.Request(url + "/")
    with urllib.request.urlopen(req, timeout=5) as resp:
        print(f"\n  ROOT PAGE: HTTP {resp.status} ({len(resp.read())} bytes)")
except Exception as e:
    print(f"\n  ROOT PAGE: FAILED - {e}")

print(f"\n{'='*50}")
print(f"RESULTS: ✅ {ok}/10 success, ❌ {fail}/10 failed")
print(f"URL: {url}")
print(f"{'='*50}")

# Cleanup
proc.kill()
proc.wait()

if ok >= 9:
    sys.exit(0)
else:
    print(f"❌ FAILED: Only {ok}/10 successful (need ≥9)")
    sys.exit(1)
