#!/usr/bin/env python3
"""端到端测试：仿真执行服务 API 接口。

用法:
    python scripts/test_pipeline.py [--url http://localhost:8100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def request(url: str, method: str = "GET", body: dict | None = None) -> dict:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"  FAIL: {e.reason}")
        return {}


def test_health(base: str) -> bool:
    print("[1/5] Health check...", end=" ")
    result = request(f"{base}/health")
    ok = result.get("status") == "ok"
    print("PASS" if ok else f"FAIL ({result})")
    return ok


def test_state(base: str) -> bool:
    print("[2/5] Robot state...", end=" ")
    state = request(f"{base}/api/state")
    required = ("base_pos", "nav_state", "grasp_state", "joint_positions", "arm_positions")
    ok = all(k in state for k in required)
    if ok:
        print(f"PASS (pos={[round(v,2) for v in state['base_pos']]}, nav={state['nav_state']})")
    else:
        print(f"FAIL ({state})")
    return ok


def test_navigate(base: str) -> bool:
    print("[3/5] Navigation...", end=" ")
    # Start navigation
    # Robot starts at (0,-10) — navigate ~3m ahead to (0,-7)
    result = request(f"{base}/api/navigate", method="POST", body={"x": 0.0, "y": -7.0})
    if result.get("status") != "accepted":
        print(f"FAIL (start: {result})")
        return False

    # Poll until arrived or timeout (longer for distant targets)
    timeout = 60.0
    deadline = time.monotonic() + timeout
    last_status = ""
    while time.monotonic() < deadline:
        status = request(f"{base}/api/navigate/status")
        s = status.get("status", "unknown")
        if s == "arrived":
            print(f"PASS (arrived)")
            return True
        if s == "error":
            print(f"FAIL (nav error)")
            return False
        if s != last_status:
            last_status = s
        time.sleep(0.5)

    print(f"FAIL (timeout, last: {last_status})")
    return False


def test_grasp(base: str) -> bool:
    print("[4/5] Grasp...", end=" ")
    result = request(f"{base}/api/grasp", method="POST")
    if result.get("status") != "accepted":
        print(f"FAIL (start: {result})")
        return False

    # Poll until success or timeout
    timeout = 60.0
    deadline = time.monotonic() + timeout
    last_status = ""
    while time.monotonic() < deadline:
        status = request(f"{base}/api/grasp/status")
        s = status.get("status", "unknown")
        if s == "success":
            print("PASS")
            return True
        if s == "error":
            print(f"FAIL (grasp error)")
            return False
        if s != last_status:
            last_status = s
        time.sleep(0.5)

    print(f"FAIL (timeout, last: {last_status})")
    return False


def test_return_home(base: str) -> bool:
    print("[5/5] Return home...", end=" ")
    result = request(f"{base}/api/navigate", method="POST", body={"x": 0.0, "y": 0.0})
    if result.get("status") != "accepted":
        print(f"FAIL (start: {result})")
        return False

    timeout = 60.0
    deadline = time.monotonic() + timeout
    last_status = ""
    while time.monotonic() < deadline:
        status = request(f"{base}/api/navigate/status")
        s = status.get("status", "unknown")
        if s == "arrived":
            print("PASS")
            return True
        if s != last_status:
            last_status = s
        time.sleep(0.5)

    print(f"FAIL (timeout)")
    return False


def main():
    parser = argparse.ArgumentParser(description="仿真执行服务接口测试")
    parser.add_argument("--url", default="http://localhost:8100")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    print(f"Testing simulation server at {base}\n")

    results = []
    for test in (test_health, test_state, test_navigate, test_grasp, test_return_home):
        try:
            results.append(test(base))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
