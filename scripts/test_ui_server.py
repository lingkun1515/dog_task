#!/usr/bin/env python3
"""UI Server 单元测试."""
import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "UI"))

from server import create_server


def test_health():
    """GET /api/health → 200."""
    resp = urllib.request.urlopen("http://127.0.0.1:8765/api/health", timeout=3)
    data = json.loads(resp.read())
    assert data["ready"], f"health should be ready: {data}"
    print("  PASS: GET /api/health → 200")


def test_task_dispatch():
    """POST /api/task/dispatch → 任务创建."""
    payload = json.dumps({"scene": "test"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/task/dispatch",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=3)
    data = json.loads(resp.read())
    assert data["status"] in ("ack", "go_to_B"), f"unexpected status: {data['status']}"
    print(f"  PASS: POST /api/task/dispatch → status={data['status']}")


def test_task_status():
    """POST /api/task/status → 更新状态."""
    payload = json.dumps({"status": "arrived_B_confirmed", "message": "test status update"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/task/status",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=3)
    data = json.loads(resp.read())
    assert data["status"] == "arrived_B_confirmed", f"status not updated: {data}"
    print("  PASS: POST /api/task/status → updated")


def test_camera_frame():
    """POST /api/camera/frame → 204."""
    import numpy as np
    import cv2
    dummy = np.zeros((120, 160, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", dummy)
    data = jpeg.tobytes()

    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/camera/frame",
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=3)
    assert resp.status == 204, f"expected 204, got {resp.status}"
    print("  PASS: POST /api/camera/frame → 204")


def test_camera_status():
    """GET /api/camera/status → connected=True."""
    resp = urllib.request.urlopen("http://127.0.0.1:8765/api/camera/status", timeout=3)
    data = json.loads(resp.read())
    assert data["connected"], "camera should be connected after frame push"
    print("  PASS: GET /api/camera/status → connected")


if __name__ == "__main__":
    print("=== UI Server 单元测试 ===")

    server = create_server("127.0.0.1", 8765, mock=False)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    try:
        test_health()
        test_task_dispatch()
        test_task_status()
        test_camera_frame()
        test_camera_status()
        print("=== ALL PASSED ===")
    finally:
        server.shutdown()
        server.server_close()
