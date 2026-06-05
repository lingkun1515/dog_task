#!/usr/bin/env python3
"""Zero-dependency demo UI server for the Husqvarna presentation.

The server is intentionally isolated from robot execution. In mock mode it
advances a product-facing task timeline. During Orin integration, the robot
controller can push the same states through POST /api/task/status.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from camera_stream import get_stream

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

PRODUCT_STATES = (
    "idle",
    "ack",
    "go_to_B",
    "arrived_B_confirmed",
    "arm_start",
    "arm_done",
    "return_A",
    "done",
    "failed",
    "paused",
    "manual_takeover",
)
ACTIVE_STATES = {
    "ack",
    "go_to_B",
    "arrived_B_confirmed",
    "arm_start",
    "arm_done",
    "return_A",
    "paused",
}
STATUS_MESSAGES = {
    "idle": "设备待命，可派发清理任务",
    "ack": "任务已接单，正在准备执行",
    "go_to_B": "设备正在前往 B 点目标区域",
    "arrived_B_confirmed": "已到达 B 点，目标确认完成",
    "arm_start": "机械臂正在执行异物清理",
    "arm_done": "异物已安全放入回收篮",
    "return_A": "设备正在返回 A 点待命区",
    "done": "清理任务已完成，结果已回传",
    "failed": "到点确认失败，已进入人工接管",
    "paused": "任务已暂停，设备保持安全状态",
    "manual_takeover": "已切换为人工接管模式",
}
MOCK_FLOW = (
    ("go_to_B", 1.4),
    ("arrived_B_confirmed", 1.2),
    ("arm_start", 1.5),
    ("arm_done", 1.2),
    ("return_A", 1.5),
    ("done", 0.0),
)


@dataclass(frozen=True)
class ServerOptions:
    mock: bool = True


class TaskStore:
    def __init__(self, options: ServerOptions) -> None:
        self.options = options
        self._lock = threading.Lock()
        self._generation = 0
        self._task = self._idle_task()
        self._rain_mode = False
        self._resume_state = "idle"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            task = deepcopy(self._task)
            task["rain_mode"] = self._rain_mode
            task["mock_mode"] = self.options.mock
            return task

    def dispatch(self, scene: str = "草坪异物清理") -> tuple[bool, dict[str, Any]]:
        with self._lock:
            if self._task["status"] in ACTIVE_STATES or self._task["status"] == "manual_takeover":
                return False, self._response("当前任务仍在执行，请勿重复派单")
            self._generation += 1
            generation = self._generation
            now = time.time()
            self._task = {
                "task_id": f"HSQ-{datetime.now():%m%d-%H%M%S}",
                "scene": scene,
                "object_name": "草坪异物",
                "status": "ack",
                "status_message": STATUS_MESSAGES["ack"],
                "device_name": "Outdoor Task Robot 01",
                "device_status": "在线 · 执行中",
                "started_at_s": now,
                "completed_at_s": None,
                "elapsed_s": 0,
                "evidence_image": "/static/assets/site-preview.svg",
                "result": None,
                "history": [self._event("ack", now)],
            }
        if self.options.mock:
            threading.Thread(
                target=self._run_mock_flow,
                args=(generation,),
                daemon=True,
                name="ui-mock-task",
            ).start()
        return True, self.snapshot()

    def update_status(
        self,
        status: str,
        *,
        message: str | None = None,
        object_name: str | None = None,
        evidence_image: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        if status not in PRODUCT_STATES:
            return False, self._response(f"不支持的任务状态: {status}")
        with self._lock:
            if self._task["task_id"] is None and status != "idle":
                return False, self._response("请先派发任务")
            if self._task["status"] == "manual_takeover" and status not in {"idle", "manual_takeover"}:
                return False, self._response("当前处于人工接管模式，请先返回待命")
            self._apply_status(status, message=message)
            if object_name:
                self._task["object_name"] = object_name
            if evidence_image:
                self._task["evidence_image"] = evidence_image
            return True, self._snapshot_locked()

    def pause(self) -> tuple[bool, dict[str, Any]]:
        with self._lock:
            if self._task["status"] not in ACTIVE_STATES or self._task["status"] == "paused":
                return False, self._response("当前没有可暂停的任务")
            self._resume_state = self._task["status"]
            self._apply_status("paused")
            return True, self._snapshot_locked()

    def resume(self) -> tuple[bool, dict[str, Any]]:
        with self._lock:
            if self._task["status"] != "paused":
                return False, self._response("当前任务未暂停")
            self._apply_status(self._resume_state)
            return True, self._snapshot_locked()

    def takeover(self) -> tuple[bool, dict[str, Any]]:
        with self._lock:
            self._generation += 1
            self._apply_status("manual_takeover")
            self._task["device_status"] = "在线 · 人工接管"
            return True, self._snapshot_locked()

    def reset(self) -> tuple[bool, dict[str, Any]]:
        with self._lock:
            self._generation += 1
            self._task = self._idle_task()
            return True, self._snapshot_locked()

    def set_rain_mode(self, enabled: bool) -> tuple[bool, dict[str, Any]]:
        with self._lock:
            self._rain_mode = bool(enabled)
            return True, self._snapshot_locked()

    def _run_mock_flow(self, generation: int) -> None:
        for status, delay_s in MOCK_FLOW:
            time.sleep(delay_s)
            while True:
                with self._lock:
                    if generation != self._generation:
                        return
                    paused = self._task["status"] == "paused"
                    takeover = self._task["status"] == "manual_takeover"
                if takeover:
                    return
                if not paused:
                    break
                time.sleep(0.2)
            with self._lock:
                if generation != self._generation:
                    return
                if self._task["status"] == "manual_takeover":
                    return
                self._apply_status(status)

    def _apply_status(self, status: str, message: str | None = None) -> None:
        now = time.time()
        self._task["status"] = status
        self._task["status_message"] = message or STATUS_MESSAGES[status]
        self._task["history"].append(self._event(status, now, message))
        if self._task["started_at_s"]:
            self._task["elapsed_s"] = round(now - self._task["started_at_s"])
        if status == "done":
            self._task["completed_at_s"] = now
            self._task["device_status"] = "在线 · 已返回待命"
            self._task["result"] = "清理完成"
        elif status in {"failed", "manual_takeover"}:
            self._task["completed_at_s"] = now
            self._task["device_status"] = "在线 · 等待人工确认"
            self._task["result"] = "已进入人工接管"
        elif status == "paused":
            self._task["device_status"] = "在线 · 已暂停"
        elif status != "idle":
            self._task["device_status"] = "在线 · 执行中"

    def _snapshot_locked(self) -> dict[str, Any]:
        task = deepcopy(self._task)
        task["rain_mode"] = self._rain_mode
        task["mock_mode"] = self.options.mock
        return task

    @staticmethod
    def _event(status: str, timestamp_s: float, message: str | None = None) -> dict[str, Any]:
        return {
            "status": status,
            "message": message or STATUS_MESSAGES[status],
            "timestamp_s": timestamp_s,
        }

    @staticmethod
    def _response(message: str) -> dict[str, Any]:
        return {"message": message}

    @staticmethod
    def _idle_task() -> dict[str, Any]:
        return {
            "task_id": None,
            "scene": "草坪异物清理",
            "object_name": "待识别",
            "status": "idle",
            "status_message": STATUS_MESSAGES["idle"],
            "device_name": "Outdoor Task Robot 01",
            "device_status": "在线 · 待命",
            "started_at_s": None,
            "completed_at_s": None,
            "elapsed_s": 0,
            "evidence_image": "/static/assets/site-preview.svg",
            "result": None,
            "history": [],
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    store: TaskStore

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json({"ready": True, "service": "husqvarna-demo-ui"})
        elif path == "/api/task":
            self._json(self.store.snapshot())
        elif path == "/api/camera/status":
            stream = get_stream()
            self._json({"connected": stream.connected})
        elif path == "/api/camera/stream":
            self._serve_mjpeg()
        elif path == "/":
            self._file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            self._file(STATIC_DIR / unquote(path.removeprefix("/static/")))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _receive_camera_frame(self) -> None:
        """接收主流程推送的 JPEG 帧"""
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 2 * 1024 * 1024:  # 限制2MB
                self._json({"message": "帧数据过大"}, HTTPStatus.BAD_REQUEST)
                return
            if size == 0:
                self._json({"message": "无数据"}, HTTPStatus.BAD_REQUEST)
                return
            jpeg_data = self.rfile.read(size)
            get_stream().push_jpeg(jpeg_data)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
        except Exception as exc:
            self._json({"message": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _serve_mjpeg(self) -> None:
        """输出MJPEG流, 保持连接直到客户端断开"""
        stream = get_stream()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for chunk in stream.generate_mjpeg():
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        # 相机帧接收(二进制JPEG, 非JSON)
        if path == "/api/camera/frame":
            self._receive_camera_frame()
            return

        payload = self._read_json()
        if payload is None:
            return
        if path == "/api/task/dispatch":
            ok, response = self.store.dispatch(str(payload.get("scene", "草坪异物清理")))
        elif path == "/api/task/status":
            ok, response = self.store.update_status(
                str(payload.get("status", "")),
                message=self._optional_text(payload, "message"),
                object_name=self._optional_text(payload, "object_name"),
                evidence_image=self._optional_text(payload, "evidence_image"),
            )
        elif path == "/api/task/pause":
            ok, response = self.store.pause()
        elif path == "/api/task/resume":
            ok, response = self.store.resume()
        elif path == "/api/task/takeover":
            ok, response = self.store.takeover()
        elif path == "/api/task/reset":
            ok, response = self.store.reset()
        elif path == "/api/mode/rain":
            ok, response = self.store.set_rain_mode(bool(payload.get("enabled", False)))
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._json(response, HTTPStatus.OK if ok else HTTPStatus.CONFLICT)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[ui] {self.address_string()} {fmt % args}")

    def _read_json(self) -> dict[str, Any] | None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 64 * 1024:
                raise ValueError("request body too large")
            body = self.rfile.read(size) if size else b"{}"
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._json({"message": f"请求格式错误: {exc}"}, HTTPStatus.BAD_REQUEST)
            return None

    def _file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(STATIC_DIR.resolve())
            content = resolved.read_bytes()
        except (OSError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(resolved.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    @staticmethod
    def _optional_text(payload: dict[str, Any], key: str) -> str | None:
        value = payload.get(key)
        return str(value) if value is not None else None


def create_server(host: str, port: int, *, mock: bool = True) -> ThreadingHTTPServer:
    handler = type("ConfiguredDemoRequestHandler", (DemoRequestHandler,), {})
    handler.store = TaskStore(ServerOptions(mock=mock))
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Husqvarna demo UI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="Wait for Orin state callbacks instead of advancing the demo timeline automatically",
    )
    args = parser.parse_args()

    server = create_server(args.host, args.port, mock=not args.no_mock)
    print(f"Husqvarna demo UI: http://127.0.0.1:{args.port}")
    print("Mode:", "mock timeline" if not args.no_mock else "Orin callback")
    print("Camera: 等待主流程推送帧数据 (D455Camera -> UI)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
