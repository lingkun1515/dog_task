"""实机 FastAPI 执行服务。

与 sim_task_server.py API 完全对齐，方便逐行对比。

启动:
    python -m execution.real_robots.real_task_server [--port 5000] [--classes bottle]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from utils.logging import setup_logging

setup_logging("execution", "logs/execution.log")

import logging

from algorithms.calibration import REAL_CALIB_FILE, load_calibration_file
from algorithms.calibration.base import CalibrationResult

logger = logging.getLogger(__name__)
from algorithms.grasp.executor import RealArmExecutor
from algorithms.grasp.planner import GraspConfig, GraspPlanner
from algorithms.kinematics.real_d1_ik import D1Kinematics
from algorithms.navigation import NavState, NavigationController
from algorithms.perception.perception import YOLODetector
from execution.real_robots.camera import RealSenseCamera

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_state = {
    "busy": False,
    "status": "idle",
    "lock": threading.Lock(),
}

_camera: RealSenseCamera | None = None
_planner: GraspPlanner | None = None
_executor: RealArmExecutor | None = None
_nav: NavigationController | None = None
_yolo_classes: list[str] = ["bottle"]
_arm_host: str = "192.168.123.100"
_arm_port: int = 8088

SAFE_PARK = [-90, 30, -10, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def init_all(
    classes: list[str] | None = None,
    arm_host: str | None = None,
    arm_port: int | None = None,
):
    global _camera, _planner, _executor, _nav, _yolo_classes, _arm_host, _arm_port

    if classes:
        _yolo_classes = classes
    if arm_host:
        _arm_host = arm_host
    if arm_port:
        _arm_port = arm_port

    print(f"[real_task] 连接机械臂 {_arm_host}:{_arm_port}...", flush=True)
    _executor = RealArmExecutor(host=_arm_host, port=_arm_port)

    print("[real_task] 启动 D455...", flush=True)
    _camera = RealSenseCamera()

    print(f"[real_task] 加载 YOLO ({_yolo_classes})...", flush=True)
    detector = YOLODetector(_camera, classes=_yolo_classes, conf=0.3)

    print("[real_task] 加载标定...", flush=True)
    calib_path = os.environ.get("CALIB_PATH", REAL_CALIB_FILE)
    try:
        calibration = load_calibration_file(calib_path)
        print(f"  标定已加载: {calib_path} (method={calibration.method})", flush=True)
    except FileNotFoundError:
        logger.warning("标定文件不存在: %s — 使用几何估算 fallback", calib_path)
        from algorithms.calibration.real_calibration import load_calibration
        calibration = load_calibration(None)

    kinematics = D1Kinematics()

    config = GraspConfig(
        approach_height=0.10,
        descend_step=0.03,
        gripper_open=65,
        gripper_close=0,
        z_overshoot=0.02,
        lift_height=0.15,
        safe_park=SAFE_PARK,
    )

    _planner = GraspPlanner(
        kinematics=kinematics,
        detector=detector,
        calibration=calibration,
        executor=_executor,
        config=config,
    )

    _nav = NavigationController(
        linear_speed=0.5,
        angular_speed=0.8,
        arrival_threshold=0.5,
        heading_threshold=0.1,
    )

    print("[real_task] 初始化完成", flush=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _camera is not None:
        _camera.stop()


app = FastAPI(title="DogTask Real Executor", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ===================================================================
# Health
# ===================================================================
@app.get("/health")
def health():
    return {"status": "ok", "camera": _camera is not None}


# ===================================================================
# Navigation（对齐 sim_task_server.py /api/navigate 系列）
# ===================================================================
@app.post("/api/navigate")
def api_navigate(body: dict):
    """开始导航到目标位置。"""
    if _nav is None:
        return JSONResponse(status_code=503, content={"detail": "not_initialized"})
    x = float(body.get("x", 0))
    y = float(body.get("y", 0))
    require_heading = bool(body.get("require_heading", True))
    arrival_threshold = body.get("arrival_threshold", None)
    if arrival_threshold is not None:
        arrival_threshold = float(arrival_threshold)
    _nav.set_target(x, y, require_heading=require_heading, arrival_threshold=arrival_threshold)
    return {"status": "accepted", "target": [x, y]}


@app.get("/api/navigate/status")
def api_navigate_status():
    """返回当前导航状态。"""
    if _nav is None:
        return {"status": "not_initialized", "target": [0, 0]}
    return {
        "status": _nav.state.value,
        "target": _nav._target.tolist(),
    }


@app.post("/api/navigate/cancel")
def api_navigate_cancel():
    """取消导航。"""
    if _nav is not None:
        _nav.cancel()
    return {"status": "cancelled"}


# ===================================================================
# Grasp（对齐 sim_task_server.py /api/grasp — 异步后台执行）
# ===================================================================
@app.post("/api/grasp")
def api_grasp():
    """启动抓取（异步后台，立即返回）。"""
    with _state["lock"]:
        if _state["busy"]:
            return JSONResponse(status_code=409, content={"status": "rejected", "detail": "busy"})
        _state["busy"] = True
        _state["status"] = "running"

    if _planner is None:
        with _state["lock"]:
            _state["busy"] = False
            _state["status"] = "idle"
        return JSONResponse(status_code=503, content={"detail": "not_initialized"})

    def _run():
        try:
            _planner.execute_full_cycle()
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            with _state["lock"]:
                _state["busy"] = False
                _state["status"] = "idle"

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "accepted"}


@app.get("/api/grasp/status")
def api_grasp_status():
    """返回当前抓取状态。"""
    return {
        "busy": _state["busy"],
        "status": _state["status"],
        "planner_state": _planner.state.value if _planner else "not_init",
    }


# ===================================================================
# Detection（对齐 sim_task_server.py — sim 从 state 获取）
# ===================================================================
@app.get("/api/detect")
def api_detect():
    """执行单次检测并返回结果。"""
    if _planner is None:
        return {"count": 0, "detections": []}
    dets = _planner._detector.detect()
    clean = []
    for d in dets:
        clean.append({
            "label": d.label,
            "center": list(d.center_pixel),
            "bbox": list(d.bbox),
            "depth_m": round(d.depth_m, 4),
            "confidence": round(d.confidence, 3),
        })
    return {"count": len(clean), "detections": clean}


# ===================================================================
# Robot state（对齐 sim_task_server.py /api/state）
# ===================================================================
@app.get("/api/state")
def api_state():
    """返回完整机器人状态快照。"""
    return {
        "busy": _state["busy"],
        "planner_state": _planner.state.value if _planner else "not_init",
        "nav_state": _nav.state.value if _nav else "not_init",
        "nav_target": _nav._target.tolist() if _nav else [0, 0],
    }


# ===================================================================
# Stop（对齐 sim_task_server.py /api/stop）
# ===================================================================
@app.post("/api/stop")
def api_stop():
    """取消所有活动（导航 + 抓取）。"""
    if _nav is not None:
        _nav.cancel()
    with _state["lock"]:
        _state["busy"] = False
    return {"status": "stopped"}


# ===================================================================
# Reset（对齐 sim_task_server.py /api/reset）
# ===================================================================
@app.post("/api/reset")
def api_reset():
    """重置状态。"""
    if _nav is not None:
        _nav.cancel()
    with _state["lock"]:
        _state["busy"] = False
        _state["status"] = "idle"
    if _executor is not None:
        _executor.move_to_joints(SAFE_PARK, mode=1, wait_time=2.0)
    return {"status": "reset"}


# ===================================================================
# MJPEG Video Feed（对齐 sim_task_server.py /api/video_feed）
# ===================================================================
def _gen_mjpeg():
    while True:
        if _camera is None:
            time.sleep(0.1)
            continue
        import cv2
        color, _ = _camera.grab()
        if color is None:
            time.sleep(0.1)
            continue
        _, buf = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.066)


@app.get("/api/video_feed")
def api_video_feed():
    return StreamingResponse(
        _gen_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ===================================================================
# Backward-compatible aliases (old paths)
# ===================================================================
@app.post("/grasp")
def api_grasp_legacy():
    return api_grasp()


@app.get("/status")
def api_status_legacy():
    return api_grasp_status()


@app.get("/detect")
def api_detect_legacy():
    return api_detect()


@app.get("/video_feed")
def api_video_feed_legacy():
    return api_video_feed()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DogTask Real Executor")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--classes", nargs="+", default=["bottle"])
    parser.add_argument("--arm-host", default="192.168.123.100")
    parser.add_argument("--arm-port", type=int, default=8088)
    args = parser.parse_args()

    init_all(classes=args.classes, arm_host=args.arm_host, arm_port=args.arm_port)

    import uvicorn
    print(f"\n[real_task] 服务启动: http://{args.host}:{args.port}", flush=True)
    print(f"  POST /api/navigate  - 导航到目标", flush=True)
    print(f"  GET  /api/navigate/status - 导航状态", flush=True)
    print(f"  POST /api/navigate/cancel - 取消导航", flush=True)
    print(f"  POST /api/grasp  - 执行抓取", flush=True)
    print(f"  GET  /api/grasp/status - 抓取状态", flush=True)
    print(f"  GET  /api/detect - 仅检测", flush=True)
    print(f"  GET  /api/state - 完整状态", flush=True)
    print(f"  GET  /api/video_feed - MJPEG 视频流", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
