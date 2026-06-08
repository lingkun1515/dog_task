"""实机 FastAPI 执行服务（替代 Orin Flask web_server.py）。

启动:
    python -m execution.robots.real_server [--port 5000] [--classes bottle]
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

from algorithms.calibration.base import CalibrationResult
from algorithms.calibration.real_calibration import load_calibration
from algorithms.grasp.executor import RealArmExecutor
from algorithms.grasp.planner import GraspConfig, GraspPlanner
from algorithms.kinematics.real_d1_ik import D1Kinematics
from algorithms.perception.real_perception import RealSenseCamera, YOLODetector

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
_yolo_classes: list[str] = ["bottle"]
_arm_host: str = "192.168.123.100"
_arm_port: int = 8088

SAFE_PARK = [-90, 30, -10, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def init_all(classes: list[str] | None = None, arm_host: str | None = None, arm_port: int | None = None):
    global _camera, _planner, _executor, _yolo_classes, _arm_host, _arm_port

    if classes:
        _yolo_classes = classes
    if arm_host:
        _arm_host = arm_host
    if arm_port:
        _arm_port = arm_port

    print(f"[real_server] 连接机械臂 {_arm_host}:{_arm_port}...", flush=True)
    _executor = RealArmExecutor(host=_arm_host, port=_arm_port)

    print("[real_server] 启动 D455...", flush=True)
    _camera = RealSenseCamera()

    print(f"[real_server] 加载 YOLO ({_yolo_classes})...", flush=True)
    detector = YOLODetector(_camera, classes=_yolo_classes, conf=0.3)

    print("[real_server] 加载标定...", flush=True)
    calib_path = os.environ.get("CALIB_PATH",
                                str(_PROJECT_ROOT / "output" / "calibration_result.json"))
    calibration = load_calibration(calib_path if os.path.exists(calib_path) else None)

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
        sim_mode=False,
    )

    print("[real_server] 初始化完成", flush=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _camera is not None:
        _camera.stop()


app = FastAPI(title="DogTask Real Executor", version="0.1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "camera": _camera is not None}


# ---------------------------------------------------------------------------
# Grasp
# ---------------------------------------------------------------------------
@app.post("/grasp")
def api_grasp():
    """阻塞式抓取：检测 → 抓取 → 返回结果。"""
    with _state["lock"]:
        if _state["busy"]:
            return JSONResponse(status_code=409, content={"success": False, "error": "busy"})
        _state["busy"] = True

    try:
        if _planner is None:
            return {"success": False, "error": "not_initialized"}
        result = _planner.execute_full_cycle()
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": "exception", "message": str(e)}
    finally:
        with _state["lock"]:
            _state["busy"] = False
            _state["status"] = "idle"


@app.get("/status")
def api_status():
    return {
        "busy": _state["busy"],
        "status": _state["status"],
        "planner_state": _planner.state.value if _planner else "not_init",
    }


@app.get("/detect")
def api_detect():
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


# ---------------------------------------------------------------------------
# MJPEG Video Feed
# ---------------------------------------------------------------------------
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


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _gen_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


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
    print(f"\n[real_server] 服务启动: http://{args.host}:{args.port}", flush=True)
    print(f"  POST /grasp  - 执行抓取", flush=True)
    print(f"  GET  /status - 查询状态", flush=True)
    print(f"  GET  /detect - 仅检测", flush=True)
    print(f"  GET  /video_feed - MJPEG 视频流", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
