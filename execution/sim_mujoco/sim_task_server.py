"""Sim 端 FastAPI 执行服务（MuJoCo 仿真）。

启动:
    python -m execution.sim_mujoco.sim_task_server --config sim_go2_piper
    python -m execution.sim_mujoco.sim_task_server --config sim_go2_d1 --render
    python -m execution.sim_mujoco.sim_task_server --config config/robots/sim_go2_d1.toml
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CLI / env-var handling (must happen BEFORE importing mujoco modules so
# that MUJOCO_GL can be set correctly).
#
# When launched as __main__ we parse CLI args and forward settings via env
# vars so that uvicorn's re-import sees the same configuration.
# ---------------------------------------------------------------------------
def _resolve_config_path(config_arg: str) -> str:
    """将 --config 参数解析为完整路径。支持 robot_id 或文件路径。"""
    p = Path(config_arg)
    if p.suffix == ".toml" and p.exists():
        return str(p.resolve())
    # 作为 robot_id 处理
    project_root = Path(__file__).resolve().parent.parent.parent
    candidate = project_root / "config" / "robots" / f"{config_arg}.toml"
    if candidate.exists():
        return str(candidate)
    return config_arg


def _parse_args():
    parser = argparse.ArgumentParser(description="DogTask MuJoCo simulation server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8100, help="Bind port")
    parser.add_argument(
        "--config", default="sim_go2_piper",
        help="Robot config: robot_id (e.g. sim_go2_d1) or .toml path",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable desktop visualisation window (default: headless EGL)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    _args = _parse_args()
    os.environ["DTS_RENDER"] = "1" if _args.render else "0"
    os.environ["DTS_HOST"] = _args.host
    os.environ["DTS_PORT"] = str(_args.port)
    os.environ["DTS_CONFIG"] = _resolve_config_path(_args.config)
else:
    # Re-import by uvicorn — read settings from env vars
    _args = argparse.Namespace(
        render=os.environ.get("DTS_RENDER", "0") == "1",
        host=os.environ.get("DTS_HOST", "0.0.0.0"),
        port=int(os.environ.get("DTS_PORT", "8100")),
        config=os.environ.get("DTS_CONFIG", ""),
    )

if not _args.render:
    os.environ.setdefault("MUJOCO_GL", "egl")
else:
    os.environ["MUJOCO_GL"] = "glfw"

# Ensure project root is on sys.path so scheduler imports work when running standalone
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from utils.logging import setup_logging

setup_logging("execution", "logs/execution.log")

import logging

from execution.sim_mujoco.scene import SimulationScene

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global scene handle — initialised during lifespan
# ---------------------------------------------------------------------------
_scene: SimulationScene | None = None


def get_scene() -> SimulationScene:
    if _scene is None:
        raise RuntimeError("Simulation scene not initialised")
    return _scene


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scene
    # In GUI mode the scene is pre-created in main() so GLFW stays on the
    # main thread; in headless mode we create it here.
    if _scene is None:
        config_path = os.environ.get("DTS_CONFIG") or None
        _scene = SimulationScene(config_path, render_mode="headless")
        _scene.start()
    yield
    if _scene is not None:
        _scene.stop()


app = FastAPI(title="DogTask Sim Executor", version="0.1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
@app.post("/api/navigate")
def navigate(body: dict[str, Any]) -> dict[str, Any]:
    """Start navigation to a target position.

    Body: {"x": float, "y": float, "require_heading": bool, "arrival_threshold": float}
    """
    scene = get_scene()
    x = float(body.get("x", 0))
    y = float(body.get("y", 0))
    require_heading = bool(body.get("require_heading", True))
    arrival_threshold = body.get("arrival_threshold", None)
    if arrival_threshold is not None:
        arrival_threshold = float(arrival_threshold)
    logger.info("导航请求: target=(%.2f, %.2f) require_heading=%s threshold=%s", x, y, require_heading, arrival_threshold)
    scene.navigate_to(x, y, require_heading=require_heading, arrival_threshold=arrival_threshold)
    return {"status": "accepted", "target": [x, y]}


@app.get("/api/navigate/status")
def navigate_status() -> dict[str, Any]:
    """Return current navigation status."""
    scene = get_scene()
    s = scene.state
    return {
        "status": s["nav_state"].value,
        "position": s["base_pos"][:2],
        "target": s["nav_target"],
    }


@app.post("/api/navigate/cancel")
def navigate_cancel() -> dict[str, str]:
    get_scene().cancel_navigation()
    return {"status": "cancelled"}


# ---------------------------------------------------------------------------
# Grasp
# ---------------------------------------------------------------------------
@app.post("/api/grasp")
def grasp() -> dict[str, str]:
    """Start the grasp sequence."""
    scene = get_scene()
    logger.info("抓取请求: 启动抓取序列")
    scene.start_grasp()
    return {"status": "accepted"}


@app.get("/api/grasp/status")
def grasp_status() -> dict[str, Any]:
    scene = get_scene()
    gs = scene.state["grasp_state"]
    return {
        "busy": scene._algo_running,
        "status": gs,
        "planner_state": scene._algo_planner.state.value if scene._algo_planner else gs,
    }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
@app.get("/api/detect")
def detect() -> dict[str, Any]:
    """执行单次检测并返回结果。"""
    scene = get_scene()
    dets: list[dict] = []
    if scene._sim_detector is not None:
        for d in scene._sim_detector.detect():
            dets.append({
                "label": d.label,
                "center": list(d.center_pixel),
                "bbox": list(d.bbox),
                "depth_m": round(d.depth_m, 4),
                "confidence": round(d.confidence, 3),
            })
    logger.debug("检测结果: %d 个目标", len(dets))
    return {"count": len(dets), "detections": dets}


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------
@app.get("/api/state")
def robot_state() -> dict[str, Any]:
    """Full robot state snapshot."""
    scene = get_scene()
    s = scene.state
    return {
        "busy": scene._algo_running,
        "base_pos": s["base_pos"][:2],
        "base_yaw": s["base_yaw"],
        "nav_state": s["nav_state"].value,
        "nav_target": s["nav_target"],
        "grasp_state": s["grasp_state"],
        "planner_state": scene._algo_planner.state.value if scene._algo_planner else "not_init",
    }


# ---------------------------------------------------------------------------
# Video feed (MJPEG)
# ---------------------------------------------------------------------------
@app.get("/api/video_feed")
def video_feed():
    """MJPEG video stream from the simulation camera."""
    scene = get_scene()

    def generate():
        while True:
            frame = scene.get_frame()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            import time

            time.sleep(0.05)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Stop all motion
# ---------------------------------------------------------------------------
@app.post("/api/stop")
def stop_all() -> dict[str, str]:
    """Cancel navigation and grasp without resetting the simulation."""
    logger.info("停止所有运动")
    get_scene().stop_all()
    return {"status": "stopped"}


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
@app.post("/api/reset")
def reset() -> dict[str, str]:
    get_scene().reset()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Backward-compatible aliases (old paths)
# ---------------------------------------------------------------------------
@app.post("/grasp")
def grasp_legacy():
    return grasp()


@app.get("/status")
def grasp_status_legacy():
    return grasp_status()


@app.get("/detect")
def detect_legacy():
    return detect()


@app.get("/video_feed")
def video_feed_legacy():
    return video_feed()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    import threading

    import uvicorn

    if _args.render:
        # GUI mode: GLFW needs the main thread, so the simulation loop runs
        # here and uvicorn runs in a daemon thread.
        global _scene
        config_path = os.environ.get("DTS_CONFIG") or None
        _scene = SimulationScene(config_path, render_mode="gui")
        _scene.enable_keyboard()

        def _serve():
            uvicorn.run(
                app,  # pass object directly — avoid re-import deadlock
                host=_args.host,
                port=_args.port,
                log_level="warning",
            )

        server_thread = threading.Thread(target=_serve, daemon=True)
        server_thread.start()
        # Let uvicorn finish startup, then block on the simulation loop.
        import time
        time.sleep(1.5)
        _scene.run()
    else:
        uvicorn.run(
            "execution.sim_mujoco.sim_task_server:app",
            host=_args.host,
            port=_args.port,
            log_level="info",
        )


if __name__ == "__main__":
    main()
