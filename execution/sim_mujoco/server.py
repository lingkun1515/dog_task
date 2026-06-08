"""FastAPI server for MuJoCo simulation execution service.

Start with:
    python -m execution.sim_mujoco.server
    python -m execution.sim_mujoco.server --port 8100 --no-render
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so scheduler imports work when running standalone
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from execution.sim_mujoco.scene import SimulationScene

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
    config_path = os.environ.get("SIM_CONFIG", None)
    _scene = SimulationScene(config_path)
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

    Body: {"x": float, "y": float}
    """
    scene = get_scene()
    x = float(body.get("x", 0))
    y = float(body.get("y", 0))
    scene.navigate_to(x, y)
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
    scene.start_grasp()
    return {"status": "accepted"}


@app.get("/api/grasp/status")
def grasp_status() -> dict[str, Any]:
    scene = get_scene()
    return {"status": scene.state["grasp_state"].value}


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------
@app.get("/api/state")
def robot_state() -> dict[str, Any]:
    """Full robot state snapshot."""
    return get_scene().state


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
# Reset
# ---------------------------------------------------------------------------
@app.post("/api/reset")
def reset() -> dict[str, str]:
    get_scene().reset()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DogTask MuJoCo simulation server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8100, help="Bind port")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "execution.sim_mujoco.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
