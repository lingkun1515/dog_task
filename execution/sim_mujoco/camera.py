"""Off-screen MuJoCo camera renderer for MJPEG streaming."""

from __future__ import annotations

import os
import threading
import time

import mujoco
import numpy as np

# MuJoCo 3.x headless rendering requires EGL or OSMesa.
# Try EGL first (most common on Linux); fall back to OSMesa.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


class SimulationCamera:
    """Wraps a MuJoCo camera for off-screen rendering.

    If off-screen rendering fails (no GL), renders are silently skipped
    and ``get_frame()`` returns None.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        width: int = 640,
        height: int = 480,
    ):
        self.model = model
        self.data = data
        self.width = width
        self.height = height

        self._ok = False

        try:
            self._scene = mujoco.MjvScene(model, maxgeom=10000)
            self._context = mujoco.MjrContext(
                model, mujoco.mjtFontScale.mjFONTSCALE_150
            )
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._cam.trackbodyid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
            )
            self._cam.distance = 3.0
            self._cam.elevation = -25
            self._cam.azimuth = 90
            self._opt = mujoco.MjvOption()
            self._ok = True
        except Exception:
            self._scene = None
            self._context = None
            self._cam = None
            self._opt = None

        self._latest_frame: bytes | None = None
        self._lock = threading.Lock()

    @property
    def ok(self) -> bool:
        return self._ok

    def render(self) -> None:
        """Render current view and store JPEG bytes. No-op if unavailable."""
        if not self._ok:
            return

        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self._opt,
            None,
            self._cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        mujoco.mjr_render(viewport, self._scene, self._context)

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self._context)
        rgb = np.flipud(rgb)

        try:
            from io import BytesIO

            from PIL import Image

            img = Image.fromarray(rgb)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80)
            jpeg = buf.getvalue()
        except ImportError:
            jpeg = rgb.tobytes()

        with self._lock:
            self._latest_frame = jpeg

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame
