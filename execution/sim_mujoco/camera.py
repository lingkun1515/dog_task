"""Off-screen MuJoCo camera renderer for MJPEG streaming."""

from __future__ import annotations

import threading

import mujoco
import numpy as np


class SimulationCamera:
    """Wraps a MuJoCo camera for off-screen rendering.

    In GUI mode, re-uses the Viewer's MjrContext (shared GL context).
    In headless mode without EGL/OSMesa, renders are silently skipped.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        width: int = 640,
        height: int = 480,
        context: mujoco.MjrContext | None = None,
    ):
        self.model = model
        self.data = data
        self.width = width
        self.height = height

        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._cam = mujoco.MjvCamera()
        # Use the body-fixed front_cam if present, otherwise fall back to tracking
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")
        if cam_id >= 0:
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._cam.fixedcamid = cam_id
        else:
            self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            track_body = -1
            for name in ("base_link", "base", "trunk"):
                track_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if track_body >= 0:
                    break
            self._cam.trackbodyid = max(track_body, 0)
            self._cam.distance = 3.0
            self._cam.elevation = -25
            self._cam.azimuth = 90
        self._opt = mujoco.MjvOption()

        self._context = context  # shared from Viewer in GUI mode
        self._own_context = False

        if self._context is None:
            # Try to create our own off-screen context (headless EGL/OSMesa)
            try:
                self._context = mujoco.MjrContext(
                    model, mujoco.mjtFontScale.mjFONTSCALE_150
                )
                self._own_context = True
            except Exception:
                self._context = None

        self._ok = self._context is not None

        self._latest_frame: bytes | None = None
        self._lock = threading.Lock()

    @property
    def ok(self) -> bool:
        return self._ok

    def render(self) -> None:
        """Render current view and store JPEG bytes. No-op if unavailable."""
        if not self._ok or self._context is None:
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
            return

        with self._lock:
            self._latest_frame = jpeg

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame
