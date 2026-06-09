"""Off-screen MuJoCo RGB-D camera renderer matching RealSense D455 specs.

D455 matching:
  - RGB FOV: 90x65(HxV), Depth FOV: 87x58
  - We match the RGB sensor: fovy=65 at 640x480 gives HFOV~86
  - Pixel-aligned RGB + Depth (simultaneous render, same viewport)
  - Depth in meters, same coordinate frame as RGB
"""

from __future__ import annotations

import logging
import threading

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


class SimRGBDCamera:
    """RGB-D camera wrapping a MuJoCo fixed camera with D455-matching FOV.

    Renders both RGB and depth simultaneously via mjr_readPixels.
    Depth buffer is converted from NDC to linear meters.

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
        cam_name: str = "front_cam",
    ):
        self.model = model
        self.data = data
        self.width = width
        self.height = height

        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._cam = mujoco.MjvCamera()
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id >= 0:
            self._cam_id = cam_id
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._cam.fixedcamid = cam_id
        else:
            self._cam_id = -1
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

        self._context = context
        self._own_context = False

        if self._context is None:
            try:
                self._context = mujoco.MjrContext(
                    model, mujoco.mjtFontScale.mjFONTSCALE_150
                )
                self._own_context = True
            except Exception:
                self._context = None

        self._ok = self._context is not None

        # Intrinsics matching D455 (fovy=65 at 640x480)
        self._update_intrinsics()

        self._latest_rgb: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None  # meters, float32
        self._latest_frame: bytes | None = None
        self._lock = threading.Lock()

    def _update_intrinsics(self) -> None:
        """Compute camera intrinsics from MJCF parameters or D455 defaults."""
        if self._cam_id >= 0:
            fovy = float(self.model.cam_fovy[self._cam_id])
            res = self.model.cam_resolution[self._cam_id]
            cam_w, cam_h = int(res[0]), int(res[1])
            # MJCF cam_resolution defaults to [0,0] when not specified
            if cam_w <= 0 or cam_h <= 0:
                cam_w, cam_h = self.width, self.height
        else:
            fovy = 65.0
            cam_w, cam_h = self.width, self.height

        fy = (cam_h / 2.0) / np.tan(np.deg2rad(fovy / 2.0))
        fx = fy  # square pixels
        self.intrinsics = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cam_w / 2.0),
            "cy": float(cam_h / 2.0),
            "width": cam_w,
            "height": cam_h,
            "fovy": fovy,
        }
        logger.info(
            "[camera] intrinsics: %dx%d fovy=%.1f° fx=%.1f fy=%.1f cx=%.1f cy=%.1f (render=%dx%d)",
            cam_w, cam_h, fovy, fx, fy, cam_w / 2.0, cam_h / 2.0,
            self.width, self.height,
        )

    @property
    def ok(self) -> bool:
        return self._ok

    def render(self) -> None:
        """Render RGB + depth for current frame. No-op if unavailable."""
        if not self._ok or self._context is None:
            return

        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjv_updateScene(
            self.model, self.data, self._opt, None, self._cam,
            mujoco.mjtCatBit.mjCAT_ALL, self._scene,
        )
        mujoco.mjr_render(viewport, self._scene, self._context)

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth_buf = np.zeros((self.height, self.width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth_buf, viewport, self._context)
        rgb = np.flipud(rgb)
        depth_buf = np.flipud(depth_buf)

        depth_m = self._depth_to_meters(depth_buf)

        jpeg = self._encode_jpeg(rgb)
        if jpeg is None:
            return

        with self._lock:
            self._latest_rgb = rgb
            self._latest_depth = depth_m
            self._latest_frame = jpeg

    def _depth_to_meters(self, depth_buf: np.ndarray) -> np.ndarray:
        """Convert MuJoCo depth buffer (NDC [0,1]) to linear depth in meters.

        MuJoCo depth buffer: 0 = near plane, 1 = far plane.
        Formula: z = znear * zfar / (zfar - depth * (zfar - znear))
        """
        extent = self.model.stat.extent
        znear = float(self.model.vis.map.znear) * extent
        zfar = float(self.model.vis.map.zfar) * extent
        # depth_buf values in [0,1]
        denom = zfar - depth_buf * (zfar - znear)
        denom = np.maximum(denom, 0.001)
        return (znear * zfar / denom).astype(np.float32)

    def get_rgb(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_rgb is not None:
                return self._latest_rgb.copy()
            return None

    def get_depth(self) -> np.ndarray | None:
        """Get latest depth image in meters (float32, (H, W))."""
        with self._lock:
            if self._latest_depth is not None:
                return self._latest_depth.copy()
            return None

    def get_intrinsics(self) -> dict:
        """Get camera intrinsics matching D455 format."""
        return dict(self.intrinsics)

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def set_rgb_frame(self, rgb: np.ndarray) -> None:
        """Replace the stored JPEG frame with an annotated RGB (for video stream only).

        注意：不覆盖 _latest_rgb，避免检测器拿到标注过的图产生反馈环。
        """
        jpeg = self._encode_jpeg(rgb)
        if jpeg is None:
            return
        with self._lock:
            self._latest_frame = jpeg

    @staticmethod
    def _encode_jpeg(rgb: np.ndarray) -> bytes | None:
        try:
            from io import BytesIO
            from PIL import Image
            img = Image.fromarray(rgb)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        except ImportError:
            return None


# Backward compatibility alias
SimulationCamera = SimRGBDCamera
