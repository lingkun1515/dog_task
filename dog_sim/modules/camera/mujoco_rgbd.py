"""MujocoCameraSim: 从共享 SimWorld 做离屏 RGBD 渲染，实现 CameraController 协议.

后台线程持续渲染，get_target() 返回最近检测结果。
支持两种检测模式:
  - "ground_truth": 直接从 MuJoCo body 位置投影（确定性，测试用）
  - "yolo": 对渲染帧跑 YOLO 检测（端到端验证感知链）
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...core.models import HealthStatus, TargetObservation, Vector3
from ..sim.world import SimWorld

logger = logging.getLogger(__name__)


class MujocoCameraSim:
    """MuJoCo RGBD 仿真相机，实现 CameraController 协议."""

    def __init__(self, config: Mapping[str, Any], world: SimWorld) -> None:
        self._world = world
        self._config = dict(config)
        self._detection_mode = self._config.get("detection_mode", "ground_truth")
        self._render_fps = self._config.get("render_fps", 30)
        self._camera_name = self._config.get("camera_name", "d455_sim")
        self._push_to_ui = self._config.get("push_to_ui", False)
        self._width = self._config.get("width", 848)
        self._height = self._config.get("height", 480)

        self._latest_observation: Optional[TargetObservation] = None
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._render_thread: Optional[threading.Thread] = None

        self._renderer = None
        self._cam_id: Optional[int] = None

    def healthcheck(self) -> HealthStatus:
        try:
            import mujoco
            self._world.camera_id(self._camera_name)
            return HealthStatus(ready=True, message=f"MujocoCameraSim ({self._detection_mode})")
        except Exception as e:
            return HealthStatus(ready=False, message=str(e))

    def get_target(self, *, require_stable: bool = False) -> Optional[TargetObservation]:
        """返回最近一帧检测到的目标."""
        if not self._running:
            self._start_render_thread()
            time.sleep(0.15)
        with self._lock:
            return self._latest_observation

    def get_raw_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取最近的 RGB 和深度帧（调试用）."""
        with self._lock:
            return self._latest_rgb, self._latest_depth

    def stop(self) -> None:
        self._running = False
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
            self._render_thread = None

    # ---- 内部渲染线程 ----

    def _start_render_thread(self) -> None:
        if self._running:
            return
        self._running = True
        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="mujoco_cam"
        )
        self._render_thread.start()

    def _render_loop(self) -> None:
        import mujoco

        self._cam_id = self._world.camera_id(self._camera_name)
        self._renderer = mujoco.Renderer(self._world.model, height=self._height, width=self._width)

        dt = 1.0 / self._render_fps
        while self._running:
            t0 = time.time()
            try:
                rgb, depth = self._render_frame()
                obs = self._detect(rgb, depth)
                with self._lock:
                    self._latest_rgb = rgb
                    self._latest_depth = depth
                    self._latest_observation = obs
                if self._push_to_ui and rgb is not None:
                    self._push_frame_to_ui(rgb)
            except Exception as e:
                logger.warning("Camera render error: %s", e)
            elapsed = time.time() - t0
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _render_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """渲染一帧 RGB + 深度."""
        import mujoco

        with self._world.read_lock():
            self._renderer.update_scene(self._world.data, camera=self._cam_id)
            rgb = self._renderer.render().copy()
            self._renderer.enable_depth_rendering()
            depth = self._renderer.render().copy()
            self._renderer.disable_depth_rendering()

        return rgb, depth

    def _detect(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[TargetObservation]:
        """检测目标."""
        if self._detection_mode == "ground_truth":
            return self._detect_ground_truth()
        elif self._detection_mode == "yolo":
            return self._detect_yolo(rgb, depth)
        return None

    def _detect_ground_truth(self) -> Optional[TargetObservation]:
        """通过 MuJoCo body 位置直接得到目标坐标（相机光学坐标系）.

        MuJoCo 相机坐标系: X-right, Y-up, Z-back (看向-Z).
        RealSense 光学坐标系: X-right, Y-down, Z-forward.
        转换: x_rs = x_mj, y_rs = -y_mj, z_rs = -z_mj
        """
        import mujoco

        model = self._world.model
        cam_id = self._cam_id

        cam_pos = self._world.data.cam_xpos[cam_id]
        cam_mat = self._world.data.cam_xmat[cam_id].reshape(3, 3)

        best_obs = None
        best_dist = float("inf")

        for i in range(model.nbody):
            if model.body_mocapid[i] >= 0:
                body_pos = self._world.data.xpos[i]
                pos_mj_cam = cam_mat.T @ (body_pos - cam_pos)

                z_forward = -pos_mj_cam[2]
                if z_forward < 0.05:
                    continue

                x_rs = float(pos_mj_cam[0])
                y_rs = float(-pos_mj_cam[1])
                z_rs = float(z_forward)

                dist = math.sqrt(x_rs**2 + y_rs**2 + z_rs**2)
                if dist < best_dist:
                    best_dist = dist
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "unknown"
                    best_obs = TargetObservation(
                        target_id=name,
                        class_name=name,
                        confidence=0.99,
                        position_m=Vector3(x=x_rs, y=y_rs, z=z_rs),
                        frame_id="camera_link",
                    )

        return best_obs

    def _detect_yolo(self, rgb: np.ndarray, depth: np.ndarray) -> Optional[TargetObservation]:
        """对渲染帧跑 YOLO 检测."""
        try:
            from .pipeline.detect import detect_targets
            results = detect_targets(rgb)
            if not results:
                return None

            best = results[0]
            cx, cy = int(best["center_x"]), int(best["center_y"])
            h, w = depth.shape[:2]
            if 0 <= cx < w and 0 <= cy < h:
                z = float(depth[cy, cx])
                if z > 0.05:
                    fx, fy, ppx, ppy = self._get_intrinsics()
                    x_cam = (cx - ppx) * z / fx
                    y_cam = (cy - ppy) * z / fy
                    class_name = best.get("class_name", "object")
                    return TargetObservation(
                        target_id=class_name,
                        class_name=class_name,
                        confidence=float(best.get("confidence", 0.8)),
                        position_m=Vector3(x=x_cam, y=y_cam, z=z),
                        frame_id="camera_link",
                    )
        except Exception as e:
            logger.warning("YOLO detection error: %s", e)
        return None

    def _get_intrinsics(self) -> Tuple[float, float, float, float]:
        """从 MuJoCo 相机参数推导内参."""
        model = self._world.model
        fovy_rad = model.cam_fovy[self._cam_id] * math.pi / 180.0
        fy = (self._height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx = self._width / 2.0
        cy = self._height / 2.0
        return fx, fy, cx, cy

    def _push_frame_to_ui(self, rgb: np.ndarray) -> None:
        """推送帧到 UI SharedFrameStream（如果 UI 模块可用）."""
        try:
            import sys, os
            ui_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "UI")
            if ui_dir not in sys.path:
                sys.path.insert(0, os.path.abspath(ui_dir))
            from camera_stream import get_stream
            import cv2
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            get_stream().push_frame(bgr)
        except ImportError:
            pass
        except Exception as e:
            logger.debug("UI push failed: %s", e)
