"""MuJoCo 仿真视频录制模块。

支持 headless EGL 模式下自主录制第三视角和第一视角视频。

用法：
    from scripts.record_video import VideoRecorder
    recorder = VideoRecorder(model, data, output_dir)
    recorder.capture_frame()  # 每个 sim step 调用
    recorder.save()  # 保存为 MP4
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import mujoco
import numpy as np

logger = logging.getLogger(__name__)


class VideoRecorder:
    """MuJoCo 仿真视频录制器。

    支持两种视角：
    - third_person: 跟踪 base_link 的第三视角
    - front_camera: 机器人前置相机（first_person）

    使用 ffmpeg 将帧序列编码为 H.264 MP4。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_dir: str | Path,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        cam_name: Optional[str] = None,
    ):
        self.model = model
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps

        # MuJoCo rendering context
        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._opt = mujoco.MjvOption()
        self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Camera setup
        self._cam = mujoco.MjvCamera()
        if cam_name:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._cam.fixedcamid = cam_id
            else:
                self._setup_tracking_camera()
        else:
            self._setup_tracking_camera()

        # Frame buffer
        self._frames: list[np.ndarray] = []
        self._frame_count = 0

    def _setup_tracking_camera(self):
        """设置跟踪 base_link 的第三视角。"""
        self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        track_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if track_body < 0:
            # Fallback to first body
            track_body = 0
        self._cam.trackbodyid = track_body
        self._cam.distance = 4.0
        self._cam.elevation = -25
        self._cam.azimuth = 135

    def capture_frame(self) -> np.ndarray:
        """捕获当前仿真状态的一帧 RGB 图像。

        Returns:
            np.ndarray: RGB 图像 (H, W, 3), uint8
        """
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjv_updateScene(
            self.model, self.data, self._opt, None, self._cam,
            mujoco.mjtCatBit.mjCAT_ALL, self._scene,
        )
        mujoco.mjr_render(viewport, self._scene, self._context)

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, viewport, self._context)
        rgb = np.flipud(rgb)  # MuJoCo renders bottom-up

        self._frames.append(rgb)
        self._frame_count += 1
        return rgb

    def save(self, filename: str = "video.mp4") -> Path:
        """将捕获的帧保存为 MP4 视频。

        Args:
            filename: 输出文件名

        Returns:
            Path: 保存的视频文件路径
        """
        if not self._frames:
            logger.warning("没有帧可保存")
            return None

        output_path = self.output_dir / filename

        # 使用 OpenCV 写入视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (self.width, self.height)
        )

        for frame in self._frames:
            # RGB -> BGR for OpenCV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()

        # 转换为 H.264 以确保兼容性
        h264_path = self.output_dir / filename.replace('.mp4', '_h264.mp4')
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', str(output_path),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                str(h264_path)
            ], capture_output=True, check=True)
            # 删除原始文件，使用 H.264 版本
            output_path.unlink()
            h264_path.rename(output_path)
            logger.info("视频已保存: %s (H.264, %d 帧)", output_path, self._frame_count)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("ffmpeg 转换失败，使用原始 MP4: %s", e)
            logger.info("视频已保存: %s (%d 帧)", output_path, self._frame_count)

        return output_path

    def reset(self):
        """清空帧缓冲。"""
        self._frames.clear()
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count


class MultiViewRecorder:
    """多视角录制器，同时录制第三视角和第一视角。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_dir: str | Path,
        front_cam_name: str = "front_cam",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.output_dir = Path(output_dir)

        # 第三视角
        self.third_person = VideoRecorder(
            model, data, output_dir,
            width=width, height=height, fps=fps,
        )

        # 第一视角（前置相机）
        self.front_camera = VideoRecorder(
            model, data, output_dir,
            width=width, height=height, fps=fps,
            cam_name=front_cam_name,
        )

    def capture_frame(self):
        """同时捕获两个视角的帧。"""
        self.third_person.capture_frame()
        self.front_camera.capture_frame()

    def save(self) -> dict[str, Path]:
        """保存两个视角的视频。

        Returns:
            dict: {"third_person": Path, "front_camera": Path}
        """
        return {
            "third_person": self.third_person.save("third_person.mp4"),
            "front_camera": self.front_camera.save("front_camera.mp4"),
        }

    def reset(self):
        """清空两个视角的帧缓冲。"""
        self.third_person.reset()
        self.front_camera.reset()
