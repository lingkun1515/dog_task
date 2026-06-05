"""UI 相机帧共享模块

不再独立操控 RealSense 硬件, 而是接收主流程(D455Camera)推送的已渲染帧
(RGB + YOLO 检测结果), 通过 MJPEG 推送到前端 UI 的 "B 点目标确认画面".

使用方式:
  主流程侧(d455.py 或编排器)调用 push_frame(bgr_image) 推送帧;
  UI server 端通过 get_stream().generate_mjpeg() 输出 MJPEG 流.
"""

from __future__ import annotations

import threading
import time
from typing import Generator

import cv2
import numpy as np


class SharedFrameStream:
    """主流程与 UI 之间的帧共享通道(无锁竞争相机硬件)"""

    def __init__(self, target_fps: int = 15):
        self._target_fps = target_fps
        self._lock = threading.Lock()
        self._frame_jpeg: bytes | None = None
        self._frame_ts: float = 0.0
        self._connected = False

    @property
    def connected(self) -> bool:
        """是否有帧数据在推送(主流程是否活跃)"""
        with self._lock:
            if not self._connected:
                return False
            # 超过5秒没有新帧才视为断连(主流程可能在做IK计算等耗时操作)
            if time.time() - self._frame_ts > 5.0:
                self._connected = False
            return self._connected

    def push_frame(self, bgr_image: np.ndarray, quality: int = 75) -> None:
        """主流程调用: 推送一帧 BGR 图像(已含 YOLO 检测标注)

        Args:
            bgr_image: OpenCV BGR格式图像(可以是 visualization 模块输出的渲染帧)
            quality: JPEG 压缩质量(1-100)
        """
        _, jpeg = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        with self._lock:
            self._frame_jpeg = jpeg.tobytes()
            self._frame_ts = time.time()
            self._connected = True

    def push_jpeg(self, jpeg_bytes: bytes) -> None:
        """直接推送已编码的 JPEG 数据"""
        with self._lock:
            self._frame_jpeg = jpeg_bytes
            self._frame_ts = time.time()
            self._connected = True

    def get_frame(self) -> bytes | None:
        """获取当前帧的 JPEG 数据"""
        with self._lock:
            return self._frame_jpeg

    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """MJPEG 流生成器, 供 HTTP 端点使用

        即使暂时没有新帧, 也持续重发最后一帧保持流不中断,
        直到主流程主动 disconnect 或超时.
        """
        interval = 1.0 / self._target_fps
        last_sent: bytes | None = None
        while True:
            frame_data = self.get_frame()
            # 有新帧就用新帧, 否则重发旧帧保持流存活
            to_send = frame_data or last_sent
            if to_send:
                last_sent = to_send
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + to_send + b"\r\n"
                )
            time.sleep(interval)

    def disconnect(self) -> None:
        """标记为断连"""
        with self._lock:
            self._connected = False
            self._frame_jpeg = None


# 全局单例
_stream_instance: SharedFrameStream | None = None
_stream_lock = threading.Lock()


def get_stream() -> SharedFrameStream:
    """获取全局共享帧流单例"""
    global _stream_instance
    with _stream_lock:
        if _stream_instance is None:
            _stream_instance = SharedFrameStream()
        return _stream_instance
