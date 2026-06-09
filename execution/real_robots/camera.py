"""D455 RealSense 相机封装 — 执行层代码（依赖 pyrealsource2 SDK）。

提供统一的 get_rgb() / get_depth() / get_intrinsics() 接口，
与 SimRGBDCamera 接口一致，YOLODetector/HSVDetector 直接使用。
"""

import time

import numpy as np


class RealSenseCamera:
    """D455 RealSense 相机封装（延迟导入 pyrealsense2）。"""

    def __init__(self, width=640, height=480, fps=30):
        import pyrealsense2 as rs
        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        rs_intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.intrinsics = {
            "fx": rs_intrinsics.fx,
            "fy": rs_intrinsics.fy,
            "cx": rs_intrinsics.ppx,
            "cy": rs_intrinsics.ppy,
            "width": rs_intrinsics.width,
            "height": rs_intrinsics.height,
        }
        for _ in range(30):
            self._pipeline.wait_for_frames()

    # ------------------------------------------------------------------
    # 统一接口（与 SimRGBDCamera 一致）
    # ------------------------------------------------------------------
    def get_rgb(self) -> np.ndarray | None:
        """获取当前帧 BGR 图像。"""
        color, _ = self.grab()
        return color

    def get_depth(self) -> np.ndarray | None:
        """获取当前深度图为 numpy 数组（米，float32）。"""
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        df = aligned.get_depth_frame()
        if not df:
            return None
        depth_mm = np.asanyarray(df.get_data())
        return depth_mm.astype(np.float32) * 0.001

    def get_intrinsics(self) -> dict:
        """获取相机内参字典。"""
        return dict(self.intrinsics)

    # ------------------------------------------------------------------
    # 原始接口（向后兼容）
    # ------------------------------------------------------------------
    def grab(self):
        """获取一帧对齐的 (color_bgr, depth_frame)。

        Returns:
            (color: (H,W,3) uint8 BGR, depth_frame: rs.depth_frame)
        """
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        if not cf or not df:
            return None, None
        return np.asanyarray(cf.get_data()), df

    def stop(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass
