"""实机感知：D455 深度相机 + YOLO / HSV 检测。

从 pick_up_trash/live_view.py 移植。
"""

import os
import time

import numpy as np

from algorithms.perception.base import Detection, ObjectDetector


# ============================================================
# HSV 检测配置（默认，可覆盖）
# ============================================================

BLOCK_HSV_LOWER = (0, 0, 0)
BLOCK_HSV_UPPER = (180, 255, 60)
BLOCK_MIN_AREA = 500
BLOCK_MAX_AREA = 20000

CLOTH_HSV_LOWER = (85, 15, 60)
CLOTH_HSV_UPPER = (130, 180, 255)
CLOTH_MIN_AREA = 3000
CLOTH_MAX_AREA = 200000

BOTTLE_HSV_LOWER = (0, 60, 60)
BOTTLE_HSV_UPPER = (15, 255, 255)
BOTTLE_MIN_AREA = 200
BOTTLE_MAX_AREA = 30000

COLORS = {
    "block": (0, 0, 255),
    "cloth": (0, 255, 255),
    "bottle": (0, 200, 0),
}


# ============================================================
# RealSense 相机封装
# ============================================================

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
        self.intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        for _ in range(30):
            self._pipeline.wait_for_frames()

    def grab(self):
        """获取一帧对齐的 (color_image, depth_frame)。"""
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

    @staticmethod
    def deproject_pixel(cx, cy, depth_frame, intrinsics, roi_size=5):
        """像素坐标 → 相机 3D 坐标（中位数深度）。"""
        depth_img = np.asanyarray(depth_frame.get_data())
        h, w = depth_img.shape
        y1, y2 = max(0, cy - roi_size), min(h, cy + roi_size)
        x1, x2 = max(0, cx - roi_size), min(w, cx + roi_size)
        roi = depth_img[y1:y2, x1:x2].astype(np.float32) * 0.001
        valid = roi[(roi > 0.1) & (roi < 3.0)]
        if len(valid) == 0:
            return None
        depth_m = float(np.median(valid))
        import pyrealsense2 as rs
        return rs.rs2_deproject_pixel_to_point(intrinsics, [float(cx), float(cy)], depth_m)

    @staticmethod
    def depth_at_pixel(depth_frame, cx, cy, roi_size=5):
        """获取像素处的中位数深度值（米）。"""
        depth_img = np.asanyarray(depth_frame.get_data())
        h, w = depth_img.shape
        y1, y2 = max(0, cy - roi_size), min(h, cy + roi_size)
        x1, x2 = max(0, cx - roi_size), min(w, cx + roi_size)
        roi = depth_img[y1:y2, x1:x2].astype(float) * 0.001
        valid = roi[(roi > 0.1) & (roi < 1.5)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))


# ============================================================
# YOLO 检测器
# ============================================================

_YOLO_MODEL = None


def _get_yolo_model(classes=None):
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO
    print("加载 YOLOv8l-worldv2 模型...", flush=True)
    _YOLO_MODEL = YOLO("yolov8l-worldv2.pt")
    if classes:
        _YOLO_MODEL.set_classes(classes)
    print("  YOLO 模型就绪", flush=True)
    return _YOLO_MODEL


class YOLODetector(ObjectDetector):
    """YOLOv8-world 检测 + 深度对齐。"""

    def __init__(
        self,
        camera: RealSenseCamera,
        classes: list[str] | None = None,
        conf: float = 0.3,
        max_depth: float = 1.5,
    ):
        self._camera = camera
        self._model = _get_yolo_model(classes)
        self._conf = conf
        self._max_depth = max_depth

    def detect(self) -> list[Detection]:
        color, df = self._camera.grab()
        if color is None:
            return []

        results = self._model.predict(color, conf=self._conf, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf_val = float(box.conf[0])
                cls_name = self._model.names[int(box.cls[0])] if self._model.names else "object"

                d = RealSenseCamera.depth_at_pixel(df, cx, cy)
                if d is None or d > self._max_depth:
                    continue

                p3d = RealSenseCamera.deproject_pixel(cx, cy, df, self._camera.intrinsics)
                detections.append(Detection(
                    label=cls_name,
                    center_pixel=(cx, cy),
                    bbox=(x1, y1, w, h),
                    depth_m=d,
                    position_cam=np.array(p3d) if p3d else None,
                    confidence=conf_val,
                ))
        return detections


# ============================================================
# HSV 颜色检测器
# ============================================================

class HSVDetector(ObjectDetector):
    """传统 CV 颜色检测（block/bottle/cloth）。

    不需要 GPU，适合作为 YOLO 的 fallback。
    """

    TARGETS = {
        "block": (BLOCK_HSV_LOWER, BLOCK_HSV_UPPER, BLOCK_MIN_AREA, BLOCK_MAX_AREA),
        "cloth": (CLOTH_HSV_LOWER, CLOTH_HSV_UPPER, CLOTH_MIN_AREA, CLOTH_MAX_AREA),
        "bottle": (BOTTLE_HSV_LOWER, BOTTLE_HSV_UPPER, BOTTLE_MIN_AREA, BOTTLE_MAX_AREA),
    }

    def __init__(self, camera: RealSenseCamera, targets=None, max_depth=1.0):
        self._camera = camera
        self._targets = targets or ["bottle"]
        self._max_depth = max_depth

    def detect(self) -> list[Detection]:
        import cv2
        color, df = self._camera.grab()
        if color is None:
            return []

        results = []
        for target in self._targets:
            cfg = self.TARGETS.get(target)
            if cfg is None:
                continue
            lower, upper, min_a, max_a = cfg

            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            if target == "bottle":
                mask[:color.shape[0] // 2, :] = 0
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                kc = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
                mask = cv2.dilate(mask, kernel, iterations=2)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_a or area > max_a:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2
                d = RealSenseCamera.depth_at_pixel(df, cx, cy)
                if d is None or d > self._max_depth:
                    continue
                p3d = RealSenseCamera.deproject_pixel(cx, cy, df, self._camera.intrinsics)
                results.append(Detection(
                    label=target,
                    center_pixel=(cx, cy),
                    bbox=(x, y, w, h),
                    depth_m=d,
                    position_cam=np.array(p3d) if p3d else None,
                ))

        return results
