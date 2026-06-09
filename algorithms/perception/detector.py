"""统一目标检测器：Sim/Real 共用同一套 YOLO / HSV 检测逻辑。

与 Sim/Real 的唯一区别仅在于帧来源：
  Sim:  SimRGBDCamera.render() → get_rgb() + get_depth()
  Real: D455.grab() → get_rgb() + get_depth()
        ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        来源不同          接口相同（都是 numpy RGB + depth meters）

一旦拿到 color (H,W,3) + depth_meters (H,W) float32 后，
YOLODetector / HSVDetector 的 detect() 逻辑 100% 相同。
"""

from __future__ import annotations

import os

import numpy as np

from algorithms.perception.base import Detection, ObjectDetector
from algorithms.perception.depth_utils import depth_at_pixel, deproject_pixel


# ============================================================
# HSV 检测配置
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

_COLORS = {
    "block": (0, 0, 255),
    "cloth": (0, 255, 255),
    "bottle": (0, 200, 0),
}


# ============================================================
# YOLO 模型加载
# ============================================================

_YOLO_MODEL = None


def _get_yolo_model(classes=None):
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO
    print("加载 YOLOv8l-worldv2 模型...", flush=True)
    _YOLO_MODEL = YOLO("yolov8l-worldv2.pt")
    if classes:
        _YOLO_MODEL.set_classes(classes)
    print("  YOLO 模型就绪", flush=True)
    return _YOLO_MODEL


# ============================================================
# 统一 YOLO 检测器 — Sim/Real 共用
# ============================================================

class YOLODetector(ObjectDetector):
    """YOLOv8-world 检测 + RGB-D 深度对齐。

    Sim/Real 共用同一个类。唯一区别是传入的 camera 不同：
      Sim:  camera = SimRGBDCamera(...)
      Real: camera = RealSenseCamera(...)
    只要 camera 提供 get_rgb() / get_depth() / get_intrinsics()，
    detect() 逻辑完全一致。
    """

    def __init__(
        self,
        camera,  # SimRGBDCamera | RealSenseCamera（必须有 get_rgb/get_depth/get_intrinsics）
        classes: list[str] | None = None,
        conf: float = 0.3,
        max_depth: float = 1.5,
    ):
        self._camera = camera
        self._model = _get_yolo_model(classes)
        self._conf = conf
        self._max_depth = max_depth
        self._intrinsics = camera.get_intrinsics()

    def detect(self) -> list[Detection]:
        # ★ 唯一区别：Sim=render 后的缓存帧，Real=D455 硬件帧 ★
        color = self._camera.get_rgb()
        if color is None:
            return []

        depth_m = self._camera.get_depth()

        # ↓ 以下 Sim/Real 100% 相同 ↓
        results = self._model.predict(color, conf=self._conf, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf_val = float(box.conf[0])
                cls_name = self._model.names[int(box.cls[0])] if self._model.names else "object"

                d = depth_at_pixel(depth_m, cx, cy)
                if d is None or d > self._max_depth:
                    continue

                p3d = deproject_pixel(cx, cy, d, self._intrinsics)
                detections.append(Detection(
                    label=cls_name,
                    center_pixel=(cx, cy),
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    depth_m=d,
                    position_cam=np.array(p3d),
                    confidence=conf_val,
                ))
        return detections


# ============================================================
# 统一 HSV 检测器 — Sim/Real 共用
# ============================================================

class HSVDetector(ObjectDetector):
    """传统 CV 颜色检测（block/bottle/cloth）。

    Sim/Real 共用同一个类。
    不需要 GPU，适合作为 YOLO 的 fallback。
    """

    TARGETS = {
        "block": (BLOCK_HSV_LOWER, BLOCK_HSV_UPPER, BLOCK_MIN_AREA, BLOCK_MAX_AREA),
        "cloth": (CLOTH_HSV_LOWER, CLOTH_HSV_UPPER, CLOTH_MIN_AREA, CLOTH_MAX_AREA),
        "bottle": (BOTTLE_HSV_LOWER, BOTTLE_HSV_UPPER, BOTTLE_MIN_AREA, BOTTLE_MAX_AREA),
    }

    def __init__(self, camera, targets=None, max_depth=1.0, color_order: str = "rgb"):
        """Args:
            camera: 帧来源（需提供 get_rgb/get_depth/get_intrinsics）
            targets: 要检测的目标类型列表
            max_depth: 最大检测深度
            color_order: 'rgb'（Sim 渲染输出）或 'bgr'（Real D455 输出）
        """
        self._camera = camera
        self._targets = targets or ["bottle"]
        self._max_depth = max_depth
        self._intrinsics = camera.get_intrinsics()
        self._cvt_code = cv2.COLOR_RGB2HSV if color_order == "rgb" else cv2.COLOR_BGR2HSV

    def detect(self) -> list[Detection]:
        import cv2

        # ★ 唯一区别：Sim=缓存帧(RGB)，Real=硬件帧(BGR) ★
        color = self._camera.get_rgb()
        if color is None:
            return []

        depth_m = self._camera.get_depth()

        # ↓ 以下 Sim/Real 100% 相同 ↓
        hsv = cv2.cvtColor(color, self._cvt_code)
        results = []
        for target in self._targets:
            cfg = self.TARGETS.get(target)
            if cfg is None:
                continue
            lower, upper, min_a, max_a = cfg

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

                d = depth_at_pixel(depth_m, cx, cy)
                if d is None or d > self._max_depth:
                    continue

                p3d = deproject_pixel(cx, cy, d, self._intrinsics)
                results.append(Detection(
                    label=target,
                    center_pixel=(cx, cy),
                    bbox=(x, y, w, h),
                    depth_m=d,
                    position_cam=np.array(p3d),
                    confidence=0.9,
                ))

        return results

    def annotate_frame(self, color: np.ndarray, depth_m: np.ndarray,
                       targets: list[dict] | None = None) -> np.ndarray:
        """在 RGB 图像上绘制检测框（用于相机推流）。

        Args:
            color: (H,W,3) uint8 图像
            depth_m: (H,W) float32 深度图（米）
            targets: [{"lower": (...), "upper": (...), "label": "..."}]
        """
        import cv2

        annotated = color.copy()
        if depth_m is None:
            return annotated

        targets = targets or []
        for cfg in targets:
            label = cfg["label"]
            lower = np.array(cfg["lower"])
            upper = np.array(cfg["upper"])
            hsv = cv2.cvtColor(color, self._cvt_code)
            mask = cv2.inRange(hsv, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 50:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2

                d = depth_at_pixel(depth_m, cx, cy)
                depth_label = f"{d:.2f}m" if d is not None else "?m"
                label_text = f"{label} {depth_label}"

                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(annotated, label_text, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(annotated, label_text, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        return annotated


# 需要 cv2，但在此处不导入（在方法内延迟导入以避免依赖问题）
import cv2  # noqa: E402
