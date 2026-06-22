"""统一感知模块：Sim/Real 共用的目标检测器。

Sim 和 Real 的唯一区别仅在于帧来源：
  Sim:  SimRGBDCamera.render() → get_rgb() + get_depth()
  Real: D455.grab() → get_rgb() + get_depth()
        ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        来源不同          接口相同（都是 numpy RGB + depth meters）

一旦拿到 color (H,W,3) + depth_meters (H,W) float32 后，
所有检测逻辑 100% 相同。

SimObjectDetector 是 MuJoCo-specific 的组合检测器：
  YOLO → HSV fallback → xpos 兜底 + annotate_frame 可视化。
"""

from __future__ import annotations

import os

import numpy as np

import logging

from algorithms.perception.base import Detection, ObjectDetector
from algorithms.perception.depth_utils import depth_at_pixel, deproject_pixel

logger = logging.getLogger(__name__)


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
    logger.info("加载 YOLOv8l-worldv2 模型...")
    _YOLO_MODEL = YOLO("yolov8l-worldv2.pt")
    if classes:
        _YOLO_MODEL.set_classes(classes)
    logger.info("YOLO 模型就绪")
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
        camera,  # SimRGBDCamera | RealSenseCamera
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
        color = self._camera.get_rgb()
        if color is None:
            return []

        depth_m = self._camera.get_depth()

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

        color = self._camera.get_rgb()
        if color is None:
            return []

        depth_m = self._camera.get_depth()

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
        """在 RGB 图像上绘制检测框（用于相机推流）。"""
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


# ============================================================================
# SimObjectDetector — MuJoCo 组合检测器（YOLO → HSV → xpos 兜底）
# ============================================================================

class SimObjectDetector(ObjectDetector):
    """Sim 专用组合检测器：YOLO → HSV → xpos 兜底。

    向后兼容旧 __init__ 签名，同时提供 annotate_frame 用于相机推流。
    """

    def __init__(
        self,
        model,  # mujoco.MjModel
        data,   # mujoco.MjData
        camera,  # SimRGBDCamera
        target_body_names: list[str],
        labels: list[str] | None = None,
        target_colors: list[dict] | None = None,
        cam_name: str = "front_cam",
    ):
        import mujoco as _mj

        self._model = model
        self._data = data
        self._camera = camera
        self._labels = labels or target_body_names
        self._cam_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_CAMERA, cam_name)

        # 目标 MuJoCo body ID（xpos 兜底用）— 可通过 set_targets 动态切换
        self._target_body_ids: list[int] = []
        self._target_labels: list[str] = []
        self._target_body_names: list[str] = list(target_body_names)
        self._refresh_target_ids()

        # HSV 检测默认配置（红+黄，覆盖常见目标颜色）
        _default_label = self._labels[0] if self._labels else "object"
        self._target_colors = target_colors or [
            {"lower": (0, 100, 50), "upper": (10, 255, 255), "label": _default_label},
            {"lower": (170, 100, 50), "upper": (180, 255, 255), "label": _default_label},
            {"lower": (20, 80, 80), "upper": (30, 255, 255), "label": _default_label},
        ]

        # 统一的 YOLO / HSV 检测器（共用类）
        self._yolo_detector: YOLODetector | None = None
        self._hsv_detector: HSVDetector | None = None

        self._intrinsics = camera.get_intrinsics()

        # 帧标注检测缓存（避免每帧都跑完整管线）
        self._cached_detections: list[Detection] = []
        self._cache_time: float = 0.0
        self._cache_interval: float = 0.5

    # ------------------------------------------------------------------
    # 动态目标切换（多任务场景切换时由 caller 调用）
    # ------------------------------------------------------------------
    def _refresh_target_ids(self) -> None:
        """根据当前 self._target_body_names 重建 body ID 列表。"""
        import mujoco as _mj
        self._target_body_ids = []
        self._target_labels = []
        for i, name in enumerate(self._target_body_names):
            bid = _mj.mj_name2id(self._model, _mj.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._target_body_ids.append(bid)
                self._target_labels.append(
                    self._labels[i] if i < len(self._labels) else name
                )
            else:
                logger.warning("[detector] body %r 不在模型中，跳过", name)

    def set_targets(self, body_names: list[str], labels: list[str] | None = None) -> None:
        """切换感知器跟踪的目标 body（场景切换时调用）。

        Args:
            body_names: 新的目标 body 名列表
            labels: 对应标签（与 body_names 等长）；缺省沿用 body 名
        """
        self._target_body_names = list(body_names)
        self._labels = list(labels) if labels is not None else list(body_names)
        self._refresh_target_ids()
        # 清除检测缓存，确保下一次 detect 用新目标
        self._cached_detections = []
        self._cache_time = 0.0
        logger.info("[detector] 目标已切换: %s", body_names)

    # ------------------------------------------------------------------
    # ObjectDetector 接口
    # ------------------------------------------------------------------
    def detect(self) -> list[Detection]:
        """检测流水线：YOLO → HSV → xpos 兜底。"""
        rgb = self._camera.get_rgb()
        if rgb is None:
            logger.debug("[detect] 无 RGB 帧，回退到 xpos")
            return self._detect_from_xpos()

        logger.debug("[detect] RGB 帧尺寸: %s dtype=%s", rgb.shape, rgb.dtype)

        # 1. YOLO
        dets = self._detect_yolo()
        if dets:
            for d in dets:
                logger.info("[detect/YOLO] %s conf=%.2f pixel=(%d,%d) bbox=(%d,%d,%d,%d) depth=%.3fm pos_cam=(%.3f,%.3f,%.3f)",
                            d.label, d.confidence,
                            d.center_pixel[0], d.center_pixel[1],
                            d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
                            d.depth_m,
                            d.position_cam[0], d.position_cam[1], d.position_cam[2])
            return dets

        # 2. HSV fallback
        dets = self._detect_hsv()
        if dets:
            # for d in dets:
            #     logger.info("[detect/HSV] %s pixel=(%d,%d) bbox=(%d,%d,%d,%d) depth=%.3fm pos_cam=(%.3f,%.3f,%.3f)",
            #                 d.label,
            #                 d.center_pixel[0], d.center_pixel[1],
            #                 d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
            #                 d.depth_m,
            #                 d.position_cam[0], d.position_cam[1], d.position_cam[2])
            return dets

        # 3. xpos 兜底
        dets = self._detect_from_xpos()
        # if dets:
        #     for d in dets:
        #         logger.info("[detect/xpos] %s pixel=(%d,%d) bbox=(%d,%d,%d,%d) depth=%.3fm pos_cam=(%.3f,%.3f,%.3f)",
        #                     d.label,
        #                     d.center_pixel[0], d.center_pixel[1],
        #                     d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3],
        #                     d.depth_m,
        #                     d.position_cam[0], d.position_cam[1], d.position_cam[2])
        # else:
        #     logger.warning("[detect] 全部管线均未检测到目标")
        return dets

    def _detect_yolo(self) -> list[Detection]:
        if self._yolo_detector is None:
            try:
                self._yolo_detector = YOLODetector(
                    camera=self._camera,
                    conf=0.3,
                )
            except Exception:
                return []
        try:
            return self._yolo_detector.detect()
        except Exception:
            return []

    def _detect_hsv(self) -> list[Detection]:
        """HSV 颜色检测（使用 _target_colors 配置，直接内联而非委托 HSVDetector）。"""
        rgb = self._camera.get_rgb()
        if rgb is None:
            return []
        depth_m = self._camera.get_depth()

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        results = []

        for cfg in self._target_colors:
            lower = np.array(cfg["lower"], dtype=np.uint8)
            upper = np.array(cfg["upper"], dtype=np.uint8)
            label = cfg["label"]

            mask = cv2.inRange(hsv, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 100 or area > 50000:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2

                d = depth_at_pixel(depth_m, cx, cy) if depth_m is not None else None
                if d is None or d > 2.0:
                    continue

                p3d = deproject_pixel(cx, cy, d, self._intrinsics)
                results.append(Detection(
                    label=label,
                    center_pixel=(cx, cy),
                    bbox=(x, y, w, h),
                    depth_m=d,
                    position_cam=np.array(p3d),
                    confidence=0.85,
                ))

        return results

    # ------------------------------------------------------------------
    # 图像标注（用于相机推流）
    # ------------------------------------------------------------------
    def annotate_frame(self, rgb: np.ndarray) -> np.ndarray:
        """在 RGB 图像上绘制检测框（使用完整 detect 管线的缓存结果）。

        没有检测结果时直接返回原图，不画任何标注。
        """
        import time

        import cv2

        now = time.time()
        if now - self._cache_time >= self._cache_interval:
            try:
                self._cached_detections = self.detect()
            except Exception as e:
                logger.error("[annotate] detect 异常: %s", e)
                self._cached_detections = []
            self._cache_time = now

        if not self._cached_detections:
            return rgb

        annotated = rgb.copy()
        img_h, img_w = annotated.shape[:2]

        for det in self._cached_detections:
            x, y, w, h = det.bbox
            cx, cy = det.center_pixel

            if not (0 <= cx < img_w and 0 <= cy < img_h):
                logger.debug("[annotate] %s pixel=(%d,%d) 超出图像边界，跳过绘制", det.label, cx, cy)
                continue

            depth_label = f"{det.depth_m:.2f}m" if det.depth_m > 0 else "?m"
            conf_label = f"{det.confidence:.0%}" if det.confidence < 1.0 else ""
            label_text = f"{det.label} {depth_label} {conf_label}".strip()

            logger.debug(
                "[annotate] %s pixel=(%d,%d) bbox=(%d,%d,%d,%d) depth=%.3fm",
                det.label, cx, cy, x, y, w, h, det.depth_m,
            )

            color = _COLORS.get(det.label, (0, 255, 0))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.circle(annotated, (cx, cy), 4, color, -1)
            cv2.putText(annotated, label_text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(annotated, label_text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        return annotated

    # ------------------------------------------------------------------
    # 兜底：xpos 直读（MuJoCo 真值）
    # ------------------------------------------------------------------
    def _detect_from_xpos(self) -> list[Detection]:
        """兜底：直接从 MuJoCo xpos 读取目标位置（上帝视角）。

        Sim ground truth：始终返回检测结果（position_cam 始终有效）。
        当 depth <= 0（球在相机后方，坐下太近时）仍返回，pixel 用 (-1,-1) 占位。
        """
        img_w = self._intrinsics.get("width", 640)
        img_h = self._intrinsics.get("height", 480)
        fx = self._intrinsics["fx"]
        fy = self._intrinsics["fy"]
        cxi = self._intrinsics["cx"]
        cyi = self._intrinsics["cy"]

        detections = []
        for bid, label in zip(self._target_body_ids, self._target_labels):
            world_pos = self._data.xpos[bid].copy()
            pos_cam = self._world_to_cam(world_pos)
            depth = float(pos_cam[2])

            if depth <= 0.01:
                # 球在相机后方（坐下时离球太近）。Sim ground truth 仍返回，
                # position_cam 有效（cam_to_arm 刚体变换不受影响），
                # pixel 用 (-1,-1) 占位（规划器不使用像素坐标）。
                logger.info("[xpos] %s 在相机后方 (depth=%.3f)，sim ground truth 仍返回", label, depth)
                detections.append(Detection(
                    label=label,
                    center_pixel=(-1, -1),
                    bbox=(-1, -1, 0, 0),
                    depth_m=abs(depth) if abs(depth) > 0.001 else 0.001,
                    position_cam=pos_cam,
                    confidence=1.0,
                ))
                continue

            px = int(fx * pos_cam[0] / depth + cxi)
            py = int(fy * pos_cam[1] / depth + cyi)

            # Sim ground truth: always return detection even if outside camera FOV
            # (position_cam is valid regardless of pixel projection)
            outside_fov = not (0 <= px < img_w and 0 <= py < img_h)
            if outside_fov:
                logger.debug("[xpos] %s pixel=(%d,%d) 超出图像 %dx%d，sim ground truth 仍返回",
                             label, px, py, img_w, img_h)

            logger.debug(
                "[xpos] %s world=(%.3f,%.3f,%.3f) cam_cv=(%.3f,%.3f,%.3f) → pixel=(%d,%d) depth=%.3fm",
                label, world_pos[0], world_pos[1], world_pos[2],
                pos_cam[0], pos_cam[1], pos_cam[2], px, py, depth,
            )

            bbox_half = max(10, int(20 / depth))
            detections.append(Detection(
                label=label,
                center_pixel=(px, py),
                bbox=(px - bbox_half, py - bbox_half, bbox_half * 2, bbox_half * 2),
                depth_m=depth,
                position_cam=pos_cam,
                confidence=1.0,
            ))
        return detections

    def _world_to_cam(self, world_pos: np.ndarray) -> np.ndarray:
        """世界坐标 → OpenCV 相机坐标（X 右、Y 下、Z 前）。"""
        cam_xpos = self._data.cam_xpos[self._cam_id]
        cam_xmat = self._data.cam_xmat[self._cam_id].reshape(3, 3)
        delta = world_pos - cam_xpos
        pos_mj = cam_xmat.T @ delta
        # MuJoCo cam (X right, Y up, Z back) → OpenCV cam (X right, Y down, Z forward)
        pos_cv = np.array([pos_mj[0], -pos_mj[1], -pos_mj[2]])
        logger.debug(
            "[world_to_cam] cam_xpos=(%.3f,%.3f,%.3f) delta=(%.3f,%.3f,%.3f) "
            "pos_mj=(%.3f,%.3f,%.3f) → pos_cv=(%.3f,%.3f,%.3f)",
            cam_xpos[0], cam_xpos[1], cam_xpos[2],
            delta[0], delta[1], delta[2],
            pos_mj[0], pos_mj[1], pos_mj[2],
            pos_cv[0], pos_cv[1], pos_cv[2],
        )
        return pos_cv
