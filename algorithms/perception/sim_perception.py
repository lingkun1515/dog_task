"""仿真感知：SimRGBDCamera RGB-D → 统一 YOLO/HSV 检测 + xpos 兜底。

Sim 和 Real 共用 detector.py 中的 YOLODetector / HSVDetector 类。
两者唯一区别：Sim 用 SimRGBDCamera，Real 用 RealSenseCamera，
只要 camera 提供 get_rgb()/get_depth()/get_intrinsics()，检测逻辑 100% 相同。

SimObjectDetector 是 MuJoCo-specific 的组合检测器：
  YOLO → HSV fallback → xpos 兜底 + annotate_frame 可视化。
"""

from __future__ import annotations

import numpy as np

from algorithms.perception.base import Detection, ObjectDetector
from algorithms.perception.depth_utils import depth_at_pixel, deproject_pixel
from algorithms.perception.detector import HSVDetector, YOLODetector


# ============================================================================
# SimObjectDetector — 组合检测器（YOLO → HSV → xpos 兜底 + annotate_frame）
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

        # 目标 MuJoCo body ID（xpos 兜底用）
        self._target_body_ids: list[int] = []
        self._target_labels: list[str] = []
        for i, name in enumerate(target_body_names):
            bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._target_body_ids.append(bid)
                self._target_labels.append(self._labels[i] if i < len(self._labels) else name)

        # HSV 检测默认配置（红+黄，覆盖常见目标颜色）
        self._target_colors = target_colors or [
            {"lower": (0, 100, 50), "upper": (10, 255, 255), "label": "ball"},
            {"lower": (170, 100, 50), "upper": (180, 255, 255), "label": "ball"},
            {"lower": (20, 80, 80), "upper": (30, 255, 255), "label": "ball"},  # 黄色
        ]

        # 统一的 YOLO / HSV 检测器（从 detector.py 共用）
        self._yolo_detector: YOLODetector | None = None
        self._hsv_detector: HSVDetector | None = None

        self._intrinsics = camera.get_intrinsics()

    # ------------------------------------------------------------------
    # ObjectDetector 接口
    # ------------------------------------------------------------------
    def detect(self) -> list[Detection]:
        """检测流水线：YOLO → HSV → xpos 兜底。

        与 real 端完全相同的 detection 数据结构。
        """
        rgb = self._camera.get_rgb()
        if rgb is None:
            return self._detect_from_xpos()

        # 1. YOLO
        dets = self._detect_yolo()
        if dets:
            for d in dets:
                print(f"[SimDetect] YOLO: {d.label} conf={d.confidence:.2f} "
                      f"pixel=({d.center_pixel[0]},{d.center_pixel[1]}) depth={d.depth_m:.3f}m")
            return dets

        # 2. HSV fallback
        dets = self._detect_hsv()
        if dets:
            for d in dets:
                print(f"[SimDetect] HSV: {d.label} pixel=({d.center_pixel[0]},{d.center_pixel[1]}) depth={d.depth_m:.3f}m")
            return dets

        # 3. 兜底 xpos 直读
        dets = self._detect_from_xpos()
        if dets:
            for d in dets:
                print(f"[SimDetect] xpos: {d.label} depth={d.depth_m:.3f}m")
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
        if self._hsv_detector is None:
            labels = [cfg["label"] for cfg in self._target_colors]
            self._hsv_detector = HSVDetector(
                camera=self._camera,
                targets=labels,
                color_order="rgb",  # Sim 渲染输出是 RGB
            )
        return self._hsv_detector.detect()

    # ------------------------------------------------------------------
    # 图像标注（用于相机推流）
    # ------------------------------------------------------------------
    def annotate_frame(self, rgb: np.ndarray) -> np.ndarray:
        """在 RGB 图像上绘制检测框。

        使用 HSV 颜色检测绘制目标边框。
        """
        import cv2

        annotated = rgb.copy()
        depth_m = self._camera.get_depth()
        if depth_m is None:
            return annotated

        found_any = False
        for cfg in self._target_colors:
            label = cfg["label"]
            lower = np.array(cfg["lower"])
            upper = np.array(cfg["upper"])
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 50:
                    continue
                found_any = True
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

    # ------------------------------------------------------------------
    # 兜底：xpos 直读（MuJoCo 真值，相机不可用或检测失败时的 fallback）
    # ------------------------------------------------------------------
    def _detect_from_xpos(self) -> list[Detection]:
        """兜底：直接从 MuJoCo xpos 读取目标位置（上帝视角）。"""
        detections = []
        for bid, label in zip(self._target_body_ids, self._target_labels):
            world_pos = self._data.xpos[bid].copy()
            pos_cam = self._world_to_cam(world_pos)
            depth = float(pos_cam[2]) if pos_cam is not None else 0.0

            if pos_cam is not None:
                fx = self._intrinsics["fx"]
                fy = self._intrinsics["fy"]
                cx = self._intrinsics["cx"]
                cy = self._intrinsics["cy"]
                z = max(pos_cam[2], 0.001)
                px = int(fx * pos_cam[0] / z + cx)
                py = int(fy * pos_cam[1] / z + cy)
            else:
                px, py = 0, 0

            detections.append(Detection(
                label=label,
                center_pixel=(px, py),
                bbox=(px - 10, py - 10, 20, 20),
                depth_m=depth,
                position_cam=pos_cam,
                confidence=1.0,
            ))
        return detections

    def _world_to_cam(self, world_pos: np.ndarray) -> np.ndarray:
        """世界坐标 → 相机坐标（用于 xpos 兜底）。"""
        cam_xpos = self._data.cam_xpos[self._cam_id]
        cam_xmat = self._data.cam_xmat[self._cam_id].reshape(3, 3)
        return cam_xmat.T @ (world_pos - cam_xpos)
