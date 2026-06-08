"""仿真感知：直接从 MuJoCo xpos 读取目标位置（上帝视角）。"""

import mujoco
import numpy as np

from algorithms.perception.base import Detection, ObjectDetector


class SimObjectDetector(ObjectDetector):
    """从 MuJoCo body 位置读取目标信息。

    不需要渲染+检测，直接从 mjData.xpos 获取精确 3D 位置。
    输出的数据结构与 real 感知完全一致，保证 GraspPlanner 可统一调用。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_body_names: list[str],
        labels: list[str] | None = None,
    ):
        self._data = data
        self._body_ids: list[int] = []
        self._labels: list[str] = []

        for i, name in enumerate(target_body_names):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._body_ids.append(bid)
                self._labels.append(labels[i] if labels else name)

    def detect(self) -> list[Detection]:
        detections = []
        for bid, label in zip(self._body_ids, self._labels):
            pos = self._data.xpos[bid].copy()
            detections.append(Detection(
                label=label,
                center_pixel=(0, 0),  # sim 不需要像素坐标
                bbox=(0, 0, 0, 0),
                depth_m=0.0,
                position_world=pos,
                confidence=1.0,
            ))
        return detections
