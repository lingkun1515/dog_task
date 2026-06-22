"""感知模块抽象基类与数据结构。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """单个检测目标。

    - sim:  position_world 有值（直接从 MuJoCo body_xpos 读取）
    - real: position_cam 有值（从深度相机反投影获得），需经标定变换到 arm 坐标系
    """

    label: str
    center_pixel: tuple[int, int]
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    depth_m: float
    position_cam: np.ndarray | None = None   # 相机坐标系 3D（real）
    position_world: np.ndarray | None = None  # 世界坐标系 3D（sim）
    confidence: float = 1.0
    body_name: str | None = None  # sim: 对应的 MuJoCo body 名（用于唯一标识/去重）


class ObjectDetector(ABC):
    """目标检测器抽象接口。"""

    @abstractmethod
    def detect(self) -> list[Detection]:
        """返回当前帧检测到的所有目标。"""
        ...
