"""手眼标定抽象基类与数据结构。"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationResult:
    """手眼标定结果：相机坐标系 → 臂基座坐标系的变换。"""

    T_cam_to_arm: np.ndarray  # 4x4 齐次变换矩阵
    method: str = "identity"
    rmse_m: float = 0.0

    def cam_to_arm(self, point_3d_cam: np.ndarray) -> np.ndarray:
        """将相机坐标系 3D 点转换到臂基座坐标系。"""
        p = np.append(point_3d_cam[:3], 1.0)
        return (self.T_cam_to_arm @ p)[:3]

    @property
    def R(self) -> np.ndarray:
        return self.T_cam_to_arm[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.T_cam_to_arm[:3, 3]
