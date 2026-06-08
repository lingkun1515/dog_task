"""机械臂运动学抽象基类。"""

from abc import ABC, abstractmethod

import numpy as np


class ArmKinematics(ABC):
    """6-DOF 机械臂正/逆运动学接口。"""

    @abstractmethod
    def get_position(self, joint_angles_deg: list[float]) -> tuple[float, float, float]:
        """正运动学：关节角度（度） → TCP 3D 位置（米）。"""
        ...

    @abstractmethod
    def forward_kinematics(self, joint_angles_deg: list[float]) -> np.ndarray:
        """正运动学：关节角度（度） → 4x4 齐次变换矩阵。"""
        ...

    @abstractmethod
    def inverse_kinematics(
        self,
        target_xyz,
        initial_angles_deg=None,
        max_iter: int = 200,
        tol: float = 0.001,
        alpha: float = 0.5,
    ) -> list[float] | None:
        """逆运动学：目标 3D 位置 → 关节角度列表（度），失败返回 None。"""
        ...
