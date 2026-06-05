"""坐标变换与机械臂可达区域(甜区)判定(core 层)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .models import Vector3


@dataclass(frozen=True)
class RigidTransform:
    """4x4 刚体变换: 把 camera_link 下的点变换到 arm_base.

    ``matrix`` 为齐次变换矩阵 T, 满足 P_arm = T @ P_cam(齐次坐标).
    """

    matrix: Tuple[Tuple[float, float, float, float], ...]

    @classmethod
    def from_value(cls, value: Sequence[Sequence[float]]) -> "RigidTransform":
        """从 JSON 配置中的 4x4 矩阵列表构造, 最后一行必须是 [0,0,0,1]."""
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("transform matrix must be 4x4")
        matrix = tuple(tuple(float(item) for item in row) for row in value)
        if matrix[3] != (0.0, 0.0, 0.0, 1.0):
            raise ValueError("transform matrix last row must be [0, 0, 0, 1]")
        return cls(matrix)

    def apply(self, point: Vector3) -> Vector3:
        """对三维点应用变换(旋转 + 平移), 返回 arm_base 下的点."""
        x, y, z = point.x, point.y, point.z
        m = self.matrix
        return Vector3(
            m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
        )


@dataclass(frozen=True)
class SweetZone:
    """机械臂保守可达区域; 抓取前必须确认目标落在此范围内.

    y 轴用 ``y_abs_max_m`` 表示左右对称: |y| <= y_abs_max_m.
    """

    x_min_m: float
    x_max_m: float
    y_abs_max_m: float   # Y 方向半宽(米), 对称约束
    z_min_m: float
    z_max_m: float

    @classmethod
    def from_dict(cls, data: dict) -> "SweetZone":
        """从 workflow.sweet_zone 配置段解析."""
        return cls(
            x_min_m=float(data["x_min_m"]),
            x_max_m=float(data["x_max_m"]),
            y_abs_max_m=float(data["y_abs_max_m"]),
            z_min_m=float(data["z_min_m"]),
            z_max_m=float(data["z_max_m"]),
        )

    def contains(self, point: Vector3) -> bool:
        """判断 arm_base 下的点是否在甜区内."""
        return (
            self.x_min_m <= point.x <= self.x_max_m
            and abs(point.y) <= self.y_abs_max_m
            and self.z_min_m <= point.z <= self.z_max_m
        )
