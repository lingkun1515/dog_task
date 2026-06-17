"""仿真 D1 运动学：D1 URDF FK/IK + MuJoCo 世界坐标系位姿。

用 D1 URDF 参数做 FK/IK 运算（与实机相同算法），
同时通过 MuJoCo 读取 arm_base 在世界坐标系中的位姿。
这样 IK/FK 与实机完全一致，保证 sim2real 行为对齐。
"""

import mujoco
import numpy as np

from algorithms.kinematics.base import ArmKinematics
from algorithms.kinematics.real_d1_ik import D1Kinematics


class SimD1Kinematics(ArmKinematics):
    """D1 运动学 + MuJoCo 世界坐标系感知。

    FK/IK 使用 D1 URDF 参数（与实机完全相同的算法），
    arm_base 世界位姿从 MuJoCo data 读取。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_base_body_name: str,
    ):
        self._model = model
        self._data = data
        self._arm_base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, arm_base_body_name
        )

    # ------------------------------------------------------------------
    # 世界坐标系属性（从 MuJoCo 读取）
    # ------------------------------------------------------------------
    @property
    def arm_base_world_pos(self) -> np.ndarray:
        return self._data.xpos[self._arm_base_body_id].copy()

    @property
    def arm_base_world_pose(self) -> np.ndarray:
        xpos = self._data.xpos[self._arm_base_body_id].copy()
        xmat = self._data.xmat[self._arm_base_body_id].copy().reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = xmat
        T[:3, 3] = xpos
        return T

    # ------------------------------------------------------------------
    # ArmKinematics 接口（委托给 D1Kinematics）
    # ------------------------------------------------------------------
    def get_position(self, joint_angles_deg: list[float]) -> tuple[float, float, float]:
        """FK → 世界坐标系 TCP 位置。"""
        pos_arm = np.array(D1Kinematics.get_position(joint_angles_deg))
        return tuple((pos_arm + self.arm_base_world_pos).tolist())

    def forward_kinematics(self, joint_angles_deg: list[float]) -> np.ndarray:
        """FK → 世界坐标系 4x4 位姿矩阵。"""
        T_arm = D1Kinematics.forward_kinematics(joint_angles_deg)
        return self.arm_base_world_pose @ T_arm

    def inverse_kinematics(
        self,
        target_xyz,
        initial_angles_deg=None,
        max_iter: int = 400,
        tol: float = 0.003,
        alpha: float = 0.5,
    ) -> list[float] | None:
        """IK：target_xyz 在 arm base 坐标系中。返回关节角度（度）。"""
        return D1Kinematics.inverse_kinematics(
            target_xyz,
            initial_angles_deg=initial_angles_deg,
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
        )
