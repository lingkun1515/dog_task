"""仿真机械臂运动学：MuJoCo FK 后端 + 数值 Jacobian IK。

与 D1 共享相同的 IK 算法（数值 Jacobian 伪逆 + DLS），
但 FK 通过 mj_forward + xpos 计算，无需 URDF 参数。

线程安全：FK 计算使用私有 MjData 实例，不干扰主仿真线程。
"""

import mujoco
import numpy as np

from algorithms.kinematics.base import ArmKinematics


class SimArmKinematics(ArmKinematics):
    """用 MuJoCo 做 FK 后端的机械臂运动学。

    IK 算法与 D1 一致：数值 Jacobian 伪逆 + 阻尼最小二乘 + 关节限位裁剪。
    所有 FK 计算在私有 MjData 上进行，与主仿真线程隔离。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tcp_body_name: str,
        arm_base_body_name: str,
        qpos_arm_slice: slice,
    ):
        self._model = model
        self._data = data                  # 主线程 data（只读访问当前状态）
        self._fk_data = mujoco.MjData(model)  # 私有 data（FK 计算，线程隔离）
        self._tcp_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tcp_body_name)
        self._arm_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm_base_body_name)
        self._qpos_arm_slice = qpos_arm_slice

        n_arm = qpos_arm_slice.stop - qpos_arm_slice.start
        jnt_range = model.jnt_range
        limits = []
        for i in range(qpos_arm_slice.start, qpos_arm_slice.stop):
            jnt_id = model.dof_jntid[i]
            if jnt_range is not None and jnt_id < len(jnt_range):
                limits.append((float(jnt_range[jnt_id][0]), float(jnt_range[jnt_id][1])))
            else:
                limits.append((-3.14, 3.14))
        self._joint_limits_rad = limits
        self._n_arm = n_arm

    # ------------------------------------------------------------------
    # MuJoCo FK（使用私有 MjData，线程安全）
    # ------------------------------------------------------------------
    def _tcp_world_pos(self, arm_qpos_rad: np.ndarray) -> np.ndarray:
        """用私有 MjData 计算 TCP 世界坐标。"""
        self._fk_data.qpos[:] = self._data.qpos
        self._fk_data.qpos[self._qpos_arm_slice] = arm_qpos_rad
        mujoco.mj_forward(self._model, self._fk_data)
        return self._fk_data.xpos[self._tcp_body_id].copy()

    def _tcp_world_pose(self, arm_qpos_rad: np.ndarray) -> np.ndarray:
        """用私有 MjData 计算 TCP 的 4x4 位姿矩阵（世界坐标系）。"""
        self._fk_data.qpos[:] = self._data.qpos
        self._fk_data.qpos[self._qpos_arm_slice] = arm_qpos_rad
        mujoco.mj_forward(self._model, self._fk_data)
        xpos = self._fk_data.xpos[self._tcp_body_id].copy()
        xmat = self._fk_data.xmat[self._tcp_body_id].copy().reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = xmat
        T[:3, 3] = xpos
        return T

    @property
    def arm_base_world_pos(self) -> np.ndarray:
        """臂基座在世界坐标系中的位置。"""
        return self._data.xpos[self._arm_base_body_id].copy()

    @property
    def arm_base_world_pose(self) -> np.ndarray:
        """臂基座在世界坐标系中的 4x4 位姿矩阵。"""
        xpos = self._data.xpos[self._arm_base_body_id].copy()
        xmat = self._data.xmat[self._arm_base_body_id].copy().reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = xmat
        T[:3, 3] = xpos
        return T

    # ------------------------------------------------------------------
    # ArmKinematics 接口
    # ------------------------------------------------------------------
    def get_position(self, joint_angles_deg: list[float]) -> tuple[float, float, float]:
        q = np.deg2rad(joint_angles_deg[:self._n_arm])
        p = self._tcp_world_pos(q)
        return tuple(p.tolist())

    def forward_kinematics(self, joint_angles_deg: list[float]) -> np.ndarray:
        q = np.deg2rad(joint_angles_deg[:self._n_arm])
        return self._tcp_world_pose(q)

    def inverse_kinematics(
        self,
        target_xyz,
        initial_angles_deg=None,
        max_iter: int = 200,
        tol: float = 0.001,
        alpha: float = 0.5,
    ) -> list[float] | None:
        """数值 IK（与 D1 同算法）：*target_xyz 在 arm base 坐标系中*。"""
        target = np.array(target_xyz)
        arm_base_pos = self._data.xpos[self._arm_base_body_id]

        if initial_angles_deg is not None:
            q = np.deg2rad(np.array(initial_angles_deg[:self._n_arm], dtype=float))
        else:
            q = np.zeros(self._n_arm)

        for _ in range(max_iter):
            tcp_world = self._tcp_world_pos(q)
            tcp_arm = tcp_world - arm_base_pos
            error = target - tcp_arm
            if np.linalg.norm(error) < tol:
                return np.rad2deg(q).tolist()

            J = self._numerical_jacobian(q)
            damping = 0.01
            JTJ = J.T @ J + damping * np.eye(self._n_arm)
            dq = np.linalg.solve(JTJ, J.T @ error) * alpha
            q = q + dq
            for i in range(self._n_arm):
                lo, hi = self._joint_limits_rad[i]
                q[i] = np.clip(q[i], lo, hi)

        tcp_world = self._tcp_world_pos(q)
        tcp_arm = tcp_world - arm_base_pos
        if np.linalg.norm(target - tcp_arm) < tol * 5:
            return np.rad2deg(q).tolist()
        return None

    # ------------------------------------------------------------------
    # 数值 Jacobian（同 D1 算法）
    # ------------------------------------------------------------------
    def _numerical_jacobian(self, angles_rad, delta=1e-5):
        J = np.zeros((3, self._n_arm))
        arm_base = self._data.xpos[self._arm_base_body_id]
        pos0 = self._tcp_world_pos(angles_rad) - arm_base

        for j in range(self._n_arm):
            q = angles_rad.copy()
            q[j] += delta
            pos1 = self._tcp_world_pos(q) - arm_base
            J[:, j] = (pos1 - pos0) / delta

        return J
