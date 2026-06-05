"""MpcController: 封装 Convex Centroidal MPC 完整管线，提供 step(cmd_vel) → torques 接口."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pinocchio as pin

from .mpc.go2_robot_data import PinGo2Model
from .mpc.gait import Gait
from .mpc.com_trajectory import ComTraj
from .mpc.centroidal_mpc import CentroidalMPC
from .mpc.leg_controller import LegController

logger = logging.getLogger(__name__)

# 关节力矩安全限幅 (N·m), leg-grouped 顺序
TORQUE_LIMITS = np.array([
    25.0, 25.0, 40.0,  # FL: hip, thigh, calf
    25.0, 25.0, 40.0,  # FR
    25.0, 25.0, 40.0,  # RL
    25.0, 25.0, 40.0,  # RR
], dtype=np.float64)

LEGS = ["FL", "FR", "RL", "RR"]


class MpcController:
    """Convex Centroidal MPC 控制器.

    封装 Pinocchio 动力学、步态调度、MPC 求解和腿部阻抗控制，
    对外暴露 step(cmd_vel, qpos, qvel) → torques 的简洁接口。
    """

    def __init__(
        self,
        control_hz: float = 50.0,
        gait_hz: float = 2.0,
        gait_duty: float = 0.6,
        mpc_horizon: int = 10,
        z_des: float = 0.27,
        verbose: bool = False,
    ) -> None:
        self._control_dt = 1.0 / control_hz
        self._gait_hz = gait_hz
        self._gait_duty = gait_duty
        self._mpc_horizon = mpc_horizon
        self._z_des = z_des
        self._verbose = verbose

        # 核心组件（初始构建，solve 首帧后 solver 固定）
        self._go2_pin = PinGo2Model()
        self._gait = Gait(frequency_hz=gait_hz, duty=gait_duty)
        self._traj = ComTraj(self._go2_pin)
        self._leg_ctrl = LegController()

        self._mpc: Optional[CentroidalMPC] = None
        self._solver_built = False
        self._sim_time = 0.0
        self._solve_time_ms = 0.0
        self._step_count = 0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def step(
        self,
        cmd_vel: np.ndarray,
        mj_qpos: np.ndarray,
        mj_qvel: np.ndarray,
    ) -> np.ndarray:
        """执行一帧 MPC 控制，返回 12 维关节力矩 (leg-grouped MuJoCo 顺序).

        Args:
            cmd_vel: [vx, vy, vyaw] 身体坐标系速度指令
            mj_qpos: MuJoCo qpos (19,)
            mj_qvel: MuJoCo qvel (18,)

        Returns:
            tau: 12 维力矩 [FL_hip,FL_thigh,FL_calf, FR_hip,..., RR_hip,RR_thigh,RR_calf]
        """
        t0 = time.perf_counter()

        # 1. MuJoCo → Pinocchio 状态转换
        q_pin, dq_pin = self._mj_to_pin(mj_qpos, mj_qvel)

        # 2. 更新 Pinocchio 动力学模型
        self._go2_pin.update_model(q_pin, dq_pin)

        # 3. 生成参考轨迹
        self._traj.generate_traj(
            self._go2_pin,
            self._gait,
            self._sim_time,
            float(cmd_vel[0]),
            float(cmd_vel[1]),
            self._z_des,
            float(cmd_vel[2]),
            self._control_dt,
        )

        # 4. 首次调用时构建 MPC solver（sparsity pattern 固定后复用）
        if not self._solver_built:
            self._mpc = CentroidalMPC(self._go2_pin, self._traj)
            self._solver_built = True

        # 5. 求解 QP，获取地面反力
        sol = self._mpc.solve_QP(self._go2_pin, self._traj, verbose=self._verbose)

        # 6. 提取第一步控制输入 u0 = [FL_fx,FL_fy,FL_fz, FR_fx,..., RR_fz]
        mpc_N = self._traj.N  # 实际 horizon（由 gait_period / dt 决定）
        u0 = np.array(sol["x"][mpc_N * 12 : mpc_N * 12 + 12], dtype=float).reshape(-1)

        self._step_count += 1
        if self._step_count <= 5 or self._step_count % 50 == 0:
            fx = u0[0::3]
            fy = u0[1::3]
            fz = u0[2::3]
            logger.info(
                "MPC step=%d forces: FL=[%.1f,%.1f,%.1f] FR=[%.1f,%.1f,%.1f] "
                "RL=[%.1f,%.1f,%.1f] RR=[%.1f,%.1f,%.1f]",
                self._step_count, fx[0], fy[0], fz[0], fx[1], fy[1], fz[1],
                fx[2], fy[2], fz[2], fx[3], fy[3], fz[3],
            )

        # 7. 逐腿计算关节力矩
        torque = np.zeros(12, dtype=np.float64)
        for i, leg in enumerate(LEGS):
            force_3d = u0[i * 3 : (i + 1) * 3].copy()
            leg_out = self._leg_ctrl.compute_leg_torque(
                leg, self._go2_pin, self._gait, force_3d, self._sim_time
            )
            torque[i * 3 : (i + 1) * 3] = leg_out.tau

        # 8. 安全限幅
        torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

        # 9. 推进仿真时间
        self._sim_time += self._control_dt

        self._solve_time_ms = (time.perf_counter() - t0) * 1e3
        return torque

    def reset(self) -> None:
        """重置 MPC 状态（站立后重新开始移动时调用）."""
        self._go2_pin = PinGo2Model()
        self._gait = Gait(frequency_hz=self._gait_hz, duty=self._gait_duty)
        self._traj = ComTraj(self._go2_pin)
        self._leg_ctrl = LegController()
        self._mpc = None
        self._solver_built = False
        self._sim_time = 0.0

    @property
    def solve_time_ms(self) -> float:
        """上一帧 QP 求解耗时 (ms)."""
        return self._solve_time_ms

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _mj_to_pin(mj_qpos: np.ndarray, mj_qvel: np.ndarray):
        """MuJoCo 状态 → Pinocchio 状态.

        MuJoCo qpos:   [x,y,z, qw,qx,qy,qz, 12 joints]
        MuJoCo qvel:   [vx_world,vy_world,vz_world, wx,wy,wz, 12 joint_vels]
        Pinocchio q:   [x,y,z, qx,qy,qz,qw, 12 joints]
        Pinocchio dq:  [vx_body,vy_body,vz_body, wx,wy,wz, 12 joint_vels]
        """
        qw, qx, qy, qz = mj_qpos[3:7]
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()

        v_world = mj_qvel[0:3]
        w_body = mj_qvel[3:6]
        v_body = R.T @ v_world

        q_pin = np.concatenate([mj_qpos[0:3], [qx, qy, qz, qw], mj_qpos[7:19]])
        dq_pin = np.concatenate([v_body, w_body, mj_qvel[6:18]])

        return q_pin, dq_pin
