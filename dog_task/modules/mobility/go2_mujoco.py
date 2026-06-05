"""Go2MujocoMobility: MuJoCo 仿真 Go2 底盘，实现 MobilityController 协议.

控制模式:
  - "rl": 用 ONNX RL 策略推理关节力矩驱动（接近真机行为）
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Mapping, Optional

import numpy as np

from ...core.models import (
    ActionResult,
    HealthStatus,
    MobilityCapabilities,
    MobilityState,
    NavOptions,
    Pose2D,
    VelocityCommand,
)
from ..sim.world import SimWorld

logger = logging.getLogger(__name__)

GO2_LEG_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

GO2_ACTUATORS = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# Isaac Lab 训练默认角度 (type-grouped: 髋×4, 大腿×4, 小腿×4)
IL_DEFAULTS = np.array([
    0.0, 0.0, 0.0, 0.0,
    1.1, 1.1, 1.1, 1.1,
    -1.8, -1.8, -1.8, -1.8,
], dtype=np.float32)

# Menagerie Go2 home-keyframe 关节角 — 用于 STANDUP/HOLD 阶段的 PD 目标。
# 大腿角度 (0.9) 与 IL_DEFAULTS (1.1) 不同: MuJoCo 物理下身重力作用下
# 0.9 rad 才是自然平衡姿态，1.1 rad 会让狗前倾后仰不稳。
MENAGERIE_HOME = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.9, 0.9, 0.9, 0.9,
    -1.8, -1.8, -1.8, -1.8,
], dtype=np.float32)


# 关节限位 (type-grouped, 95% of motor range)
IL_LIMITS = np.array([
    [-0.837, 0.837], [-0.837, 0.837], [-0.837, 0.837], [-0.837, 0.837],
    [-3.490, 1.570], [-3.490, 1.570], [-4.530, 1.570], [-4.530, 1.570],
    [-2.720, -0.837], [-2.720, -0.837], [-2.720, -0.837], [-2.720, -0.837],
], dtype=np.float32) * 0.95

# 力矩安全限幅 (type-grouped, N·m)
IL_TORQUE_LIMITS = np.array([
    25.0, 25.0, 25.0, 25.0,
    25.0, 25.0, 25.0, 25.0,
    40.0, 40.0, 40.0, 40.0,
], dtype=np.float32)

# 关节顺序重映射 (gather 语义: result = data[indices])
# IL (type-grouped): FL_hip,FR_hip,RL_hip,RR_hip | FL_thigh,FR_thigh,RL_thigh,RR_thigh | FL_calf,FR_calf,RL_calf,RR_calf
# MJ (leg-grouped):  FL_hip,FL_thigh,FL_calf | FR_hip,FR_thigh,FR_calf | RL_hip,RL_thigh,RL_calf | RR_hip,RR_thigh,RR_calf
IL_TO_MJ = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=int)   # mj = il[IL_TO_MJ]
MJ_TO_IL = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=int)  # il = mj[MJ_TO_IL]

ACTION_SCALE = 0.25


class Go2MujocoMobility:
    """MuJoCo 仿真 Go2 底盘."""

    def __init__(self, config: Mapping[str, Any], world: SimWorld) -> None:
        self._world = world
        self._config = dict(config)
        self._control_mode = self._config.get("control_mode", "rl")
        self._stop_velocity_threshold = self._config.get("stop_velocity_threshold", 0.05)

        self._cmd_vel = np.zeros(3)  # [vx, vy, vyaw]
        self._posture = "stand"
        self._lock = threading.Lock()

        self._rl_policy = None
        self._last_action = np.zeros(12)
        self._control_hz = self._config.get("control_hz", 50)
        self._kp = self._config.get("kp", 20.0)
        self._kd = self._config.get("kd", 0.5)

        self._running = False
        self._control_thread: Optional[threading.Thread] = None

    def healthcheck(self) -> HealthStatus:
        try:
            self._world.joint_id("FL_hip_joint")
            import importlib.util
            if importlib.util.find_spec("onnxruntime") is None:
                return HealthStatus(ready=False, message="onnxruntime not installed")
            from ...config import project_path
            import os
            model_path = str(project_path(self._config.get("rl_model", "assets/rl_models/flat_policy_v6.onnx")))
            if not os.path.exists(model_path):
                return HealthStatus(ready=False, message=f"RL model not found: {model_path}")
            return HealthStatus(ready=True, message="Go2 MuJoCo (rl)")
        except Exception as e:
            return HealthStatus(ready=False, message=str(e))

    def capabilities(self) -> MobilityCapabilities:
        return MobilityCapabilities(
            supports_velocity=True,
            supports_strafe=True,
            supports_global_nav=True,
            supports_docking=False,
            supports_posture=True,
        )

    def get_state(self) -> MobilityState:
        qpos = self._world.get_freejoint_qpos("root")
        qvel = self._world.get_freejoint_qvel("root")
        x, y, z = qpos[0], qpos[1], qpos[2]
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        speed = np.linalg.norm(qvel[0:3])
        is_stopped = speed < self._stop_velocity_threshold

        return MobilityState(
            is_stopped=is_stopped,
            posture=self._posture,
            pose=Pose2D(x_m=float(x), y_m=float(y), yaw_rad=float(yaw)),
        )

    def set_velocity(self, command: VelocityCommand) -> ActionResult:
        with self._lock:
            self._cmd_vel = np.array([
                command.linear_x_mps,
                command.linear_y_mps,
                command.angular_z_rps,
            ])
        if not self._running:
            self._start_control_loop()
        return ActionResult(success=True, message="velocity set")

    def stop(self) -> ActionResult:
        with self._lock:
            self._cmd_vel = np.zeros(3)
        return ActionResult(success=True, message="stopped")

    def is_stopped(self) -> bool:
        return self.get_state().is_stopped

    def set_posture(self, posture: str) -> ActionResult:
        if posture not in ("stand", "stand_down"):
            return ActionResult(success=False, message=f"unknown posture: {posture}")
        self._posture = posture

        if posture == "stand_down":
            return ActionResult(success=True, message="posture=stand_down (ignored in RL mode)")
        self._apply_posture_rl()
        return ActionResult(success=True, message="posture=stand")

    def _apply_posture_rl(self) -> None:
        """RL 模式下设置姿态：暂停控制循环，用高增益 PD 收敛到 MENAGERIE_HOME."""
        was_running = self._running
        self._running = False
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
            self._control_thread = None

        # MENAGERIE_HOME 用于站立（大腿 0.9，MuJoCo 自然平衡姿态）
        target_q_mj = MENAGERIE_HOME[IL_TO_MJ]
        stand_kp, stand_kd = 80.0, 2.0

        for _ in range(500):
            current_q = self._world.get_qpos(GO2_LEG_JOINTS)
            current_dq = self._world.get_qvel(GO2_LEG_JOINTS)
            err = np.max(np.abs(target_q_mj - current_q))
            if err < 0.01:
                break
            tau = stand_kp * (target_q_mj - current_q) - stand_kd * current_dq
            self._world.set_ctrl(GO2_ACTUATORS, tau)
            self._world.step(10)
        self._world.forward()

        if was_running:
            self._start_control_loop()

    def go_to(self, goal: Pose2D, options: NavOptions = None) -> ActionResult:
        state = self.get_state()
        dx = goal.x_m - state.pose.x_m
        dy = goal.y_m - state.pose.y_m
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            return ActionResult(success=True, message="already at goal")

        target_yaw = math.atan2(dy, dx)
        speed = min(0.5, dist)
        yaw_err = self._wrap_angle(target_yaw - state.pose.yaw_rad)
        vyaw = np.clip(2.0 * yaw_err, -1.0, 1.0)

        vx = speed * math.cos(yaw_err)
        vy = speed * math.sin(yaw_err)
        self.set_velocity(VelocityCommand(linear_x_mps=vx, linear_y_mps=vy, angular_z_rps=vyaw))

        timeout = dist / 0.3 + 3.0
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.1)
            state = self.get_state()
            if math.hypot(goal.x_m - state.pose.x_m, goal.y_m - state.pose.y_m) < 0.1:
                self.stop()
                return ActionResult(success=True, message="reached goal")
            dx2 = goal.x_m - state.pose.x_m
            dy2 = goal.y_m - state.pose.y_m
            target_yaw2 = math.atan2(dy2, dx2)
            yaw_err2 = self._wrap_angle(target_yaw2 - state.pose.yaw_rad)
            sp = min(0.5, math.hypot(dx2, dy2))
            self.set_velocity(VelocityCommand(
                linear_x_mps=sp * math.cos(yaw_err2),
                linear_y_mps=sp * math.sin(yaw_err2),
                angular_z_rps=float(np.clip(2.0 * yaw_err2, -1.0, 1.0)),
            ))

        self.stop()
        return ActionResult(success=False, message="go_to timeout")

    def dock(self, dock_id: Optional[str] = None) -> ActionResult:
        return ActionResult(success=False, message="docking not supported in sim")

    def emergency_stop(self) -> ActionResult:
        self.stop()
        return ActionResult(success=True, message="emergency stop")

    # ---- 内部控制循环 ----

    def _start_control_loop(self) -> None:
        if self._running:
            return
        self._running = True
        self._control_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="go2_ctrl"
        )
        self._control_thread.start()

    def _control_loop(self) -> None:
        dt = 1.0 / self._control_hz
        while self._running:
            with self._lock:
                cmd = self._cmd_vel.copy()
            self._rl_step(cmd)
            time.sleep(dt)

    def _rl_step(self, cmd: np.ndarray) -> None:
        """RL 策略推理 + PD 力矩控制."""
        if self._rl_policy is None:
            self._load_rl_policy()

        obs = self._build_observation(cmd)
        action = self._rl_policy.infer(obs)
        self._last_action = action.copy()

        # action 是 type-grouped (IL), 计算目标角度后重排为 leg-grouped
        target_q_il = IL_DEFAULTS + ACTION_SCALE * action
        target_q_il = np.clip(target_q_il, IL_LIMITS[:, 0], IL_LIMITS[:, 1])
        target_q_mj = target_q_il[IL_TO_MJ]

        current_q = self._world.get_qpos(GO2_LEG_JOINTS)
        current_dq = self._world.get_qvel(GO2_LEG_JOINTS)

        tau = self._kp * (target_q_mj - current_q) - self._kd * current_dq
        tau = np.clip(tau, -IL_TORQUE_LIMITS[IL_TO_MJ], IL_TORQUE_LIMITS[IL_TO_MJ])
        self._world.set_ctrl(GO2_ACTUATORS, tau)

        substeps = int(round(1.0 / (self._control_hz * self._world.model.opt.timestep)))
        self._world.step(substeps)

    def _build_observation(self, cmd: np.ndarray) -> np.ndarray:
        """构建 45 维观测向量 (Isaac Lab type-grouped 顺序)."""
        base_angvel = self._world.get_sensor("imu_gyro")
        base_quat = self._world.get_freejoint_qpos("root")[3:7]
        proj_grav = self._projected_gravity(base_quat)
        joint_pos_mj = self._world.get_qpos(GO2_LEG_JOINTS)
        joint_vel_mj = self._world.get_qvel(GO2_LEG_JOINTS)

        # leg-grouped → type-grouped (IL order)
        joint_pos_il = joint_pos_mj[MJ_TO_IL]
        joint_vel_il = joint_vel_mj[MJ_TO_IL]

        obs = np.concatenate([
            base_angvel,                    # 0-2
            proj_grav,                      # 3-5
            cmd,                            # 6-8
            joint_pos_il - IL_DEFAULTS,     # 9-20
            joint_vel_il,                   # 21-32
            self._last_action,              # 33-44
        ])
        return obs

    @staticmethod
    def _projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
        """将重力方向投影到 body 坐标系 (Isaac Lab 约定: 单位向量)."""
        w, x, y, z = quat_wxyz
        # R_body2world^T @ [0, 0, -1] = body 坐标系中的重力方向
        gx = 2.0 * (x * z - w * y)
        gy = 2.0 * (y * z + w * x)
        gz = 1.0 - 2.0 * (x * x + y * y)
        return np.array([-gx, -gy, -gz])

    def _load_rl_policy(self) -> None:
        from ..sim.rl_policy import RLPolicy
        from ...config import project_path
        model_path = str(project_path(self._config.get("rl_model", "assets/rl_models/flat_policy_v6.onnx")))
        self._rl_policy = RLPolicy(model_path, obs_dim=45)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi
