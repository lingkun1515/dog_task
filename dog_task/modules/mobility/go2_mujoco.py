"""Go2MujocoMobility: MuJoCo 仿真 Go2 底盘，实现 MobilityController 协议.

支持两种控制模式:
  - "kinematic": 直接对 base_link freejoint 做速度积分（无腿部物理）
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

GO2_DEFAULT_ANGLES = np.array([0, 0.9, -1.8] * 4)

ACTION_SCALE = 0.25


class Go2MujocoMobility:
    """MuJoCo 仿真 Go2 底盘."""

    def __init__(self, config: Mapping[str, Any], world: SimWorld) -> None:
        self._world = world
        self._config = dict(config)
        self._control_mode = self._config.get("control_mode", "kinematic")
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
            if self._control_mode == "rl":
                import importlib.util
                if importlib.util.find_spec("onnxruntime") is None:
                    return HealthStatus(ready=False, message="onnxruntime not installed (required for rl mode)")
                from ...config import project_path
                import os
                model_path = str(project_path(self._config.get("rl_model", "assets/rl_models/flat_policy_v5.onnx")))
                if not os.path.exists(model_path):
                    return HealthStatus(ready=False, message=f"RL model not found: {model_path}")
            return HealthStatus(ready=True, message=f"Go2 MuJoCo ({self._control_mode})")
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
        if self._control_mode == "kinematic":
            qpos = self._world.get_freejoint_qpos("root")
            if posture == "stand":
                qpos[2] = 0.27
                leg_angles = GO2_DEFAULT_ANGLES
            else:
                qpos[2] = 0.15
                leg_angles = np.array([0.0, 1.57, -2.5] * 4)
            self._world.set_freejoint_qpos(qpos, "root")
            self._world.set_qpos(GO2_LEG_JOINTS, leg_angles)
            self._world.forward()
        return ActionResult(success=True, message=f"posture={posture}")

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
        if self._control_mode == "rl":
            self._world.set_ctrl(GO2_ACTUATORS, np.zeros(12))
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
            if self._control_mode == "kinematic":
                self._kinematic_step(cmd, dt)
            else:
                self._rl_step(cmd)
            time.sleep(dt)

    def _kinematic_step(self, cmd: np.ndarray, dt: float) -> None:
        """运动学积分: 直接更新 freejoint qpos 和 qvel."""
        qpos = self._world.get_freejoint_qpos("root")
        x, y, z = qpos[0], qpos[1], qpos[2]
        qw, qx, qy, qz_val = qpos[3], qpos[4], qpos[5], qpos[6]
        yaw = math.atan2(2.0 * (qw * qz_val + qx * qy), 1.0 - 2.0 * (qy * qy + qz_val * qz_val))

        vx_world = cmd[0] * math.cos(yaw) - cmd[1] * math.sin(yaw)
        vy_world = cmd[0] * math.sin(yaw) + cmd[1] * math.cos(yaw)
        vyaw = cmd[2]

        new_x = x + vx_world * dt
        new_y = y + vy_world * dt
        new_yaw = yaw + vyaw * dt

        cy = math.cos(new_yaw / 2)
        sy = math.sin(new_yaw / 2)
        new_qpos = np.array([new_x, new_y, z, cy, 0, 0, sy])
        self._world.set_freejoint_qpos(new_qpos, "root")

        vx = (new_x - x) / dt if dt > 0 else 0.0
        vy = (new_y - y) / dt if dt > 0 else 0.0
        vyaw_actual = (new_yaw - yaw) / dt if dt > 0 else 0.0
        self._world.set_freejoint_qvel(np.array([vx, vy, 0.0, 0.0, 0.0, vyaw_actual]), "root")

        self._world.set_qpos(GO2_LEG_JOINTS, GO2_DEFAULT_ANGLES)
        self._world.forward()

    def _rl_step(self, cmd: np.ndarray) -> None:
        """RL 策略推理 + PD 力矩控制."""
        if self._rl_policy is None:
            self._load_rl_policy()

        obs = self._build_observation(cmd)
        action = self._rl_policy.infer(obs)
        self._last_action = action.copy()

        target_q = GO2_DEFAULT_ANGLES + ACTION_SCALE * action
        current_q = self._world.get_qpos(GO2_LEG_JOINTS)
        current_dq = self._world.get_qvel(GO2_LEG_JOINTS)

        tau = self._kp * (target_q - current_q) - self._kd * current_dq
        self._world.set_ctrl(GO2_ACTUATORS, tau)

        substeps = int(round(1.0 / (self._control_hz * self._world.model.opt.timestep)))
        self._world.step(substeps)

    def _build_observation(self, cmd: np.ndarray) -> np.ndarray:
        """构建 45 维观测向量."""
        base_angvel = self._world.get_sensor("imu_gyro")
        base_quat = self._world.get_freejoint_qpos("root")[3:7]
        proj_grav = self._projected_gravity(base_quat)
        joint_pos = self._world.get_qpos(GO2_LEG_JOINTS)
        joint_vel = self._world.get_qvel(GO2_LEG_JOINTS)

        obs = np.concatenate([
            base_angvel,
            proj_grav,
            cmd,
            joint_pos - GO2_DEFAULT_ANGLES,
            joint_vel,
            self._last_action,
        ])
        return obs

    @staticmethod
    def _projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
        """将重力向量投影到 body 坐标系 (MuJoCo quat=[w,x,y,z])."""
        w, x, y, z = quat_wxyz
        gx = 2.0 * (x * z - w * y)
        gy = 2.0 * (y * z + w * x)
        gz = 1.0 - 2.0 * (x * x + y * y)
        return np.array([gx, gy, gz]) * (-9.81)

    def _load_rl_policy(self) -> None:
        try:
            from ..sim.rl_policy import RLPolicy
            from ...config import project_path
            model_path = str(project_path(self._config.get("rl_model", "assets/rl_models/flat_policy_v5.onnx")))
            self._rl_policy = RLPolicy(model_path, obs_dim=45)
        except Exception as e:
            logger.error("RL policy load failed, falling back to kinematic: %s", e)
            self._control_mode = "kinematic"

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi
