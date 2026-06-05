"""UnifiedMujocoArm: 在共享 SimWorld 中实现 ArmController 协议.

复用 D1Chain IK 做轨迹规划，通过 SimWorld 的 set_qpos/step 执行运动。
不自建场景、不录视频（相机模块负责可视化）。
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from ...config import project_path
from ...core.models import (
    ActionResult,
    ArmCapabilities,
    HealthStatus,
    PickRequest,
    Reachability,
    TaughtPickRequest,
)
from ..sim.world import SimWorld

logger = logging.getLogger(__name__)

_DEFAULT_URDF = "assets/d1/d1.urdf"

_ARM_JOINTS = ["arm_Joint1", "arm_Joint2", "arm_Joint3", "arm_Joint4", "arm_Joint5", "arm_Joint6"]
_GRIPPER_JOINTS = ["arm_Joint_L", "arm_Joint_R"]

_GRIPPER_OPEN_SERVO = 50.0
_GRIPPER_CLOSE_SERVO = 20.0

_CARRY_TAGS = {"LIFT", "PLACE_PRE", "PLACE"}


class UnifiedMujocoArm:
    """在共享 SimWorld 中的 D1 机械臂控制器."""

    def __init__(self, config: Mapping[str, Any], world: SimWorld) -> None:
        self._world = world
        self._config = dict(config)
        urdf = self._config.get("urdf")
        self._urdf_path = project_path(urdf) if urdf else _DEFAULT_URDF
        self._tcp_offset_m = float(self._config.get("tcp_offset_m", 0.12))
        self._approach_dir = tuple(float(v) for v in self._config.get("approach_dir", [0, 0, -1]))
        self._approach_height_m = float(self._config.get("approach_height_m", 0.15))
        self._replay_speed = float(self._config.get("replay_speed", 1.0))
        self._seconds_per_segment = float(self._config.get("seconds_per_segment", 1.0))
        self._fps = int(self._config.get("fps", 30))
        self._chain: Any = None

    def healthcheck(self) -> HealthStatus:
        if not self._urdf_path.is_file():
            return HealthStatus(False, f"D1 URDF not found: {self._urdf_path}")
        try:
            from .ik.d1_kinematics import D1Chain  # noqa: F401
        except ImportError as e:
            return HealthStatus(False, f"d1_kinematics import failed: {e}")
        try:
            self._world.joint_id("arm_Joint1")
        except KeyError:
            return HealthStatus(False, "arm_Joint1 not found in SimWorld model")
        return HealthStatus(True, "UnifiedMujocoArm ready")

    def capabilities(self) -> ArmCapabilities:
        return ArmCapabilities(dof=6, supports_taught_replay=True, grasp_policies=("top_down",))

    def estimate_reachability(self, request: PickRequest) -> Reachability:
        return Reachability(True, "defers to IK in pick_and_place")

    def safe_home(self) -> ActionResult:
        self._apply_joints(np.zeros(6), gripper_frac=1.0)
        return ActionResult(True, "arm returned to home")

    def stop(self) -> ActionResult:
        return ActionResult(True, "unified arm stop (no-op)")

    def pick_and_place(self, request: PickRequest, context=None) -> ActionResult:
        """IK 规划抓放路点并在 SimWorld 中执行."""
        chain = self._ensure_chain()
        approach = list(request.approach_dir.as_list())
        height = request.approach_height_m or self._approach_height_m
        target = np.asarray(request.target_arm_base_m.as_list(), dtype=float)
        basket = np.asarray(request.basket_arm_base_m.as_list(), dtype=float)
        up = np.array([0.0, 0.0, height])

        open_frac = self._servo_to_frac(_GRIPPER_OPEN_SERVO)
        close_frac = self._servo_to_frac(_GRIPPER_CLOSE_SERVO)

        plan = [
            ("PRE_GRASP", target + up, open_frac),
            ("GRASP", target, open_frac),
            ("CLOSE", target, close_frac),
            ("LIFT", target + up, close_frac),
            ("PLACE_PRE", basket + up, close_frac),
            ("PLACE", basket, close_frac),
            ("RELEASE", basket, open_frac),
            ("RETREAT", basket + up, open_frac),
        ]

        keyframes = [(np.zeros(6), open_frac, "HOME")]
        seed = np.zeros(6)
        all_reachable = True
        for tag, xyz, grip in plan:
            q, reachable, pos_err = self._solve(chain, xyz, approach, seed)
            seed = q
            all_reachable = all_reachable and reachable
            keyframes.append((q, grip, tag))

        self._execute_trajectory(keyframes)

        if all_reachable:
            return ActionResult(True, "pick_and_place completed in SimWorld")
        return ActionResult(True, "pick_and_place completed (some waypoints unreachable)")

    def replay_taught_pick(self, request: TaughtPickRequest) -> ActionResult:
        """FK 回放教学点位."""
        grasp_q = np.deg2rad(request.grasp_joints_deg)
        place_q = np.deg2rad(request.place_joints_deg)
        open_frac = self._servo_to_frac(_GRIPPER_OPEN_SERVO)
        close_frac = self._servo_to_frac(_GRIPPER_CLOSE_SERVO)

        keyframes = [
            (np.zeros(6), open_frac, "HOME"),
            (grasp_q, open_frac, "PRE_GRASP"),
            (grasp_q, close_frac, "GRASP_CLOSE"),
            (np.zeros(6), close_frac, "LIFT"),
            (place_q, close_frac, "PLACE"),
            (place_q, open_frac, "RELEASE"),
            (np.zeros(6), open_frac, "RETREAT"),
        ]
        self._execute_trajectory(keyframes)
        return ActionResult(True, "taught replay completed in SimWorld")

    # ---- 内部方法 ----

    def _execute_trajectory(self, keyframes: List[Tuple[np.ndarray, float, str]]) -> None:
        """逐段插值执行轨迹."""
        frames_per_seg = int(self._fps * self._seconds_per_segment / self._replay_speed)
        steps_per_frame = max(1, int(round(1.0 / (self._fps * self._world.model.opt.timestep))))

        prev_q = keyframes[0][0]
        prev_grip = keyframes[0][1]
        self._apply_joints(prev_q, prev_grip)

        for i in range(1, len(keyframes)):
            target_q, target_grip, tag = keyframes[i]
            for f in range(frames_per_seg):
                alpha = (f + 1) / frames_per_seg
                q = prev_q + alpha * (target_q - prev_q)
                grip = prev_grip + alpha * (target_grip - prev_grip)
                self._apply_joints(q, grip)
                self._world.step(steps_per_frame)
            prev_q = target_q
            prev_grip = target_grip
            logger.debug("waypoint %s reached", tag)

    def _apply_joints(self, q_rad: np.ndarray, gripper_frac: float) -> None:
        """设置 D1 关节角度."""
        self._world.set_qpos(_ARM_JOINTS, q_rad)
        grip_rad = gripper_frac * 0.04
        self._world.set_qpos(_GRIPPER_JOINTS, np.array([grip_rad, grip_rad]))

    def _ensure_chain(self):
        if self._chain is not None:
            return self._chain
        from .ik.d1_kinematics import D1Chain

        self._chain = D1Chain(str(self._urdf_path), tcp_offset=(self._tcp_offset_m, 0.0, 0.0))
        return self._chain

    def _solve(self, chain, xyz, approach, seed):
        q_rad, info = chain.ik(
            target_pos=xyz.tolist(),
            approach_dir=approach,
            seed=seed,
        )
        if q_rad is None:
            return seed, False, 999.0
        pos_err = info.get("pos_err", 0.0)
        return q_rad, True, pos_err

    @staticmethod
    def _servo_to_frac(servo: float) -> float:
        return servo / _GRIPPER_OPEN_SERVO
