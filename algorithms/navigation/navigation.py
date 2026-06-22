"""Heading-based navigation controller with optional final heading alignment.

Sim/Real 共用同一个导航逻辑。
唯一区别在于底盘速度命令的执行方式：
  Sim:  RL policy 或 sliding mode（MuJoCo 仿真步进）
  Real: ROS2 / 串口底盘命令（实机执行）
"""

from __future__ import annotations

import logging
import math
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class NavState(Enum):
    IDLE = "idle"
    MOVE = "move"
    ALIGN = "align"
    ARRIVED = "arrived"
    ERROR = "error"


class NavigationController:
    """Point-to-point navigation with optional final heading alignment.

    行为：
    - MOVE 阶段：边走边转（比例角速度修正），不会停车原地转
    - ALIGN 阶段：仅在到达目标点后原地旋转对齐 goal_heading
    """

    def __init__(
        self,
        linear_speed: float = 0.5,
        angular_speed: float = 0.8,
        arrival_threshold: float = 0.5,
        heading_threshold: float = 0.1,
        fine_threshold: float = 0.08,
        stop_turn_threshold: float = 0.25,
        align_loose_threshold: float = 0.25,
        align_max_steps: int = 150,
    ) -> None:
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.arrival_threshold = arrival_threshold
        self._default_arrival_threshold = arrival_threshold
        self.heading_threshold = heading_threshold
        self.fine_threshold = fine_threshold   # camera-safe distance: stop creeping here
        self.stop_turn_threshold = stop_turn_threshold  # rad: stop forward when heading error exceeds this
        # ALIGN 阶段：严格阈值（heading_threshold）若因 RL policy 稳态震荡无法收敛，
        # 退化为宽松阈值（align_loose_threshold）+ 最少稳定步数（align_max_steps 后强制 ARRIVED）
        self.align_loose_threshold = align_loose_threshold
        self.align_max_steps = align_max_steps
        self._max_creep_steps = 60
        self._creep_steps = 0
        self._heading_aligned_steps = 0
        self._min_aligned_steps = 3  # must stay aligned this many steps before walking

        self._state = NavState.IDLE
        self._target = np.zeros(2)
        self._require_heading = False
        self._goal_heading: float | None = None
        self._step_count = 0
        self._heading_stable_count = 0
        self._align_step_count = 0  # 进入 ALIGN 后的步数计数

    @property
    def state(self) -> NavState:
        return self._state

    def set_target(
        self, x: float, y: float,
        require_heading: bool = False,
        goal_heading: float | None = None,
        arrival_threshold: float | None = None,
    ) -> None:
        """设置导航目标。

        Args:
            x, y: 目标位置
            require_heading: 到达后是否需要对齐朝向
            goal_heading: 目标朝向角(rad)。None=面朝目标方向，数值=固定角度
            arrival_threshold: 到达判定距离覆盖
        """
        self._target = np.array([x, y])
        self._require_heading = require_heading
        self._goal_heading = goal_heading
        if arrival_threshold is None:
            self.arrival_threshold = self._default_arrival_threshold
        else:
            self.arrival_threshold = arrival_threshold
        self._state = NavState.MOVE
        self._step_count = 0
        self._heading_stable_count = 0
        self._creep_steps = 0
        self._heading_aligned_steps = 0
        self._align_step_count = 0

    def cancel(self) -> None:
        self._state = NavState.IDLE
        self._step_count = 0
        self._heading_stable_count = 0
        self._creep_steps = 0
        self._heading_aligned_steps = 0
        self._align_step_count = 0

    def start_heading_align(self, goal_heading: float) -> None:
        """Rotate in-place to align heading without forward movement."""
        self._goal_heading = goal_heading
        self._require_heading = True
        self._heading_stable_count = 0
        self._align_step_count = 0
        self._step_count = 999
        self.arrival_threshold = self._default_arrival_threshold + 1.0
        self._state = NavState.MOVE  # Must set MOVE so update() doesn't early-return

    def update(
        self, base_pos: np.ndarray, base_yaw: float, dt: float
    ) -> np.ndarray:
        """Compute velocity command for one control step.

        Returns:
            np.ndarray: [vx, vy, vyaw] velocity command
        """
        if self._state in (NavState.IDLE, NavState.ARRIVED, NavState.ERROR):
            return np.zeros(3)

        self._step_count += 1

        dx = self._target[0] - base_pos[0]
        dy = self._target[1] - base_pos[1]
        distance = math.hypot(dx, dy)

        min_steps = 20

        # 行进中：边走边转，不停车
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - base_yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # 到达判定
        if distance < self.arrival_threshold and self._step_count >= min_steps:
            # Fine approach: creep toward target at reduced speed
            if distance > self.fine_threshold and self._creep_steps < self._max_creep_steps:
                self._creep_steps += 1
                vx = min(0.15, distance * 0.5)
                vyaw = np.clip(heading_error * 1.5, -0.3, 0.3)
                self._state = NavState.MOVE
                return np.array([vx, 0.0, vyaw])
            if not self._require_heading:
                self._state = NavState.ARRIVED
                return np.zeros(3)
            # 到达位置后，原地旋转对齐 goal_heading
            return self._align_heading(base_yaw, math.atan2(dy, dx))

        # Stop-and-turn: heading error 大时停车原地转，避免绕圈浪费路径
        if abs(heading_error) > self.stop_turn_threshold:
            self._heading_aligned_steps = 0
            vyaw = np.clip(heading_error * 2.0, -self.angular_speed, self.angular_speed)
            self._state = NavState.MOVE
            return np.array([0.0, 0.0, vyaw])

        # 航向足够正，走直线（带轻微修正）
        self._heading_aligned_steps += 1
        vx = min(self.linear_speed, distance * 0.8)
        vyaw = np.clip(heading_error * 1.5, -0.3, 0.3)

        self._state = NavState.MOVE
        return np.array([vx, 0.0, vyaw])

    def _align_heading(self, base_yaw: float, move_heading: float) -> np.ndarray:
        """到达后原地旋转对齐 goal_heading。

        收敛策略（避免 RL policy 稳态震荡导致永久卡死）：
          1. 严格阈值 heading_threshold 内连续 10 步 → ARRIVED
          2. 宽松阈值 align_loose_threshold 内连续 20 步 → ARRIVED
          3. ALIGN 阶段超过 align_max_steps → 强制 ARRIVED（兜底）
        """
        final_heading = self._goal_heading if self._goal_heading is not None else move_heading
        final_error = final_heading - base_yaw
        final_error = math.atan2(math.sin(final_error), math.cos(final_error))

        self._align_step_count += 1
        abs_err = abs(final_error)

        # 兜底：超步数强制到达（无论朝向误差）
        if self._align_step_count >= self.align_max_steps:
            logger.debug(
                "[nav] ALIGN 超步数 %d 强制 ARRIVED, heading_err=%.1f°",
                self._align_step_count, math.degrees(abs_err),
            )
            self._state = NavState.ARRIVED
            return np.zeros(3)

        # 严格阈值累积
        if abs_err < self.heading_threshold:
            self._heading_stable_count += 1
        # 宽松阈值累积（独立计数）
        if abs_err < self.align_loose_threshold:
            self._heading_aligned_steps += 1
        else:
            # 误差超出宽松阈值时重置严格计数（避免震荡时误判收敛）
            if abs_err >= self.heading_threshold:
                self._heading_stable_count = 0

        # 严格收敛：连续 10 步
        if self._heading_stable_count >= 10:
            self._state = NavState.ARRIVED
            return np.zeros(3)
        # 宽松收敛：连续 20 步在 ~14° 内
        if self._heading_aligned_steps >= 20:
            logger.debug(
                "[nav] ALIGN 宽松收敛 ARRIVED, heading_err=%.1f°",
                math.degrees(abs_err),
            )
            self._state = NavState.ARRIVED
            return np.zeros(3)

        self._state = NavState.ALIGN
        # 比例控制，最小角速度地板避免小误差不动
        vyaw = np.clip(final_error * 2.0, -self.angular_speed, self.angular_speed)
        if abs(vyaw) < 0.05 and abs_err > self.heading_threshold:
            vyaw = math.copysign(0.05, final_error)
        return np.array([0.0, 0.0, vyaw])
