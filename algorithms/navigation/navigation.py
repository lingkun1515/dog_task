"""Heading-based navigation controller with optional final heading alignment.

Sim/Real 共用同一个导航逻辑。
唯一区别在于底盘速度命令的执行方式：
  Sim:  RL policy 或 sliding mode（MuJoCo 仿真步进）
  Real: ROS2 / 串口底盘命令（实机执行）
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np


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
    ):
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.arrival_threshold = arrival_threshold
        self._default_arrival_threshold = arrival_threshold
        self.heading_threshold = heading_threshold

        self._state = NavState.IDLE
        self._target = np.zeros(2)
        self._require_heading = False
        self._goal_heading: float | None = None
        self._step_count = 0
        self._heading_stable_count = 0

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

    def cancel(self) -> None:
        self._state = NavState.IDLE
        self._step_count = 0
        self._heading_stable_count = 0

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

        # 到达判定
        if distance < self.arrival_threshold and self._step_count >= min_steps:
            if not self._require_heading:
                self._state = NavState.ARRIVED
                return np.zeros(3)
            # 到达位置后，原地旋转对齐 goal_heading
            return self._align_heading(base_yaw, math.atan2(dy, dx))

        # 行进中：边走边转，不停车
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - base_yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # 线速度：远处全速，接近时按距离线性减速
        vx = min(self.linear_speed, distance * 0.8)
        # 角速度大时适当降低线速度避免绕圈
        if abs(heading_error) > 0.5:
            vx *= 0.3
        elif abs(heading_error) > 0.3:
            vx *= 0.6

        # 角速度：比例控制
        vyaw = np.clip(heading_error * 2.5, -self.angular_speed, self.angular_speed)

        self._state = NavState.MOVE
        return np.array([vx, 0.0, vyaw])

    def _align_heading(self, base_yaw: float, move_heading: float) -> np.ndarray:
        """到达后原地旋转对齐 goal_heading。"""
        final_heading = self._goal_heading if self._goal_heading is not None else move_heading
        final_error = final_heading - base_yaw
        final_error = math.atan2(math.sin(final_error), math.cos(final_error))

        if abs(final_error) < self.heading_threshold:
            self._heading_stable_count += 1
        else:
            self._heading_stable_count = 0

        if self._heading_stable_count >= 10:
            self._state = NavState.ARRIVED
            return np.zeros(3)

        self._state = NavState.ALIGN
        vyaw = np.clip(final_error * 2.0, -self.angular_speed, self.angular_speed)
        return np.array([0.0, 0.0, vyaw])
