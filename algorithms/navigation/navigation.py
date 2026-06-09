"""Heading-based navigation controller with optional goal heading.

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
    ALIGN = "align"
    MOVE = "move"
    ARRIVED = "arrived"
    ERROR = "error"


class NavigationController:
    """Point-to-point navigation with optional heading requirement."""

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
        self.heading_threshold = heading_threshold

        self._state = NavState.IDLE
        self._target = np.zeros(2)
        self._require_heading = True
        self._step_count = 0
        self._heading_stable_count = 0

    @property
    def state(self) -> NavState:
        return self._state

    def set_target(
        self, x: float, y: float,
        require_heading: bool = True,
        arrival_threshold: float | None = None,
    ) -> None:
        self._target = np.array([x, y])
        self._require_heading = require_heading
        if arrival_threshold is not None:
            self.arrival_threshold = arrival_threshold
        self._state = NavState.ALIGN if require_heading else NavState.MOVE
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
        target_heading = math.atan2(dy, dx)

        heading_error = target_heading - base_yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        min_steps = 20
        min_stable = 15

        heading_ok = abs(heading_error) < self.heading_threshold
        if heading_ok:
            self._heading_stable_count += 1
        else:
            self._heading_stable_count = 0

        if distance < self.arrival_threshold and self._step_count >= min_steps:
            if not self._require_heading:
                self._state = NavState.ARRIVED
                return np.zeros(3)
            if heading_ok and self._heading_stable_count >= min_stable:
                self._state = NavState.ARRIVED
                return np.zeros(3)
            vyaw = self.angular_speed if heading_error > 0 else -self.angular_speed
            return np.array([0.0, 0.0, vyaw])

        if self._require_heading and abs(heading_error) > self.heading_threshold:
            self._state = NavState.ALIGN
            vyaw = self.angular_speed if heading_error > 0 else -self.angular_speed
            return np.array([0.0, 0.0, vyaw])

        self._state = NavState.MOVE
        vx = min(self.linear_speed, distance * 0.5)
        vyaw = heading_error * 2.0
        return np.array([vx, 0.0, vyaw])
