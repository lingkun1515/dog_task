"""Simple heading-based navigation controller.

Uses direct base velocity control (sliding mode) to navigate the robot
toward a target (x, y) position. No RL policy required.

State machine: IDLE -> ALIGN -> MOVE -> ARRIVED
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
    """Heading-based point-to-point navigation."""

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

    @property
    def state(self) -> NavState:
        return self._state

    def set_target(self, x: float, y: float) -> None:
        """Set navigation target and begin."""
        self._target = np.array([x, y])
        self._state = NavState.ALIGN

    def cancel(self) -> None:
        """Cancel navigation."""
        self._state = NavState.IDLE

    def update(
        self, base_pos: np.ndarray, base_yaw: float, dt: float
    ) -> np.ndarray:
        """Compute velocity command for one control step.

        Returns:
            np.ndarray: [vx, vy, vyaw] velocity command
        """
        if self._state in (NavState.IDLE, NavState.ARRIVED, NavState.ERROR):
            return np.zeros(3)

        # Vector from robot to target
        dx = self._target[0] - base_pos[0]
        dy = self._target[1] - base_pos[1]
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)

        if distance < self.arrival_threshold:
            self._state = NavState.ARRIVED
            return np.zeros(3)

        # Heading error (wrapped to [-pi, pi])
        heading_error = target_heading - base_yaw
        heading_error = math.atan2(
            math.sin(heading_error), math.cos(heading_error)
        )

        if abs(heading_error) > self.heading_threshold:
            self._state = NavState.ALIGN
            vyaw = (
                self.angular_speed if heading_error > 0 else -self.angular_speed
            )
            return np.array([0.0, 0.0, vyaw])
        else:
            self._state = NavState.MOVE
            # Forward velocity proportional to distance (capped)
            vx = min(self.linear_speed, distance * 0.5)
            # Small correction while moving
            vyaw = heading_error * 2.0
            return np.array([vx, 0.0, vyaw])
