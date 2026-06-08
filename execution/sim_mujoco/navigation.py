"""Heading-based navigation controller with optional goal heading."""

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

    @property
    def state(self) -> NavState:
        return self._state

    def set_target(self, x: float, y: float, require_heading: bool = True, arrival_threshold: float | None = None) -> None:
        """Set navigation target.

        Args:
            x, y: Target position.
            require_heading: If False, only distance matters (no heading alignment).
            arrival_threshold: Override default arrival distance threshold.
        """
        self._target = np.array([x, y])
        self._require_heading = require_heading
        if arrival_threshold is not None:
            self.arrival_threshold = arrival_threshold
        self._state = NavState.ALIGN if require_heading else NavState.MOVE

    def cancel(self) -> None:
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

        dx = self._target[0] - base_pos[0]
        dy = self._target[1] - base_pos[1]
        distance = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)

        heading_error = target_heading - base_yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        if distance < self.arrival_threshold:
            if not self._require_heading:
                self._state = NavState.ARRIVED
                return np.zeros(3)
            # With heading requirement: also check heading alignment
            if abs(heading_error) < self.heading_threshold:
                self._state = NavState.ARRIVED
                return np.zeros(3)
            # Close enough but wrong heading — align in place
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
