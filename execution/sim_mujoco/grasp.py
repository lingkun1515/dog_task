"""Grasp demo: open-loop arm joint trajectory."""

from __future__ import annotations

import time
from enum import Enum

import numpy as np


class GraspState(Enum):
    IDLE = "idle"
    MOVING_TO_GRASP = "moving_to_grasp"
    HOLDING = "holding"
    RETURNING = "returning"
    SUCCESS = "success"
    ERROR = "error"


class GraspController:
    """Open-loop grasp using predefined arm joint angles."""

    def __init__(
        self,
        grasp_angles: list[float],
        default_angles: list[float],
        step_duration: float = 0.02,
        hold_duration: float = 1.0,
    ):
        self.grasp_angles = np.array(grasp_angles, dtype=np.float64)
        self.default_arm = np.array(default_angles[12:], dtype=np.float64)
        self.step_duration = step_duration
        self.hold_duration = hold_duration

        self._state = GraspState.IDLE
        self._start_time = 0.0
        self._current_target = self.default_arm.copy()

    @property
    def state(self) -> GraspState:
        return self._state

    @property
    def current_target(self) -> np.ndarray:
        return self._current_target.copy()

    def start(self) -> None:
        """Begin grasp sequence (restartable from any terminal state)."""
        if self._state in (GraspState.MOVING_TO_GRASP, GraspState.HOLDING):
            return  # already in progress
        self._state = GraspState.MOVING_TO_GRASP
        self._start_time = time.monotonic()
        self._current_target = self.grasp_angles.copy()

    def cancel(self) -> None:
        self._state = GraspState.IDLE
        self._current_target = self.default_arm.copy()

    def update(self, current_arm: np.ndarray) -> GraspState:
        """Check progress and transition states.

        Call this each frame after setting the arm target via PD.
        """
        now = time.monotonic()

        if self._state == GraspState.MOVING_TO_GRASP:
            error = np.max(np.abs(current_arm - self.grasp_angles))
            if error < 0.35:
                self._state = GraspState.HOLDING
                self._start_time = now

        elif self._state == GraspState.HOLDING:
            if now - self._start_time > self.hold_duration:
                self._state = GraspState.RETURNING
                self._start_time = now
                self._current_target = self.default_arm.copy()

        elif self._state == GraspState.RETURNING:
            error = np.max(np.abs(current_arm - self.default_arm))
            if error < 0.35:
                self._state = GraspState.SUCCESS

        return self._state
