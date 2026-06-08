"""Simulation scene: orchestrates RobotSim, controllers, and camera in a background loop."""

from __future__ import annotations

import threading
import time
from copy import deepcopy

import numpy as np

from execution.sim_mujoco.camera import SimulationCamera
from execution.sim_mujoco.grasp import GraspController, GraspState
from execution.sim_mujoco.navigation import NavState, NavigationController
from execution.sim_mujoco.robot_loader import RobotSim, move_base


class SimulationScene:
    """Manages the MuJoCo simulation loop and controller orchestration."""

    def __init__(self, config_path: str | None = None):
        self.robot = RobotSim(config_path)
        cfg = self.robot.cfg

        self.nav = NavigationController(
            linear_speed=cfg.get("nav_linear_speed", 0.5),
            angular_speed=cfg.get("nav_angular_speed", 0.8),
            arrival_threshold=cfg.get("nav_arrival_threshold", 0.5),
            heading_threshold=cfg.get("nav_heading_threshold", 0.1),
        )

        grasp_angles = cfg.get("grasp_arm_angles", [0.3, 1.5, -2.2, 0.5, -1.0, 0.0])
        default_angles = cfg.get("default_angles", self.robot.default_angles.tolist())
        self.grasp = GraspController(
            grasp_angles=grasp_angles,
            default_angles=default_angles,
            step_duration=cfg.get("grasp_step_duration", 0.02),
            hold_duration=cfg.get("grasp_hold_duration", 1.0),
        )

        self.camera = SimulationCamera(
            model=self.robot.model,
            data=self.robot.data,
            width=cfg.get("camera_width", 640),
            height=cfg.get("camera_height", 480),
        )

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        self._state_snapshot: dict = {}
        self._refresh_snapshot()

    # ------------------------------------------------------------------
    # Thread-safe state snapshot
    # ------------------------------------------------------------------
    def _refresh_snapshot(self) -> None:
        with self._lock:
            self._state_snapshot = {
                "base_pos": self.robot.base_position.tolist(),
                "base_yaw": float(self.robot.base_yaw),
                "joint_positions": self.robot.joint_positions.tolist(),
                "arm_positions": self.robot.get_arm_positions().tolist(),
                "nav_state": self.nav.state,
                "nav_target": self.nav._target.tolist(),
                "grasp_state": self.grasp.state,
            }

    @property
    def state(self) -> dict:
        with self._lock:
            return deepcopy(self._state_snapshot)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def reset(self) -> None:
        was_running = self._running
        if was_running:
            self.stop()
        self.robot.reset()
        self.nav.cancel()
        self.grasp.cancel()
        self._refresh_snapshot()
        if was_running:
            self.start()

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        decimation = self.robot.control_decimation
        physics_dt = self.robot.model.opt.timestep
        ctrl_dt = physics_dt * decimation
        step_counter = 0
        render_mod = max(decimation * 3, 4)

        next_wake = time.perf_counter()

        while self._running:
            # --- rate-limit to real-time ---
            now = time.perf_counter()
            if now < next_wake:
                time.sleep(min(next_wake - now, physics_dt * 0.8))
                continue

            if step_counter % decimation == 0:
                # -- navigation (kinematic sliding) --
                nav_cmd = self.nav.update(
                    self.robot.base_position,
                    self.robot.base_yaw,
                    ctrl_dt,
                )
                # Transform robot-frame velocity → world-frame displacement
                yaw = self.robot.base_yaw
                c = np.cos(yaw)
                s = np.sin(yaw)
                world_dx = float(nav_cmd[0] * c - nav_cmd[1] * s)
                world_dy = float(nav_cmd[0] * s + nav_cmd[1] * c)
                move_base(
                    self.robot.data,
                    world_dx,
                    world_dy,
                    float(nav_cmd[2]),
                    ctrl_dt,
                )

                # -- grasp --
                arm_pos = self.robot.get_arm_positions()
                self.grasp.update(arm_pos)
                self.robot.set_arm_target(self.grasp.current_target)

                # -- snapshot state for HTTP reads --
                self._refresh_snapshot()

                # -- render camera (throttled) --
                if step_counter % render_mod == 0:
                    self.camera.render()

            self.robot.step()
            step_counter += 1

            # schedule next physics step
            next_wake += physics_dt
            # reset clock if running behind by more than one step
            if next_wake < time.perf_counter() - physics_dt:
                next_wake = time.perf_counter() + physics_dt

    # ------------------------------------------------------------------
    # Command helpers (called from server endpoints)
    # ------------------------------------------------------------------
    def navigate_to(self, x: float, y: float) -> None:
        self.nav.set_target(x, y)

    def cancel_navigation(self) -> None:
        self.nav.cancel()

    def start_grasp(self) -> None:
        self.grasp.start()

    def cancel_grasp(self) -> None:
        self.grasp.cancel()

    def get_frame(self) -> bytes | None:
        return self.camera.get_frame()
