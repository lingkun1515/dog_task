"""Simulation scene: orchestrates RobotSim, RL policy, controllers, and camera."""

from __future__ import annotations

import threading
import time
from copy import deepcopy

import numpy as np

from execution.sim_mujoco.camera import SimulationCamera
from execution.sim_mujoco.grasp import GraspController, GraspState
from execution.sim_mujoco.navigation import NavState, NavigationController
from execution.sim_mujoco.policy_runner import PolicyRunner
from execution.sim_mujoco.robot_loader import RobotSim
from execution.sim_mujoco.viewer import PassiveViewer

# Algorithm-based grasp (optional — lazy import)
_ALGO_AVAILABLE = False
try:
    from algorithms.calibration.sim_calibration import create_sim_calibration
    from algorithms.grasp.executor import SimArmExecutor
    from algorithms.grasp.planner import GraspConfig, GraspPlanner
    from algorithms.kinematics.sim_arm_ik import SimArmKinematics
    from algorithms.perception.sim_perception import SimObjectDetector

    _ALGO_AVAILABLE = True
except ImportError:
    pass


class SimulationScene:
    """Manages the MuJoCo simulation loop and controller orchestration."""

    def __init__(self, config_path: str | None = None, render_mode: str = "headless"):
        self.robot = RobotSim(config_path)
        cfg = self.robot.cfg

        self._render_mode = render_mode

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

        # ---- Algorithm-based grasp (New!) ----
        self._algo_planner: GraspPlanner | None = None
        self._algo_thread: threading.Thread | None = None
        self._algo_running = False

        algo_cfg = cfg.get("algorithms")
        if algo_cfg and _ALGO_AVAILABLE:
            try:
                tcp_body = algo_cfg["tcp_body"]
                arm_base_body = algo_cfg["arm_base_body"]
                target_bodies = algo_cfg["target_bodies"]
                target_labels = algo_cfg.get("target_labels", target_bodies)

                kinematics = SimArmKinematics(
                    model=self.robot.model,
                    data=self.robot.data,
                    tcp_body_name=tcp_body,
                    arm_base_body_name=arm_base_body,
                    qpos_arm_slice=slice(19, 25),
                )
                detector = SimObjectDetector(
                    model=self.robot.model,
                    data=self.robot.data,
                    target_body_names=target_bodies,
                    labels=target_labels,
                )
                calibration = create_sim_calibration(
                    model=self.robot.model,
                    data=self.robot.data,
                    camera_name="front_cam",
                    arm_base_body_name=arm_base_body,
                )
                executor = SimArmExecutor(self.robot)

                grasp_config = GraspConfig(
                    approach_height=algo_cfg.get("approach_height", 0.12),
                    descend_step=algo_cfg.get("descend_step", 0.015),
                    gripper_open=algo_cfg.get("gripper_open", 65),
                    gripper_close=algo_cfg.get("gripper_close", 0),
                    safe_park=algo_cfg.get("safe_park_angles", [0.0, 1.0, -0.8, 0.0, -0.3, 0.0]),
                )
                self._algo_planner = GraspPlanner(
                    kinematics=kinematics,
                    detector=detector,
                    calibration=calibration,
                    executor=executor,
                    config=grasp_config,
                    sim_mode=True,
                )
                print("[scene] Algorithm-based grasp enabled (GraspPlanner)")
            except Exception as e:
                print(f"[scene] Algorithm grasp init failed: {e} — falling back to open-loop")
                self._algo_planner = None

        # ---- RL policy (leg locomotion) ----
        try:
            self.policy = PolicyRunner(
                kps=cfg.get("kps", None),
                kds=cfg.get("kds", None),
                default_angles=default_angles,
                action_scale=cfg.get("action_scale", None),
                base_ang_vel_scale=cfg.get("base_ang_vel_scale", 0.2),
                joint_vel_scale=cfg.get("joint_vel_scale", 0.05),
                num_hist=cfg.get("num_hist", 3),
            )
            self._use_policy = True
        except Exception as e:
            print(f"[scene] RL policy unavailable: {e} — falling back to sliding mode")
            self._use_policy = False
            self.policy = None
            self.robot.kps = np.array(cfg.get("kps", self.robot.kps.tolist()), dtype=np.float64)
            self.robot.kds = np.array(cfg.get("kds", self.robot.kds.tolist()), dtype=np.float64)
            self.robot.default_angles = np.array(default_angles, dtype=np.float64)
            self.robot._target_dof_pos = self.robot.default_angles.copy()

        # Viewer BEFORE camera so the GL context is active for MjrContext.
        self._viewer: PassiveViewer | None = None
        if self._render_mode == "gui":
            try:
                self._viewer = PassiveViewer(
                    self.robot.model,
                    self.robot.data,
                    width=cfg.get("window_width", 1200),
                    height=cfg.get("window_height", 900),
                )
            except RuntimeError as e:
                print(f"[scene] GUI window unavailable: {e} — falling back to headless")
                self._render_mode = "headless"

        # Share Viewer's GL context with camera for off-screen rendering in GUI mode
        viewer_context = self._viewer._context if self._viewer is not None else None
        self.camera = SimulationCamera(
            model=self.robot.model,
            data=self.robot.data,
            width=cfg.get("camera_width", 640),
            height=cfg.get("camera_height", 480),
            context=viewer_context,
        )

        # ---- keyboard teleop (only in GUI mode) ----
        self._kb_controller = None
        self._kb_enabled = False

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
            # Map algorithm grasp state to legacy GraspState
            from execution.sim_mujoco.grasp import GraspState as GS
            _state_map = {
                "idle": GS.IDLE,
                "detecting": GS.MOVING_TO_GRASP,
                "moving_above": GS.MOVING_TO_GRASP,
                "descending": GS.MOVING_TO_GRASP,
                "gripping": GS.HOLDING,
                "lifting": GS.RETURNING,
                "parking": GS.RETURNING,
                "success": GS.SUCCESS,
                "error": GS.ERROR,
            }
            if self._algo_planner is not None and self._algo_planner.state.value != "idle":
                gs = _state_map.get(self._algo_planner.state.value, GS.IDLE)
            elif self._algo_running and self._algo_planner is not None:
                gs = _state_map.get(self._algo_planner.state.value, GS.IDLE)
            else:
                gs = self.grasp.state

            self._state_snapshot = {
                "base_pos": self.robot.base_position.tolist(),
                "base_yaw": float(self.robot.base_yaw),
                "joint_positions": self.robot.joint_positions.tolist(),
                "arm_positions": self.robot.get_arm_positions().tolist(),
                "nav_state": self.nav.state,
                "nav_target": self.nav._target.tolist(),
                "grasp_state": gs,
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

    def run(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop()

    def stop(self) -> None:
        self._running = False
        self._algo_running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._algo_thread is not None:
            self._algo_thread.join(timeout=5.0)
            self._algo_thread = None
        if self._kb_controller is not None:
            self._kb_controller.close()
            self._kb_controller = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def reset(self) -> None:
        was_running = self._running
        if was_running:
            self.stop()
        self.robot.reset()
        self.nav.cancel()
        self.grasp.cancel()
        if self.policy is not None:
            self.policy.reset()
        self._refresh_snapshot()
        if was_running:
            self.start()

    # ------------------------------------------------------------------
    # Keyboard teleop (lazy init — only in GUI mode)
    # ------------------------------------------------------------------
    def enable_keyboard(self) -> None:
        """Enable keyboard teleoperation (requires pynput)."""
        if self._kb_enabled:
            return
        try:
            from execution.sim_mujoco.keyboard_teleop import KeyboardTeleop
            self._kb_controller = KeyboardTeleop()
            self._kb_enabled = True
            print("[scene] Keyboard teleop enabled (WASD=move, QE=turn, IJKL=arm, UO=height, Space=grasp)")
        except ImportError as e:
            print(f"[scene] Keyboard teleop unavailable: {e}")

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
            now = time.perf_counter()
            if now < next_wake:
                time.sleep(min(next_wake - now, physics_dt * 0.8))
                continue

            if step_counter % decimation == 0:
                # -- keyboard teleop override --
                if self._kb_controller is not None and self._viewer is not None:
                    kb_cmd = self._kb_controller.get_command()
                    kb_vel = kb_cmd["velocity"]  # [vx, vy, vyaw]
                    kb_pos = kb_cmd["pos"]       # [px, py, pz, qw, qx, qy, qz]

                    # If keyboard is actively used (non-zero command), override nav
                    if any(abs(v) > 0.001 for v in kb_vel) or self._kb_controller.has_motion():
                        self.nav.cancel()

                    # Check grasp trigger
                    if self._kb_controller.consume_grasp():
                        if self.grasp.state in (GraspState.IDLE, GraspState.SUCCESS, GraspState.ERROR):
                            self.grasp.start()
                else:
                    kb_vel = np.zeros(3)
                    kb_pos = np.array([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])

                # -- navigation (provides velocity command) --
                if self.nav.state in (NavState.IDLE, NavState.ARRIVED):
                    # Use keyboard velocity if active, otherwise zero
                    if self._kb_enabled and any(abs(v) > 0.001 for v in kb_vel):
                        vel_cmd = np.array(kb_vel, dtype=np.float64)
                    else:
                        vel_cmd = np.zeros(3, dtype=np.float64)
                else:
                    vel_cmd = self.nav.update(
                        self.robot.base_position,
                        self.robot.base_yaw,
                        ctrl_dt,
                    )

                # Default arm pose for policy pos_command
                if self._kb_enabled and self._kb_controller is not None:
                    pos_cmd = np.array(kb_pos, dtype=np.float64)
                else:
                    pos_cmd = np.array([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

                # -- grasp --
                arm_pos = self.robot.get_arm_positions()
                if not self._algo_running:
                    self.grasp.update(arm_pos)

                # -- RL policy or sliding --
                if self._use_policy and self.policy is not None:
                    target_dof = self.policy.step(
                        qpos_joints=self.robot.joint_positions,
                        qvel_joints=self.robot.joint_velocities,
                        base_quat_wxyz=self.robot.data.qpos[3:7],
                        base_ang_vel=self.robot.data.qvel[3:6],
                        vel_command=vel_cmd,
                        pos_command=pos_cmd,
                    )
                    self.robot._target_dof_pos = target_dof
                else:
                    # Sliding mode (fallback) — translate base directly
                    yaw = self.robot.base_yaw
                    c = np.cos(yaw)
                    s = np.sin(yaw)
                    world_dx = float(vel_cmd[0] * c - vel_cmd[1] * s)
                    world_dy = float(vel_cmd[0] * s + vel_cmd[1] * c)
                    from execution.sim_mujoco.robot_loader import move_base
                    move_base(
                        self.robot.data,
                        world_dx,
                        world_dy,
                        float(vel_cmd[2]),
                        ctrl_dt,
                    )

                # Apply arm targets: algo takes priority, otherwise old grasp controller
                if self._algo_running and self.robot._algo_arm_target is not None:
                    self.robot._target_dof_pos[12:18] = self.robot._algo_arm_target
                else:
                    self.robot.set_arm_target(self.grasp.current_target)

                # -- snapshot state for HTTP reads --
                self._refresh_snapshot()

                # -- render camera (throttled) --
                if step_counter % render_mod == 0:
                    self.camera.render()
                    if self._viewer is not None:
                        if not self._viewer.sync(self.robot.model, self.robot.data):
                            self._running = False

            # PD control + physics step (every step regardless of mode)
            if self._use_policy and self.policy is not None:
                # PD is handled inside step() using _target_dof_pos
                self.robot.step()
            else:
                self.robot.step()

            step_counter += 1

            next_wake += physics_dt
            if next_wake < time.perf_counter() - physics_dt:
                next_wake = time.perf_counter() + physics_dt

    # ------------------------------------------------------------------
    # Command helpers (called from server endpoints)
    # ------------------------------------------------------------------
    def navigate_to(self, x: float, y: float, require_heading: bool = True, arrival_threshold: float | None = None) -> None:
        self.nav.set_target(x, y, require_heading=require_heading, arrival_threshold=arrival_threshold)
        self._refresh_snapshot()

    def cancel_navigation(self) -> None:
        self.nav.cancel()
        self._refresh_snapshot()

    def start_grasp(self) -> None:
        if self._algo_planner is not None:
            self._start_algo_grasp()
        else:
            self.grasp.start()
        self._refresh_snapshot()

    def _start_algo_grasp(self) -> None:
        """Run algorithm grasp in background thread."""
        if self._algo_running:
            return
        self._algo_running = True

        def _run():
            try:
                self._algo_planner.execute_full_cycle()
            except Exception as e:
                print(f"[scene] Algorithm grasp error: {e}")
            finally:
                self._algo_running = False

        self._algo_thread = threading.Thread(target=_run, daemon=True)
        self._algo_thread.start()

    def cancel_grasp(self) -> None:
        self._algo_running = False
        self.grasp.cancel()
        self._refresh_snapshot()

    def stop_all(self) -> None:
        """Cancel all active controllers (navigation + grasp)."""
        self.nav.cancel()
        self._algo_running = False
        self.grasp.cancel()
        self._refresh_snapshot()

    def get_frame(self) -> bytes | None:
        return self.camera.get_frame()
