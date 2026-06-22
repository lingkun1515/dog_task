"""Simulation scene: orchestrates RobotSim, RL policy, controllers, and camera."""

from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy

import numpy as np

logger = logging.getLogger(__name__)

from algorithms.calibration import SIM_CALIB_FILE, load_calibration_file
from algorithms.calibration.sim_calibration import create_sim_calibration
from algorithms.grasp.executor import SimArmExecutor
from algorithms.grasp.planner import GraspConfig, GraspPlanner, GraspState as PlannerState
from algorithms.kinematics.sim_arm_ik import SimArmKinematics
from algorithms.navigation import NavState, NavigationController
from algorithms.perception.perception import SimObjectDetector
from execution.sim_mujoco.camera import SimRGBDCamera
from execution.sim_mujoco.policy_runner import PolicyRunner
from execution.sim_mujoco.robot_loader import RobotSim
from execution.sim_mujoco.task_scenes import (
    ALL_SCENES,
    SCENE_GOLF_BALL,
    SCENE_LAWN_DEBRIS,
    SCENE_MATERIAL_DROP,
    SCENE_RAIN_INSPECT,
    TaskOutcome,
    TaskResult,
    TaskSceneManager,
)
from execution.sim_mujoco.viewer import PassiveViewer


class SimulationScene:
    """Manages the MuJoCo simulation loop and controller orchestration.

    Uses the same algorithm-based grasp pipeline as real robot (no open-loop fallback).
    """

    def __init__(self, config_path: str | None = None, render_mode: str = "headless"):
        self.robot = RobotSim(config_path)
        cfg = self.robot.cfg
        self._render_mode = render_mode
        self._arm_rl_enabled = cfg.get("arm_rl_enabled", True)

        default_angles = cfg.get("default_angles", self.robot.default_angles.tolist())

        # ---- Navigation ----
        self.nav = NavigationController(
            linear_speed=cfg.get("nav_linear_speed", 0.5),
            angular_speed=cfg.get("nav_angular_speed", 0.8),
            arrival_threshold=cfg.get("arrival_threshold", cfg.get("nav_arrival_threshold", 0.5)),
            heading_threshold=cfg.get("nav_heading_threshold", 0.1),
        )

        # ---- Arm RL switch ----
        self._arm_rl_enabled = cfg.get("arm_rl_enabled", True)

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
            logger.info("RL policy 已加载")
        except Exception as e:
            logger.warning("RL policy 不可用: %s — 回退到滑动模式", e)
            self._use_policy = False
            self.policy = None
            self.robot.kps = np.array(cfg.get("kps", self.robot.kps.tolist()), dtype=np.float64)
            self.robot.kds = np.array(cfg.get("kds", self.robot.kds.tolist()), dtype=np.float64)
            self.robot.default_angles = np.array(default_angles, dtype=np.float64)
            self.robot._target_dof_pos = self.robot.default_angles.copy()

        # ---- Viewer (must be before camera for GL context) ----
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
                logger.warning("GUI 窗口不可用: %s — 回退到 headless", e)
                self._render_mode = "headless"

        # ---- Camera (needs viewer GL context) ----
        viewer_context = self._viewer._context if self._viewer is not None else None
        self.camera = SimRGBDCamera(
            model=self.robot.model,
            data=self.robot.data,
            width=cfg.get("camera_width", 640),
            height=cfg.get("camera_height", 480),
            context=viewer_context,
            cam_name=cfg.get("camera_name", "front_cam"),
        )

        # ---- Algorithm-based grasp (same pipeline as real robot) ----
        self._algo_planner: GraspPlanner | None = None
        self._algo_thread: threading.Thread | None = None
        self._algo_running = False
        self._detect_enabled = False  # 检测标注开关：仅抓取阶段激活
        self._sim_detector: SimObjectDetector | None = None
        self._sit_override: np.ndarray | None = None  # sit pose override during grasp
        self._standing_pose: np.ndarray | None = None

        algo_cfg = cfg.get("algorithms")
        if not algo_cfg:
            raise RuntimeError("config 中缺少 'algorithms' 段，无法初始化抓取管线")

        arm_base_body = algo_cfg["arm_base_body"]
        target_bodies = algo_cfg["target_bodies"]
        target_labels = algo_cfg.get("target_labels", target_bodies)

        # ---- Task scene manager (handles multi-scene geometry) ----
        self.task_scene_mgr = TaskSceneManager(self.robot.model, self.robot.data)
        # 默认场景：lawn_debris（保持向后兼容）
        self.task_scene_mgr.reset_to_default()
        self._active_scene: str = SCENE_LAWN_DEBRIS

        self._sim_detector = SimObjectDetector(
            model=self.robot.model,
            data=self.robot.data,
            camera=self.camera,
            target_body_names=target_bodies,
            labels=target_labels,
        )

        arm_kinematics_type = algo_cfg.get("arm_kinematics", "d1")
        if arm_kinematics_type == "d1":
            from algorithms.kinematics.sim_d1_ik import SimD1Kinematics
            kinematics = SimD1Kinematics(
                model=self.robot.model,
                data=self.robot.data,
                arm_base_body_name=arm_base_body,
            )
        else:
            tcp_body = algo_cfg.get("tcp_body", arm_base_body)
            kinematics = SimArmKinematics(
                model=self.robot.model,
                data=self.robot.data,
                tcp_body_name=tcp_body,
                arm_base_body_name=arm_base_body,
                qpos_arm_slice=slice(19, 25),
            )

        try:
            calibration = load_calibration_file(SIM_CALIB_FILE)
            logger.info("仿真标定已从文件加载 (method=%s)", calibration.method)
        except FileNotFoundError:
            calibration = create_sim_calibration(
                model=self.robot.model,
                data=self.robot.data,
                camera_name="front_cam",
                arm_base_body_name=arm_base_body,
            )
            logger.info("仿真标定已计算并保存到 %s", SIM_CALIB_FILE)
        executor = SimArmExecutor(self.robot)

        grasp_config = GraspConfig(
            approach_height=algo_cfg.get("approach_height", 0.18),
            descend_step=algo_cfg.get("descend_step", 0.015),
            gripper_open=algo_cfg.get("gripper_open", 65),
            gripper_close=algo_cfg.get("gripper_close", 0),
            safe_park=algo_cfg.get("safe_park_angles", [0.0, 1.0, -0.8, 0.0, -0.3, 0.0]),
            move_wait=algo_cfg.get("move_wait", 2.0),
            gripper_wait=algo_cfg.get("gripper_wait", 0.5),
            max_attempts=algo_cfg.get("max_attempts", 3),
        )
        self._algo_planner = GraspPlanner(
            kinematics=kinematics,
            detector=self._sim_detector,
            calibration=calibration,
            executor=executor,
            config=grasp_config,
        )
        # Wire up reposition callback for IK failure recovery
        self._algo_planner.reposition_fn = self._reposition_for_grasp
        logger.info("算法抓取管线已初始化 (与实机一致)")

        # ---- Sit-down pose for grasp precision ----
        default_legs = np.array(default_angles[:12], dtype=np.float64)
        self._standing_pose = default_legs.copy()
        self._sit_pose = default_legs.copy()
        # Rear legs fold more (thigh increases, calf more negative)
        self._sit_pose[6] = 0.0    # RR_hip
        self._sit_pose[7] = 1.5    # RR_thigh (standing=1.0)
        self._sit_pose[8] = -2.2   # RR_calf  (standing=-1.5)
        self._sit_pose[9] = 0.0    # RL_hip
        self._sit_pose[10] = 1.5   # RL_thigh
        self._sit_pose[11] = -2.2  # RL_calf
        # Front legs slightly more bent for stability
        self._sit_pose[1] = 1.0    # FR_thigh (standing=0.8)
        self._sit_pose[2] = -1.7   # FR_calf  (standing=-1.5)
        self._sit_pose[4] = 1.0    # FL_thigh
        self._sit_pose[5] = -1.7   # FL_calf
        logger.info("坐下姿态已配置")

        # ---- keyboard teleop (only in GUI mode) ----
        self._kb_controller = None
        self._kb_enabled = False

        # ---- Task execution state ----
        self._last_task_result: TaskResult | None = None
        # golf_ball 场景：已回收的目标体（用于 multi-grasp 进度）
        self._collected_targets: list[str] = []

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
            if self._algo_planner is not None:
                gs = self._algo_planner.state.value
            else:
                gs = "idle"

            self._state_snapshot = {
                "base_pos": self.robot.base_position.tolist(),
                "base_yaw": float(self.robot.base_yaw),
                "joint_positions": self.robot.joint_positions.tolist(),
                "arm_positions": self.robot.get_arm_positions().tolist(),
                "nav_state": self.nav.state,
                "nav_target": self.nav._target.tolist(),
                "grasp_state": gs,
                "active_scene": self._active_scene,
                "task_result": (
                    self._last_task_result.to_dict()
                    if self._last_task_result is not None else None
                ),
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
        if self.policy is not None:
            self.policy.reset()
        # 重置场景几何到默认 lawn_debris
        self.task_scene_mgr.reset_to_default()
        self._active_scene = SCENE_LAWN_DEBRIS
        self._sim_detector.set_targets(["target_sphere"], ["ball"])
        self._last_task_result = None
        self._collected_targets.clear()
        self._refresh_snapshot()
        if was_running:
            self.start()

    # ------------------------------------------------------------------
    # Keyboard teleop (lazy init — only in GUI mode)
    # ------------------------------------------------------------------
    def enable_keyboard(self) -> None:
        if self._kb_enabled:
            return
        try:
            from execution.sim_mujoco.keyboard_teleop import KeyboardTeleop
            self._kb_controller = KeyboardTeleop()
            self._kb_enabled = True
            logger.info("键盘遥控已启用 (WASD=移动, QE=转向, Space=抓取)")
        except ImportError as e:
            logger.warning("键盘遥控不可用: %s", e)

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
                    kb_vel = kb_cmd["velocity"]
                    kb_pos = kb_cmd["pos"]

                    if any(abs(v) > 0.001 for v in kb_vel) or self._kb_controller.has_motion():
                        self.nav.cancel()

                    if self._kb_controller.consume_grasp():
                        self.start_grasp()
                else:
                    kb_vel = np.zeros(3)
                    kb_pos = np.array([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])

                # -- navigation (provides velocity command) --
                if self.nav.state in (NavState.IDLE, NavState.ARRIVED):
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
                    # RL 不控制手臂时，强制保持默认收缩姿态
                    if not self._arm_rl_enabled:
                        self.robot._target_dof_pos[12:18] = self.robot.default_angles[12:18]
                    # Sit-down override: replace leg targets during grasp
                    if self._sit_override is not None:
                        self.robot._target_dof_pos[0:12] = self._sit_override
                else:
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

                # Apply arm targets from algorithm grasp (overrides RL/default)
                if self._algo_running and self.robot._algo_arm_target is not None:
                    self.robot._target_dof_pos[12:18] = self.robot._algo_arm_target

                # -- snapshot state for HTTP reads --
                self._refresh_snapshot()

                # -- render camera (throttled) --
                if step_counter % render_mod == 0:
                    self.camera.render()
                    if self._sim_detector is not None and self._detect_enabled:
                        try:
                            rgb = self.camera.get_rgb()
                            if rgb is not None:
                                annotated = self._sim_detector.annotate_frame(rgb)
                                self.camera.set_rgb_frame(annotated)
                        except Exception as e:
                            if step_counter % (render_mod * 30) == 0:
                                logger.debug("帧标注异常: %s", e)
                    if self._viewer is not None:
                        if not self._viewer.sync(self.robot.model, self.robot.data):
                            self._running = False

            # Physics step
            self.robot.step()
            # Apply gripper constraint (ball follows TCP when gripper closed)
            self.robot.apply_gripper_constraint()
            step_counter += 1

            next_wake += physics_dt
            if next_wake < time.perf_counter() - physics_dt:
                next_wake = time.perf_counter() + physics_dt

    # ------------------------------------------------------------------
    # Command helpers (called from server endpoints)
    # ------------------------------------------------------------------
    def navigate_to(self, x: float, y: float, require_heading: bool = False, goal_heading: float | None = None, arrival_threshold: float | None = None) -> None:
        logger.info("导航目标设置: (%.2f, %.2f) heading=%s", x, y, goal_heading)
        self.nav.set_target(x, y, require_heading=require_heading, goal_heading=goal_heading, arrival_threshold=arrival_threshold)
        self._refresh_snapshot()

    def cancel_navigation(self) -> None:
        self.nav.cancel()
        self._refresh_snapshot()

    def start_heading_align(self, goal_heading: float) -> None:
        """Pure in-place heading alignment (no forward movement)."""
        self.nav.start_heading_align(goal_heading)
        self._refresh_snapshot()

    # ------------------------------------------------------------------
    # 任务场景配置（被 /api/scene/setup 端点调用）
    # ------------------------------------------------------------------
    def configure_scene(
        self,
        scene: str,
        target_x: float,
        target_y: float,
        home_x: float,
        home_y: float,
    ) -> None:
        """激活指定任务场景：重定位 body + 切换感知器目标。

        Args:
            scene: 场景名（ALL_SCENES 之一）
            target_x/y: 任务目标点（场景几何体摆放参考点）
            home_x/y: 充电桩位置（material_drop 的 payload 初始位置）
        """
        if scene not in ALL_SCENES:
            raise ValueError(f"未知场景: {scene!r}")

        logger.info(
            "[configure_scene] scene=%s target=(%.2f,%.2f) home=(%.2f,%.2f)",
            scene, target_x, target_y, home_x, home_y,
        )
        # 估算目标 z（球体/方块半径，贴近地面）
        target_z = 0.04 if scene == SCENE_LAWN_DEBRIS else 0.03
        home_z = 0.05

        self.task_scene_mgr.setup_scene(
            scene,
            target_pos=(target_x, target_y, target_z),
            home_pos=(home_x, home_y, home_z),
        )
        self._active_scene = scene

        # 切换感知器目标集
        self._sim_detector.set_targets(
            self.task_scene_mgr.target_body_names,
            self.task_scene_mgr.target_labels,
        )
        self._last_task_result = None
        self._refresh_snapshot()

    def enable_detect(self) -> None:
        """启用检测标注（调度端进入抓取准备阶段时调用）。"""
        self._detect_enabled = True

    def disable_detect(self) -> None:
        """关闭检测标注。"""
        self._detect_enabled = False

    def start_grasp(self) -> None:
        """启动抓取/作业流程。根据当前活动场景分发：
          - lawn_debris / golf_ball : 走标准抓取管线（单次 / 多目标）
          - material_drop           : 走投放流程（从 home 抓起方块 → 携带到 target → 释放）
          - rain_inspect            : 巡检流程（不抓取，仅检测+记录）
        """
        if self._algo_running:
            logger.warning("算法抓取已在运行中 — 忽略重复请求")
            return
        logger.info("启动作业流程: scene=%s", self._active_scene)
        self._detect_enabled = True
        self._algo_running = True
        self._last_task_result = None
        self._collected_targets.clear()

        # 抓取/投放类场景需要坐下；巡检场景保持站立
        if self.task_scene_mgr.requires_grasp or self.task_scene_mgr.requires_drop:
            self._sit_override = self._sit_pose.copy()

        # 根据场景选择 worker
        if self.task_scene_mgr.requires_inspect:
            worker = self._run_inspect
        elif self._active_scene == SCENE_GOLF_BALL:
            worker = self._run_multi_grasp
        elif self.task_scene_mgr.requires_drop:
            worker = self._run_material_drop
        else:
            worker = self._run_single_grasp

        def _run():
            try:
                # 抓取/投放场景：等待坐下稳定
                if self.task_scene_mgr.requires_grasp or self.task_scene_mgr.requires_drop:
                    logger.info("等待坐下稳定...")
                    time.sleep(2.5)
                    logger.info("坐下完成，开始作业")
                worker()
            except Exception as e:
                logger.error("作业流程异常: %s", e, exc_info=True)
                self._last_task_result = TaskResult(
                    scene=self._active_scene,
                    outcome=TaskOutcome.FAILED,
                    message=f"异常: {e}",
                )
            finally:
                self._algo_running = False
                self._detect_enabled = False
                # Restore standing pose
                self._sit_override = None
                self._algo_planner._state = PlannerState.IDLE
                logger.info("作业结束，恢复站立姿态: result=%s",
                            self._last_task_result.outcome.value if self._last_task_result else "n/a")
                self._refresh_snapshot()

        self._algo_thread = threading.Thread(target=_run, daemon=True)
        self._algo_thread.start()
        self._refresh_snapshot()

    # ------------------------------------------------------------------
    # 场景化作业实现
    # ------------------------------------------------------------------
    def _run_single_grasp(self) -> None:
        """lawn_debris：单目标抓取（原有 execute_full_cycle）。"""
        logger.info("开始单目标抓取 (lawn_debris)")
        self._algo_planner.execute_full_cycle()
        ok = self._algo_planner.state == PlannerState.SUCCESS
        self._last_task_result = TaskResult(
            scene=SCENE_LAWN_DEBRIS,
            outcome=TaskOutcome.SUCCESS if ok else TaskOutcome.FAILED,
            details={"planner_state": self._algo_planner.state.value},
            message="单目标抓取成功" if ok else "单目标抓取失败",
        )

    def _run_multi_grasp(self) -> None:
        """golf_ball：逐个抓取多目标，成功一个就在原地保持，记录进度。

        简化策略：复用单次 execute_full_cycle，但每次只感知当前未被收集
        的 body（通过 detector.set_targets 缩小目标集）。每次成功抓取后，
        把该 body 从感知列表移除（视为已入篮）。
        """
        from algorithms.grasp.planner import GraspState as GS
        total = len(self.task_scene_mgr.target_body_names)
        logger.info("开始多目标回收 (golf_ball): 共 %d 个目标", total)

        # 初始目标集：所有 golf 球
        active_bodies = list(self.task_scene_mgr.target_body_names)
        active_labels = list(self.task_scene_mgr.target_labels)
        success_count = 0

        for idx in range(total):
            if not active_bodies:
                break
            # 缩小感知器目标集
            self._sim_detector.set_targets(active_bodies, active_labels)
            self._algo_planner._state = GS.IDLE
            logger.info("[golf] 第 %d/%d 个目标，剩余感知: %s",
                        idx + 1, total, active_bodies)

            self._algo_planner.execute_full_cycle()
            if self._algo_planner.state == GS.SUCCESS:
                collected = active_bodies.pop(0)
                self._collected_targets.append(collected)
                success_count += 1
                logger.info("[golf] 已回收 %s (%d/%d)", collected, success_count, total)
                # 把已回收的球移到 park 区（视觉上「入篮」）
                self.task_scene_mgr._relocate_body(collected, (100.0, 100.0, -5.0))
                import mujoco
                mujoco.mj_forward(self.robot.model, self.robot.data)
                # 张开夹爪，准备下一次
                cfg = self._algo_planner._config
                self.robot._gripper_closed = False
                time.sleep(0.3)
            else:
                logger.warning("[golf] 目标 %d 抓取失败 (state=%s)，跳过",
                              idx + 1, self._algo_planner.state.value)
                # 跳过该球（从感知集移除，避免循环卡死）
                active_bodies.pop(0)

        # 恢复默认感知目标集
        self._sim_detector.set_targets(
            self.task_scene_mgr.target_body_names,
            self.task_scene_mgr.target_labels,
        )

        if success_count == total:
            outcome = TaskOutcome.SUCCESS
            msg = f"全部 {total} 个目标回收成功"
        elif success_count > 0:
            outcome = TaskOutcome.PARTIAL
            msg = f"部分成功: {success_count}/{total}"
        else:
            outcome = TaskOutcome.FAILED
            msg = f"全部 {total} 个目标均失败"

        self._last_task_result = TaskResult(
            scene=SCENE_GOLF_BALL,
            outcome=outcome,
            details={
                "total": total,
                "collected": success_count,
                "collected_bodies": list(self._collected_targets),
            },
            message=msg,
        )

    def _run_material_drop(self) -> None:
        """material_drop：投放流程。

        简化策略：payload 初始在 home（充电桩）。机器人到 target 后启动
        本流程，机械臂下降→「虚拟携带」payload（用 gripperConstraint 把
        payload 绑定到 TCP）→抬起→原地释放（payload 落到 target）。
        实际仿真中机器人已在 target，所以这里做的是「让方块从 home 传送
        到 target 上方 → 释放」，模拟「携带到达」的语义。
        """
        import mujoco
        logger.info("开始物料投放 (material_drop)")

        # 1. 把 payload_box 从 home 传送到机械臂 TCP（模拟已携带）
        tcp_site_id = mujoco.mj_name2id(
            self.robot.model, mujoco.mjtObj.mjOBJ_SITE, "d1_tcp"
        )
        if tcp_site_id < 0:
            self._last_task_result = TaskResult(
                scene=SCENE_MATERIAL_DROP,
                outcome=TaskOutcome.FAILED,
                message="找不到 d1_tcp site，无法投放",
            )
            return

        # 用 RobotSim 的 gripper constraint 把 payload 绑定到 TCP
        payload_bid = mujoco.mj_name2id(
            self.robot.model, mujoco.mjtObj.mjOBJ_BODY, "payload_box"
        )
        if payload_bid < 0:
            self._last_task_result = TaskResult(
                scene=SCENE_MATERIAL_DROP,
                outcome=TaskOutcome.FAILED,
                message="找不到 payload_box",
            )
            return

        # 临时改写 robot 的抓取目标为 payload_box
        original_ball_id = self.robot._grasp_target_body_id
        original_qpos_addr = self.robot._ball_qpos_addr
        original_qvel_addr = self.robot._ball_qvel_addr

        jnt_id = self.robot.model.body_jntadr[payload_bid]
        self.robot._grasp_target_body_id = payload_bid
        self.robot._ball_qpos_addr = int(self.robot.model.jnt_qposadr[jnt_id])
        self.robot._ball_qvel_addr = int(self.robot.model.jnt_dofadr[jnt_id])

        try:
            # 「抓取」payload（绑定到 TCP）
            self.robot._gripper_closed = True
            time.sleep(1.0)  # 让约束稳定
            logger.info("[drop] payload 已绑定到 TCP")

            # 抬升一点，模拟携带
            cfg = self._algo_planner._config
            self._algo_planner._executor.move_to_joints(
                cfg.safe_park + [cfg.gripper_open], mode=1, wait_time=cfg.move_wait,
            )

            # 「释放」payload：松开夹爪，payload 自由下落
            self.robot._gripper_closed = False
            time.sleep(1.5)  # 等待物理下落
            logger.info("[drop] payload 已释放")

            # 检查 payload 是否落在 target 附近
            payload_pos = self.robot.data.xpos[payload_bid].copy()
            base_pos = self.robot.base_position
            drop_err = float(np.linalg.norm(payload_pos[:2] - base_pos[:2]))

            if drop_err < 0.5:
                outcome = TaskOutcome.SUCCESS
                msg = f"物料投放成功（落点偏差 {drop_err*100:.1f}cm）"
            else:
                outcome = TaskOutcome.PARTIAL
                msg = f"物料投放偏移较大（{drop_err*100:.1f}cm）"

            self._last_task_result = TaskResult(
                scene=SCENE_MATERIAL_DROP,
                outcome=outcome,
                details={
                    "drop_position": payload_pos.tolist(),
                    "drop_error_m": drop_err,
                },
                message=msg,
            )
        finally:
            # 恢复 robot 的原始抓取目标（target_sphere）
            self.robot._grasp_target_body_id = original_ball_id
            self.robot._ball_qpos_addr = original_qpos_addr
            self.robot._ball_qvel_addr = original_qvel_addr
            self.robot._gripper_closed = False

    def _run_inspect(self) -> None:
        """rain_inspect：巡检流程。不抓取，仅检测积水点+记录结果。"""
        logger.info("开始巡检 (rain_inspect)")
        cfg = self._algo_planner._config

        # 把感知器目标切换到积水点
        original_bodies = list(self._sim_detector._target_body_names)
        original_labels = list(self._sim_detector._labels)
        self._sim_detector.set_targets(
            self.task_scene_mgr.target_body_names,
            self.task_scene_mgr.target_labels,
        )

        try:
            # 执行 3 次检测，按 body_name 去重（每个目标体计一次）
            detections_log: list[dict] = []
            detected_bodies: set[str] = set()
            for i in range(3):
                dets = self._sim_detector.detect()
                for d in dets:
                    key = d.body_name or d.label  # 优先 body_name 去重
                    detected_bodies.add(key)
                    detections_log.append({
                        "body_name": d.body_name,
                        "label": d.label,
                        "depth_m": float(d.depth_m),
                        "confidence": float(d.confidence),
                    })
                time.sleep(0.2)

            # 机械臂做一个「扫描」动作（safe_park → 前伸 → 归位）作为可视化
            self._algo_planner._executor.move_to_joints(
                cfg.safe_park + [cfg.gripper_open], mode=1, wait_time=cfg.move_wait,
            )
            time.sleep(0.5)

            unique_count = len(detected_bodies)
            logger.info("[inspect] 检测到 %d 个标记（按 body 去重 %d）",
                       len(detections_log), unique_count)

            self._last_task_result = TaskResult(
                scene=SCENE_RAIN_INSPECT,
                outcome=TaskOutcome.SUCCESS,
                details={
                    "detections": detections_log,
                    "unique_bodies": sorted(detected_bodies),
                    "puddle_count": unique_count,
                },
                message=f"巡检完成：发现 {unique_count} 个积水点",
            )
        finally:
            # 恢复感知器目标
            self._sim_detector.set_targets(original_bodies, original_labels)

    def cancel_grasp(self) -> None:
        self._algo_running = False
        self._refresh_snapshot()

    def _reposition_for_grasp(self) -> None:
        """Stand up → wait → sit down → wait to change arm-ball geometry on IK failure."""
        logger.info("[reposition] 站起...")
        self._sit_override = None
        time.sleep(2.0)
        logger.info("[reposition] 再次坐下...")
        self._sit_override = self._sit_pose.copy()
        time.sleep(2.5)
        logger.info("[reposition] 坐下稳定，继续抓取")

    def stop_all(self) -> None:
        self.nav.cancel()
        self._algo_running = False
        self._refresh_snapshot()

    def get_frame(self) -> bytes | None:
        return self.camera.get_frame()
