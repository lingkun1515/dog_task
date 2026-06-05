"""工作流编排器: 固定底座与移动闭环两种 Demo 的状态机."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

from ..core.interfaces import ArmController, CameraController, MobilityController
from ..core.models import (
    ActionResult,
    PickRequest,
    TargetObservation,
    TaskEvent,
    Vector3,
    VelocityCommand,
    WorkflowState,
)
from ..core.transforms import RigidTransform, SweetZone


@dataclass(frozen=True)
class WorkflowSettings:
    """从 JSON ``workflow`` 段解析出的工作流参数."""

    basket_arm_base_m: Vector3
    sweet_zone: SweetZone
    grasp_offset_by_class: Mapping[str, Vector3]
    approach_height_m: float = 0.10
    approach_distance_m: float = 0.70
    image_center_tolerance_px: float = 40.0
    stop_settle_s: float = 0.8
    max_steps: int = 50
    stable_timeout_s: float = 8.0
    search_turn_rps: float = 0.15
    approach_speed_mps: float = 0.12
    align_speed_mps: float = 0.06
    align_turn_rps: float = 0.12
    go2_forward_in_arm_base_xy: tuple[float, float] = (1.0, 0.0)
    go2_left_in_arm_base_xy: tuple[float, float] = (0.0, 1.0)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "WorkflowSettings":
        """从配置字典构造; 缺失字段使用类默认值."""
        offsets = {
            str(name): Vector3.from_value(value, f"grasp_offset_by_class.{name}")
            for name, value in dict(data.get("grasp_offset_by_class", {})).items()
        }
        forward = tuple(float(item) for item in data.get("go2_forward_in_arm_base_xy", [1, 0]))
        left = tuple(float(item) for item in data.get("go2_left_in_arm_base_xy", [0, 1]))
        if len(forward) != 2 or len(left) != 2:
            raise ValueError("Go2 direction vectors must contain exactly two values")
        return cls(
            basket_arm_base_m=Vector3.from_value(data["basket_arm_base_m"], "basket_arm_base_m"),
            sweet_zone=SweetZone.from_dict(dict(data["sweet_zone"])),
            grasp_offset_by_class=offsets,
            approach_height_m=float(data.get("approach_height_m", 0.15)),
            approach_distance_m=float(data.get("approach_distance_m", 0.70)),
            image_center_tolerance_px=float(data.get("image_center_tolerance_px", 40.0)),
            stop_settle_s=float(data.get("stop_settle_s", 0.8)),
            max_steps=int(data.get("max_steps", 50)),
            stable_timeout_s=float(data.get("stable_timeout_s", 8.0)),
            search_turn_rps=float(data.get("search_turn_rps", 0.15)),
            approach_speed_mps=float(data.get("approach_speed_mps", 0.12)),
            align_speed_mps=float(data.get("align_speed_mps", 0.06)),
            align_turn_rps=float(data.get("align_turn_rps", 0.12)),
            go2_forward_in_arm_base_xy=forward,
            go2_left_in_arm_base_xy=left,
        )


class DemoOrchestrator:
    """固定底座(fixed-once)与移动闭环(mobile-once)的轻量状态机编排器."""

    def __init__(
        self,
        *,
        arm: ArmController,
        camera: CameraController,
        mobility: MobilityController,
        camera_to_arm_base: RigidTransform,
        settings: WorkflowSettings,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
        ui_base_url: str = "http://127.0.0.1:8765",
    ) -> None:
        self.arm = arm
        self.camera = camera
        self.mobility = mobility
        self.camera_to_arm_base = camera_to_arm_base
        self.settings = settings
        self.sleep = sleep
        self.log = log
        self._running = True
        self._ui_base_url = ui_base_url
        self._ui_task_dispatched = False

    def healthcheck(self) -> dict[str, object]:
        """分别检查三个模块, 返回 name -> HealthStatus 字典."""
        return {
            "arm": self.arm.healthcheck(),
            "camera": self.camera.healthcheck(),
            "mobility": self.mobility.healthcheck(),
        }

    def request_stop(self) -> None:
        """外部调用此方法可优雅退出 run_fixed_base_once 的无限轮询."""
        self._running = False

    def run_fixed_base_once(self) -> ActionResult:
        """固定底座无限轮询抓取: 持续读取相机 -> 可达性判断 -> 抓放, 直到成功或外部中断."""
        not_ready = self._not_ready("arm", "camera")
        if not_ready:
            return ActionResult(False, not_ready)

        self._running = True
        poll_interval = 0.5
        self._ensure_ui_task_dispatched()
        self._sync_ui_status("go_to_B", "仿真开始，搜索目标...")

        while self._running:
            observation = self.camera.get_target(require_stable=True)
            if observation is None:
                self.log("未检测到目标, 继续轮询...")
                self.sleep(poll_interval)
                continue

            target = self._target_in_arm_base(observation)

            reachability = self.arm.estimate_reachability(
                PickRequest(
                    target_arm_base_m=target,
                    basket_arm_base_m=self.settings.basket_arm_base_m,
                    class_name=observation.class_name,
                    approach_height_m=self.settings.approach_height_m,
                )
            )
            if not reachability.reachable:
                self.log(f"目标不可达: {reachability.message}, 继续轮询...")
                self.sleep(poll_interval)
                continue

            self._release_camera_view()
            result = self._pick_and_place(observation, target)
            if result.success:
                self._sync_ui_status("done", "抓取放置完成")
                return result
            self._sync_ui_status("go_to_B", f"抓取失败: {result.message}, 继续轮询...")
            self.log(f"抓取失败: {result.message}, 继续轮询新目标...")
            self.sleep(poll_interval)

        self._sync_ui_status("failed", "轮询被外部中断")
        return ActionResult(False, "轮询被外部中断(request_stop)")

    def run_walk_and_pick(
        self,
        walk_distance_m: float = 5.0,
        walk_speed_mps: float = 1.0,
    ) -> ActionResult:
        """完整流程: Go2 站立 -> 前进指定距离 -> 趴下 -> 摄像头识别 -> 机械臂抓取."""
        not_ready = self._not_ready("arm", "camera", "mobility")
        if not_ready:
            return ActionResult(False, not_ready)

        self.log("[1/5] Go2 站立...")
        self._ensure_ui_task_dispatched()
        self._sync_ui_status("go_to_B", "Go2 站立准备前进")
        stand_result = self.mobility.set_posture("stand")
        if not stand_result.success:
            return ActionResult(False, f"站立失败: {stand_result.message}")

        self.log(f"[2/5] Go2 前进 {walk_distance_m}m ...")
        walk_fn = getattr(self.mobility, "walk_forward", None)
        if callable(walk_fn):
            walk_result = walk_fn(walk_distance_m, speed_mps=walk_speed_mps, log=self.log)
        else:
            walk_result = self._walk_by_velocity(walk_distance_m, walk_speed_mps)
        if not walk_result.success:
            return ActionResult(False, f"前进失败: {walk_result.message}")

        self.log("[3/5] Go2 趴下, 准备抓取...")
        self._sync_ui_status("arrived_B_confirmed", "Go2 到达目标区域，准备抓取")
        down_result = self.mobility.set_posture("stand_down")
        if not down_result.success:
            return ActionResult(False, f"趴下失败: {down_result.message}")
        self.sleep(1.0)

        self.log("[4/5] 摄像头开始识别目标...")
        self.log("[5/5] 识别到目标后自动抓取投放...")
        pick_result = self.run_fixed_base_once()

        return pick_result

    def _walk_by_velocity(self, distance_m: float, speed_mps: float) -> ActionResult:
        """用 set_velocity + 里程计闭环实现定距前进."""
        import math as _math

        start_state = self.mobility.get_state()
        start_x, start_y = start_state.pose.x_m, start_state.pose.y_m
        self.log(f"  使用 set_velocity 前进 {distance_m:.1f}m (speed={speed_mps:.1f}m/s)")
        self.mobility.set_velocity(VelocityCommand(linear_x_mps=speed_mps))

        timeout_s = distance_m / speed_mps * 2.0 + 3.0
        t0 = self._time_monotonic()
        while self._time_monotonic() - t0 < timeout_s:
            self.sleep(0.05)
            state = self.mobility.get_state()
            traveled = _math.hypot(state.pose.x_m - start_x, state.pose.y_m - start_y)
            if traveled >= distance_m:
                break

        self.mobility.stop()
        self.sleep(0.5)
        final_state = self.mobility.get_state()
        actual_dist = _math.hypot(final_state.pose.x_m - start_x, final_state.pose.y_m - start_y)
        self.log(f"  实际前进: {actual_dist:.2f}m (目标: {distance_m:.1f}m)")
        return ActionResult(True, f"前进 {actual_dist:.2f}m 完成")

    @staticmethod
    def _time_monotonic() -> float:
        import time as _time
        return _time.monotonic()

    def _release_camera_view(self) -> None:
        """若摄像头驱动支持, 关闭实时 OpenCV 预览窗口."""
        closer = getattr(self.camera, "close_visualization", None)
        if callable(closer):
            closer()

    def run_mobile_single(self) -> ActionResult:
        """移动闭环: Go2 搜索/靠近/对准/停稳/重定位后抓取一次."""
        not_ready = self._not_ready("arm", "camera", "mobility")
        if not_ready:
            return ActionResult(False, not_ready)
        state = WorkflowState.SEARCH
        observation: TargetObservation | None = None
        target: Vector3 | None = None
        self._ensure_ui_task_dispatched()
        self._sync_ui_status("go_to_B", "移动闭环开始，搜索目标...")
        try:
            for step in range(1, self.settings.max_steps + 1):
                self.log(f"[{step:02d}] {state.value}")
                if state == WorkflowState.SEARCH:
                    observation = self.camera.get_target()
                    if observation is None:
                        self.mobility.set_velocity(
                            VelocityCommand(angular_z_rps=self.settings.search_turn_rps)
                        )
                        continue
                    state = WorkflowState.APPROACH

                elif state == WorkflowState.APPROACH:
                    observation = self.camera.get_target()
                    if observation is None:
                        state = WorkflowState.SEARCH
                        continue
                    if observation.depth_m > self.settings.approach_distance_m:
                        self.mobility.set_velocity(
                            VelocityCommand(
                                linear_x_mps=self.settings.approach_speed_mps,
                                angular_z_rps=self._image_turn(observation),
                            )
                        )
                        continue
                    state = WorkflowState.CENTER

                elif state == WorkflowState.CENTER:
                    observation = self.camera.get_target()
                    if observation is None:
                        state = WorkflowState.SEARCH
                        continue
                    turn = self._image_turn(observation)
                    if turn != 0.0:
                        self.mobility.set_velocity(VelocityCommand(angular_z_rps=turn))
                        continue
                    state = WorkflowState.ALIGN

                elif state == WorkflowState.ALIGN:
                    observation = self.camera.get_target()
                    if observation is None:
                        state = WorkflowState.SEARCH
                        continue
                    target = self._target_in_arm_base(observation)
                    if self.settings.sweet_zone.contains(target):
                        state = WorkflowState.STOP
                        continue
                    zone = self.settings.sweet_zone
                    if not zone.z_min_m <= target.z <= zone.z_max_m:
                        return ActionResult(False, f"target height outside arm sweet zone: {target.z}")
                    self.mobility.set_velocity(self._alignment_command(target))

                elif state == WorkflowState.STOP:
                    stop_result = self.mobility.stop()
                    if not stop_result.success:
                        return stop_result
                    self.sleep(self.settings.stop_settle_s)
                    if not self.mobility.is_stopped():
                        return ActionResult(False, "Go2 did not report a stopped state")
                    state = WorkflowState.RECHECK

                elif state == WorkflowState.RECHECK:
                    observation = self.camera.get_target(require_stable=True)
                    if observation is None:
                        state = WorkflowState.ALIGN
                        continue
                    target = self._target_in_arm_base(observation)
                    if not self.settings.sweet_zone.contains(target):
                        state = WorkflowState.ALIGN
                        continue
                    state = WorkflowState.PICK_AND_PLACE

                elif state == WorkflowState.PICK_AND_PLACE:
                    assert observation is not None and target is not None
                    self._sync_ui_status("arrived_B_confirmed", "目标已定位，开始抓取")
                    result = self._pick_and_place(observation, target)
                    if not result.success:
                        self._sync_ui_status("failed", result.message)
                        return result
                    self.log(f"[{step:02d}] {WorkflowState.DONE.value}")
                    self._sync_ui_status("done", "移动闭环任务完成")
                    return result
        finally:
            self.mobility.stop()
        return ActionResult(False, f"workflow exceeded {self.settings.max_steps} steps")

    def _acquire_stable_target(self) -> TargetObservation | None:
        """轮询相机直到得到稳定目标, 或超过 stable_timeout_s."""
        observation = self.camera.get_target(require_stable=True)
        if observation is not None:
            return observation
        deadline = time.monotonic() + self.settings.stable_timeout_s
        while time.monotonic() < deadline:
            observation = self.camera.get_target(require_stable=True)
            if observation is not None:
                return observation
        return None

    def _not_ready(self, *names: str) -> str | None:
        """若指定模块未就绪, 拼接错误信息; 全部就绪则返回 None."""
        statuses = self.healthcheck()
        failures = [f"{name}: {statuses[name].message}" for name in names if not statuses[name].ready]
        return "; ".join(failures) if failures else None

    def _sync_ui_status(self, status: str, message: str | None = None) -> None:
        """POST /api/task/status 同步任务状态到 UI 服务器."""
        try:
            import json
            import urllib.request
            payload = json.dumps({"status": status, "message": message or ""}).encode("utf-8")
            req = urllib.request.Request(
                f"{self._ui_base_url}/api/task/status",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=1.0)
        except Exception:
            pass

    def _ensure_ui_task_dispatched(self) -> None:
        """首次调用时向 UI 派发任务."""
        if self._ui_task_dispatched:
            return
        self._ui_task_dispatched = True
        try:
            import json
            import urllib.request
            payload = json.dumps({"scene": "仿真异物清理"}).encode("utf-8")
            req = urllib.request.Request(
                f"{self._ui_base_url}/api/task/dispatch",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=1.0)
        except Exception:
            pass

    def _target_in_arm_base(self, observation: TargetObservation) -> Vector3:
        """camera_link 观测 -> 外参变换 -> 类别抓取偏置 -> arm_base 抓取点."""
        if observation.frame_id != "camera_link":
            raise ValueError(f"expected camera_link observation, got {observation.frame_id}")
        cam_pt = observation.position_m
        self.log(
            f"[坐标] camera_link: x={cam_pt.x:.4f} y={cam_pt.y:.4f} z={cam_pt.z:.4f} "
            f"(class={observation.class_name})"
        )
        raw_target = self.camera_to_arm_base.apply(cam_pt)
        offset = self.settings.grasp_offset_by_class.get(
            observation.class_name, Vector3(0.0, 0.0, 0.0)
        )
        final = raw_target.shifted(offset)
        self.log(
            f"[坐标] arm_base: x={final.x:.4f} y={final.y:.4f} z={final.z:.4f} "
            f"(raw={raw_target.as_list()}, offset={offset.as_list()})"
        )
        return final

    def _image_turn(self, observation: TargetObservation) -> float:
        """根据目标在图像中的水平偏差, 计算 Go2 应施加的角速度."""
        error = observation.image_error_x_px
        if abs(error) <= self.settings.image_center_tolerance_px:
            return 0.0
        return -self.settings.align_turn_rps if error < 0.0 else self.settings.align_turn_rps

    def _alignment_command(self, target: Vector3) -> VelocityCommand:
        """把 arm_base 下目标相对甜区中心的误差, 映射为 Go2 前进或转向指令."""
        zone = self.settings.sweet_zone
        x_error = target.x - (zone.x_min_m + zone.x_max_m) / 2.0
        y_error = target.y
        forward_x, forward_y = self.settings.go2_forward_in_arm_base_xy
        left_x, left_y = self.settings.go2_left_in_arm_base_xy
        forward_error = x_error * forward_x + y_error * forward_y
        left_error = x_error * left_x + y_error * left_y
        if abs(forward_error) > 0.01:
            speed = self.settings.align_speed_mps if forward_error > 0.0 else -self.settings.align_speed_mps
            return VelocityCommand(linear_x_mps=speed)
        turn = self.settings.align_turn_rps if left_error > 0.0 else -self.settings.align_turn_rps
        return VelocityCommand(angular_z_rps=turn)

    def _pick_and_place(self, observation: TargetObservation, target: Vector3) -> ActionResult:
        """记录日志并调用机械臂执行一次抓放(传入 GraspContext 供重抓时刷新坐标)."""
        self.log(
            f"pick target id={observation.target_id} "
            f"class={observation.class_name} arm_base={target.as_list()}"
        )
        self._sync_ui_status("arm_start", f"机械臂开始抓取 {observation.class_name}")
        request = PickRequest(
            target_arm_base_m=target,
            basket_arm_base_m=self.settings.basket_arm_base_m,
            class_name=observation.class_name,
            approach_height_m=self.settings.approach_height_m,
        )
        context = _OrchestratorGraspContext(self)
        return self.arm.pick_and_place(request, context=context)


class _OrchestratorGraspContext:
    """编排器提供给机械臂的 GraspContext 实现.

    重抓时通过 refresh_target() 重新读取相机并转换坐标,
    避免机械臂直接依赖相机模块.
    """

    def __init__(self, orchestrator: DemoOrchestrator) -> None:
        self._orch = orchestrator

    def refresh_target(self, *, require_stable: bool = True) -> Vector3 | None:
        obs = self._orch.camera.get_target(require_stable=require_stable)
        if obs is None:
            return None
        return self._orch._target_in_arm_base(obs)

    def emit_event(self, event: TaskEvent) -> None:
        self._orch.log(f"[事件] {event.state}: {event.message}")
