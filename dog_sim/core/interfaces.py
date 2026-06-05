"""稳定接口(Protocol)层(core 层).

各负责人只需让自己的模块实现对应 Protocol, 编排器即可联调.
主流程只依赖这些协议, 不依赖具体硬件品牌; 换机械臂/底盘/相机时只新增适配器.
不要随意修改方法签名, 否则三人并行开发会互相阻塞.
"""

from __future__ import annotations

from typing import Optional, Protocol

from .models import (
    ActionResult,
    ArmCapabilities,
    GraspOutcome,
    HealthStatus,
    MobilityCapabilities,
    MobilityState,
    NavOptions,
    PickRequest,
    Pose2D,
    Reachability,
    TargetObservation,
    TaskEvent,
    TaughtPickRequest,
    Vector3,
    VelocityCommand,
)


class GraspContext(Protocol):
    """抓取上下文: 机械臂在重抓循环中通过此接口获取最新目标坐标与发布事件.

    编排器在调用 pick_and_place 时传入实现了本协议的对象, 从而让机械臂
    在重抓失败后能请求刷新坐标(间接调用相机), 同时向 EventBus 推送事件,
    避免机械臂直接依赖相机或事件总线.
    """

    def refresh_target(self, *, require_stable: bool = True) -> Optional[Vector3]:
        """请求编排器重新获取目标坐标(arm_base 系, 米).

        返回 None 表示暂无可用坐标, 机械臂应沿用上一次的目标.
        """
        ...

    def emit_event(self, event: TaskEvent) -> None:
        """向外部推送任务事件(如 GRASP_RETRYING / ARM_PLACING)."""
        ...


class ArmController(Protocol):
    """机械臂模块契约(机械臂负责人实现).

    编排器只调用 ``pick_and_place``; 坐标已在 arm_base 系, 机械臂不关心视觉细节.
    抓取/重抓/放置等闭环逻辑全部封装在适配器内部.
    """

    def healthcheck(self) -> HealthStatus:
        """检查 URDF/依赖/硬件二进制是否就绪."""
        ...

    def capabilities(self) -> ArmCapabilities:
        """声明自由度、是否支持示教回放、可用抓取策略等能力."""
        ...

    def estimate_reachability(self, request: PickRequest) -> Reachability:
        """判断目标是否可达(不真正执行), 供编排器在调整底盘前预判."""
        ...

    def pick_and_place(
        self,
        request: PickRequest,
        context: Optional[GraspContext] = None,
    ) -> ActionResult:
        """从目标点抓取并投放到收纳框(内部完成规划/抓取/验证/重抓/放置).

        context: 可选的抓取上下文, 用于重抓时刷新坐标与推送事件.
        """
        ...

    def replay_taught_pick(self, request: TaughtPickRequest) -> ActionResult:
        """按示教关节角回放抓放(可选能力)."""
        ...

    def safe_home(self) -> ActionResult:
        """回到安全位."""
        ...

    def stop(self) -> ActionResult:
        """停止当前动作."""
        ...


class FeedbackArmController(Protocol):
    """带夹爪反馈的细粒度机械臂契约, 支持"抓取失败重抓"流程.

    与 ArmController.pick_and_place 不同, 这里把"整套抓放"拆成可分别调用的三步,
    以便在提起(LIFT)之后、放置(PLACE)之前插入夹爪反馈判断:
      1. grasp_and_lift: 接近 -> 下降 -> 闭合 -> 提起, 返回是否抓住的反馈.
      2. place_to_basket: 仅在确认抓住后调用, 把物体放入收纳框.
      3. open_gripper_and_retreat: 重抓耗尽或异常时, 张开夹爪并安全退回.
    """

    def healthcheck(self) -> HealthStatus:
        ...

    def grasp_and_lift(
        self,
        target_arm_base_m: Vector3,
        *,
        approach_dir: Vector3,
        approach_height_m: float,
    ) -> GraspOutcome:
        """执行一次"接近-下降-闭合-提起", 并返回夹爪反馈判定结果."""
        ...

    def place_to_basket(
        self,
        basket_arm_base_m: Vector3,
        *,
        approach_height_m: float,
    ) -> ActionResult:
        """把已抓住的物体放入收纳框并松开夹爪、退回."""
        ...

    def open_gripper_and_retreat(self) -> ActionResult:
        """张开夹爪并安全退回, 用于重抓失败或中止时复位."""
        ...


class TargetProvider(Protocol):
    """目标坐标来源契约, 适配"摄像头自动检测"或"手动输入".

    机械臂不关心坐标从何而来; 重抓失败时用 ``next_target`` 重新确认坐标.
    """

    def next_target(self, last: Optional[Vector3]) -> Optional[Vector3]:
        """返回最新目标坐标(arm_base, 米).

        参数:
            last: 上一次使用的坐标; 无新坐标时调用方应沿用 last.
        返回:
            新坐标; 若暂无新坐标返回 None.
        """
        ...


class CameraController(Protocol):
    """摄像头模块契约(D455 负责人实现).

    只输出 camera_link 下的观测; 到 arm_base 的变换由编排器负责.
    """

    def healthcheck(self) -> HealthStatus:
        """检查 RealSense/YOLO 等依赖与模型权重."""
        ...

    def get_target(self, *, require_stable: bool = False) -> Optional[TargetObservation]:
        """获取当前帧目标观测.

        参数:
            require_stable: True 时仅返回多帧稳定的观测, 否则可能返回不稳定点.
        """
        ...


class MobilityController(Protocol):
    """移动底盘模块契约(移动平台负责人实现).

    不仅支持低速速度控制, 还要支持姿态切换、停止、(可选)导航与返航等产品动作.
    具体支持哪些动作通过 ``capabilities()`` 暴露, 避免小车被迫实现趴下.
    """

    def healthcheck(self) -> HealthStatus:
        """检查底盘 SDK 连接等."""
        ...

    def capabilities(self) -> MobilityCapabilities:
        """声明是否支持速度/侧移/全局导航/返航/姿态切换."""
        ...

    def get_state(self) -> MobilityState:
        """返回底盘状态(是否停稳、姿态、位姿、电量等)."""
        ...

    def set_velocity(self, command: VelocityCommand) -> ActionResult:
        """设置底盘速度(低速前进/后退/转向)."""
        ...

    def stop(self) -> ActionResult:
        """立即停止(零速度)."""
        ...

    def is_stopped(self) -> bool:
        """报告底盘是否已停稳(移动后 RECHECK 前必须为 True)."""
        ...

    def set_posture(self, posture: str) -> ActionResult:
        """切换姿态(stand/stand_down/sit 等), 不支持的平台返回失败."""
        ...

    def go_to(self, goal: Pose2D, options: NavOptions) -> ActionResult:
        """导航到目标位姿(需 supports_global_nav, 第一版可不支持)."""
        ...

    def dock(self, dock_id: Optional[str] = None) -> ActionResult:
        """返航/对接充电桩(需 supports_docking, 第一版可不支持)."""
        ...

    def emergency_stop(self) -> ActionResult:
        """急停(立即停止底盘, 必要时阻尼保护)."""
        ...
