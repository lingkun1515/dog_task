"""框架的共享数据结构(core 层).

所有模块(感知/机械臂/移动/编排/UI/Agent)通过这里定义的类型交换数据,
避免直接依赖彼此实现. 本层不依赖任何硬件 SDK, 也不绑定具体品牌.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Mapping, Optional, Sequence, Tuple


def _require_length(values: Sequence[Any], length: int, name: str) -> None:
    """校验序列长度, 用于解析配置或 JSON 中的坐标/ bbox."""
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(values)}")


@dataclass(frozen=True)
class Vector3:
    """三维向量, 单位通常为米, 用于各坐标系下的点或方向."""

    x: float
    y: float
    z: float

    @classmethod
    def from_value(cls, value: Sequence[float], name: str = "vector") -> "Vector3":
        """从 ``[x, y, z]`` 列表或元组构造, 长度必须为 3."""
        _require_length(value, 3, name)
        return cls(*(float(item) for item in value))

    def as_list(self) -> list[float]:
        """转为 ``[x, y, z]`` 列表, 便于传给子进程或日志."""
        return [self.x, self.y, self.z]

    def shifted(self, offset: "Vector3") -> "Vector3":
        """点坐标加上偏置向量(如按类别的抓取补偿)."""
        return Vector3(self.x + offset.x, self.y + offset.y, self.z + offset.z)


@dataclass(frozen=True)
class Pose2D:
    """移动平台在地图/里程计平面上的位姿(导航与巡逻使用)."""

    x_m: float
    y_m: float
    yaw_rad: float
    frame_id: str = "map"


@dataclass(frozen=True)
class Pose3D:
    """带坐标系标注的三维位姿(位置 + 可选朝向).

    用于跨模块传递抓取点/放置点等, ``frame_id`` 显式标注坐标系防止误用.
    朝向暂用接近方向向量表示, 后续可扩展为四元数.
    """

    position: Vector3
    frame_id: str = "arm_base"
    approach_dir: Optional[Vector3] = None


@dataclass(frozen=True)
class TargetObservation:
    """摄像头侧的目标观测(尚未转换到机械臂坐标系).

    真实 D455 模块必须在 ``camera_link`` 坐标系下发布位置;
    ``frame_id`` 显式标注坐标系, 防止误用未变换的点.
    """

    target_id: str          # 唯一标识, 如 "bottle-3"(类别 + ByteTrack id)
    class_name: str         # 映射后的抓取类别, 如 bottle / can / tennis_ball
    confidence: float       # 检测置信度 0~1
    position_m: Vector3     # 目标参考点在 camera_link 下的坐标(米)
    frame_id: str = "camera_link"
    bbox_px: Optional[Tuple[int, int, int, int]] = None  # 像素框 (x1,y1,x2,y2)
    image_size_px: Tuple[int, int] = (640, 480)            # 图像宽高, 用于算居中误差
    stable: bool = False    # 是否已通过多帧稳定性判定
    captured_at_s: float = field(default_factory=time)     # 观测时间戳

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetObservation":
        """从 mock 配置 JSON 中的观测字典构造(用于 demo.mock.*.json)."""
        bbox = data.get("bbox_px")
        image_size = data.get("image_size_px", [640, 480])
        if bbox is not None:
            _require_length(bbox, 4, "bbox_px")
            bbox = tuple(int(item) for item in bbox)
        _require_length(image_size, 2, "image_size_px")
        return cls(
            target_id=str(data["target_id"]),
            class_name=str(data["class_name"]),
            confidence=float(data["confidence"]),
            position_m=Vector3.from_value(data["position_m"], "position_m"),
            frame_id=str(data.get("frame_id", "camera_link")),
            bbox_px=bbox,
            image_size_px=(int(image_size[0]), int(image_size[1])),
            stable=bool(data.get("stable", False)),
            captured_at_s=float(data.get("captured_at_s", time())),
        )

    @property
    def depth_m(self) -> float:
        """相机系下目标的深度, 即 ``position_m.z``(Z 向前, 越大越远).

        仅对 ``frame_id == camera_link`` 有意义; mobile 流程里用于判断 Go2 是否够近.
        """
        if self.frame_id != "camera_link":
            raise ValueError("depth_m is only defined for camera_link observations")
        return self.position_m.z

    @property
    def image_error_x_px(self) -> float:
        """目标 bbox 中心相对图像水平中心的像素偏差.

        正值 = 目标在画面右侧, 编排器据此让 Go2 左转(负 angular_z).
        无 bbox 时返回 0(视为已居中).
        """
        if self.bbox_px is None:
            return 0.0
        x1, _, x2, _ = self.bbox_px
        return (x1 + x2) / 2.0 - self.image_size_px[0] / 2.0


@dataclass(frozen=True)
class RetryPolicy:
    """重抓策略参数(机械臂内部抓取策略使用)."""

    max_attempts: int = 3                       # 总尝试次数(含首次)
    refresh_target_each_attempt: bool = True    # 每次重抓前是否重新确认坐标


@dataclass(frozen=True)
class PickRequest:
    """编排器交给机械臂的一次抓放请求(坐标已在 arm_base 系).

    使用通用 arm_base 坐标系命名, 不绑定具体机械臂品牌;
    换机械臂时适配器自行把 arm_base 映射到自己的基座定义.
    ``grasp_policy`` / ``place_policy`` / ``retry`` 为通用字段, 不写具体品牌专属参数.
    """

    target_arm_base_m: Vector3   # 抓取点(arm_base, 米)
    basket_arm_base_m: Vector3   # 收纳框放置点(arm_base, 米)
    class_name: str              # 物体类别(日志/仿真命名用)
    approach_dir: Vector3 = Vector3(0.0, 0.0, -1.0)  # 夹爪接近方向, 默认顶抓向下
    approach_height_m: float = 0.15                  # 抓取/放置点上方的预接近高度(米)
    grasp_policy: str = "auto"        # auto / top_down / side / taught / suction
    place_policy: str = "fixed_basket"
    retry: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class TaughtPickRequest:
    """示教回放: 直接用关节角而非笛卡尔坐标."""

    grasp_joints_deg: Tuple[float, float, float, float, float, float]  # 6 关节抓取角(度)
    place_joints_deg: Tuple[float, float, float, float, float, float]  # 6 关节放置角(度)


@dataclass(frozen=True)
class Reachability:
    """机械臂对某次抓取请求的可达性判定结果."""

    reachable: bool
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspOutcome:
    """单次"抓取并提起"后的反馈结果, 用于判断是否真的抓住物体.

    判据: 夹爪闭合后, 若夹住物体则物体卡住夹爪, 实际反馈角(gripper_angle_deg)
    明显大于指令闭合角(commanded_close_deg); 若扑空, 反馈角会合到接近指令角.
    因此 grabbed 由 "反馈角 - 指令角 > 阈值" 决定(见 retry_grasp.decide_grabbed).
    """

    grabbed: bool
    gripper_angle_deg: float      # 夹爪实际反馈角 servo6(度)
    commanded_close_deg: float    # 本次下发的夹爪闭合指令角(度)
    message: str = ""


@dataclass(frozen=True)
class VelocityCommand:
    """底盘速度指令(编排器只使用这三个分量, 与具体底盘无关)."""

    linear_x_mps: float = 0.0     # 前进速度(米/秒), 正=前
    linear_y_mps: float = 0.0     # 侧向速度(米/秒), 第一版通常不用
    angular_z_rps: float = 0.0    # 绕竖轴角速度(弧度/秒), 正=逆时针(左转)


class Posture(str, Enum):
    """移动平台姿态(Go2 等支持, 小车通常不支持)."""

    STAND = "stand"             # 站立
    STAND_DOWN = "stand_down"   # 趴下/降低机身, 方便机械臂抓取
    SIT = "sit"                 # 坐下
    RISE_SIT = "rise_sit"       # 从坐姿恢复


@dataclass(frozen=True)
class NavOptions:
    """导航/路径跟随的可选参数(由具体底盘适配器解释)."""

    max_speed_mps: Optional[float] = None
    tolerance_m: float = 0.1
    timeout_s: Optional[float] = None


@dataclass(frozen=True)
class MobilityCapabilities:
    """移动平台能力声明; 主流程据此判断能否调用某动作, 避免小车被迫实现趴下."""

    supports_velocity: bool = True
    supports_strafe: bool = False
    supports_global_nav: bool = False
    supports_docking: bool = False
    supports_posture: Tuple[str, ...] = ()   # 如 ("stand", "stand_down", "sit")


@dataclass(frozen=True)
class MobilityState:
    """移动平台状态反馈."""

    is_stopped: bool
    posture: str = Posture.STAND.value
    pose: Optional[Pose2D] = None
    battery_ratio: Optional[float] = None    # 0~1, 无电量反馈时为 None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmCapabilities:
    """机械臂能力声明."""

    dof: int = 6
    supports_taught_replay: bool = False
    grasp_policies: Tuple[str, ...] = ("auto",)
    max_reach_m: Optional[float] = None


@dataclass(frozen=True)
class ActionResult:
    """任意模块执行一次动作后的统一返回格式."""

    success: bool
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)  # 附加信息, 如子进程 stdout


@dataclass(frozen=True)
class HealthStatus:
    """模块就绪检查结果."""

    ready: bool
    message: str


class WorkflowState(str, Enum):
    """mobile-once 移动闭环状态机的各阶段(demo 编排器使用)."""

    SEARCH = "SEARCH"                   # 原地旋转找目标
    APPROACH = "APPROACH"               # 目标够远时前进靠近
    CENTER = "CENTER"                   # 把目标对准图像中心
    ALIGN = "ALIGN"                     # 微调底盘使目标进入机械臂甜区
    STOP = "STOP"                       # 发送停止并等待停稳
    RECHECK = "RECHECK"                 # 停稳后重新获取稳定观测
    PICK_AND_PLACE = "PICK_AND_PLACE"   # 调用机械臂抓放
    DONE = "DONE"
    FAILED = "FAILED"


class TaskState(str, Enum):
    """产品级任务状态机(TaskOrchestrator 使用, 见框架文档 §5).

    第一版未必全部用到, 先把命名固定下来, UI/Agent 与编排器据此对齐.
    """

    IDLE = "IDLE"
    TASK_ACCEPTED = "TASK_ACCEPTED"
    PRECHECK = "PRECHECK"
    PATROLLING = "PATROLLING"
    TARGET_DETECTED = "TARGET_DETECTED"
    APPROACHING_TARGET = "APPROACHING_TARGET"
    ALIGNING_FOR_ARM = "ALIGNING_FOR_ARM"
    PREPARE_GRASP_POSTURE = "PREPARE_GRASP_POSTURE"
    RECHECK_TARGET = "RECHECK_TARGET"
    ARM_PICKING = "ARM_PICKING"
    ARM_PLACING = "ARM_PLACING"
    OBJECT_STORED = "OBJECT_STORED"
    RESUME_PATROL = "RESUME_PATROL"
    RETURN_DOCK = "RETURN_DOCK"
    STAND_DOWN_IDLE = "STAND_DOWN_IDLE"
    DONE = "DONE"
    # 异常分支
    TARGET_LOST = "TARGET_LOST"
    OUT_OF_REACH = "OUT_OF_REACH"
    GRASP_RETRYING = "GRASP_RETRYING"
    GRASP_FAILED = "GRASP_FAILED"
    LOW_BATTERY = "LOW_BATTERY"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    MANUAL_TAKEOVER = "MANUAL_TAKEOVER"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True)
class TaskSpec:
    """UI 或 Agent 创建的结构化任务(不直接调用硬件, 见框架文档 §6)."""

    task_id: str
    intent: str                                  # 如 patrol_pickup_foreign_object
    area_id: Optional[str] = None
    patrol_route_id: Optional[str] = None
    target_classes: Tuple[str, ...] = ()
    pickup_enabled: bool = True
    finish_policy: str = "continue_patrol"       # continue_patrol / return_dock / stand_down_idle
    max_duration_s: Optional[int] = None


@dataclass(frozen=True)
class TaskCommand:
    """任务级控制命令(暂停/取消/返航/趴下等), 走安全白名单, 不含原始速度/关节角."""

    command: str                                 # pause/resume/cancel/return_dock/...
    finish_policy: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskEvent:
    """编排器向 UI/Agent 回传的统一事件(见框架文档 §12.2)."""

    task_id: str
    state: str
    message: str = ""
    object_name: Optional[str] = None
    confidence: Optional[float] = None
    robot_pose: Optional[Pose2D] = None
    target_pose: Optional[Pose3D] = None
    evidence_image: Optional[str] = None
    timestamp_s: float = field(default_factory=time)
