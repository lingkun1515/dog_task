"""任务场景管理器：为不同任务类型动态管理 MuJoCo 几何体。

支持的场景（与 scheduler.config.SUPPORTED_TASK_SCENES 对齐）：
  - lawn_debris   : 单目标抓取（黄色球体），最简形式
  - golf_ball     : 多目标回收（5 个白色高尔夫球），需逐个抓取
  - rain_inspect  : 巡检（无目标体，仅导航+录像+检测积水点）
  - material_drop : 物料投放（从 home 携带方块到 target 释放）

实现说明：
  MuJoCo 编译后的 MjModel 不能动态增删 body。本管理器采用「预放置 +
  运行时重定位」策略：所有场景所需的 body 都已在 scene.xml 中预定义
  （初始停放在地下 y=100 远处），激活场景时通过 qpos 写入把它们移到
  作业区，停用场景时移回 park 区。

  这与 robot_loader.apply_gripper_constraint 写入 ball qpos 的做法一致，
  sim2real 时该模块仅作为 sim 的辅助层，不影响 real 端逻辑。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 场景枚举（与 scheduler.config.SUPPORTED_TASK_SCENES 一致）
# ---------------------------------------------------------------------------
SCENE_LAWN_DEBRIS = "lawn_debris"
SCENE_GOLF_BALL = "golf_ball"
SCENE_RAIN_INSPECT = "rain_inspect"
SCENE_MATERIAL_DROP = "material_drop"

ALL_SCENES = (SCENE_LAWN_DEBRIS, SCENE_GOLF_BALL, SCENE_RAIN_INSPECT, SCENE_MATERIAL_DROP)

# 不活跃 body 的统一停放位置（地下远处，相机看不到）
_PARK_POS = (100.0, 100.0, -5.0)
_PARK_QUAT = (1.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 场景几何预设（描述激活时各 body 的目标位置）
# ---------------------------------------------------------------------------
@dataclass
class SceneBodySpec:
    """激活场景时某 body 应放置到的位置/姿态。"""
    body_name: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    label: str = ""
    # 是否带 freejoint（freejoint body 通过 qpos 写入；无 joint 的静态 body 用 mj_forward 重算）
    is_free: bool = True


def _lawn_debris_specs(target_pos: tuple[float, float, float]) -> list[SceneBodySpec]:
    """lawn_debris: 单黄色球体放在 target。"""
    return [
        SceneBodySpec(
            body_name="target_sphere",
            pos=target_pos,
            label="debris",
        )
    ]


def _golf_ball_specs(target_pos: tuple[float, float, float]) -> list[SceneBodySpec]:
    """golf_ball: 5 个白色球散布在 target 周围。"""
    cx, cy, cz = target_pos
    offsets = [
        (0.0, 0.0),
        (0.15, 0.10),
        (-0.12, 0.14),
        (0.10, -0.16),
        (-0.18, -0.08),
    ]
    return [
        SceneBodySpec(
            body_name=f"golf_ball_{i}",
            pos=(cx + ox, cy + oy, 0.018),  # 球半径
            label="golf",
        )
        for i, (ox, oy) in enumerate(offsets)
    ]


def _material_drop_specs(target_pos: tuple[float, float, float]) -> list[SceneBodySpec]:
    """material_drop: 红色物料方块。激活后由 caller 重定位到 home。"""
    return [
        SceneBodySpec(
            body_name="payload_box",
            pos=target_pos,
            label="payload",
        )
    ]


def _rain_inspect_specs(target_pos: tuple[float, float, float]) -> list[SceneBodySpec]:
    """rain_inspect: 3 个静态积水点（在 scene.xml 中已预放置在 target 附近）。

    它们不带 freejoint，激活/停用通过移动它们会被 mj_forward 自动校正，
    因此这里返回空 specs（无需重定位）——积水点位置已在 MJCF 固化。
    """
    return []


_SCENE_SPEC_BUILDERS = {
    SCENE_LAWN_DEBRIS: _lawn_debris_specs,
    SCENE_GOLF_BALL: _golf_ball_specs,
    SCENE_RAIN_INSPECT: _rain_inspect_specs,
    SCENE_MATERIAL_DROP: _material_drop_specs,
}

# 各场景需要从 park 区移出的 body 全集（用于 clear 时统一回 park）
_SCENE_BODY_REGISTRY = {
    SCENE_LAWN_DEBRIS: ["target_sphere"],
    SCENE_GOLF_BALL: [f"golf_ball_{i}" for i in range(5)],
    SCENE_RAIN_INSPECT: ["puddle_0", "puddle_1", "puddle_2"],
    SCENE_MATERIAL_DROP: ["payload_box"],
}

# park 区所有预放置 body（首次初始化时把除 target_sphere 外的全部归位）
_PARKABLE_BODIES = (
    [f"golf_ball_{i}" for i in range(5)]
    + ["payload_box"]
)


# ---------------------------------------------------------------------------
# 任务场景管理器
# ---------------------------------------------------------------------------
class TaskSceneManager:
    """管理运行时 MuJoCo 场景物体的位置，并提供场景特定的语义。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model = model
        self._data = data
        self._current_scene: str | None = None
        self._body_qpos_addr: dict[str, tuple[int, int]] = {}  # body_name -> (qpos_addr, qvel_addr)
        self._build_body_addr_cache()

    def _build_body_addr_cache(self) -> None:
        """缓存所有管理范围内的 body 的 qpos/qvel 地址。"""
        for name in _PARKABLE_BODIES + ["target_sphere"]:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            jnt_adr = self._model.body_jntadr[bid]
            jnt_num = self._model.body_jntnum[bid]
            if jnt_num == 0:
                # 静态 body（无 freejoint）—— 用 xpos/xquat 直接读，写入靠 mj_forward
                self._body_qpos_addr[name] = (-1, -1)
                continue
            jnt_id = jnt_adr
            qpos_addr = int(self._model.jnt_qposadr[jnt_id])
            qvel_addr = int(self._model.jnt_dofadr[jnt_id])
            self._body_qpos_addr[name] = (qpos_addr, qvel_addr)

    @property
    def current_scene(self) -> str | None:
        return self._current_scene

    @property
    def target_body_names(self) -> list[str]:
        """返回当前场景下，应被检测器/抓取器感知的目标 body 名。"""
        if self._current_scene == SCENE_GOLF_BALL:
            return [f"golf_ball_{i}" for i in range(5)]
        if self._current_scene == SCENE_LAWN_DEBRIS:
            return ["target_sphere"]
        if self._current_scene == SCENE_MATERIAL_DROP:
            return ["payload_box"]
        if self._current_scene == SCENE_RAIN_INSPECT:
            return ["puddle_0", "puddle_1", "puddle_2"]
        # 默认 lawn_debris 行为（向后兼容旧调用）
        return ["target_sphere"]

    @property
    def target_labels(self) -> list[str]:
        """检测标签（显示在视频帧上）。"""
        if self._current_scene == SCENE_GOLF_BALL:
            return ["golf"] * 5
        if self._current_scene == SCENE_MATERIAL_DROP:
            return ["payload"]
        if self._current_scene == SCENE_RAIN_INSPECT:
            return ["puddle"] * 3
        return ["ball"]  # lawn_debris & default

    @property
    def requires_grasp(self) -> bool:
        """该场景是否走标准抓取管线（lawn_debris / golf_ball）。"""
        return self._current_scene in (SCENE_LAWN_DEBRIS, SCENE_GOLF_BALL)

    @property
    def requires_drop(self) -> bool:
        """该场景是否走物料投放管线。"""
        return self._current_scene == SCENE_MATERIAL_DROP

    @property
    def requires_inspect(self) -> bool:
        """该场景是否走巡检管线。"""
        return self._current_scene == SCENE_RAIN_INSPECT

    # ------------------------------------------------------------------
    # 场景几何激活/停用
    # ------------------------------------------------------------------
    def setup_scene(
        self,
        scene: str,
        target_pos: tuple[float, float, float],
        home_pos: tuple[float, float, float],
    ) -> list[SceneBodySpec]:
        """激活任务场景：把相关 body 从 park 区移动到作业区。

        Args:
            scene: 场景名（ALL_SCENES 之一）
            target_pos: (x, y, z) 任务目标点
            home_pos: (x, y, z) 充电桩/起点（material_drop 用）
        Returns:
            实际激活的 SceneBodySpec 列表
        """
        if scene not in ALL_SCENES:
            raise ValueError(f"未知场景: {scene!r}，支持: {list(ALL_SCENES)}")

        # 先清理所有 body（归位 park）
        self.park_all_bodies()

        builder = _SCENE_SPEC_BUILDERS.get(scene)
        if builder is None:
            raise ValueError(f"场景 {scene!r} 没有几何构造器")

        specs = builder(target_pos)

        # material_drop 特殊：payload 初始放在 home，不是 target
        if scene == SCENE_MATERIAL_DROP and specs:
            specs[0] = SceneBodySpec(
                body_name=specs[0].body_name,
                pos=(home_pos[0], home_pos[1], 0.05),
                label=specs[0].label,
            )

        self._current_scene = scene

        # 应用 specs：把每个 body 写入指定 qpos
        for spec in specs:
            self._relocate_body(spec.body_name, spec.pos, spec.quat)

        # mj_forward 让世界坐标系更新（xpos 反映新位置）
        mujoco.mj_forward(self._model, self._data)

        logger.info(
            "[TaskScene] 场景已激活: %s，目标体: %s",
            scene, self.target_body_names,
        )
        return specs

    def park_all_bodies(self) -> None:
        """把所有预放置 body 归位到 park 区（为下一次 setup 准备）。"""
        for name in _PARKABLE_BODIES:
            self._relocate_body(name, _PARK_POS, _PARK_QUAT)
        # target_sphere 归位只在切换非 lawn_debris 场景时执行（由 setup 重新放置）
        # 这里不归位 target_sphere，因为 lawn_debris 是默认场景
        mujoco.mj_forward(self._model, self._data)

    def _relocate_body(
        self,
        body_name: str,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """写入 body 的 qpos（freejoint）以重定位。"""
        addrs = self._body_qpos_addr.get(body_name)
        if addrs is None:
            logger.debug("[TaskScene] body %s 不在管理范围内，跳过", body_name)
            return
        qpos_addr, qvel_addr = addrs
        if qpos_addr < 0:
            return  # 静态 body 无 qpos 可写
        self._data.qpos[qpos_addr:qpos_addr + 3] = pos
        self._data.qpos[qpos_addr + 3:qpos_addr + 7] = quat
        # 速度归零
        self._data.qvel[qvel_addr:qvel_addr + 6] = 0.0

    def deactivate_scene(self) -> None:
        """停用当前场景（仅清理状态标记，几何清理由下次 setup 完成）。"""
        logger.debug("[TaskScene] 停用场景: %s", self._current_scene)
        self._current_scene = None

    def reset_to_default(self) -> None:
        """重置为默认 lawn_debris 场景（保持向后兼容）。"""
        # target_sphere 已在 MJCF 初始位置，park 其它 body
        for name in _PARKABLE_BODIES:
            self._relocate_body(name, _PARK_POS, _PARK_QUAT)
        self._current_scene = SCENE_LAWN_DEBRIS
        mujoco.mj_forward(self._model, self._data)
        logger.info("[TaskScene] 已重置为默认场景: lawn_debris")

    def get_target_positions(self) -> dict[str, np.ndarray]:
        """返回当前场景下所有目标体的世界坐标（用于分析/日志）。"""
        result: dict[str, np.ndarray] = {}
        for name in self.target_body_names:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                result[name] = self._data.xpos[bid].copy()
        return result


# ---------------------------------------------------------------------------
# 场景特定的任务执行结果
# ---------------------------------------------------------------------------
class TaskOutcome(Enum):
    """场景任务的执行结果。"""
    SUCCESS = "success"
    PARTIAL = "partial"     # 部分目标完成（如 golf_ball 只抓到 3/5）
    FAILED = "failed"
    SKIPPED = "skipped"     # 不需要执行（如该场景无抓取）


@dataclass
class TaskResult:
    """任务执行结果摘要。"""
    scene: str
    outcome: TaskOutcome
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "outcome": self.outcome.value,
            "details": self.details,
            "message": self.message,
        }
