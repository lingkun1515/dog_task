"""按 JSON 配置装配编排器及各模块驱动实例.

SimWorld 单例管理, 确保 go2_mujoco / mujoco_unified / mujoco_rgbd 三个驱动
共享同一个物理世界.

DogTaskSim 仅包含仿真驱动; 真机 driver (d1 / d1_subprocess / d455 / go2) 在
此仓库中不可用, 调用时会给出明确错误提示. 未来如需切回真机, 只需:
  1. 拷贝对应的实机驱动文件到对应目录
  2. 在对应 __init__.py 中恢复导出
  3. 去掉下面分支中的 raise 即可
"""

from __future__ import annotations

from typing import Any, Mapping

from ..config import project_path, section
from ..core.models import TargetObservation, Vector3
from ..core.transforms import RigidTransform
from ..modules.arm import MockArm, UnifiedMujocoArm
from ..modules.camera import MockCamera, MujocoCameraSim
from ..modules.mobility import Go2MujocoMobility, MockMobility
from .demo_orchestrator import DemoOrchestrator, WorkflowSettings


_sim_world = None


def _get_or_build_sim_world(full_config: Mapping[str, Any]):
    """获取或构建共享 SimWorld 单例."""
    global _sim_world
    if _sim_world is not None:
        return _sim_world
    from ..modules.sim import SceneBuilder, SimWorld

    sim_config = section(full_config, "sim_world")
    builder = SceneBuilder(sim_config)
    model, data = builder.build()
    _sim_world = SimWorld(model, data)
    return _sim_world


def build_orchestrator(config: Mapping[str, Any]) -> DemoOrchestrator:
    """根据完整 JSON 配置构建 fixed-once / mobile-once 用的编排器.

    读取段: arm, camera, mobility, camera_to_arm_base, workflow.
    可选段: sim_world (统一仿真模式).
    """
    transform_config = section(config, "camera_to_arm_base")
    if transform_config.get("placeholder"):
        raise ValueError("replace the placeholder camera_to_arm_base matrix before running")
    transform = RigidTransform.from_value(transform_config["matrix"])
    return DemoOrchestrator(
        arm=_build_arm(section(config, "arm"), config),
        camera=_build_camera(section(config, "camera"), config),
        mobility=_build_mobility(section(config, "mobility"), config),
        camera_to_arm_base=transform,
        settings=WorkflowSettings.from_dict(section(config, "workflow")),
    )


def _build_arm(config: Mapping[str, Any], full_config: Mapping[str, Any] = None):
    """按 arm.driver 创建机械臂适配器: mock / mujoco_unified."""
    driver = config.get("driver")
    if driver == "mock":
        return MockArm()
    if driver == "mujoco_unified":
        world = _get_or_build_sim_world(full_config)
        return UnifiedMujocoArm(config, world)
    if driver in ("d1", "d1_subprocess", "mujoco"):
        raise ValueError(
            f"arm driver '{driver}' is not available in DogTaskSim (simulation-only). "
            "Use 'mujoco_unified' or 'mock'."
        )
    raise ValueError(f"unknown arm driver: {driver}")


def _build_camera(config: Mapping[str, Any], full_config: Mapping[str, Any] = None):
    """按 camera.driver 创建摄像头: mock / mujoco_rgbd."""
    driver = config.get("driver")
    if driver == "mock":
        observations = [
            None if item is None else TargetObservation.from_dict(item)
            for item in config.get("observations", [])
        ]
        return MockCamera(observations, repeat_last=bool(config.get("repeat_last", True)))
    if driver == "mujoco_rgbd":
        world = _get_or_build_sim_world(full_config)
        return MujocoCameraSim(config, world)
    if driver == "d455":
        raise ValueError(
            "camera driver 'd455' is not available in DogTaskSim (simulation-only). "
            "Use 'mujoco_rgbd' or 'mock'."
        )
    raise ValueError(f"unknown camera driver: {driver}")


def _build_mobility(config: Mapping[str, Any], full_config: Mapping[str, Any] = None):
    """按 mobility.driver 创建底盘: mock / go2_mujoco."""
    driver = config.get("driver")
    if driver == "mock":
        return MockMobility()
    if driver == "go2_mujoco":
        world = _get_or_build_sim_world(full_config)
        return Go2MujocoMobility(config, world)
    if driver == "go2":
        raise ValueError(
            "mobility driver 'go2' is not available in DogTaskSim (simulation-only). "
            "Use 'go2_mujoco' or 'mock'."
        )
    raise ValueError(f"unknown mobility driver: {driver}")
