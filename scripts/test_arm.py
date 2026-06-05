#!/usr/bin/env python3
"""UnifiedMujocoArm 单元测试."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.arm.mujoco_unified import UnifiedMujocoArm
from dog_task.core.models import PickRequest, Vector3


def build_arm_world():
    """构建只含 D1 臂的最小场景（通过 SceneBuilder 挂到 Go2 上）."""
    config = {
        "go2_xml": "assets/go2/go2_standalone.xml",
        "d1_urdf": "assets/d1/d1.urdf",
        "arm_mount_pos": [0.0, 0.0, 0.10],
        "arm_mount_quat": [1, 0, 0, 0],
        "camera_pos": [0.30, 0.0, 0.05],
        "camera_fovy": 58,
        "camera_resolution": [848, 480],
    }
    builder = SceneBuilder(config)
    model, data = builder.build()
    world = SimWorld(model, data)
    world.reset_to_keyframe("home")
    world.forward()
    return world


def test_healthcheck(world):
    arm = UnifiedMujocoArm({"tcp_offset_m": 0.12}, world)
    status = arm.healthcheck()
    assert status.ready, f"healthcheck should be ready: {status.message}"
    print("  PASS: healthcheck ready")


def test_safe_home(world):
    arm = UnifiedMujocoArm({"tcp_offset_m": 0.12}, world)
    result = arm.safe_home()
    assert result.success, f"safe_home failed: {result.message}"

    # 验证关节归零
    for name in ["arm_Joint1", "arm_Joint2", "arm_Joint3", "arm_Joint4", "arm_Joint5", "arm_Joint6"]:
        val = world.get_qpos_by_joint(name)
        assert abs(val) < 0.01, f"{name} should be ~0, got {val}"
    print("  PASS: safe_home zeros all joints")


def test_pick_and_place(world):
    arm = UnifiedMujocoArm(
        {"tcp_offset_m": 0.12, "approach_dir": [0, 0, -1],
         "approach_height_m": 0.08, "seconds_per_segment": 0.15, "fps": 30},
        world,
    )
    arm.safe_home()

    # 用 IK 可达的目标点请求
    request = PickRequest(
        target_arm_base_m=Vector3(x=0.15, y=0.0, z=0.05),
        basket_arm_base_m=Vector3(x=0.12, y=0.22, z=0.04),
        class_name="bottle",
        approach_height_m=0.08,
    )
    result = arm.pick_and_place(request)
    assert result.success, f"pick_and_place failed: {result.message}"
    print(f"  PASS: pick_and_place -> {result.message}")


def test_gripper_within_limit(world):
    """验证 gripper 开合不超出 URDF 限位 (bug fix #1.4)."""
    arm = UnifiedMujocoArm({"tcp_offset_m": 0.12}, world)
    arm.safe_home()

    # 全开
    arm._apply_joints(np.zeros(6), gripper_frac=1.0)
    for name in ["arm_Joint_L", "arm_Joint_R"]:
        val = world.get_qpos_by_joint(name)
        assert val < 0.034, f"{name} exceeded limit: {val}"

    # 全关
    close_frac = arm._servo_to_frac(20.0)
    arm._apply_joints(np.zeros(6), gripper_frac=close_frac)
    for name in ["arm_Joint_L", "arm_Joint_R"]:
        val = world.get_qpos_by_joint(name)
        assert val >= -0.001, f"{name} negative: {val}"

    print("  PASS: gripper within URDF limits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="弹出 MuJoCo 3D 可视化窗口")
    args = parser.parse_args()

    if not args.viewer:
        os.environ["MUJOCO_GL"] = "egl"

    print("=== UnifiedMujocoArm 单元测试 ===")
    w = build_arm_world()

    viewer_handle = None
    if args.viewer:
        import time
        import mujoco.viewer
        viewer_handle = mujoco.viewer.launch_passive(w.model, w.data)
        print("  MuJoCo viewer 已打开")

    def sync_v():
        if viewer_handle and viewer_handle.is_running():
            viewer_handle.sync()

    test_healthcheck(w)
    sync_v()
    test_safe_home(w)
    sync_v()
    test_pick_and_place(w)
    sync_v()
    test_gripper_within_limit(w)
    sync_v()
    print("=== ALL PASSED ===")

    if viewer_handle and viewer_handle.is_running():
        print("\n[MuJoCo Viewer] 关闭窗口退出...")
        while viewer_handle.is_running():
            sync_v()
            time.sleep(0.02)
