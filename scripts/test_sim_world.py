#!/usr/bin/env python3
"""SimWorld + SceneBuilder 单元测试."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["MUJOCO_GL"] = "egl"

from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
import numpy as np


def test_build_minimal():
    """构建最小场景并验证编译成功."""
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
    assert model.nq > 0, "model should have DOFs"
    assert data.qpos.shape[0] == model.nq
    print(f"  PASS: nq={model.nq}, nbody={model.nbody}")
    return model, data


def test_reset_keyframe(model, data):
    """验证 keyframe 重置后 Go2 高度和 D1 关节值."""
    world = SimWorld(model, data)
    world.reset_to_keyframe("home")
    world.forward()

    qpos = world.get_freejoint_qpos("root")
    assert qpos[2] > 0.05, f"Go2 should be above ground, got z={qpos[2]:.3f}"

    # D1 臂关节应在 0 附近
    try:
        j1 = world.get_qpos_by_joint("arm_Joint1")
        assert abs(j1) < 0.01, f"arm_Joint1 should be ~0, got {j1}"
    except KeyError:
        pass
    print(f"  PASS: keyframe home z={qpos[2]:.3f}")


def test_locking(model, data):
    """验证 set_qpos 加锁保护."""
    world = SimWorld(model, data)
    world.reset_to_keyframe("home")

    go2_legs = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    angles = np.array([0, 0.9, -1.8] * 4)
    world.set_qpos(go2_legs, angles)
    world.forward()

    readback = world.get_qpos(go2_legs)
    assert np.allclose(readback, angles, atol=1e-6), f"qpos readback mismatch: {readback}"
    print("  PASS: set_qpos/get_qpos consistency")


def test_read_lock(model, data):
    """验证 read_lock 上下文管理器."""
    world = SimWorld(model, data)
    results = []

    def reader():
        with world.read_lock():
            results.append(1)

    import threading
    t = threading.Thread(target=reader)
    t.start()
    t.join(timeout=1.0)
    assert len(results) == 1
    print("  PASS: read_lock context manager")


def test_freejoint_qvel_update():
    """验证 kinematic 模式下 qvel 被正确更新 (bug fix #1.1)."""
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

    # 模拟 kinematic 模式下的 qvel 写入
    vx, vy, vyaw = 0.3, 0.1, 0.05
    world.set_freejoint_qvel(np.array([vx, vy, 0.0, 0.0, 0.0, vyaw]), "root")
    qvel = world.get_freejoint_qvel("root")
    assert abs(qvel[0] - vx) < 1e-6, f"qvel vx mismatch: {qvel[0]}"
    assert abs(qvel[1] - vy) < 1e-6, f"qvel vy mismatch: {qvel[1]}"
    assert abs(qvel[5] - vyaw) < 1e-6, f"qvel vyaw mismatch: {qvel[5]}"
    print("  PASS: freejoint qvel update")


if __name__ == "__main__":
    print("=== SimWorld + SceneBuilder 单元测试 ===")
    model, data = test_build_minimal()
    test_reset_keyframe(model, data)
    test_locking(model, data)
    test_read_lock(model, data)
    test_freejoint_qvel_update()
    print("=== ALL PASSED ===")
