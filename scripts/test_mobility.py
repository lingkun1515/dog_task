#!/usr/bin/env python3
"""Go2MujocoMobility RL 运控单元测试."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.mobility.go2_mujoco import Go2MujocoMobility
from dog_task.core.models import VelocityCommand


def build_world():
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


def cleanup_mob(mob):
    """停止并等待控制线程退出."""
    mob._running = False
    mob._cmd_vel = np.zeros(3)
    if mob._control_thread is not None:
        mob._control_thread.join(timeout=2.0)


def test_healthcheck(world):
    mob = Go2MujocoMobility({}, world)
    try:
        status = mob.healthcheck()
        assert status.ready, f"healthcheck should be ready: {status.message}"
        print("  PASS: healthcheck ready")
    finally:
        cleanup_mob(mob)


def test_rl_walk(world, viewer=None):
    """RL 运控行走 + 里程计测试."""
    mob = Go2MujocoMobility({"control_hz": 50}, world)
    try:
        health = mob.healthcheck()
        assert health.ready, f"RL healthcheck failed: {health.message}"
        mob.set_posture("stand")
        time.sleep(0.5)
        if viewer:
            viewer.sync()

        start_state = mob.get_state()
        start_x, start_y = start_state.pose.x_m, start_state.pose.y_m
        mob.set_velocity(VelocityCommand(linear_x_mps=0.5, linear_y_mps=0.0, angular_z_rps=0.0))

        t0 = time.time()
        dist = 0.0
        while dist < 0.1 and time.time() - t0 < 15:
            time.sleep(0.05)
            if viewer:
                viewer.sync()
            state = mob.get_state()
            dist = np.hypot(state.pose.x_m - start_x, state.pose.y_m - start_y)

        mob.stop()
        time.sleep(0.3)
        if viewer:
            viewer.sync()
        assert dist > 0.03, f"RL walk distance too small: {dist:.3f}m"
        print(f"  PASS: RL walked {dist:.3f}m in {time.time() - t0:.1f}s")
    finally:
        cleanup_mob(mob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="弹出 MuJoCo 3D 可视化窗口")
    args = parser.parse_args()

    if not args.viewer:
        os.environ["MUJOCO_GL"] = "egl"

    print("=== Go2MujocoMobility 单元测试 ===")
    w = build_world()

    viewer_handle = None
    if args.viewer:
        import mujoco.viewer
        viewer_handle = mujoco.viewer.launch_passive(w.model, w.data)
        print("  MuJoCo viewer 已打开")

    def sync_v():
        if viewer_handle and viewer_handle.is_running():
            viewer_handle.sync()

    test_healthcheck(w)
    test_rl_walk(w, viewer_handle)
    print("=== ALL PASSED ===")

    if viewer_handle and viewer_handle.is_running():
        print("\n[MuJoCo Viewer] 关闭窗口退出...")
        while viewer_handle.is_running():
            sync_v()
            time.sleep(0.02)
