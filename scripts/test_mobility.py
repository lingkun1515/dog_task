#!/usr/bin/env python3
"""Go2MujocoMobility 单元测试 (kinematic + RL 模式)."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.mobility.go2_mujoco import Go2MujocoMobility
from dog_task.core.models import VelocityCommand, Pose2D


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
    mob = Go2MujocoMobility({"control_mode": "kinematic"}, world)
    try:
        status = mob.healthcheck()
        assert status.ready, f"healthcheck should be ready: {status.message}"
        print("  PASS: healthcheck ready")
    finally:
        cleanup_mob(mob)


def test_set_velocity_and_state(world):
    """验证 set_velocity → qvel 更新 → get_state 位移变化."""
    mob = Go2MujocoMobility({"control_mode": "kinematic"}, world)
    try:
        state0 = mob.get_state()
        assert state0.is_stopped, "initial state should be stopped"

        result = mob.set_velocity(VelocityCommand(linear_x_mps=0.5, linear_y_mps=0.0, angular_z_rps=0.0))
        assert result.success, f"set_velocity failed: {result.message}"
        time.sleep(0.5)

        state1 = mob.get_state()
        dx = state1.pose.x_m - state0.pose.x_m
        assert dx > 0.1, f"Should have moved forward ~0.25m, got dx={dx:.3f}"
        print(f"  PASS: moved forward dx={dx:.3f}m in 0.5s")
    finally:
        cleanup_mob(mob)


def test_stop_and_is_stopped(world):
    """验证 stop 后 is_stopped 正确反映状态 (bug fix #1.1)."""
    mob = Go2MujocoMobility({"control_mode": "kinematic"}, world)
    try:
        mob.set_velocity(VelocityCommand(linear_x_mps=0.5))
        time.sleep(0.3)
        mob.stop()
        time.sleep(0.5)

        state = mob.get_state()
        assert state.is_stopped, f"is_stopped should be True after stop, got {state.is_stopped}"
        print(f"  PASS: stop -> is_stopped={state.is_stopped}")
    finally:
        cleanup_mob(mob)


def test_set_posture(world):
    """验证 stand_down 姿势切换."""
    mob = Go2MujocoMobility({"control_mode": "kinematic"}, world)
    try:
        mob.set_posture("stand")
        time.sleep(0.1)
        state = mob.get_state()
        assert state.posture == "stand"

        result = mob.set_posture("stand_down")
        assert result.success, f"stand_down failed: {result.message}"
        time.sleep(0.1)

        qpos = world.get_freejoint_qpos("root")
        assert qpos[2] < 0.20, f"stand_down height should be ~0.15, got z={qpos[2]:.3f}"
        print(f"  PASS: stand_down z={qpos[2]:.3f}")
    finally:
        cleanup_mob(mob)


def test_go_to(world):
    """验证 go_to 导航到目标点."""
    mob = Go2MujocoMobility({"control_mode": "kinematic"}, world)
    try:
        state0 = mob.get_state()
        goal = Pose2D(x_m=state0.pose.x_m + 1.0, y_m=state0.pose.y_m, yaw_rad=0.0)

        result = mob.go_to(goal)
        assert result.success, f"go_to failed: {result.message}"

        state1 = mob.get_state()
        dist = np.hypot(state1.pose.x_m - goal.x_m, state1.pose.y_m - goal.y_m)
        assert dist < 0.15, f"should be within 0.15m of goal, got dist={dist:.3f}"
        print(f"  PASS: go_to goal within {dist:.3f}m")
    finally:
        cleanup_mob(mob)


def test_rl_walk(world, viewer=None):
    """RL 运控行走 + 里程计测试."""
    mob = Go2MujocoMobility({"control_mode": "rl", "control_hz": 50}, world)
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
    parser.add_argument("--rl", action="store_true", help="测试 RL 运控模式")
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

    if args.rl:
        test_rl_walk(w, viewer_handle)
    else:
        test_healthcheck(w)
        test_set_velocity_and_state(w)
        test_stop_and_is_stopped(w)
        test_set_posture(w)
        test_go_to(w)
    print("=== ALL PASSED ===")

    if viewer_handle and viewer_handle.is_running():
        print("\n[MuJoCo Viewer] 关闭窗口退出...")
        while viewer_handle.is_running():
            sync_v()
            time.sleep(0.02)
