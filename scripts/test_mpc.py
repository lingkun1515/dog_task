#!/usr/bin/env python3
"""MPC 运控仿真测试脚本.

用法:
  conda activate mower
  python3 scripts/test_mpc.py                 # 无头模式
  python3 scripts/test_mpc.py --viewer        # 弹 MuJoCo 交互窗口
  python3 scripts/test_mpc.py --walk 1.0      # 前进指定米数
"""
import argparse
import logging
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.mobility.go2_mujoco import Go2MujocoMobility
from dog_task.core.models import VelocityCommand


def build_world(viewer=False):
    config_world = {
        "go2_xml": "assets/go2/go2_mpc.xml",
        "d1_urdf": "assets/d1/d1.urdf",
        "arm_mount_pos": [0.0, 0.0, 0.10],
        "arm_mount_quat": [1, 0, 0, 0],
        "camera_pos": [0.30, 0.0, 0.05],
        "camera_fovy": 58,
        "camera_xyaxes": "0 -1 0 0.0872 0 0.9962",
        "camera_resolution": [848, 480],
        "viewer": viewer,
        "environment": {
            "ground_size": [8, 8],
        },
    }
    builder = SceneBuilder(config_world)
    model, data = builder.build()
    world = SimWorld(model, data)
    world.reset_to_keyframe("home")
    world.forward()
    return world


def run(viewer=False, walk_dist=0.5, walk_speed=0.3):
    if not viewer:
        os.environ["MUJOCO_GL"] = "egl"

    print("=== 构建 MPC 仿真世界 ===", flush=True)
    world = build_world(viewer)
    print(f"  model: nq={world.model.nq}, nv={world.model.nv}, nu={world.model.nu}", flush=True)
    print(f"  timestep={world.model.opt.timestep}", flush=True)

    mob_config = {
        "control_mode": "mpc",
        "control_hz": 50,
        "stop_velocity_threshold": 0.05,
        "mpc": {
            "gait_hz": 3.0,
            "gait_duty": 0.6,
            "mpc_horizon": 10,
            "z_des": 0.27,
            "verbose": False,
        },
    }
    mob = Go2MujocoMobility(mob_config, world)

    # 健康检查
    print("\n[healthcheck]", flush=True)
    status = mob.healthcheck()
    print(f"  ready={status.ready}, msg={status.message}", flush=True)
    if not status.ready:
        print("FAILED: healthcheck 未就绪", flush=True)
        return

    # 启动 viewer
    viewer_handle = None
    if viewer:
        import mujoco.viewer
        viewer_handle = mujoco.viewer.launch_passive(world.model, world.data)
        print("  MuJoCo viewer 已打开", flush=True)

    def sync_viewer():
        if viewer_handle and viewer_handle.is_running():
            viewer_handle.sync()

    # --- Phase 1: 站立 ---
    print("\n[Phase 1] 站立...", flush=True)
    result = mob.set_posture("stand")
    print(f"  {result.message}", flush=True)
    time.sleep(0.5)
    sync_viewer()

    state = mob.get_state()
    print(f"  初始位姿: x={state.pose.x_m:.3f}, y={state.pose.y_m:.3f}, z={mob._world.get_freejoint_qpos('root')[2]:.3f}",
          flush=True)

    # --- Phase 2: 前进 ---
    print(f"\n[Phase 2] MPC 前进 {walk_dist}m @ {walk_speed}m/s...", flush=True)
    mob.set_velocity(VelocityCommand(
        linear_x_mps=walk_speed,
        linear_y_mps=0.0,
        angular_z_rps=0.0,
    ))

    start_state = mob.get_state()
    start_x, start_y = start_state.pose.x_m, start_state.pose.y_m
    t0 = time.time()
    dist_traveled = 0.0
    timeout = walk_dist / max(walk_speed, 0.1) + 10.0

    while dist_traveled < walk_dist and time.time() - t0 < timeout:
        time.sleep(0.02)
        sync_viewer()
        state = mob.get_state()
        dist_traveled = math.hypot(state.pose.x_m - start_x, state.pose.y_m - start_y)

        if int((time.time() - t0) * 10) % 10 == 0:
            print(f"\r  位移: {dist_traveled:.3f}m / {walk_dist}m  "
                  f"mpc_time={mob._mpc_ctrl.solve_time_ms:.1f}ms", end="", flush=True)

    mob.stop()
    time.sleep(0.2)
    sync_viewer()
    print(f"\n  最终位移: {dist_traveled:.3f}m, 耗时: {time.time()-t0:.1f}s", flush=True)

    # --- Phase 3: 趴下 ---
    print("\n[Phase 3] 趴下...", flush=True)
    mob.set_posture("stand_down")
    time.sleep(0.5)
    sync_viewer()

    state = mob.get_state()
    z = mob._world.get_freejoint_qpos("root")[2]
    print(f"  高度: z={z:.3f}m", flush=True)

    # 保持 viewer
    if viewer_handle and viewer_handle.is_running():
        print("\n[完成] Viewer 保持打开, 关闭窗口退出", flush=True)
        while viewer_handle.is_running():
            sync_viewer()
            time.sleep(0.05)

    mob._running = False
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="弹出 MuJoCo 交互窗口")
    parser.add_argument("--walk", type=float, default=0.8, help="前进距离 (m)")
    parser.add_argument("--speed", type=float, default=0.4, help="前进速度 (m/s)")
    args = parser.parse_args()
    run(viewer=args.viewer, walk_dist=args.walk, walk_speed=args.speed)
