#!/usr/bin/env python3
"""walk-and-pick 仿真调试脚本.

用法:
  conda activate mower
  python3 scripts/test_walk_and_pick.py              # 无头模式
  python3 scripts/test_walk_and_pick.py --viewer     # 弹 MuJoCo 交互窗口
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.mobility.go2_mujoco import Go2MujocoMobility
from dog_task.modules.arm.mujoco_unified import UnifiedMujocoArm
from dog_task.modules.camera.mujoco_rgbd import MujocoCameraSim
from dog_task.core.models import VelocityCommand


def build_world(viewer=False):
    config_world = {
        "go2_xml": "assets/go2/go2_standalone.xml",
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
            "objects": [
                {
                    "type": "sphere", "name": "bottle",
                    "pos": [1.0, 0.0, 0.03], "size": 0.025,
                    "rgba": [0.2, 0.8, 0.1, 1.0], "mocap": True,
                },
                {
                    "type": "box", "name": "basket_vis",
                    "pos": [0.12, 0.25, 0.03],
                    "size": [0.08, 0.06, 0.04],
                    "rgba": [0.3, 0.3, 0.8, 0.5], "mocap": False,
                },
            ],
        },
    }
    builder = SceneBuilder(config_world)
    model, data = builder.build()
    world = SimWorld(model, data)
    world.reset_to_keyframe("home")
    world.forward()
    return world


def run(viewer=False):
    if not viewer:
        os.environ["MUJOCO_GL"] = "egl"

    print("=== 构建仿真世界 ===", flush=True)
    world = build_world(viewer)
    print(f"  nq={world.model.nq}, nbody={world.model.nbody}", flush=True)

    mob = Go2MujocoMobility({"control_mode": "kinematic"}, world)
    cam = MujocoCameraSim(
        {"detection_mode": "ground_truth", "render_fps": 10, "width": 848, "height": 480},
        world,
    )
    arm = UnifiedMujocoArm(
        {"tcp_offset_m": 0.12, "approach_dir": [0, 0, -1], "approach_height_m": 0.08,
         "seconds_per_segment": 0.3, "fps": 30},
        world,
    )

    # 启动 viewer（如果需要）
    viewer_handle = None
    if viewer:
        import mujoco.viewer
        viewer_handle = mujoco.viewer.launch_passive(world.model, world.data)
        print("  MuJoCo viewer 已打开", flush=True)

    def sync_viewer():
        if viewer_handle and viewer_handle.is_running():
            viewer_handle.sync()

    # --- Phase 1: 初始检测 ---
    print("\n[Phase 1] 初始位置检测...", flush=True)
    time.sleep(0.5)
    obs = cam.get_target()
    if obs:
        print(f"  检测到: {obs.class_name}, 距离 z={obs.position_m.z:.3f}m", flush=True)
    else:
        print("  未检测到目标", flush=True)

    # --- Phase 2: 前进 ---
    walk_dist = 0.6  # 走 0.6m，目标在 1.0m，走完后目标在前方约 0.4m
    speed = 0.3
    duration = walk_dist / speed
    print(f"\n[Phase 2] Go2 前进 {walk_dist}m (speed={speed}m/s, ~{duration:.1f}s)...", flush=True)

    mob.set_velocity(VelocityCommand(linear_x_mps=speed, linear_y_mps=0.0, angular_z_rps=0.0))
    t0 = time.time()
    while time.time() - t0 < duration:
        time.sleep(0.05)
        sync_viewer()
    mob.stop()
    time.sleep(0.2)
    sync_viewer()

    state = mob.get_state()
    print(f"  到达位置: x={state.pose.x_m:.3f}m", flush=True)

    # --- Phase 3: 趴下 ---
    print("\n[Phase 3] 趴下...", flush=True)
    mob.set_posture("stand_down")
    time.sleep(0.3)
    sync_viewer()

    # --- Phase 4: 检测目标 ---
    print("\n[Phase 4] 相机检测目标...", flush=True)
    time.sleep(0.5)
    obs = cam.get_target()
    if obs:
        print(f"  检测到: {obs.class_name}", flush=True)
        print(f"  相机坐标: x={obs.position_m.x:.3f}, y={obs.position_m.y:.3f}, z={obs.position_m.z:.3f}", flush=True)
    else:
        print("  未检测到目标! 流程终止", flush=True)
        cam.stop()
        mob._running = False
        return

    # --- Phase 5: 抓取 ---
    print("\n[Phase 5] 手臂抓取...", flush=True)
    from dog_task.core.models import PickRequest, Vector3
    target_arm = obs.position_m  # 简化: camera_to_arm_base = identity
    request = PickRequest(
        target_arm_base_m=target_arm,
        basket_arm_base_m=Vector3(x=0.12, y=0.22, z=0.04),
        class_name=obs.class_name,
        approach_height_m=0.08,
    )
    result = arm.pick_and_place(request)
    sync_viewer()
    print(f"  结果: {'OK' if result.success else 'FAILED'} - {result.message}", flush=True)

    # 保持 viewer 打开
    if viewer_handle and viewer_handle.is_running():
        print("\n[完成] Viewer 保持打开, 关闭窗口退出", flush=True)
        while viewer_handle.is_running():
            sync_viewer()
            time.sleep(0.05)

    cam.stop()
    mob._running = False
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="弹出 MuJoCo 交互窗口")
    args = parser.parse_args()
    run(viewer=args.viewer)
