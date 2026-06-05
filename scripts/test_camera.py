#!/usr/bin/env python3
"""MujocoCameraSim 单元测试."""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.camera.mujoco_rgbd import MujocoCameraSim


def build_world():
    config = {
        "go2_xml": "assets/go2/go2_standalone.xml",
        "d1_urdf": "assets/d1/d1.urdf",
        "arm_mount_pos": [0.0, 0.0, 0.10],
        "arm_mount_quat": [1, 0, 0, 0],
        "camera_pos": [0.30, 0.0, 0.05],
        "camera_fovy": 58,
        "camera_resolution": [848, 480],
        "environment": {
            "ground_size": [8, 8],
            "objects": [
                {
                    "type": "sphere", "name": "test_target",
                    "pos": [1.0, 0.0, 0.03], "size": 0.025,
                    "rgba": [0.95, 0.25, 0.15, 1.0], "mocap": True,
                },
            ],
        },
    }
    builder = SceneBuilder(config)
    model, data = builder.build()
    world = SimWorld(model, data)
    world.reset_to_keyframe("home")
    world.forward()
    return world


def test_healthcheck(world):
    cam = MujocoCameraSim({"detection_mode": "ground_truth"}, world)
    status = cam.healthcheck()
    assert status.ready, f"healthcheck should be ready: {status.message}"
    print("  PASS: healthcheck ready")


def test_get_target_ground_truth(world):
    """验证 ground_truth 模式能检测到 mocap body."""
    cam = MujocoCameraSim(
        {"detection_mode": "ground_truth", "render_fps": 10, "width": 848, "height": 480},
        world,
    )
    try:
        obs = cam.get_target()
        assert obs is not None, "should detect mocap target"
        assert obs.class_name == "test_target", f"expected test_target, got {obs.class_name}"
        assert obs.position_m.z > 0, "target should be in front of camera (positive z)"
        print(f"  PASS: detected '{obs.class_name}' at "
              f"x={obs.position_m.x:.3f} y={obs.position_m.y:.3f} z={obs.position_m.z:.3f}")
    finally:
        cam.stop()


def test_rgbd_render(world):
    """验证 RGB 和深度帧渲染成功."""
    cam = MujocoCameraSim(
        {"detection_mode": "ground_truth", "render_fps": 10, "width": 640, "height": 360},
        world,
    )
    try:
        cam.get_target()  # 启动渲染
        time.sleep(0.3)
        rgb, depth = cam.get_raw_rgbd()
        assert rgb is not None, "RGB frame should not be None"
        assert depth is not None, "Depth frame should not be None"
        assert rgb.shape[:2] == (360, 640), f"RGB shape mismatch: {rgb.shape}"
        assert rgb.dtype == np.uint8, f"RGB dtype should be uint8, got {rgb.dtype}"
        print(f"  PASS: RGB={rgb.shape} depth={depth.shape}")
    finally:
        cam.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="弹出 MuJoCo 3D 可视化窗口")
    args = parser.parse_args()

    if not args.viewer:
        os.environ["MUJOCO_GL"] = "egl"

    print("=== MujocoCameraSim 单元测试 ===")
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
    sync_v()
    test_get_target_ground_truth(w)
    sync_v()
    test_rgbd_render(w)
    sync_v()
    print("=== ALL PASSED ===")

    if viewer_handle and viewer_handle.is_running():
        print("\n[MuJoCo Viewer] 关闭窗口退出...")
        while viewer_handle.is_running():
            sync_v()
            time.sleep(0.02)
