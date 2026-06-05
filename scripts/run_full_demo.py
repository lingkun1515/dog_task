#!/usr/bin/env python3
"""小白一键全流程演示: UI 浏览器界面 + MuJoCo 3D 可视化 + Go2 前进抓取.

用法:
  conda activate mower
  python3 scripts/run_full_demo.py

然后:
  1. 浏览器打开 http://127.0.0.1:8765
  2. 点击 "一键派发清理任务" 按钮
  3. 观看 MuJoCo 3D 窗口中的狗行走 + 机械臂抓取
  4. UI 界面会实时显示任务进度和仿真相机画面
"""
import json
import os
import sys
import threading
import time
import urllib.request
import webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
from mujoco import viewer as mujoco_viewer
import numpy as np

from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.mobility.go2_mujoco import Go2MujocoMobility
from dog_task.modules.arm.mujoco_unified import UnifiedMujocoArm
from dog_task.modules.camera.mujoco_rgbd import MujocoCameraSim
from dog_task.core.models import VelocityCommand, PickRequest

UI_PORT = 8765
UI_BASE = f"http://127.0.0.1:{UI_PORT}"


def start_ui_server():
    """在后台线程启动 UI HTTP 服务器."""
    UI_DIR = os.path.join(os.path.dirname(__file__), "..", "UI")
    sys.path.insert(0, UI_DIR)
    import server as ui_server
    srv = ui_server.create_server("127.0.0.1", UI_PORT, mock=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="ui-server")
    t.start()
    return srv


def sync_ui_status(status, message=""):
    """POST /api/task/status 同步状态到 UI."""
    try:
        payload = json.dumps({"status": status, "message": message}).encode()
        req = urllib.request.Request(
            f"{UI_BASE}/api/task/status", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=1.0)
    except Exception:
        pass


def dispatch_ui_task():
    """向 UI 派发任务."""
    try:
        payload = json.dumps({"scene": "草坪异物清理"}).encode()
        req = urllib.request.Request(
            f"{UI_BASE}/api/task/dispatch", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=1.0)
    except Exception:
        pass


def build_world():
    config = {
        "go2_xml": "assets/go2/go2_standalone.xml",
        "d1_urdf": "assets/d1/d1.urdf",
        "arm_mount_pos": [0.0, 0.0, 0.10],
        "arm_mount_quat": [1, 0, 0, 0],
        "camera_pos": [0.30, 0.0, 0.05],
        "camera_fovy": 58,
        "camera_xyaxes": "0 -1 0 0.0872 0 0.9962",
        "camera_resolution": [848, 480],
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
    builder = SceneBuilder(config)
    model, data = builder.build()
    world = SimWorld(model, data)
    world.reset_to_keyframe("home")
    world.forward()
    return world


def print_step(n, total, text):
    print(f"\n{'='*50}")
    print(f"  [{n}/{total}] {text}")
    print(f"{'='*50}", flush=True)


def main():
    print("=" * 50)
    print("  DogTaskSim 全流程演示")
    print("  Go2 + D1 + D455 仿真 + UI + 3D 可视化")
    print("=" * 50)

    # 1. 启动 UI
    print("\n[启动] UI 服务器...", end=" ", flush=True)
    server = start_ui_server()
    time.sleep(0.5)
    print(f"OK → {UI_BASE}")

    # 2. 构建场景
    print("[启动] 构建 MuJoCo 仿真场景...", end=" ", flush=True)
    world = build_world()
    print(f"OK (nq={world.model.nq}, nbody={world.model.nbody})")

    # 3. 初始化模块
    mob = Go2MujocoMobility({"control_mode": "kinematic"}, world)
    cam = MujocoCameraSim(
        {"detection_mode": "ground_truth", "render_fps": 15,
         "width": 848, "height": 480, "push_to_ui": True},
        world,
    )
    arm = UnifiedMujocoArm(
        {"tcp_offset_m": 0.12, "approach_dir": [0, 0, -1],
         "approach_height_m": 0.08, "seconds_per_segment": 0.3, "fps": 30},
        world,
    )

    # 4. 打开 MuJoCo 3D 可视化窗口
    print("[启动] MuJoCo 3D 可视化窗口...", end=" ", flush=True)
    viewer = mujoco_viewer.launch_passive(world.model, world.data)
    print("OK (关闭窗口即结束演示)")

    def sync_v():
        if viewer.is_running():
            viewer.sync()

    # 5. 向 UI 派发任务
    dispatch_ui_task()

    TOTAL_STEPS = 7
    step = 0

    # --- Step 1: 初始状态 ---
    step += 1
    print_step(step, TOTAL_STEPS, "初始姿态 — 狗站立，等待命令")
    sync_ui_status("ack", "设备就绪，等待派发任务")
    for _ in range(30):
        time.sleep(0.05)
        sync_v()

    # --- Step 2: 前进 ---
    step += 1
    walk_dist, speed = 0.6, 0.3
    duration = walk_dist / speed
    print_step(step, TOTAL_STEPS, f"Go2 前进 {walk_dist}m (速度 {speed}m/s)")
    sync_ui_status("go_to_B", "设备正在前往 B 点目标区域")

    mob.set_velocity(VelocityCommand(linear_x_mps=speed, linear_y_mps=0.0, angular_z_rps=0.0))
    t0 = time.time()
    while time.time() - t0 < duration:
        time.sleep(0.05)
        sync_v()
    mob.stop()
    time.sleep(0.3)
    state = mob.get_state()
    print(f"  当前位置: x={state.pose.x_m:.2f}m", flush=True)

    # --- Step 3: 趴下 ---
    step += 1
    print_step(step, TOTAL_STEPS, "Go2 趴下，准备机械臂作业")
    sync_ui_status("arrived_B_confirmed", "已到达 B 点，目标确认完成")
    mob.set_posture("stand_down")
    for _ in range(10):
        time.sleep(0.05)
        sync_v()

    # --- Step 4: 相机检测 ---
    step += 1
    print_step(step, TOTAL_STEPS, "D455 仿真相机检测目标")
    time.sleep(0.5)
    obs = cam.get_target()
    if obs:
        print(f"  检测到: {obs.class_name} @ "
              f"x={obs.position_m.x:.2f} y={obs.position_m.y:.2f} z={obs.position_m.z:.2f}m")
    else:
        print("  未检测到目标！")
        cam.stop()
        mob._running = False
        viewer.close()
        server.shutdown()
        return

    # --- Step 5: 机械臂抓取 ---
    step += 1
    print_step(step, TOTAL_STEPS, "D1 机械臂执行抓取放置")
    sync_ui_status("arm_start", "机械臂正在执行异物清理")

    from dog_task.core.models import Vector3
    request = PickRequest(
        target_arm_base_m=obs.position_m,
        basket_arm_base_m=Vector3(x=0.12, y=0.22, z=0.04),
        class_name=obs.class_name,
        approach_height_m=0.08,
    )
    result = arm.pick_and_place(request)
    for _ in range(10):
        sync_v()

    # --- Step 6: 完成 ---
    step += 1
    print_step(step, TOTAL_STEPS, f"抓取结果: {'成功' if result.success else '失败'} — {result.message}")
    if result.success:
        sync_ui_status("arm_done", "异物已安全放入回收篮")
        time.sleep(1.0)
        sync_ui_status("done", "清理任务已完成，结果已回传")
    else:
        sync_ui_status("failed", result.message)

    # --- Step 7: 保持运行 ---
    step += 1
    print_step(step, TOTAL_STEPS, "演示完成！")
    print(f"""
    ╔══════════════════════════════════════════╗
    ║  🎯 演示完成！                          ║
    ║                                        ║
    ║  UI 界面:  {UI_BASE}           ║
    ║  3D 窗口:  关闭 MuJoCo 窗口即退出       ║
    ║                                        ║
    ║  💡 提示:                              ║
    ║  - 鼠标拖拽 3D 窗口可旋转视角           ║
    ║  - 滚轮缩放                            ║
    ║  - 右键拖拽平移                         ║
    ╚══════════════════════════════════════════╝
    """)

    while viewer.is_running():
        sync_v()
        time.sleep(0.05)

    cam.stop()
    mob._running = False
    server.shutdown()
    print("再见！")


if __name__ == "__main__":
    main()
