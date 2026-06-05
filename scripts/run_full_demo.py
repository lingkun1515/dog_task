#!/usr/bin/env python3
"""交互式全流程演示: UI 浏览器界面 + MuJoCo 3D 可视化 + Go2 前进抓取.

与剧本式演示不同，本脚本模拟真实运行环境：
  - 仿真进程轮询 UI 服务端的任务状态
  - 用户在浏览器点击 "一键派发清理任务" 后，仿真才真正执行
  - 每个阶段只有仿真实际完成后，才 POST 状态更新到 UI
  - 支持暂停/继续/人工接管/重置 等 UI 按钮交互

用法:
  conda activate mower
  python3 scripts/run_full_demo.py

然后浏览器打开 http://127.0.0.1:8765，点击按钮即可体验。
"""
import json
import math
import os
import sys
import threading
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["MUJOCO_GL"] = "egl"

from mujoco import viewer as mujoco_viewer
import numpy as np
import mujoco

from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.mobility.go2_mujoco import Go2MujocoMobility
from dog_task.modules.arm.mujoco_unified import UnifiedMujocoArm
from dog_task.modules.camera.mujoco_rgbd import MujocoCameraSim
from dog_task.core.models import VelocityCommand, PickRequest

UI_PORT = 8765
UI_BASE = f"http://127.0.0.1:{UI_PORT}"
POLL_INTERVAL = 0.5  # 轮询 UI 任务状态的间隔


def start_ui_server():
    """在后台线程启动 UI HTTP 服务器 (mock=False, 等待仿真推送状态)."""
    UI_DIR = os.path.join(os.path.dirname(__file__), "..", "UI")
    sys.path.insert(0, UI_DIR)
    import server as ui_server
    srv = ui_server.create_server("127.0.0.1", UI_PORT, mock=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="ui-server")
    t.start()
    return srv


def ui_get(path: str) -> dict:
    """GET 请求 UI 服务."""
    try:
        req = urllib.request.Request(f"{UI_BASE}{path}")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def ui_post(path: str, payload: dict) -> dict:
    """POST 请求 UI 服务."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{UI_BASE}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def ui_sync_status(status: str, message: str = "") -> dict:
    """推送任务状态到 UI."""
    return ui_post("/api/task/status", {"status": status, "message": message})


def ui_get_task() -> dict:
    """获取当前任务快照."""
    return ui_get("/api/task")


def ui_reset_task():
    """重置任务到 idle."""
    ui_post("/api/task/reset", {})


def build_world():
    config = {
        "go2_xml": "assets/go2/go2_standalone.xml",
        "d1_urdf": "assets/d1/d1.urdf",
        "arm_mount_pos": [0.0, 0.0, 0.10],
        "arm_mount_quat": [1, 0, 0, 0],
        "camera_pos": [0.30, 0.0, 0.05],
        "camera_fovy": 58,
        "camera_xyaxes": "0 -1 0 0.3907 0 0.9205",
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


def execute_workflow(world, mob, cam, arm, sync_v):
    """执行完整的清理任务流程，每个阶段完成后推送 UI 状态。返回 True 表示成功."""
    from dog_task.core.models import Vector3

    # --- 阶段 1: 前往目标点 ---
    print("  [执行] Go2 前进 0.35m (RL 运控 + 里程计闭环)...")
    ui_sync_status("go_to_B", "设备正在前往 B 点目标区域")
    walk_dist = 0.35
    start_state = mob.get_state()
    start_x, start_y = start_state.pose.x_m, start_state.pose.y_m
    mob.set_velocity(VelocityCommand(linear_x_mps=0.5, linear_y_mps=0.0, angular_z_rps=0.0))

    t0 = time.time()
    dist_traveled = 0.0
    while dist_traveled < walk_dist:
        time.sleep(0.05)
        sync_v()
        state = mob.get_state()
        dist_traveled = math.hypot(state.pose.x_m - start_x, state.pose.y_m - start_y)
        task = ui_get_task()
        if task.get("status") in ("paused", "manual_takeover"):
            mob.stop()
            print("  [暂停] 任务被 UI 暂停或接管")
            return False
        if time.time() - t0 > 30:
            print("  [超时] 行走超过 30 秒")
            break
    mob.stop()
    time.sleep(0.3)

    state = mob.get_state()
    print(f"  [到达] x={state.pose.x_m:.2f}m (实际位移 {dist_traveled:.2f}m)")

    # --- 阶段 2: 到点确认 ---
    print("  [执行] 到点确认 (RL 模式跳过趴下)...")
    ui_sync_status("arrived_B_confirmed", "已到达 B 点，目标确认完成")
    mob.set_posture("stand_down")  # RL 模式忽略，保持站立
    for _ in range(10):
        time.sleep(0.05)
        sync_v()

    # --- 阶段 3: 目标检测 ---
    print("  [执行] D455 相机检测目标...")
    ui_sync_status("arm_start", "机械臂正在执行异物清理")
    time.sleep(0.5)
    obs = cam.get_target()
    if not obs:
        print("  [失败] 未检测到目标")
        ui_sync_status("failed", "到点确认失败，已进入人工接管")
        return False

    print(f"  [检测] {obs.class_name} @ "
          f"x={obs.position_m.x:.2f} y={obs.position_m.y:.2f} z={obs.position_m.z:.2f}m")

    # --- 阶段 4: 机械臂抓取 ---
    print("  [执行] D1 机械臂抓取放置...")
    request = PickRequest(
        target_arm_base_m=obs.position_m,
        basket_arm_base_m=Vector3(x=0.12, y=0.22, z=0.04),
        class_name=obs.class_name,
        approach_height_m=0.08,
    )
    result = arm.pick_and_place(request)
    for _ in range(10):
        sync_v()

    # --- 阶段 5: 任务完成 ---
    if result.success:
        print(f"  [完成] 抓取成功 — {result.message}")
        ui_sync_status("arm_done", "异物已安全放入回收篮")
        time.sleep(0.5)
        ui_sync_status("done", "清理任务已完成，结果已回传")
    else:
        print(f"  [失败] {result.message}")
        ui_sync_status("failed", result.message)

    return True


def main():
    print("=" * 50)
    print("  DogTaskSim 交互式全流程演示")
    print("  Go2 + D1 + D455 仿真 + UI + 3D 可视化")
    print("=" * 50)

    # 1. 启动 UI
    print("\n[启动] UI 服务器 (mock=off, 等待仿真推送)...", end=" ", flush=True)
    server = start_ui_server()
    time.sleep(0.5)
    print(f"OK → {UI_BASE}")

    # 2. 构建场景
    print("[启动] 构建 MuJoCo 仿真场景...", end=" ", flush=True)
    world = build_world()
    print(f"OK (nq={world.model.nq}, nbody={world.model.nbody})")

    # 3. 初始化模块
    mob = Go2MujocoMobility({"control_mode": "mpc", "control_hz": 50}, world)
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

    # 4. 相机渲染器先启动 (占 EGL context), 再开 3D 窗口
    print("[启动] 仿真相机渲染器...", end=" ", flush=True)
    cam.get_target()
    time.sleep(0.2)
    print("OK")

    # 5. MuJoCo 3D 可视化
    print("[启动] MuJoCo 3D 可视化窗口...", end=" ", flush=True)
    viewer = mujoco_viewer.launch_passive(world.model, world.data)
    print("OK (关闭窗口即结束演示)")

    def sync_v():
        if viewer.is_running():
            viewer.sync()

    # 6. 进入交互式事件循环, 轮询 UI 任务状态
    print(f"""
╔══════════════════════════════════════════════╗
║  🟢 系统就绪，等待用户操作                  ║
║                                            ║
║  浏览器打开: {UI_BASE}              ║
║  点击 "一键派发清理任务" 启动任务           ║
║  关闭 3D 窗口退出                          ║
╚══════════════════════════════════════════════╝
""")

    last_task_id = None
    ui_sync_status("idle", "设备待命，等待派发任务")

    while viewer.is_running():
        sync_v()

        try:
            task = ui_get_task()
        except Exception:
            time.sleep(POLL_INTERVAL)
            continue

        status = task.get("status", "idle")
        task_id = task.get("task_id")

        # 检测到新任务派发
        if task_id and task_id != last_task_id and status == "ack":
            last_task_id = task_id
            print(f"\n{'='*50}")
            print(f"  [任务派发] {task_id} — {task.get('scene', '')}")
            print(f"{'='*50}")

            # 重置场景到初始状态
            world.reset_to_keyframe("home")
            world.forward()
            time.sleep(0.2)

            success = execute_workflow(world, mob, cam, arm, sync_v)
            if success:
                last_task_id = None  # 允许下次重新派发

        # 检测到手动重置 (UI 点 "返回待命界面")
        if status == "idle" and last_task_id is not None:
            last_task_id = None
            print("\n  [重置] 任务已重置，等待新任务...")
            world.reset_to_keyframe("home")
            world.forward()
            mob.stop()
            time.sleep(0.5)

        time.sleep(POLL_INTERVAL)

    cam.stop()
    mob._running = False
    server.shutdown()
    print("再见！")


if __name__ == "__main__":
    main()
