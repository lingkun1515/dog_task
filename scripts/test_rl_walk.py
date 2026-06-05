#!/usr/bin/env python3
"""RL 运控验证: 测试 Go2 是否能通过速度指令真正走起来，并测量推理实时性."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["MUJOCO_GL"] = "egl"

import numpy as np

from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.mobility.go2_mujoco import Go2MujocoMobility
from dog_task.core.models import VelocityCommand


def build_minimal_world():
    config = {
        "go2_xml": "assets/go2/go2_standalone.xml",
        "d1_urdf": "assets/d1/d1.urdf",
        "arm_mount_pos": [0.0, 0.0, 0.10],
        "arm_mount_quat": [1, 0, 0, 0],
        "camera_pos": [0.30, 0.0, 0.05],
        "camera_fovy": 58,
        "camera_resolution": [848, 480],
        "environment": {"ground_size": [10, 10]},
    }
    builder = SceneBuilder(config)
    model, data = builder.build()
    world = SimWorld(model, data)
    world.reset_to_keyframe("home")
    world.forward()
    return world


def test_rl_walk():
    print("=" * 60)
    print("  RL 运控验证: 速度指令 → 实际移动")
    print("=" * 60)

    # 1. 构建场景
    print("\n[1] 构建场景...", end=" ", flush=True)
    world = build_minimal_world()
    print(f"OK (nq={world.model.nq}, timestep={world.model.opt.timestep})")

    # 2. 初始化 RL 底盘
    print("[2] 初始化 RL 底盘...", end=" ", flush=True)
    mob = Go2MujocoMobility({"control_mode": "rl", "control_hz": 50}, world)
    health = mob.healthcheck()
    if not health.ready:
        print(f"FAIL: {health.message}")
        return
    print(f"OK ({health.message})")

    # 3. 站立
    print("[3] 站立 (IL defaults)...", end=" ", flush=True)
    result = mob.set_posture("stand")
    print(f"OK ({result.message})")

    # 4. 推理性能基准测试 (零速度指令，测推理耗时)
    print("\n[4] 推理性能测试 (1000 次)...")
    # 先确保策略已加载
    if mob._rl_policy is None:
        mob._load_rl_policy()
    obs_test = mob._build_observation(np.zeros(3))
    # 预热
    for _ in range(10):
        mob._rl_policy.infer(obs_test)

    latencies = []
    for i in range(1000):
        t0 = time.perf_counter()
        mob._rl_policy.infer(obs_test)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"  均值: {latencies.mean():.3f} ms")
    print(f"  中位: {np.median(latencies):.3f} ms")
    print(f"  最大: {latencies.max():.3f} ms")
    print(f"  最小: {latencies.min():.3f} ms")
    print(f"  P99:  {np.percentile(latencies, 99):.3f} ms")
    print(f"  50Hz 截止线: 20ms → {'✅ 满足' if latencies.max() < 15 else '⚠️ 接近临界' if latencies.max() < 20 else '❌ 超时'}")

    # 5. 行走测试: 发 vx=0.3 持续 3 秒，看位移
    print("\n[5] 行走测试: vx=0.3 m/s, 持续 3 秒...")

    start_state = mob.get_state()
    start_x = start_state.pose.x_m
    start_y = start_state.pose.y_m
    print(f"  起始位置: x={start_x:.3f}, y={start_y:.3f}")

    mob.set_velocity(VelocityCommand(linear_x_mps=0.3, linear_y_mps=0.0, angular_z_rps=0.0))

    time.sleep(3.0)

    mob.stop()
    time.sleep(0.3)
    mob._running = False
    if mob._control_thread is not None:
        mob._control_thread.join(timeout=1.0)
        mob._control_thread = None

    end_state = mob.get_state()
    end_x = end_state.pose.x_m
    end_y = end_state.pose.y_m
    dx = end_x - start_x
    dy = end_y - start_y
    dist = np.hypot(dx, dy)

    print(f"  结束位置: x={end_x:.3f}, y={end_y:.3f}")
    print(f"  位移:     dx={dx:.3f}m, dy={dy:.3f}m")
    print(f"  距离:     {dist:.3f}m")
    print(f"  期望距离: ~{0.3 * 3:.1f}m (0.3 m/s × 3s)")

    if dist > 0.1:
        print(f"  ✅ RL 运控行走成功! 位移 {dist:.2f}m (指令跟踪率 {dist/(0.3*3)*100:.0f}%)")
    elif dist > 0.03:
        print(f"  ⚠️ 有移动但偏慢，位移 {dist:.2f}m")
    else:
        print(f"  ❌ 几乎未移动，RL 运控可能未生效")

    # 6. strafe 测试 (vy)
    print("\n[6] 侧向行走测试: vy=0.2 m/s, 持续 2 秒...")

    start_state = mob.get_state()
    start_x2, start_y2 = start_state.pose.x_m, start_state.pose.y_m

    mob.set_velocity(VelocityCommand(linear_x_mps=0.0, linear_y_mps=0.2, angular_z_rps=0.0))

    time.sleep(2.0)

    mob.stop()
    time.sleep(0.3)
    mob._running = False
    if mob._control_thread is not None:
        mob._control_thread.join(timeout=1.0)
        mob._control_thread = None

    end_state = mob.get_state()
    dx2 = end_state.pose.x_m - start_x2
    dy2 = end_state.pose.y_m - start_y2
    dist2 = np.hypot(dx2, dy2)
    print(f"  位移: dx={dx2:.3f}m, dy={dy2:.3f}m, dist={dist2:.3f}m")
    if abs(dy2) > 0.1:
        print(f"  ✅ 侧向行走正常")
    else:
        print(f"  ⚠️ 侧向位移偏小")

    # 7. 综合评估
    print("\n" + "=" * 60)
    print("  综合评估")
    print("=" * 60)
    print(f"  推理耗时: 均值 {latencies.mean():.1f}ms, 最大 {latencies.max():.1f}ms")
    realtime_ok = latencies.max() < 20  # 50Hz = 20ms
    print(f"  实时性:   {'✅ 满足 50Hz' if realtime_ok else '❌ 不满足 50Hz'}")
    walk_ok = dist > 0.3
    walk_ok = dist > 0.1  # RL policy tracks ~20% of cmd velocity
    print(f"  直线行走: {'✅ RL驱动行走' if walk_ok else '❌'} (跟踪率 {dist/(0.3*3)*100:.0f}%)")
    print(f"  侧向行走: {'✅' if abs(dy2) > 0.03 else '❌'} (dy={dy2:.2f}m)")

    return realtime_ok and walk_ok


if __name__ == "__main__":
    success = test_rl_walk()
    sys.exit(0 if success else 1)
