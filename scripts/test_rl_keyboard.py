#!/usr/bin/env python3
"""RL 运控键盘控制 Demo — GLFW 渲染 + RL 推理，单线程，无多线程冲突.

用法:
  conda activate mower
  python3 scripts/test_rl_keyboard.py

键盘（在 MuJoCo 窗口中操作）:
  按住 W/S: 前进/后退 (vx 持续增减)
  按住 A/D: 左转/右转 (vyaw 持续增减)
  按住 Q/E: 左移/右移 (vy 持续增减)
  Space:     急停 (速度归零)
  R:         重新站立
  Esc:       退出
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import mujoco
from mujoco.glfw import glfw          # MuJoCo 自带的 GLFW 绑定

from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.sim.rl_policy import RLPolicy
from dog_task.modules.mobility.go2_mujoco import (
    GO2_LEG_JOINTS, GO2_ACTUATORS,
    IL_DEFAULTS, MENAGERIE_HOME,
    IL_TO_MJ, MJ_TO_IL,
    IL_LIMITS, IL_TORQUE_LIMITS,
    ACTION_SCALE,
)
from dog_task.config import project_path


# ── 观测 & 重力 ──────────────────────────────────────────────────

def _build_obs(world, cmd, last_action):
    base_angvel = world.get_sensor("imu_gyro")
    base_quat = world.get_freejoint_qpos("root")[3:7]
    proj_grav = _proj_gravity(base_quat)

    joint_pos_mj = world.get_qpos(GO2_LEG_JOINTS)
    joint_vel_mj = world.get_qvel(GO2_LEG_JOINTS)
    joint_pos_il = joint_pos_mj[MJ_TO_IL]
    joint_vel_il = joint_vel_mj[MJ_TO_IL]

    return np.concatenate([
        base_angvel,
        proj_grav,
        cmd,
        joint_pos_il - IL_DEFAULTS,
        joint_vel_il,
        last_action,
    ])


def _proj_gravity(quat_wxyz):
    w, x, y, z = quat_wxyz
    gx = 2.0 * (x * z - w * y)
    gy = 2.0 * (y * z + w * x)
    gz = 1.0 - 2.0 * (x * x + y * y)
    return np.array([-gx, -gy, -gz])


def stand_up(world):
    """高增益 PD 站起到 IL_DEFAULTS (大腿=1.1, 匹配策略训练姿态, 支撑 D1 臂重量)."""
    target_q_mj = IL_DEFAULTS[IL_TO_MJ]
    kp, kd = 80.0, 2.0
    for _ in range(500):
        current_q = world.get_qpos(GO2_LEG_JOINTS)
        current_dq = world.get_qvel(GO2_LEG_JOINTS)
        if np.max(np.abs(target_q_mj - current_q)) < 0.01:
            break
        tau = kp * (target_q_mj - current_q) - kd * current_dq
        world.set_ctrl(GO2_ACTUATORS, tau)
        world.step(10)
    world.forward()


# ── GLFW 窗口 & 渲染 ─────────────────────────────────────────────

_one_shot_keys = set()
_key_held = {}  # key -> bool, 通过回调追踪所有按键状态

# 鼠标拖拽状态
_mouse_button_left = False
_mouse_button_right = False
_mouse_button_middle = False
_mouse_last_x = 0.0
_mouse_last_y = 0.0


def _key_callback(window, key, scancode, action, mods):
    """追踪所有按键状态 + 单次事件."""
    _key_held[key] = (action != glfw.RELEASE)
    if action == glfw.PRESS:
        _one_shot_keys.add(key)


def _mouse_button_callback(window, button, action, mods):
    """跟踪鼠标按键状态."""
    global _mouse_button_left, _mouse_button_right, _mouse_button_middle
    pressed = (action == glfw.PRESS)
    if button == glfw.MOUSE_BUTTON_LEFT:
        _mouse_button_left = pressed
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        _mouse_button_right = pressed
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        _mouse_button_middle = pressed


def _mouse_move_callback(window, xpos, ypos):
    """鼠标拖拽 → mjv_moveCamera 旋转/平移."""
    global _mouse_last_x, _mouse_last_y
    if not (_mouse_button_left or _mouse_button_right or _mouse_button_middle):
        _mouse_last_x, _mouse_last_y = xpos, ypos
        return

    dx = xpos - _mouse_last_x
    dy = ypos - _mouse_last_y
    _mouse_last_x, _mouse_last_y = xpos, ypos

    w, h = glfw.get_framebuffer_size(window)
    model = _win_model
    scn = _win_scn
    cam = _win_cam

    if _mouse_button_left:
        # 左键: 旋转 (横向=rotate_H, 纵向=rotate_V)
        if abs(dx) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                                  dx / w * 2.0, 0.0, scn, cam)
        if abs(dy) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                                  0.0, dy / h * 2.0, scn, cam)
    elif _mouse_button_right:
        # 右键: 平移
        if abs(dx) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                                  dx / w * 0.5, 0.0, scn, cam)
        if abs(dy) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V,
                                  0.0, -dy / h * 0.5, scn, cam)
    elif _mouse_button_middle:
        # 中键: 缩放
        if abs(dy) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                  0.0, dy / h * 2.0, scn, cam)


def _scroll_callback(window, xoffset, yoffset):
    """滚轮缩放."""
    scn = _win_scn
    cam = _win_cam
    model = _win_model
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                          0.0, -yoffset * 0.05, scn, cam)


# 模块级变量，供鼠标回调访问场景和相机
_win_model = None
_win_scn = None
_win_cam = None


def _init_window(model, data):
    global _win_model, _win_scn, _win_cam

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    window = glfw.create_window(1280, 720, "Go2 RL 键盘控制 — WASD 行走 | 鼠标拖拽视角", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window create failed")

    glfw.make_context_current(window)
    glfw.swap_interval(0)  # 不限制帧率

    # 键盘
    glfw.set_key_callback(window, _key_callback)
    # 鼠标
    glfw.set_mouse_button_callback(window, _mouse_button_callback)
    glfw.set_cursor_pos_callback(window, _mouse_move_callback)
    glfw.set_scroll_callback(window, _scroll_callback)

    scn = mujoco.MjvScene(model, maxgeom=10000)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat = (0.5, 0, 0.2)
    cam.distance = 2.5
    cam.elevation = -20
    cam.azimuth = 180

    _win_model = model
    _win_scn = scn
    _win_cam = cam

    opt = mujoco.MjvOption()
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    return window, scn, cam, opt, ctx


def _render(window, scn, cam, opt, model, data, ctx):
    w, h = glfw.get_framebuffer_size(window)
    vp = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    mujoco.mjr_render(vp, scn, ctx)


# ── 主循环 ───────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  RL 运控键盘控制 Demo (GLFW)")
    print("  按住 W/S: vx   按住 A/D: vyaw   按住 Q/E: vy")
    print("  Space: 停   R: 站   Esc: 退")
    print("  鼠标: 左键旋转 | 右键平移 | 滚轮缩放")
    print("=" * 55)

    # 1. 场景
    print("[1] 构建场景...", end=" ", flush=True)
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
    print(f"OK (nq={model.nq})")

    # 2. RL 策略 v6
    print("[2] 加载 flat_policy_v6...", end=" ", flush=True)
    policy = RLPolicy(str(project_path("assets/rl_models/flat_policy_v6.onnx")), obs_dim=45)
    print("OK")

    # 3. 窗口
    print("[3] GLFW 窗口...", end=" ", flush=True)
    window, scn, cam, opt, ctx = _init_window(model, data)
    print("OK")

    # ── 三段式控制 (对齐 go2_rl_policy_node.py) ──
    # STANDUP (前 stand_up_dur 秒): 高增益 PD → MENAGERIE_HOME, 强置 cmd=0
    # HOLD    (接下来 cmd_hold_dur 秒): 软增益 PD → MENAGERIE_HOME, 强置 cmd=0, 策略预热 last_action
    # POLICY  (之后): 软增益 PD → IL_DEFAULTS + action*0.25, 使用实际 cmd_vel
    cmd_vel = np.zeros(3)
    last_action = np.zeros(12)
    ctrl_dt = 0.02
    sub = int(round(ctrl_dt / model.opt.timestep))
    last_rl = data.time - ctrl_dt         # 确保首次迭代即进入 RL 推理
    steps = 0
    stand_up_dur = 3.0                     # wall-time 高增益站立
    cmd_hold_dur = 5.0                     # wall-time 策略预热 (force cmd=0)
    t_wall_start = time.time()
    phase = "START"
    phase_announced = {"STANDUP": False, "HOLD": False, "POLICY": False}

    print(f"\n[5] 主循环 — STANDUP({stand_up_dur:.0f}s) → HOLD({cmd_hold_dur:.0f}s) → POLICY")
    print("  在窗口中按住 WASD 控制 Go2\n")

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # ── 单次按键 ──
        for k in list(_one_shot_keys):
            if k == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif k == glfw.KEY_SPACE:
                cmd_vel[:] = 0.0
                print(f"  ⏹ 急停  wall_t={time.time() - t_wall_start:.1f}s")
            elif k == glfw.KEY_R:
                cmd_vel[:] = 0.0
                stand_up(world)
                last_action[:] = 0.0
                t_wall_start = time.time()  # 重置阶段计时
                for p in phase_announced:
                    phase_announced[p] = False
                print(f"  ↩ 重新站立  wall_t={time.time() - t_wall_start:.1f}s")
        _one_shot_keys.clear()

        # ── 持续按住: 通过回调追踪的 _key_held 状态 — 速度渐变 ──
        dv = 0.015
        if _key_held.get(glfw.KEY_W):
            cmd_vel[0] = min(cmd_vel[0] + dv, 1.0)
        if _key_held.get(glfw.KEY_S):
            cmd_vel[0] = max(cmd_vel[0] - dv, -1.0)
        if _key_held.get(glfw.KEY_A):
            cmd_vel[2] = min(cmd_vel[2] + 0.03, 2.0)
        if _key_held.get(glfw.KEY_D):
            cmd_vel[2] = max(cmd_vel[2] - 0.03, -2.0)
        if _key_held.get(glfw.KEY_Q):
            cmd_vel[1] = min(cmd_vel[1] + 0.015, 0.5)
        if _key_held.get(glfw.KEY_E):
            cmd_vel[1] = max(cmd_vel[1] - 0.015, -0.5)

        # ── RL 推理 (50Hz) ──
        now = data.time
        if now - last_rl >= ctrl_dt - 1e-6:
            last_rl = now

            # 用 wall time 做阶段切换（与参考 ROS 节点一致）
            elapsed_wall = time.time() - t_wall_start
            elapsed_into_policy = elapsed_wall - stand_up_dur
            in_standup = elapsed_wall < stand_up_dur
            in_hold = not in_standup and elapsed_into_policy < cmd_hold_dur

            # 阶段切换广播
            if in_standup and not phase_announced["STANDUP"]:
                phase_announced["STANDUP"] = True
                phase = "STANDUP"
                print(f"  [{phase}] kp=80/kd=2, target=IL_DEFAULTS, cmd=0  (wall_t={elapsed_wall:.1f}s)")
            elif in_hold and not phase_announced["HOLD"]:
                phase_announced["HOLD"] = True
                phase = "HOLD"
                print(f"  [{phase}] kp=20/kd=0.5, target=IL_DEFAULTS, cmd=0  策略预热中...  (wall_t={elapsed_wall:.1f}s)")
            elif not in_standup and not in_hold and not phase_announced["POLICY"]:
                phase_announced["POLICY"] = True
                phase = "POLICY"
                print(f"  [{phase}] kp=20/kd=0.5, target=IL_DEFAULTS+action, cmd=user  (wall_t={elapsed_wall:.1f}s)")

            # force cmd=0 during STANDUP and HOLD
            if in_standup or in_hold:
                feed_cmd = np.zeros(3)
            else:
                feed_cmd = cmd_vel.copy()
                if np.all(np.abs(feed_cmd) < 0.05):
                    feed_cmd[:] = 0.0

            # 构建观测 → 策略推理（三个阶段都跑推理，用于预热 last_action）
            obs = _build_obs(world, feed_cmd, last_action)
            action = policy.infer(obs)
            last_action = action.copy()

            # PD 目标: STANDUP/HOLD 用 IL_DEFAULTS (thigh=1.1) 确保关节角在策略训练分布内
            # MENAGERIE_HOME (thigh=0.9) 在 D1 臂增重下 kp=20 撑不住，导致关节偏离训练分布 → 策略输出爆炸
            if in_standup:
                target_q_mj = IL_DEFAULTS[IL_TO_MJ]
                kp_now, kd_now = 80.0, 2.0
            elif in_hold:
                target_q_mj = IL_DEFAULTS[IL_TO_MJ]
                kp_now, kd_now = 20.0, 0.5
            else:
                tq = IL_DEFAULTS + ACTION_SCALE * action
                tq = np.clip(tq, IL_LIMITS[:, 0], IL_LIMITS[:, 1])
                target_q_mj = tq[IL_TO_MJ]
                kp_now, kd_now = 20.0, 0.5

            cq = world.get_qpos(GO2_LEG_JOINTS)
            cdq = world.get_qvel(GO2_LEG_JOINTS)
            tau = kp_now * (target_q_mj - cq) - kd_now * cdq
            tau = np.clip(tau, -IL_TORQUE_LIMITS[IL_TO_MJ], IL_TORQUE_LIMITS[IL_TO_MJ])
            world.set_ctrl(GO2_ACTUATORS, tau)
            world.step(sub)
            steps += 1

            # 诊断 (每 100 RL步 ≈ 2秒)
            if steps % 100 == 0:
                cq_il = cq[MJ_TO_IL]
                tq_il = target_q_mj[MJ_TO_IL]
                print(f"  [{phase}] steps={steps} sim_t={data.time:.1f}s wall_t={elapsed_wall:.1f}s "
                      f"cmd=({cmd_vel[0]:+.2f},{cmd_vel[1]:+.2f},{cmd_vel[2]:+.2f}) "
                      f"feed=({feed_cmd[0]:+.2f},{feed_cmd[1]:+.2f},{feed_cmd[2]:+.2f}) "
                      f"|τ|={np.max(np.abs(tau)):.1f}Nm "
                      f"err_thigh={np.max(np.abs(tq_il[4:8] - cq_il[4:8])):.4f}rad",
                      flush=True)

        # ── 渲染 ──
        _render(window, scn, cam, opt, model, data, ctx)
        glfw.swap_buffers(window)

        time.sleep(0.001)

    glfw.destroy_window(window)
    glfw.terminate()
    print(f"\n  总步数: {steps}, 模拟时间: {data.time:.1f}s")


if __name__ == "__main__":
    main()
