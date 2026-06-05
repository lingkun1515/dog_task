#!/usr/bin/env python3
"""MPC 运控键盘控制 + 可视化 Demo — GLFW 渲染 + MPC 力矩控制，单线程.

用法:
  conda activate mower
  python3 scripts/test_mpc_keyboard.py

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

import logging
import time
import numpy as np
import mujoco
from mujoco.glfw import glfw

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from dog_task.modules.sim.scene_builder import SceneBuilder
from dog_task.modules.sim.world import SimWorld
from dog_task.modules.mobility.mpc_controller import MpcController
from dog_task.modules.mobility.go2_mujoco import (
    GO2_LEG_JOINTS, GO2_ACTUATORS, MPC_STAND_ANGLES, MPC_SIT_ANGLES,
)

# ── GLFW 窗口 & 输入 ────────────────────────────────────────────────

_one_shot_keys = set()
_key_held = {}
_mouse_button_left = False
_mouse_button_right = False
_mouse_button_middle = False
_mouse_last_x = 0.0
_mouse_last_y = 0.0

_win_model = None
_win_scn = None
_win_cam = None


def _key_callback(window, key, scancode, action, mods):
    _key_held[key] = (action != glfw.RELEASE)
    if action == glfw.PRESS:
        _one_shot_keys.add(key)


def _mouse_button_callback(window, button, action, mods):
    global _mouse_button_left, _mouse_button_right, _mouse_button_middle
    pressed = (action == glfw.PRESS)
    if button == glfw.MOUSE_BUTTON_LEFT:
        _mouse_button_left = pressed
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        _mouse_button_right = pressed
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        _mouse_button_middle = pressed


def _mouse_move_callback(window, xpos, ypos):
    global _mouse_last_x, _mouse_last_y
    if not (_mouse_button_left or _mouse_button_right or _mouse_button_middle):
        _mouse_last_x, _mouse_last_y = xpos, ypos
        return
    dx = xpos - _mouse_last_x
    dy = ypos - _mouse_last_y
    _mouse_last_x, _mouse_last_y = xpos, ypos
    w, h = glfw.get_framebuffer_size(window)
    model, scn, cam = _win_model, _win_scn, _win_cam
    if _mouse_button_left:
        if abs(dx) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx / w * 2.0, 0.0, scn, cam)
        if abs(dy) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0.0, dy / h * 2.0, scn, cam)
    elif _mouse_button_right:
        if abs(dx) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_H, dx / w * 0.5, 0.0, scn, cam)
        if abs(dy) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_MOVE_V, 0.0, -dy / h * 0.5, scn, cam)
    elif _mouse_button_middle:
        if abs(dy) > 0:
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, dy / h * 2.0, scn, cam)


def _scroll_callback(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(_win_model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                          0.0, -yoffset * 0.05, _win_scn, _win_cam)


def _init_window(model, data):
    global _win_model, _win_scn, _win_cam
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    window = glfw.create_window(1280, 720, "Go2 MPC 键盘控制 — WASD 行走 | 鼠标拖拽视角", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window create failed")
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    glfw.set_key_callback(window, _key_callback)
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

    _win_model = model; _win_scn = scn; _win_cam = cam
    opt = mujoco.MjvOption()
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    return window, scn, cam, opt, ctx


def _render(window, scn, cam, opt, model, data, ctx):
    w, h = glfw.get_framebuffer_size(window)
    vp = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    mujoco.mjr_render(vp, scn, ctx)


def _stand_up(world, target_q, kp=80.0, kd=2.0, max_iter=500):
    """PD 收敛到目标关节角，返回是否步进过."""
    stepped = False
    for _ in range(max_iter):
        cq = world.get_qpos(GO2_LEG_JOINTS)
        cdq = world.get_qvel(GO2_LEG_JOINTS)
        if np.max(np.abs(target_q - cq)) < 0.01:
            break
        tau = kp * (target_q - cq) - kd * cdq
        world.set_ctrl(GO2_ACTUATORS, tau)
        world.step(10)
        stepped = True
    world.forward()
    return stepped


def _stance_hold(world, target_q, hold_sim_s, ctrl_dt, sub, kp=60.0, kd=1.5):
    """姿态稳定期：PD 保持关节角不松手，跑足 hold_sim_s 秒物理."""
    n_steps = int(hold_sim_s / ctrl_dt)
    for i in range(n_steps):
        cq = world.get_qpos(GO2_LEG_JOINTS)
        cdq = world.get_qvel(GO2_LEG_JOINTS)
        tau = kp * (target_q - cq) - kd * cdq
        world.set_ctrl(GO2_ACTUATORS, tau)
        world.step(sub)
    world.forward()


def _get_state(world):
    """读取 freejoint 状态，返回 (x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz)."""
    qpos = world.get_freejoint_qpos("root")
    qvel = world.get_freejoint_qvel("root")
    return qpos, qvel


# ── 主循环 ───────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  MPC 运控键盘控制 Demo (GLFW)")
    print("  按住 W/S: vx   按住 A/D: vyaw   按住 Q/E: vy")
    print("  Space: 停   R: 重新站立   Esc: 退")
    print("  鼠标: 左键旋转 | 右键平移 | 滚轮缩放")
    print("=" * 55)

    # 1. 场景
    print("[1] 构建场景...", end=" ", flush=True)
    config_world = {
        "go2_xml": "assets/go2/go2_mpc.xml",
        "d1_urdf": "assets/d1/d1.urdf",
        "arm_mount_pos": [0.0, 0.0, 0.10],
        "arm_mount_quat": [1, 0, 0, 0],
        "camera_pos": [0.30, 0.0, 0.05],
        "camera_fovy": 58,
        "camera_resolution": [848, 480],
        "environment": {"ground_size": [10, 10]},
    }
    builder = SceneBuilder(config_world)
    model, data = builder.build()
    world = SimWorld(model, data)
    world.reset_to_keyframe("home")
    world.forward()
    print(f"OK (nq={model.nq})")

    # 2. MPC 控制器
    print("[2] 初始化 MPC...", end=" ", flush=True)
    control_hz = 50.0
    ctrl_dt = 1.0 / control_hz
    sub = int(round(ctrl_dt / model.opt.timestep))
    mpc = MpcController(
        control_hz=control_hz, gait_hz=2.0, gait_duty=0.6,
        mpc_horizon=10, z_des=0.27, verbose=False,
    )
    print("OK")

    # 3. 窗口
    print("[3] GLFW 窗口...", end=" ", flush=True)
    window, scn, cam, opt, ctx = _init_window(model, data)
    print("OK")

    # 4. 站立 + 姿态保持
    print("[4] 站立...", end=" ", flush=True)
    _stand_up(world, MPC_STAND_ANGLES)
    print("", flush=True)
    print("[4.5] 姿态稳定保持 (PD, 1.5s 仿真实时)...", end=" ", flush=True)
    _stance_hold(world, MPC_STAND_ANGLES, hold_sim_s=1.5, ctrl_dt=ctrl_dt, sub=sub)
    mpc.reset()
    print("OK")

    # ── 状态变量 ──
    cmd_vel = np.zeros(3)
    last_mpc = data.time - ctrl_dt      # 确保首帧立即进入 MPC 步进
    steps = 0
    t_start = time.time()

    print("\n[5] 主循环 — 在窗口中按住 WASD 控制 Go2\n")

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # ── 单次按键 ──
        for k in list(_one_shot_keys):
            if k == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif k == glfw.KEY_SPACE:
                cmd_vel[:] = 0.0
                print(f"  ⏹ 急停  wall_t={time.time() - t_start:.1f}s  cmd=({cmd_vel[0]:+.2f},{cmd_vel[1]:+.2f},{cmd_vel[2]:+.2f})", flush=True)
            elif k == glfw.KEY_R:
                cmd_vel[:] = 0.0
                _stand_up(world, MPC_STAND_ANGLES)
                _stance_hold(world, MPC_STAND_ANGLES, hold_sim_s=1.0, ctrl_dt=ctrl_dt, sub=sub)
                mpc.reset()
                last_mpc = data.time - ctrl_dt
                print(f"  ↩ 重新站立  wall_t={time.time() - t_start:.1f}s", flush=True)
        _one_shot_keys.clear()

        # ── 持续按住 → 速度渐变 ──
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

        # ── MPC 步进 (50Hz) ──
        now = data.time
        if now - last_mpc >= ctrl_dt - 1e-6:
            last_mpc = now

            qpos, qvel = _get_state(world)
            mj_qpos = np.concatenate([qpos, world.get_qpos(GO2_LEG_JOINTS)])
            mj_qvel = np.concatenate([qvel, world.get_qvel(GO2_LEG_JOINTS)])

            torque = mpc.step(cmd_vel, mj_qpos, mj_qvel)
            world.set_ctrl(GO2_ACTUATORS, torque)
            world.step(sub)
            steps += 1

            # 诊断 (每 100 步 ≈ 2s)
            if steps % 100 == 0:
                p = qpos[:3]
                print(f"  [MPC] steps={steps} sim_t={data.time:.1f}s "
                      f"cmd=({cmd_vel[0]:+.2f},{cmd_vel[1]:+.2f},{cmd_vel[2]:+.2f}) "
                      f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                      f"qp={mpc.solve_time_ms:.1f}ms",
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
