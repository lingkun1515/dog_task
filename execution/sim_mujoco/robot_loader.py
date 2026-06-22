"""MuJoCo robot model loader, PD controller, and state access."""

from __future__ import annotations

import os
from pathlib import Path

# Off-screen rendering requires EGL or OSMesa — set before mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | None = None) -> dict:
    """Load simulation config from TOML (preferred) or YAML (legacy)."""
    if config_path is None:
        config_path = str(_PROJECT_ROOT / "config" / "robots" / "sim_go2_piper.toml")

    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    # Support robot_id shorthand: try config/robots/<id>.toml
    if not path.exists() and path.suffix == "":
        candidate = _PROJECT_ROOT / "config" / "robots" / f"{config_path}.toml"
        if candidate.exists():
            path = candidate

    if path.suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    else:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


def resolve_path(relative_path: str) -> str:
    """Resolve a path relative to the project root."""
    return str(_PROJECT_ROOT / relative_path)


def load_model(xml_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load MuJoCo model and create data."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_base_position(data: mujoco.MjData) -> np.ndarray:
    """Get base (x, y, z) position in world frame."""
    return data.qpos[0:3].copy()


def get_base_yaw(data: mujoco.MjData) -> float:
    """Get base yaw angle from quaternion."""
    qw, qx, qy, qz = data.qpos[3:7]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def move_base(data: mujoco.MjData, dx: float, dy: float, dyaw: float, dt: float) -> None:
    """Directly translate/rotate base in world frame (kinematic sliding mode).

    Modifies ``data.qpos`` for the base freejoint so the robot glides
    without needing leg locomotion.  PD control keeps legs in stance.
    """
    # --- translation ---
    data.qpos[0] += dx * dt
    data.qpos[1] += dy * dt

    # --- yaw rotation (quaternion * rot_z(dyaw*dt)) ---
    half = dyaw * dt * 0.5
    c = np.cos(half)
    s = np.sin(half)
    qw, qx, qy, qz = data.qpos[3:7]
    data.qpos[3] = c * qw - s * qz
    data.qpos[4] = c * qx - s * qy
    data.qpos[5] = c * qy + s * qx
    data.qpos[6] = c * qz + s * qw


class RobotSim:
    """Manages a MuJoCo robot simulation instance."""

    def __init__(self, config_path: str | None = None):
        cfg = load_config(config_path)
        self.cfg = cfg

        xml_path = resolve_path(cfg["xml_path"])
        self.model, self.data = load_model(xml_path)
        self.model.opt.timestep = cfg["simulation_dt"]

        self.kps = np.array(cfg["kps"], dtype=np.float64)
        self.kds = np.array(cfg["kds"], dtype=np.float64)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float64)
        self.control_decimation = cfg["control_decimation"]

        # Number of actuated joints (nq after the base freejoint)
        # nu = 18（12 legs + 6 arm）或 20（L2 物理夹爪 +2 finger motor）
        self._n_actuated = self.model.nu
        # PD 控制的关节数（legs + arm，不含 finger）
        self._n_pd_joints = len(self.default_angles)  # 18
        # qpos start index for actuated joints (7 = base freejoint)
        self._qpos_start = 7
        # qvel start index (6 = base freejoint velocity)
        self._qvel_start = 6

        self._step_counter = 0
        self._target_dof_pos = self.default_angles.copy()
        self._algo_arm_target: np.ndarray | None = None  # set by algo grasp thread

        # L2 物理夹爪：手指 motor 力矩（前 18 个 ctrl 是 PD，后 2 个是 finger）
        self._has_finger_actuators = self.model.nu > self._n_pd_joints
        # 力矩方向：finger_l 负值=外扩（qpos 减小），finger_r 正值=外扩（qpos 增大）
        # 闭合时反向：finger_l 正值（qpos 增大→向内），finger_r 负值（qpos 减小→向内）
        self._finger_open_torque = 1.5     # 张开（l:负方向, r:正方向）
        self._finger_close_torque = -2.5   # 闭合

        # 手指 slide joint qpos 地址（用于初始化为张开位置）
        if self._has_finger_actuators:
            self._finger_l_qpos_addr = int(
                self.model.jnt_qposadr[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "d1_finger_l")]
            )
            self._finger_r_qpos_addr = int(
                self.model.jnt_qposadr[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, "d1_finger_r")]
            )
            # 初始化为张开（range 端点）
            self._init_finger_open()
        else:
            self._finger_l_qpos_addr = -1
            self._finger_r_qpos_addr = -1

        # ---- Sim gripper coupling ----
        # L1 改造：用 MuJoCo weld equality 替换硬 qpos 绑定。
        # 夹爪闭合时激活 weld（球通过物理约束跟随 TCP，松手会掉落）。
        self._gripper_closed: bool = False
        self._grasp_target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_sphere"
        )
        self._tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "d1_tcp"
        )
        if self._grasp_target_body_id >= 0:
            _jnt_id = self.model.body_jntadr[self._grasp_target_body_id]
            self._ball_qpos_addr: int = int(self.model.jnt_qposadr[_jnt_id])
            self._ball_qvel_addr: int = int(self.model.jnt_dofadr[_jnt_id])
        else:
            self._ball_qpos_addr = -1
            self._ball_qvel_addr = -1

        # weld equality 索引（L1 软约束）
        self._grasp_weld_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld"
        )

    def _init_finger_open(self) -> None:
        """把手指 slide joint 初始化到张开位置（range 外扩端）。"""
        if self._finger_l_qpos_addr < 0:
            return
        # finger_l range=[-0.034, 0]，张开=qpos=-0.034（最负）
        self.data.qpos[self._finger_l_qpos_addr] = -0.034
        # finger_r range=[0, 0.034]，张开=qpos=0.034（最正）
        self.data.qpos[self._finger_r_qpos_addr] = 0.034
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._step_counter = 0
        self._target_dof_pos = self.default_angles.copy()
        self._algo_arm_target = None
        self._gripper_closed = False
        # 手指初始化为张开
        if self._has_finger_actuators:
            self._init_finger_open()

    def step(self) -> None:
        """Run one physics step with PD control for legs+arm, torque for fingers."""
        # PD 控制：legs(12) + arm(6) = 18 关节
        q_end = self._qpos_start + self._n_pd_joints
        v_end = self._qvel_start + self._n_pd_joints
        tau_pd = pd_control(
            self._target_dof_pos[:self._n_pd_joints],
            self.data.qpos[self._qpos_start : q_end],
            self.kps[:self._n_pd_joints],
            np.zeros(self._n_pd_joints),
            self.data.qvel[self._qvel_start : v_end],
            self.kds[:self._n_pd_joints],
        )
        # 写入 ctrl：前 18 个是 PD tau
        self.data.ctrl[:self._n_pd_joints] = tau_pd
        # 手指 motor（L2 物理）：按 _gripper_closed 状态给力矩
        # 张开：finger_l 负力矩（qpos→-0.034），finger_r 正力矩（qpos→+0.034）
        # 闭合：finger_l 正力矩（qpos→0），finger_r 负力矩（qpos→0）
        if self._has_finger_actuators:
            if self._gripper_closed:
                self.data.ctrl[self._n_pd_joints] = -self._finger_close_torque      # l: 正→向内
                self.data.ctrl[self._n_pd_joints + 1] = self._finger_close_torque   # r: 负→向内
            else:
                self.data.ctrl[self._n_pd_joints] = -self._finger_open_torque       # l: 负→外扩
                self.data.ctrl[self._n_pd_joints + 1] = self._finger_open_torque    # r: 正→外扩
        mujoco.mj_step(self.model, self.data)
        self._step_counter += 1

    def set_arm_target(self, angles: np.ndarray) -> None:
        """Set target joint angles for the arm (last 6 of 18 DOF)."""
        self._target_dof_pos[12:] = angles

    def get_arm_positions(self) -> np.ndarray:
        """Get current arm joint positions (last 6 of 18 actuated joints)."""
        arm_start = self._qpos_start + 12  # 7 + 12 = 19
        arm_end = arm_start + 6
        return self.data.qpos[arm_start:arm_end].copy()

    def get_body_position(self, body_name: str) -> np.ndarray:
        """Get a body's (x, y, z) position in world frame."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        return self.data.xpos[body_id].copy()

    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get a body's (xpos, xmat) in world frame. xmat is 3x3."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")
        return (
            self.data.xpos[body_id].copy(),
            self.data.xmat[body_id].copy().reshape(3, 3),
        )

    @property
    def base_position(self) -> np.ndarray:
        return get_base_position(self.data)

    @property
    def base_yaw(self) -> float:
        return get_base_yaw(self.data)

    @property
    def joint_positions(self) -> np.ndarray:
        """All actuated joint positions (18 DOF: 12 legs + 6 arm)."""
        q_end = self._qpos_start + self._n_actuated
        return self.data.qpos[self._qpos_start : q_end].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        v_end = self._qvel_start + self._n_actuated
        return self.data.qvel[self._qvel_start : v_end].copy()
    def set_gripper(self, angle: float) -> None:
        """控制夹爪开合。angle<=10 视为闭合。

        L2 物理夹爪：闭合时手指 motor 施加内向力矩，靠摩擦接触夹球；
        张开时手指外扩，球自由下落。
        L1 weld 兼容：若无 finger actuator（旧模型），回退到 weld equality。
        """
        self._gripper_closed = (angle <= 10.0)
        # L2 物理夹爪：手指力矩由 step() 根据 _gripper_closed 自动施加
        if self._has_finger_actuators:
            return  # 物理夹爪，无需 weld
        # L1 回退：weld equality
        self._update_grasp_weld()

    def _update_grasp_weld(self) -> None:
        """根据 _gripper_closed 状态激活/停用 weld equality。"""
        if self._grasp_weld_id < 0 or self._grasp_target_body_id < 0:
            return  # 无 weld 定义或无目标体，回退到无约束
        if self._gripper_closed:
            # 激活 weld 前，先把 weld 的 relpose 设为「当前球相对 TCP 的位姿」，
            # 否则 weld 会把球硬拉到 anchor 默认位置（跳跃）。
            self._set_weld_relpose_to_current()
            self.data.eq_active[self._grasp_weld_id] = 1
        else:
            self.data.eq_active[self._grasp_weld_id] = 0

    def _set_weld_relpose_to_current(self) -> None:
        """把 weld 的目标相对位姿设为「当前球相对 d1_link6 的位姿」。

        MuJoCo weld eq_data 布局：
          [0:3]   = anchor（body2 局部坐标的锚点，body1 的对应点要对齐到这里）
          [3]     = torquescale
          [4:8]   = relquat（body1 相对 body2 的期望姿态，w,x,y,z 顺序？实际 [3:7]）
          [7:10]  = relpos（body1 相对 body2 的期望位置，body2 局部）
          [10]    = torquescale2 (unused)

        实测有效布局：eq_data[0:3]=anchor, eq_data[3:7]=relquat(0,0,0,1),
        eq_data[7:10]=relpos。
        """
        import mujoco as _mj
        link6_id = _mj.mj_name2id(self.model, _mj.mjtObj.mjOBJ_BODY, "d1_link6")
        if link6_id < 0:
            return
        # 球当前世界位姿
        ball_pos = self.data.xpos[self._grasp_target_body_id].copy()
        # d1_link6 世界位姿
        link6_pos = self.data.xpos[link6_id].copy()
        link6_mat = self.data.xmat[link6_id].reshape(3, 3).copy()
        # 球相对 link6 的位置（link6 局部）
        rel_pos = link6_mat.T @ (ball_pos - link6_pos)

        wid = self._grasp_weld_id
        self.model.eq_data[wid][0:3] = [0.0, 0.0, 0.0]      # anchor = link6 原点
        self.model.eq_data[wid][3:7] = [0.0, 0.0, 0.0, 1.0]  # relquat identity
        self.model.eq_data[wid][7:10] = rel_pos               # relpos = 球相对 link6

    def apply_gripper_constraint(self) -> None:
        """兼容旧接口：L1 改造后 weld 由 MuJoCo solver 自动求解，本方法为空操作。

        保留是为了不破坏 scene.py 的 _loop 调用链。实际约束由 weld equality
        在 mj_step 中自动施加。
        """
        return
