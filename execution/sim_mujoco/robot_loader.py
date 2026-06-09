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
        self._n_actuated = self.model.nu  # 18 for Go2+Piper
        # qpos start index for actuated joints (7 = base freejoint)
        self._qpos_start = 7
        # qvel start index (6 = base freejoint velocity)
        self._qvel_start = 6

        self._step_counter = 0
        self._target_dof_pos = self.default_angles.copy()
        self._algo_arm_target: np.ndarray | None = None  # set by algo grasp thread

    def reset(self) -> None:
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._step_counter = 0
        self._target_dof_pos = self.default_angles.copy()
        self._algo_arm_target = None

    def step(self) -> None:
        """Run one physics step with PD control."""
        q_end = self._qpos_start + self._n_actuated
        v_end = self._qvel_start + self._n_actuated

        tau = pd_control(
            self._target_dof_pos,
            self.data.qpos[self._qpos_start : q_end],
            self.kps,
            np.zeros(self._n_actuated),
            self.data.qvel[self._qvel_start : v_end],
            self.kds,
        )
        self.data.ctrl[:] = tau
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
