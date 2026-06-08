"""RL policy inference for Go2+Piper locomotion.

Loads a TorchScript policy and produces joint position targets from
velocity / position commands.  Integrates with the existing PD controller
in ``robot_loader.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


# IsaacLab joint order → MuJoCo joint order (legs only, indices 0-11)
# Isaac: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
# MuJoCo: FR(3-5), FL(0-2), RR(9-11), RL(6-8)
ISAAC_TO_MUJOCO_LEG = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Inverse: MuJoCo → Isaac
_MUJOCO_TO_ISAAC_LEG = [0] * 12
for _isaac_idx, _mj_idx in enumerate(ISAAC_TO_MUJOCO_LEG):
    _MUJOCO_TO_ISAAC_LEG[_mj_idx] = _isaac_idx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _roll_append(buf: torch.Tensor, new: torch.Tensor, chunk: int) -> torch.Tensor:
    return torch.cat([buf, new], dim=-1)[:, chunk:]


def _gravity_vector(quat_wxyz: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat_wxyz
    return torch.tensor([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ])


class PolicyRunner:
    """TorchScript RL policy wrapper for Go2+Piper."""

    def __init__(
        self,
        policy_path: str | None = None,
        kps: list[float] | None = None,
        kds: list[float] | None = None,
        default_angles: list[float] | None = None,
        action_scale: list[float] | None = None,
        base_ang_vel_scale: float = 0.2,
        joint_vel_scale: float = 0.05,
        num_hist: int = 3,
    ):
        if policy_path is None:
            policy_path = str(_PROJECT_ROOT / "assets" / "policies" / "go2_piper" / "policy.pt")

        self._policy = torch.jit.load(policy_path)
        self._policy.eval()

        self._action_scale = torch.tensor(
            action_scale or [0.25] * 18, dtype=torch.float32
        )
        self._default_angles = torch.tensor(
            default_angles or [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                               0.1, 1.0, -1.5, -0.1, 1.0, -1.5,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
        self._base_ang_vel_scale = base_ang_vel_scale
        self._joint_vel_scale = joint_vel_scale
        self._num_hist = num_hist

        self._action = torch.zeros(1, 18, dtype=torch.float32)

        # History observation buffers
        self._base_ang_vel_obs = torch.zeros(1, 3 * num_hist)
        self._joint_pos_obs = torch.zeros(1, 18 * num_hist)
        self._joint_vel_obs = torch.zeros(1, 18 * num_hist)
        self._actions_obs = torch.zeros(1, 18 * num_hist)
        self._projected_gravity_obs = torch.zeros(1, 3 * num_hist)
        self._vel_command_obs = torch.zeros(1, 3 * num_hist)
        self._pos_command_obs = torch.zeros(1, 7 * num_hist)

        self._target_dof_pos = self._default_angles.clone()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear history buffers (call after simulation reset)."""
        self._action.zero_()
        self._target_dof_pos = self._default_angles.clone()
        for buf_name in (
            "_base_ang_vel_obs", "_joint_pos_obs", "_joint_vel_obs",
            "_actions_obs", "_projected_gravity_obs",
            "_vel_command_obs", "_pos_command_obs",
        ):
            getattr(self, buf_name).zero_()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        qpos_joints: np.ndarray,       # (18,) all actuated joints
        qvel_joints: np.ndarray,       # (18,)
        base_quat_wxyz: np.ndarray,    # (4,)  [qw, qx, qy, qz]
        base_ang_vel: np.ndarray,      # (3,)
        vel_command: np.ndarray,       # (3,)  [vx, vy, vyaw]
        pos_command: np.ndarray,       # (7,)  [px, py, pz, qw, qx, qy, qz]
    ) -> np.ndarray:
        """Run one policy inference step.

        Returns:
            np.ndarray: (18,) target joint positions for PD control.
        """
        qj = torch.from_numpy(qpos_joints.astype(np.float32)).unsqueeze(0)
        dqj = (
            torch.from_numpy(qvel_joints.astype(np.float32)).unsqueeze(0)
            * self._joint_vel_scale
        )
        quat = torch.from_numpy(base_quat_wxyz.astype(np.float32))
        omega = (
            torch.from_numpy(base_ang_vel.astype(np.float32)).unsqueeze(0)
            * self._base_ang_vel_scale
        )
        vel_cmd = torch.from_numpy(vel_command.astype(np.float32)).unsqueeze(0)
        pos_cmd = torch.from_numpy(pos_command.astype(np.float32)).unsqueeze(0)

        qj_rel = qj - self._default_angles
        gravity = _gravity_vector(quat).unsqueeze(0)

        # Remap leg observations: MuJoCo → IsaacLab order
        leg_pos_mj = qj_rel[:, :12]
        leg_vel_mj = dqj[:, :12]
        leg_pos_isaac = leg_pos_mj[:, _MUJOCO_TO_ISAAC_LEG]
        leg_vel_isaac = leg_vel_mj[:, _MUJOCO_TO_ISAAC_LEG]

        # Update history buffers
        self._base_ang_vel_obs = _roll_append(
            self._base_ang_vel_obs, omega, 3
        )
        self._projected_gravity_obs = _roll_append(
            self._projected_gravity_obs, gravity, 3
        )
        self._joint_pos_obs = _roll_append(
            self._joint_pos_obs,
            torch.cat([leg_pos_isaac, qj_rel[:, 12:]], dim=-1),
            18,
        )
        self._joint_vel_obs = _roll_append(
            self._joint_vel_obs,
            torch.cat([leg_vel_isaac, dqj[:, 12:]], dim=-1),
            18,
        )
        self._actions_obs = _roll_append(
            self._actions_obs, self._action, 18
        )
        self._vel_command_obs = _roll_append(
            self._vel_command_obs, vel_cmd, 3
        )
        self._pos_command_obs = _roll_append(
            self._pos_command_obs, pos_cmd, 7
        )

        hist_obs = torch.cat([
            self._base_ang_vel_obs,
            self._projected_gravity_obs,
            self._joint_pos_obs,
            self._joint_vel_obs,
            self._actions_obs,
            self._vel_command_obs,
            self._pos_command_obs,
        ], dim=-1).float().clamp(-100.0, 100.0)

        action = self._policy(hist_obs).clamp(-20.0, 20.0)
        self._action = action

        # Remap leg actions back: IsaacLab → MuJoCo
        leg_action_isaac = action[:, :12]
        arm_action = action[:, 12:]
        leg_action_mj = leg_action_isaac[:, ISAAC_TO_MUJOCO_LEG]
        action_out = torch.cat([leg_action_mj, arm_action], dim=-1)

        self._target_dof_pos = (
            action_out * self._action_scale + self._default_angles
        )
        return self._target_dof_pos.numpy().flatten().astype(np.float64)

    @property
    def target_dof_pos(self) -> np.ndarray:
        """Current target joint positions (18,)."""
        return self._target_dof_pos.numpy().flatten().astype(np.float64)
