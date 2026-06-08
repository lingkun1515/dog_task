"""D1 机械臂运动学（从 pick_up_trash/kinematics.py 移植）。

基于 URDF 参数构建运动学链，使用数值 Jacobian 伪逆做 IK。
"""

import numpy as np


# ============================================================
# 基础旋转矩阵
# ============================================================

def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy_to_matrix(roll, pitch, yaw):
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _make_transform(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def _joint_transform(xyz, rpy, axis, angle_rad):
    T_origin = _make_transform(xyz, rpy)
    ax = np.array(axis)
    if np.allclose(ax, [0, 0, 1]):
        R_joint = _rot_z(angle_rad)
    elif np.allclose(ax, [0, 0, -1]):
        R_joint = _rot_z(-angle_rad)
    elif np.allclose(ax, [0, 1, 0]):
        R_joint = _rot_y(angle_rad)
    elif np.allclose(ax, [0, -1, 0]):
        R_joint = _rot_y(-angle_rad)
    elif np.allclose(ax, [1, 0, 0]):
        R_joint = _rot_x(angle_rad)
    elif np.allclose(ax, [-1, 0, 0]):
        R_joint = _rot_x(-angle_rad)
    else:
        ax = ax / np.linalg.norm(ax)
        K = np.array([
            [0, -ax[2], ax[1]],
            [ax[2], 0, -ax[0]],
            [-ax[1], ax[0], 0],
        ])
        R_joint = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
    T_joint = np.eye(4)
    T_joint[:3, :3] = R_joint
    return T_origin @ T_joint


# ============================================================
# D1 URDF 参数
# ============================================================

JOINTS = [
    {"name": "Joint1", "xyz": [0, 0, 0.0533], "rpy": [0, 0, -3.1416], "axis": [0, 0, 1]},
    {"name": "Joint2", "xyz": [0, 0.028, 0.0563], "rpy": [1.5708, 0, -3.1416], "axis": [0, 0, -1]},
    {"name": "Joint3", "xyz": [0, 0.2693, 0.0009], "rpy": [0, 0, 0], "axis": [0, 0, -1]},
    {"name": "Joint4", "xyz": [0.0577, 0.042, -0.0275], "rpy": [-1.5708, 0, -1.5708], "axis": [0, 0, 1]},
    {"name": "Joint5", "xyz": [-0.0001, -0.0237, 0.14018], "rpy": [1.5708, -1.5708, 0], "axis": [0, 0, -1]},
    {"name": "Joint6", "xyz": [0.0825, -0.0010782, -0.023822], "rpy": [-1.5708, 0, -1.5708], "axis": [0, 0, -1]},
]

GRIPPER_TCP_OFFSET = _make_transform(
    xyz=[-0.00562, 0.0, 0.0706],
    rpy=[-1.5708, -1.5708, 0],
)

JOINT_LIMITS_RAD = [
    (-2.35, 2.35),
    (-1.57, 1.57),
    (-1.57, 1.57),
    (-2.35, 2.35),
    (-1.57, 1.57),
    (-2.35, 2.35),
]


def _numerical_jacobian(angles_rad, get_pos_fn, delta=1e-5):
    J = np.zeros((3, 6))
    angles_deg = np.rad2deg(angles_rad)
    pos0 = np.array(get_pos_fn(angles_deg))
    for j in range(6):
        q = angles_rad.copy()
        q[j] += delta
        pos1 = np.array(get_pos_fn(np.rad2deg(q)))
        J[:, j] = (pos1 - pos0) / delta
    return J


# ============================================================
# D1Kinematics 类
# ============================================================

class D1Kinematics:
    """D1 机械臂运动学（URDF 参数 + 数值 Jacobian）。

    实现 ArmKinematics 的鸭子类型接口（不继承，避免耦合）。
    """

    @staticmethod
    def forward_kinematics(joint_angles_deg):
        angles_rad = np.deg2rad(joint_angles_deg[:6])
        T = np.eye(4)
        for i, joint in enumerate(JOINTS):
            T = T @ _joint_transform(
                xyz=joint["xyz"], rpy=joint["rpy"],
                axis=joint["axis"], angle_rad=angles_rad[i],
            )
        return T @ GRIPPER_TCP_OFFSET

    @staticmethod
    def get_position(joint_angles_deg):
        T = D1Kinematics.forward_kinematics(joint_angles_deg)
        return tuple(T[:3, 3])

    @staticmethod
    def get_link_positions(joint_angles_deg):
        angles_rad = np.deg2rad(joint_angles_deg[:6])
        positions = [(0, 0, 0)]
        T = np.eye(4)
        for i, joint in enumerate(JOINTS):
            T = T @ _joint_transform(
                xyz=joint["xyz"], rpy=joint["rpy"],
                axis=joint["axis"], angle_rad=angles_rad[i],
            )
            positions.append(tuple(T[:3, 3]))
        T = T @ GRIPPER_TCP_OFFSET
        positions.append(tuple(T[:3, 3]))
        return positions

    @staticmethod
    def inverse_kinematics(target_xyz, initial_angles_deg=None, max_iter=200,
                           tol=0.001, alpha=0.5):
        target = np.array(target_xyz)
        if initial_angles_deg is not None:
            q = np.deg2rad(np.array(initial_angles_deg[:6], dtype=float))
        else:
            q = np.zeros(6)

        for _ in range(max_iter):
            current_pos = np.array(D1Kinematics.get_position(np.rad2deg(q)))
            error = target - current_pos
            if np.linalg.norm(error) < tol:
                return np.rad2deg(q).tolist()

            J = _numerical_jacobian(q, D1Kinematics.get_position)
            damping = 0.01
            JTJ = J.T @ J + damping * np.eye(6)
            dq = np.linalg.solve(JTJ, J.T @ error) * alpha
            q = q + dq
            for i in range(6):
                lo, hi = JOINT_LIMITS_RAD[i]
                q[i] = np.clip(q[i], lo, hi)

        final_pos = np.array(D1Kinematics.get_position(np.rad2deg(q)))
        if np.linalg.norm(target - final_pos) < tol * 5:
            return np.rad2deg(q).tolist()
        return None
