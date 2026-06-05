"""
D1 机械臂运动学模块: URDF 解析 + 正运动学(FK) + 数值逆解(IK).

设计要点:
- 直接从 URDF 读取关节链, 与硬件模型保持一致, 切换型号只需换 URDF 路径;
- 仅把 6 个 revolute 关节作为可解变量, 夹爪(prismatic)单独控制;
- IK 用 scipy 的 least_squares(信赖域 + 关节限位), 支持纯位置或位置+姿态;
- 输出统一为"DDS 角度(度)", 通过 JointMapping 处理 URDF 关节值与
  下发角度之间的符号/零位差异(默认恒等, 实测后可校正).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares


# --------------------------- 基础数学工具 ---------------------------

def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy(固定轴 XYZ)转旋转矩阵: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_angle_to_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """绕单位轴 axis 旋转 theta 的旋转矩阵(罗德里格斯公式)."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = a
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def make_transform(rot: np.ndarray, trans: Sequence[float]) -> np.ndarray:
    """由 3x3 旋转和平移组装 4x4 齐次矩阵."""
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = trans
    return t


def matrix_to_rotvec(rot: np.ndarray) -> np.ndarray:
    """旋转矩阵转旋转向量(轴*角), 用于姿态误差度量."""
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    if abs(angle - math.pi) < 1e-6:
        # 接近 pi 时数值不稳, 取对称矩阵主特征向量
        d = np.diag(rot)
        k = int(np.argmax(d))
        axis = rot[:, k].copy()
        axis[k] += 1.0
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * angle
    rv = np.array([rot[2, 1] - rot[1, 2],
                   rot[0, 2] - rot[2, 0],
                   rot[1, 0] - rot[0, 1]])
    return rv / (2.0 * math.sin(angle)) * angle


# --------------------------- URDF 解析 ---------------------------

@dataclass
class UrdfJoint:
    name: str
    jtype: str            # revolute / prismatic / fixed
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _parse_floats(text: Optional[str], default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def parse_urdf_joints(urdf_path: str) -> List[UrdfJoint]:
    """读取 URDF 中全部 joint."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints: List[UrdfJoint] = []
    for j in root.findall("joint"):
        origin = j.find("origin")
        axis_el = j.find("axis")
        limit = j.find("limit")
        joints.append(UrdfJoint(
            name=j.get("name"),
            jtype=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            origin_xyz=_parse_floats(origin.get("xyz") if origin is not None else None, [0, 0, 0]),
            origin_rpy=_parse_floats(origin.get("rpy") if origin is not None else None, [0, 0, 0]),
            axis=_parse_floats(axis_el.get("xyz") if axis_el is not None else None, [0, 0, 1]),
            lower=float(limit.get("lower")) if (limit is not None and limit.get("lower")) else -math.pi,
            upper=float(limit.get("upper")) if (limit is not None and limit.get("upper")) else math.pi,
        ))
    return joints


# --------------------------- 关节映射(URDF <-> DDS) ---------------------------

@dataclass
class JointMapping:
    """URDF 关节值(弧度)与下发角度(度)之间的映射.

    下发角度(度) = sign * degrees(urdf_q) + offset_deg
    默认恒等映射. 若实测发现某关节方向相反或零位不同, 改这里即可.
    """
    sign: np.ndarray = field(default_factory=lambda: np.ones(6))
    offset_deg: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def urdf_to_dds_deg(self, q_rad: np.ndarray) -> np.ndarray:
        return self.sign * np.degrees(q_rad) + self.offset_deg

    def dds_deg_to_urdf(self, dds_deg: np.ndarray) -> np.ndarray:
        return np.radians((dds_deg - self.offset_deg) / self.sign)


# --------------------------- 运动学链 ---------------------------

class D1Chain:
    """D1 的 6 自由度运动学链(末端到夹爪 TCP)."""

    def __init__(self, urdf_path: str, tcp_offset: Sequence[float] = (0.12, 0.0, 0.0),
                 tool_axis: int = 0, mapping: Optional[JointMapping] = None):
        """
        tcp_offset: 在法兰(最后一个 revolute 关节子坐标系)下, 从法兰到夹爪夹持中心的偏移(米).
                    实测确认: D1 夹爪沿法兰 X 轴伸出, 故默认 (0.12,0,0), 需按实物标定长度.
        tool_axis:  夹爪"接近方向/指向"对应的法兰局部轴(0=X,1=Y,2=Z). 实测 D1 为 X 轴(0).
        """
        self.urdf_path = urdf_path
        self.all_joints = parse_urdf_joints(urdf_path)
        self.revolute = self._build_revolute_chain()
        if len(self.revolute) != 6:
            raise ValueError(f"期望 6 个 revolute 关节, 实际解析到 {len(self.revolute)} 个")
        self.tcp_offset = np.array(tcp_offset, dtype=float)
        self.tool_axis = int(tool_axis)
        self.mapping = mapping or JointMapping()
        self.lower = np.array([j.lower for j in self.revolute])
        self.upper = np.array([j.upper for j in self.revolute])

    def _build_revolute_chain(self) -> List[UrdfJoint]:
        """从 base_link 出发, 沿 revolute 关节逐级向下, 收集 6 个臂关节."""
        by_parent = {}
        for j in self.all_joints:
            by_parent.setdefault(j.parent, []).append(j)
        # 找到根链接(只作 parent, 不作任何 joint 的 child)
        children = {j.child for j in self.all_joints}
        roots = [j.parent for j in self.all_joints if j.parent not in children]
        cur = roots[0] if roots else "base_link"
        chain: List[UrdfJoint] = []
        while True:
            nxt = [j for j in by_parent.get(cur, []) if j.jtype == "revolute"]
            if not nxt:
                break
            chain.append(nxt[0])
            cur = nxt[0].child
        return chain

    def fk(self, q_rad: np.ndarray, to_tcp: bool = True) -> np.ndarray:
        """正运动学: 输入 6 个关节弧度, 返回 TCP 的 4x4 位姿."""
        t = np.eye(4)
        for i, j in enumerate(self.revolute):
            local = make_transform(rpy_to_matrix(*j.origin_rpy), j.origin_xyz)
            rot = axis_angle_to_matrix(j.axis, q_rad[i])
            t = t @ local @ make_transform(rot, [0, 0, 0])
        if to_tcp:
            t = t @ make_transform(np.eye(3), self.tcp_offset)
        return t

    def fk_position(self, q_rad: np.ndarray) -> np.ndarray:
        return self.fk(q_rad)[:3, 3]

    # ------------------- 逆解 -------------------

    def ik(self,
           target_pos: Sequence[float],
           target_rpy: Optional[Sequence[float]] = None,
           approach_dir: Optional[Sequence[float]] = None,
           seed: Optional[np.ndarray] = None,
           ori_weight: float = 1.0,
           n_restarts: int = 12,
           pos_tol: float = 1e-3,
           ori_tol: float = 1e-2) -> Tuple[Optional[np.ndarray], dict]:
        """数值逆解.

        target_pos:   目标位置 [x,y,z] (米, base_link 坐标系).
        target_rpy:   可选完整目标姿态 [roll,pitch,yaw] (弧度), 约束 6 自由度.
        approach_dir: 可选夹爪接近方向(世界系单位向量, 工具 z 轴指向),
                      只约束接近方向, 绕夹爪轴的转角自由(更适合抓取, 更易可达).
                      与 target_rpy 二选一; 都为 None 时只解位置.
        返回: (q_rad 或 None, info字典). q_rad 为 6 个关节弧度.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = rpy_to_matrix(*target_rpy) if target_rpy is not None else None
        appr = None
        if approach_dir is not None:
            appr = np.asarray(approach_dir, dtype=float)
            appr = appr / (np.linalg.norm(appr) + 1e-12)

        def residual(q):
            t = self.fk(q)
            err_p = t[:3, 3] - target_pos
            if appr is not None:
                err_z = (t[:3, self.tool_axis] - appr) * ori_weight
                return np.concatenate([err_p, err_z])
            if target_rot is None:
                return err_p
            err_r = matrix_to_rotvec(target_rot @ t[:3, :3].T) * ori_weight
            return np.concatenate([err_p, err_r])

        rng = np.random.default_rng(0)
        mid = (self.lower + self.upper) / 2.0
        best_q, best_cost = None, np.inf
        info = {"attempts": 0, "pos_err": None, "ori_err": None}

        for k in range(max(1, n_restarts)):
            if k == 0 and seed is not None:
                x0 = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
            elif k == 0:
                x0 = mid.copy()
            else:
                x0 = rng.uniform(self.lower, self.upper)
            info["attempts"] += 1
            sol = least_squares(residual, x0, bounds=(self.lower, self.upper),
                                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
            cost = float(np.sum(sol.fun ** 2))
            if cost < best_cost:
                best_cost, best_q = cost, sol.x
            # 满足精度即提前结束
            t = self.fk(sol.x)
            pe = np.linalg.norm(t[:3, 3] - target_pos)
            oe = self._ori_error(t, target_rot, appr, self.tool_axis)
            if pe <= pos_tol and oe <= ori_tol:
                best_q = sol.x
                info.update(pos_err=pe, ori_err=oe)
                return best_q, info

        # 未达精度, 返回最优近似并标注误差
        if best_q is not None:
            t = self.fk(best_q)
            info["pos_err"] = float(np.linalg.norm(t[:3, 3] - target_pos))
            info["ori_err"] = self._ori_error(t, target_rot, appr, self.tool_axis)
        return best_q, info

    @staticmethod
    def _ori_error(t: np.ndarray, target_rot, appr, tool_axis: int = 0) -> float:
        """姿态误差: 接近向量模式用工具轴夹角, 完整姿态模式用旋转向量模长."""
        if appr is not None:
            cosang = float(np.clip(np.dot(t[:3, tool_axis], appr), -1.0, 1.0))
            return math.acos(cosang)
        if target_rot is None:
            return 0.0
        return float(np.linalg.norm(matrix_to_rotvec(target_rot @ t[:3, :3].T)))

    def in_limits(self, q_rad: np.ndarray, margin: float = 0.0) -> bool:
        return bool(np.all(q_rad >= self.lower + margin) and np.all(q_rad <= self.upper - margin))

    def to_dds_deg(self, q_rad: np.ndarray) -> np.ndarray:
        """6 个关节弧度 -> DDS 下发角度(度, 对应 angle0~angle5)."""
        return self.mapping.urdf_to_dds_deg(q_rad)
