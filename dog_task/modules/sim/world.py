"""SimWorld: 线程安全的 MuJoCo 物理世界单例.

拥有 mjModel/mjData，提供后台物理步进线程和线程安全的读写接口。
Camera 模块通过 read_lock 获取一致性快照做离屏渲染；
Mobility/Arm 模块通过 set_ctrl/set_qpos 写入控制指令。
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np


class SimWorld:
    """统一 MuJoCo 物理世界."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self._lock = threading.RLock()
        self._stepping = False
        self._step_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._joint_id_cache: Dict[str, int] = {}
        self._actuator_id_cache: Dict[str, int] = {}
        self._body_id_cache: Dict[str, int] = {}
        self._sensor_cache: Dict[str, Tuple[int, int]] = {}
        self._cam_id_cache: Dict[str, int] = {}

    @contextmanager
    def read_lock(self):
        """上下文管理器，获取锁用于读取 mjData 状态（如相机渲染）."""
        with self._lock:
            yield

    def step(self, n: int = 1) -> None:
        """前进 n 个物理步."""
        with self._lock:
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)

    def start_stepping(self, hz: float = 500.0) -> None:
        """启动后台物理步进线程."""
        if self._stepping:
            return
        self._stepping = True
        self._stop_event.clear()
        dt = self.model.opt.timestep
        steps_per_tick = max(1, int(round(1.0 / (hz * dt))))
        sleep_s = steps_per_tick * dt

        def _loop():
            while not self._stop_event.is_set():
                with self._lock:
                    for _ in range(steps_per_tick):
                        mujoco.mj_step(self.model, self.data)
                time.sleep(sleep_s * 0.8)

        self._step_thread = threading.Thread(target=_loop, daemon=True, name="sim_physics")
        self._step_thread.start()

    def stop_stepping(self) -> None:
        """停止后台物理步进线程."""
        if not self._stepping:
            return
        self._stop_event.set()
        if self._step_thread is not None:
            self._step_thread.join(timeout=2.0)
        self._stepping = False
        self._step_thread = None

    @property
    def is_stepping(self) -> bool:
        return self._stepping

    @property
    def time(self) -> float:
        return self.data.time

    # ---- 名称 → ID 映射（带缓存）----

    def joint_id(self, name: str) -> int:
        if name not in self._joint_id_cache:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise KeyError(f"joint '{name}' not found in model")
            self._joint_id_cache[name] = jid
        return self._joint_id_cache[name]

    def actuator_id(self, name: str) -> int:
        if name not in self._actuator_id_cache:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                raise KeyError(f"actuator '{name}' not found in model")
            self._actuator_id_cache[name] = aid
        return self._actuator_id_cache[name]

    def body_id(self, name: str) -> int:
        if name not in self._body_id_cache:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                raise KeyError(f"body '{name}' not found in model")
            self._body_id_cache[name] = bid
        return self._body_id_cache[name]

    def camera_id(self, name: str) -> int:
        if name not in self._cam_id_cache:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid == -1:
                raise KeyError(f"camera '{name}' not found in model")
            self._cam_id_cache[name] = cid
        return self._cam_id_cache[name]

    def _sensor_adr(self, name: str) -> Tuple[int, int]:
        if name not in self._sensor_cache:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid == -1:
                raise KeyError(f"sensor '{name}' not found in model")
            adr = self.model.sensor_adr[sid]
            dim = self.model.sensor_dim[sid]
            self._sensor_cache[name] = (adr, dim)
        return self._sensor_cache[name]

    # ---- 传感器/关节/body 读写 ----

    def get_sensor(self, name: str) -> np.ndarray:
        """读取传感器值."""
        adr, dim = self._sensor_adr(name)
        return self.data.sensordata[adr:adr + dim].copy()

    def get_qpos_by_joint(self, name: str) -> float:
        """读取单个 hinge 关节位置."""
        jid = self.joint_id(name)
        adr = self.model.jnt_qposadr[jid]
        return float(self.data.qpos[adr])

    def get_qpos(self, names: Sequence[str]) -> np.ndarray:
        """批量读取关节位置（仅 hinge 关节）."""
        result = np.empty(len(names))
        for i, name in enumerate(names):
            jid = self.joint_id(name)
            adr = self.model.jnt_qposadr[jid]
            result[i] = self.data.qpos[adr]
        return result

    def set_qpos(self, names: Sequence[str], values: np.ndarray) -> None:
        """批量设置关节位置（仅 hinge 关节）."""
        with self._lock:
            for i, name in enumerate(names):
                jid = self.joint_id(name)
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr] = values[i]

    def get_qvel(self, names: Sequence[str]) -> np.ndarray:
        """批量读取关节速度."""
        result = np.empty(len(names))
        for i, name in enumerate(names):
            jid = self.joint_id(name)
            adr = self.model.jnt_dofadr[jid]
            result[i] = self.data.qvel[adr]
        return result

    def set_ctrl(self, names: Sequence[str], values: np.ndarray) -> None:
        """按名称设置 actuator 控制信号."""
        with self._lock:
            for i, name in enumerate(names):
                aid = self.actuator_id(name)
                self.data.ctrl[aid] = values[i]

    def get_body_xpos(self, name: str) -> np.ndarray:
        """读取 body 世界位置."""
        bid = self.body_id(name)
        return self.data.xpos[bid].copy()

    def get_body_xquat(self, name: str) -> np.ndarray:
        """读取 body 世界四元数."""
        bid = self.body_id(name)
        return self.data.xquat[bid].copy()

    def get_body_xmat(self, name: str) -> np.ndarray:
        """读取 body 世界旋转矩阵 (3x3)."""
        bid = self.body_id(name)
        return self.data.xmat[bid].reshape(3, 3).copy()

    def set_mocap_pos(self, name: str, pos: np.ndarray) -> None:
        """设置 mocap body 位置."""
        with self._lock:
            bid = self.body_id(name)
            mocap_id = self.model.body_mocapid[bid]
            if mocap_id == -1:
                raise ValueError(f"body '{name}' is not a mocap body")
            self.data.mocap_pos[mocap_id] = pos

    def set_mocap_quat(self, name: str, quat: np.ndarray) -> None:
        """设置 mocap body 四元数."""
        with self._lock:
            bid = self.body_id(name)
            mocap_id = self.model.body_mocapid[bid]
            if mocap_id == -1:
                raise ValueError(f"body '{name}' is not a mocap body")
            self.data.mocap_quat[mocap_id] = quat

    def reset_to_keyframe(self, name: str = "home") -> None:
        """重置到指定 keyframe."""
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid == -1:
            raise KeyError(f"keyframe '{name}' not found")
        with self._lock:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)

    def forward(self) -> None:
        """调用 mj_forward 更新运动学（不做物理步进）."""
        with self._lock:
            mujoco.mj_forward(self.model, self.data)

    # ---- freejoint 直接操作（kinematic 底盘模式用）----

    def get_freejoint_qpos(self, joint_name: str = "root") -> np.ndarray:
        """获取 freejoint 的 7D qpos [x,y,z,qw,qx,qy,qz]."""
        jid = self.joint_id(joint_name)
        adr = self.model.jnt_qposadr[jid]
        return self.data.qpos[adr:adr + 7].copy()

    def set_freejoint_qpos(self, qpos7: np.ndarray, joint_name: str = "root") -> None:
        """设置 freejoint 的 7D qpos."""
        with self._lock:
            jid = self.joint_id(joint_name)
            adr = self.model.jnt_qposadr[jid]
            self.data.qpos[adr:adr + 7] = qpos7

    def get_freejoint_qvel(self, joint_name: str = "root") -> np.ndarray:
        """获取 freejoint 的 6D qvel [vx,vy,vz,wx,wy,wz]."""
        jid = self.joint_id(joint_name)
        adr = self.model.jnt_dofadr[jid]
        return self.data.qvel[adr:adr + 6].copy()

    def set_freejoint_qvel(self, qvel6: np.ndarray, joint_name: str = "root") -> None:
        """设置 freejoint 的 6D qvel."""
        with self._lock:
            jid = self.joint_id(joint_name)
            adr = self.model.jnt_dofadr[jid]
            self.data.qvel[adr:adr + 6] = qvel6
