"""机械臂执行器抽象 + Sim/Real 实现。"""

from abc import ABC, abstractmethod

import numpy as np


class ArmExecutor(ABC):
    """机械臂执行器抽象接口。"""

    @abstractmethod
    def move_to_joints(self, angles: list[float], mode: int = 1, wait_time: float = 2.0) -> None:
        """移动到目标关节角度并等待。"""
        ...

    @abstractmethod
    def set_gripper(self, angle: float) -> None:
        """控制夹爪开合（0=闭合, 65=张开）。"""
        ...

    @abstractmethod
    def get_current_joints(self) -> list[float]:
        """获取当前关节角度（度）。"""
        ...

    @abstractmethod
    def wait_until_reached(
        self, target: list[float], threshold: float = 3.0, timeout: float = 20.0
    ) -> bool:
        """等待机械臂到达目标角度。"""
        ...


class SimArmExecutor(ArmExecutor):
    """仿真执行器：通过 RobotSim 的 PD 控制驱动 MuJoCo arm。

    不直接等待（仿真在 step 循环中推进），move_to_joints 仅设 target。
    等待由仿真步进自然完成。
    """

    def __init__(self, robot):
        """robot: RobotSim 实例。"""
        self._robot = robot
        self._arm_qpos_slice = slice(19, 25)  # qpos[19:25] = 6 arm joints

    def move_to_joints(self, angles: list[float], mode: int = 1, wait_time: float = 2.0) -> None:
        # IK 返回的是度，MuJoCo PD 控制需要弧度
        self._robot._algo_arm_target = np.deg2rad(np.array(angles[:6], dtype=np.float64))
        # 给仿真 PD 控制器时间推进（step 循环在另一个线程）
        import time
        time.sleep(wait_time)

    def set_gripper(self, angle: float) -> None:
        # Sim gripper: angle=0 means closed, angle=65 means open.
        # When closed, signal RobotSim to force ball to TCP position.
        self._robot._gripper_closed = (angle <= 10.0)

    def get_current_joints(self) -> list[float]:
        q = self._robot.data.qpos[self._arm_qpos_slice]
        return np.rad2deg(q).tolist()

    def wait_until_reached(
        self, target: list[float], threshold: float = 3.0, timeout: float = 20.0
    ) -> bool:
        # 仿真中 PD 快速收敛，短暂等待后检查
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            cur = np.array(self.get_current_joints())
            tgt = np.array(target[:6])
            if float(np.max(np.abs(cur - tgt))) < threshold:
                return True
            time.sleep(0.05)
        return False


class RealArmExecutor(ArmExecutor):
    """实机执行器：封装 D1 ArmController HTTP 客户端。

    依赖 pick_up_trash/arm_control.py 的网络协议（:8088）。
    """

    def __init__(self, host: str = "192.168.123.100", port: int = 8088):
        from algorithms.kinematics.real_d1_ik import D1Kinematics
        self._kinematics = D1Kinematics

        import requests
        self._requests = requests
        self._base_url = f"http://{host}:{port}"
        self._timeout = 5

    def move_to_joints(self, angles: list[float], mode: int = 1, wait_time: float = 2.0) -> None:
        try:
            self._requests.post(
                f"{self._base_url}/api/joints",
                json={"angles": list(angles[:6]), "mode": mode},
                timeout=self._timeout,
            )
        except Exception:
            pass
        import time
        time.sleep(wait_time)

    def set_gripper(self, angle: float) -> None:
        try:
            self._requests.post(
                f"{self._base_url}/api/gripper",
                json={"angle": float(angle)},
                timeout=self._timeout,
            )
        except Exception:
            pass

    def get_current_joints(self) -> list[float]:
        try:
            r = self._requests.get(f"{self._base_url}/api/status", timeout=2)
            return r.json()["angles"][:6]
        except Exception:
            return [0.0] * 6

    def wait_until_reached(
        self, target: list[float], threshold: float = 3.0, timeout: float = 20.0
    ) -> bool:
        import time
        t0 = time.time()
        target_arr = np.array(target[:6])
        ok_streak = 0
        while time.time() - t0 < timeout:
            try:
                cur = np.array(self.get_current_joints())
                if np.max(np.abs(cur - target_arr)) < threshold:
                    ok_streak += 1
                    if ok_streak >= 3:
                        return True
                else:
                    ok_streak = 0
            except Exception:
                ok_streak = 0
            time.sleep(0.5)
        return False
