"""统一抓取规划器：感知 → 坐标变换 → IK → 轨迹执行。

Sim 和 Real 共用同一个 GraspPlanner 类，仅底层组件不同。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from algorithms.calibration.base import CalibrationResult
from algorithms.grasp.executor import ArmExecutor
from algorithms.kinematics.base import ArmKinematics
from algorithms.perception.base import Detection, ObjectDetector


class GraspState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    MOVING_ABOVE = "moving_above"
    DESCENDING = "descending"
    GRIPPING = "gripping"
    LIFTING = "lifting"
    PARKING = "parking"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class GraspConfig:
    """抓取参数。"""

    approach_height: float = 0.12       # 目标上方高度（米）
    descend_step: float = 0.015         # 每次下降步长（米）
    gripper_open: float = 65            # 夹爪张开角度
    gripper_close: float = 0            # 夹爪闭合角度
    z_overshoot: float = 0.02           # 下降过冲（米）
    lift_height: float = 0.15           # 提起高度（米）
    safe_park: list[float] = field(default_factory=lambda: [-90, 30, -10, 0, 0, 0])
    max_attempts: int = 1               # 最多重试次数


class GraspPlanner:
    """统一抓取流程：Sim/Real 共用。

    区分点：坐标变换来源不同。
    - Sim:  target_arm = arm_base_pose⁻¹ @ detection.position_world
    - Real: target_arm = calibration.cam_to_arm(detection.position_cam)
    """

    def __init__(
        self,
        kinematics: ArmKinematics,
        detector: ObjectDetector,
        calibration: CalibrationResult,
        executor: ArmExecutor,
        config: GraspConfig | None = None,
        sim_mode: bool = False,
    ):
        self._kinematics = kinematics
        self._detector = detector
        self._calibration = calibration
        self._executor = executor
        self._config = config or GraspConfig()
        self._sim_mode = sim_mode

        self._state = GraspState.IDLE
        self._status_msg = ""
        self._last_result: dict = {}

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    @property
    def state(self) -> GraspState:
        return self._state

    @property
    def status(self) -> str:
        return self._status_msg

    @property
    def last_result(self) -> dict:
        return self._last_result

    # ------------------------------------------------------------------
    # 完整抓取流程
    # ------------------------------------------------------------------
    def execute_full_cycle(self) -> dict:
        """执行一次完整抓取：感知 → IK → 下降 → 夹取 → 提起 → 归位。

        Returns:
            {"success": bool, "target": {...} | "error": str}
        """
        cfg = self._config

        # === 1. 检测 ===
        self._state = GraspState.DETECTING
        self._status_msg = "detecting"
        detections = self._detector.detect()
        if not detections:
            self._set_error("no_target", "未检测到目标")
            return self._last_result

        # 选最近的目标
        best = self._pick_best_target(detections)
        if best is None:
            self._set_error("no_reachable", "无可达目标")
            return self._last_result

        target_arm = self._transform_to_arm(best)
        if target_arm is None:
            self._set_error("transform_failed", "坐标变换失败")
            return self._last_result

        print(f"[GraspPlanner] 目标 arm 坐标: ({target_arm[0]:.4f}, {target_arm[1]:.4f}, {target_arm[2]:.4f})")

        # === 2. 开夹爪 ===
        self._status_msg = "open gripper"
        self._executor.set_gripper(cfg.gripper_open)
        time.sleep(0.5)

        # === 3. IK 目标上方 ===
        above = np.array([target_arm[0], target_arm[1], target_arm[2] + cfg.approach_height])
        ik_above = self._solve_ik(above)
        if ik_above is None:
            self._set_error("ik_failed", "IK 无法到达目标上方")
            return self._last_result

        fk_world = np.array(self._kinematics.get_position(ik_above))
        fk_arm = fk_world - np.array(self._kinematics.arm_base_world_pos)
        print(f"[GraspPlanner] IK above: {[f'{a:.1f}' for a in ik_above]}, FK err={np.linalg.norm(fk_arm - above)*1000:.1f}mm")

        # === 4. 移动到上方 ===
        self._state = GraspState.MOVING_ABOVE
        self._status_msg = "moving above target"
        self._executor.move_to_joints(ik_above + [cfg.gripper_open], mode=1, wait_time=3.0 if self._sim_mode else 1.0)
        if not self._sim_mode:
            self._executor.wait_until_reached(ik_above, threshold=2.0, timeout=8.0)
        time.sleep(0.5)

        # === 5. 逐步下降 ===
        self._state = GraspState.DESCENDING
        current_z = float(above[2])
        min_z = float(target_arm[2]) - cfg.z_overshoot
        prev_ik = ik_above

        for _ in range(20):
            current_z -= cfg.descend_step
            if current_z < min_z:
                current_z = min_z

            self._status_msg = f"descending z={current_z:.3f}"
            tp = [target_arm[0], target_arm[1], current_z]
            ik_step = self._kinematics.inverse_kinematics(tp, initial_angles_deg=prev_ik)
            if ik_step is None:
                break
            self._executor.move_to_joints(ik_step + [cfg.gripper_open], mode=1, wait_time=1.0 if self._sim_mode else 0.5)
            prev_ik = ik_step
            if current_z <= min_z:
                break

        time.sleep(0.3 if self._sim_mode else 1.0)

        # === 6. 闭合夹爪 ===
        self._state = GraspState.GRIPPING
        self._status_msg = "gripping"
        self._executor.set_gripper(cfg.gripper_close)
        time.sleep(0.5 if self._sim_mode else 1.5)
        self._executor.set_gripper(cfg.gripper_close)  # 二次确保
        time.sleep(0.5 if self._sim_mode else 1.0)

        # === 7. 提起 ===
        self._state = GraspState.LIFTING
        self._status_msg = "lifting"
        lift = [target_arm[0], target_arm[1], target_arm[2] + cfg.lift_height]
        ik_lift = self._kinematics.inverse_kinematics(lift, initial_angles_deg=prev_ik)
        if ik_lift:
            self._executor.move_to_joints(ik_lift + [cfg.gripper_close], mode=1, wait_time=2.0)
        else:
            self._executor.move_to_joints(cfg.safe_park + [cfg.gripper_close], mode=1, wait_time=2.0)

        # === 8. 归位 ===
        self._state = GraspState.PARKING
        self._status_msg = "parking"
        self._executor.move_to_joints(cfg.safe_park + [cfg.gripper_close], mode=1, wait_time=2.0)
        if not self._sim_mode:
            self._executor.wait_until_reached(cfg.safe_park, threshold=3.0, timeout=10.0)

        self._state = GraspState.SUCCESS
        self._status_msg = "success"
        self._last_result = {
            "success": True,
            "target": {
                "label": best.label,
                "conf": best.confidence,
                "pixel": list(best.center_pixel),
                "depth_m": best.depth_m,
                "arm_xyz": [round(float(v), 4) for v in target_arm],
            },
        }
        print("[GraspPlanner] 抓取完成")
        return self._last_result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _pick_best_target(self, detections: list[Detection]) -> Detection | None:
        """选最近的目标（sim 模式下选 Y 最近，real 按深度）。"""
        if not detections:
            return None
        if self._sim_mode:
            return detections[0]  # sim 通常只有一个目标
        return min(detections, key=lambda d: d.depth_m)

    def _transform_to_arm(self, detection: Detection) -> np.ndarray | None:
        """将 detection 坐标转换到 arm base 坐标系。"""
        if self._sim_mode and detection.position_world is not None:
            # 需要 SimArmKinematics 的 arm_base_world_pose
            if hasattr(self._kinematics, "arm_base_world_pose"):
                T_world_arm = np.linalg.inv(self._kinematics.arm_base_world_pose)
                p = np.append(detection.position_world[:3], 1.0)
                return (T_world_arm @ p)[:3]
            return detection.position_world[:3]

        if detection.position_cam is not None:
            return self._calibration.cam_to_arm(detection.position_cam)

        return None

    def _solve_ik(self, target_xyz: np.ndarray) -> list[float] | None:
        """多初始种子尝试求解 IK。"""
        seeds = [
            [-90, 45, -30, 0, -10, 0],
            [-90, 55, -40, 0, -15, 0],
            [-90, 65, -50, 0, -20, 0],
            [-80, 50, -30, 0, -10, 0],
            [-100, 50, -30, 0, -10, 0],
        ]
        for init in seeds:
            result = self._kinematics.inverse_kinematics(target_xyz.tolist(), initial_angles_deg=init)
            if result is not None:
                return result
        return None

    def _set_error(self, error: str, message: str) -> None:
        self._state = GraspState.ERROR
        self._status_msg = error
        self._last_result = {"success": False, "error": error, "message": message}
