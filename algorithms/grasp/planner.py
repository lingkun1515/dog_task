"""统一抓取规划器：感知 → 坐标变换 → IK → 轨迹执行。

Sim 和 Real 共用同一个 GraspPlanner 类，仅底层组件不同：
  - Sim:  calibration = create_sim_calibration(model, data)  # 从 MJCF 推算
  - Real: calibration = load_calibration("output/calibration_result.json")  # ArUco 标定

检测数据流（Sim/Real 完全一致）：
  RGB-D → YOLO/HSV → depth_at_pixel → deproject_pixel → Detection.position_cam
                              ↑ 共享 depth_utils（唯一区别是深度来源）

坐标变换（Sim/Real 完全一致）：
  Detection.position_cam → calibration.cam_to_arm() → arm base frame → IK
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
    move_wait: float = 2.0              # 移动后等待时间（秒）
    gripper_wait: float = 0.5           # 夹爪动作等待时间（秒）


class GraspPlanner:
    """统一抓取流程：Sim/Real 共用。

    区分点仅在校准来源不同：
    - Sim:  create_sim_calibration() → CalibrationResult
    - Real: load_calibration_file() → CalibrationResult
    """

    def __init__(
        self,
        kinematics: ArmKinematics,
        detector: ObjectDetector,
        calibration: CalibrationResult,
        executor: ArmExecutor,
        config: GraspConfig | None = None,
    ):
        self._kinematics = kinematics
        self._detector = detector
        self._calibration = calibration
        self._executor = executor
        self._config = config or GraspConfig()

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
    #
    # 对齐 pick_up_trash/web_server.py:do_grasp_cycle 逻辑：
    #   detect×3 → pick_best → SAFE_PARK → IK above → move above
    #   → descend step × N → grip×2 → lift → park
    #
    # Sim/Real 差异仅在校准（T_cam_to_arm 来源不同），
    # detect→deproject→transform→IK→execute 流程完全一致。
    # ------------------------------------------------------------------
    def execute_full_cycle(self) -> dict:
        """执行一次完整抓取：感知 → IK → 下降 → 夹取 → 提起 → 归位。

        Returns:
            {"success": bool, "target": {...} | "error": str}
        """
        cfg = self._config

        for attempt in range(cfg.max_attempts):
            # === 1. 检测（3次） ===
            self._state = GraspState.DETECTING
            self._status_msg = "detecting"

            if attempt > 0:
                print(f"[GraspPlanner] 重试抓取 (第{attempt+1}次)，重新检测定位")
                self._executor.set_gripper(cfg.gripper_open)
                self._executor.move_to_joints(cfg.safe_park + [cfg.gripper_open], mode=1, wait_time=cfg.move_wait)
                time.sleep(1.0)

            all_dets: list[Detection] = []
            for i in range(3):
                dets = self._detector.detect()
                all_dets.extend(dets)
                if i < 2:
                    time.sleep(0.1)

            if not all_dets:
                if attempt == 0:
                    self._set_error("no_target", "未检测到目标")
                    return self._last_result
                break

            # 选最近目标
            best = self._pick_best_target(all_dets)
            if best is None:
                if attempt == 0:
                    self._set_error("no_reachable", "无可达目标")
                    return self._last_result
                break

            print(f"[GraspPlanner] 检测到 {best.label} conf={best.confidence:.2f} "
                  f"pixel=({best.center_pixel[0]},{best.center_pixel[1]}) depth={best.depth_m:.3f}m")

            # === 2. 坐标变换：camera 3D → arm base frame ===
            # Sim/Real 统一路径：Detection.position_cam → CalibrationResult.cam_to_arm()
            target_arm = self._transform_to_arm(best)
            if target_arm is None:
                self._set_error("transform_failed", "坐标变换失败")
                return self._last_result

            print(f"[GraspPlanner] arm 坐标: ({target_arm[0]:.4f}, {target_arm[1]:.4f}, {target_arm[2]:.4f})")

            # === 3. SAFE_PARK ===
            self._state = GraspState.PARKING
            self._status_msg = "parking"
            self._executor.set_gripper(cfg.gripper_open)
            self._executor.move_to_joints(cfg.safe_park + [cfg.gripper_open], mode=1, wait_time=cfg.move_wait)
            self._executor.wait_until_reached(cfg.safe_park, threshold=2.0, timeout=8.0)
            time.sleep(cfg.gripper_wait)

            # === 4. IK 目标上方 ===
            above = np.array([target_arm[0], target_arm[1], target_arm[2] + cfg.approach_height])
            ik_above = self._solve_ik(above)
            if ik_above is None:
                if attempt == 0:
                    self._set_error("ik_failed", "IK 无法到达目标上方")
                    return self._last_result
                break

            fk_world = np.array(self._kinematics.get_position(ik_above))
            fk_arm = fk_world - np.array(self._kinematics.arm_base_world_pos)
            print(f"[GraspPlanner] IK OK, FK误差={np.linalg.norm(fk_arm - above)*1000:.1f}mm")

            # === 5. 移动到上方 ===
            self._state = GraspState.MOVING_ABOVE
            self._status_msg = "moving above target"
            self._executor.move_to_joints(ik_above + [cfg.gripper_open], mode=1, wait_time=cfg.move_wait)
            self._executor.wait_until_reached(ik_above, threshold=2.0, timeout=8.0)
            time.sleep(cfg.gripper_wait)

            # === 6. 逐步下降 ===
            self._state = GraspState.DESCENDING
            current_z = float(above[2])
            min_z = float(target_arm[2]) - cfg.z_overshoot
            prev_ik = ik_above

            for step in range(10):
                current_z -= cfg.descend_step
                if current_z < min_z:
                    current_z = min_z

                self._status_msg = f"descending step={step} z={current_z:.3f}"
                tp = [target_arm[0], target_arm[1], current_z]
                ik_step = self._kinematics.inverse_kinematics(tp, initial_angles_deg=prev_ik)
                if ik_step is None:
                    print(f"[GraspPlanner] 下降步{step} IK 失败，停止下降")
                    break
                self._executor.move_to_joints(ik_step + [cfg.gripper_open], mode=1, wait_time=cfg.move_wait)
                prev_ik = ik_step
                if current_z <= min_z:
                    break

            time.sleep(cfg.gripper_wait)

            # === 7. 闭合夹爪（二次确保） ===
            self._state = GraspState.GRIPPING
            self._status_msg = "gripping"
            self._executor.set_gripper(cfg.gripper_close)
            time.sleep(cfg.gripper_wait)
            self._executor.set_gripper(cfg.gripper_close)
            time.sleep(cfg.gripper_wait)
            print("[GraspPlanner] 夹爪闭合完成")

            # === 8. 提起 ===
            self._state = GraspState.LIFTING
            self._status_msg = "lifting"
            lift = [target_arm[0], target_arm[1], target_arm[2] + cfg.lift_height]
            ik_lift = self._kinematics.inverse_kinematics(lift, initial_angles_deg=prev_ik)
            if ik_lift:
                self._executor.move_to_joints(ik_lift + [cfg.gripper_close], mode=1, wait_time=cfg.move_wait)
            else:
                self._executor.move_to_joints(cfg.safe_park + [cfg.gripper_close], mode=1, wait_time=cfg.move_wait)

            # === 9. 归位 ===
            self._state = GraspState.PARKING
            self._status_msg = "parking"
            self._executor.move_to_joints(cfg.safe_park + [cfg.gripper_close], mode=1, wait_time=cfg.move_wait)
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

        return self._last_result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _pick_best_target(self, detections: list[Detection]) -> Detection | None:
        """选最近的目标（按 depth_m 排序）。"""
        if not detections:
            return None
        return min(detections, key=lambda d: d.depth_m if d.depth_m > 0 else float("inf"))

    def _transform_to_arm(self, detection: Detection) -> np.ndarray | None:
        """将 detection 坐标转到 arm base 坐标系。

        Sim/Real 统一路径：
          detection.position_cam → self._calibration.cam_to_arm() → arm base 坐标系
        """
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
        print(f"[GraspPlanner] IK 所有种子均失败: target_arm={target_xyz.tolist()}, "
              f"arm_base_world={self._kinematics.arm_base_world_pos.tolist()}")
        return None

    def _set_error(self, error: str, message: str) -> None:
        self._state = GraspState.ERROR
        self._status_msg = error
        self._last_result = {"success": False, "error": error, "message": message}
