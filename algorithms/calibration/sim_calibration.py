"""仿真手眼标定：从 MJCF camera 位姿推算 T_cam_to_arm。

输出的 T_cam_to_arm 将 OpenCV 相机坐标系（X 右、Y 下、Z 前）中的点
映射到机械臂基座坐标系中，与实机 ArUco 标定产出格式一致。
"""

import json
import os

import mujoco
import numpy as np

from algorithms.calibration.base import CalibrationResult

# MuJoCo 相机坐标系 → OpenCV 相机坐标系的转换矩阵
# MuJoCo: X right, Y up,   Z backward
# OpenCV:  X right, Y down, Z forward
# 转换: P_cv = R @ P_mj,  其中 R = diag(1, -1, -1)
# 反向: P_mj = R @ P_cv   (R 是自逆矩阵)
_R_CV_TO_MJ = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=np.float64)

_R_CV_TO_MJ_4x4 = np.eye(4)
_R_CV_TO_MJ_4x4[:3, :3] = _R_CV_TO_MJ

SIM_CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SIM_CALIB_FILE = os.path.join(SIM_CALIB_DIR, "sim_calibration_result.json")


def create_sim_calibration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = "front_cam",
    arm_base_body_name: str = "Piper",
    save_to_file: bool = True,
) -> CalibrationResult:
    """从 MJCF 中的 camera 位姿和 arm base body 位姿计算 T_cam_to_arm。

    输出的 T_cam_to_arm 将 OpenCV 相机坐标系中的 3D 点映射到机械臂基座坐标系。
    这与 deproject_pixel() 的输出约定一致（z 正方向为相机前方）。
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    arm_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm_base_body_name)

    if cam_id >= 0:
        cam_xpos = data.cam_xpos[cam_id].copy()
        cam_xmat = data.cam_xmat[cam_id].copy().reshape(3, 3)
    else:
        cam_xpos = np.array(data.xpos[arm_base_id])
        cam_xmat = np.eye(3)

    arm_base_pos = data.xpos[arm_base_id].copy()
    arm_base_xmat = data.xmat[arm_base_id].copy().reshape(3, 3)

    # T_cam_world: MuJoCo 相机在世界坐标系中的位姿
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = cam_xmat
    T_cam_world[:3, 3] = cam_xpos

    # inv(T_arm_world): 世界 → 机械臂基座
    R_inv = arm_base_xmat.T
    T_arm_world_inv = np.eye(4)
    T_arm_world_inv[:3, :3] = R_inv
    T_arm_world_inv[:3, 3] = -R_inv @ arm_base_pos

    # T_mj_cam_to_arm: MuJoCo 相机坐标系 → 机械臂基座坐标系
    T_mj_cam_to_arm = T_arm_world_inv @ T_cam_world

    # T_cam_to_arm: OpenCV 相机坐标系 → 机械臂基座坐标系
    # P_arm = T_mj_cam_to_arm @ P_mj = T_mj_cam_to_arm @ R_cv_to_mj @ P_cv
    T_cam_to_arm = T_mj_cam_to_arm @ _R_CV_TO_MJ_4x4

    if save_to_file:
        _save_calibration(T_cam_to_arm)

    return CalibrationResult(
        T_cam_to_arm=T_cam_to_arm,
        method="mjcf_exact",
        rmse_m=0.0,
    )


def load_sim_calibration(calib_path: str | None = None) -> CalibrationResult | None:
    """从文件加载仿真标定结果。

    Args:
        calib_path: 标定文件路径，默认 output/sim_calibration_result.json

    Returns:
        CalibrationResult 或 None（文件不存在时）
    """
    path = calib_path or SIM_CALIB_FILE
    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)
    T = np.array(data["T_cam_to_arm"])
    return CalibrationResult(
        T_cam_to_arm=T,
        method=data.get("method", "loaded_sim"),
        rmse_m=data.get("rmse_m", 0.0),
    )


def _save_calibration(T_cam_to_arm: np.ndarray) -> None:
    """保存仿真标定结果到 JSON 文件。"""
    os.makedirs(SIM_CALIB_DIR, exist_ok=True)
    result = {
        "T_cam_to_arm": T_cam_to_arm.tolist(),
        "R": T_cam_to_arm[:3, :3].tolist(),
        "t": T_cam_to_arm[:3, 3].tolist(),
        "rmse_m": 0.0,
        "method": "mjcf_exact",
        "note": "OpenCV camera frame (z forward, y down) -> arm base frame",
    }
    with open(SIM_CALIB_FILE, "w") as f:
        json.dump(result, f, indent=2)
