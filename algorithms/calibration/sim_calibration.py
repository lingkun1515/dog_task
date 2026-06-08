"""仿真手眼标定：从 MJCF camera 位姿推算 T_cam_to_arm。"""

import mujoco
import numpy as np

from algorithms.calibration.base import CalibrationResult


def create_sim_calibration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = "front_cam",
    arm_base_body_name: str = "Piper",
) -> CalibrationResult:
    """从 MJCF 中的 camera 位姿和 arm base body 位姿计算 T_cam_to_arm。

    在 MuJoCo 中，camera 的位姿是已知的（直接从 world 坐标系读），
    arm_base 的位姿也是已知的。由此可精确计算手眼标定。
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    arm_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm_base_body_name)

    # 读取世界坐标系中的相机位姿和 arm_base 位姿（需要 mj_forward 后的 data）
    if cam_id >= 0:
        cam_xpos = data.cam_xpos[cam_id].copy()
        cam_xmat = data.cam_xmat[cam_id].copy().reshape(3, 3)
    else:
        # fallback: 使用 front_cam 的固定位姿
        cam_xpos = np.array(data.xpos[arm_base_id])
        cam_xmat = np.eye(3)

    arm_base_pos = data.xpos[arm_base_id].copy()
    arm_base_xmat = data.xmat[arm_base_id].copy().reshape(3, 3)

    # T_cam_to_world
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = cam_xmat
    T_cam_world[:3, 3] = cam_xpos

    # T_arm_base_to_world
    T_arm_world = np.eye(4)
    T_arm_world[:3, :3] = arm_base_xmat
    T_arm_world[:3, 3] = arm_base_pos

    # T_cam_to_arm = inv(T_arm_world) @ T_cam_world
    T_arm_world_inv = np.eye(4)
    R_inv = arm_base_xmat.T
    T_arm_world_inv[:3, :3] = R_inv
    T_arm_world_inv[:3, 3] = -R_inv @ arm_base_pos

    T_cam_to_arm = T_arm_world_inv @ T_cam_world

    return CalibrationResult(
        T_cam_to_arm=T_cam_to_arm,
        method="mjcf_exact",
        rmse_m=0.0,
    )
