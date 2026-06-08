"""实机手眼标定：ArUco 标记 + 几何估算。

从 pick_up_trash/calibrate.py 移植。
"""

import json
import os
import time

import numpy as np

from algorithms.calibration.base import CalibrationResult

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output")

# 物理约束基础旋转矩阵（cam_X→arm_-X, cam_Y→arm_-Z, cam_Z→arm_-Y）
_R_BASE = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, -1, 0],
], dtype=float)


def solve_rigid_transform(points_A, points_B):
    """SVD 求解 A→B 的刚体变换: B = R @ A + t。"""
    A = np.array(points_A)
    B = np.array(points_B)
    cA = np.mean(A, axis=0)
    cB = np.mean(B, axis=0)
    H = (A - cA).T @ (B - cB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t


def solve_constrained_transform(points_cam, points_arm):
    """物理约束标定：搜索最优 tilt/pan/roll 使 RMSE 最小。"""
    cam = np.array(points_cam)
    arm = np.array(points_arm)

    best_rmse = 1e9
    best_R, best_t = None, None

    for stage, (lo, hi, step) in enumerate([(-10, 11, 2), (None, None, None)]):
        if stage == 1:
            ranges = [
                np.arange(best_angles[0] - 2, best_angles[0] + 2.1, 0.2),
                np.arange(best_angles[1] - 2, best_angles[1] + 2.1, 0.2),
                np.arange(best_angles[2] - 2, best_angles[2] + 2.1, 0.2),
            ]
        else:
            ranges = [np.arange(lo, hi, step)] * 3
            best_angles = (0, 0, 0)

        for tilt_d in ranges[0]:
            for pan_d in ranges[1]:
                for roll_d in ranges[2]:
                    tilt, pan, roll = np.radians(tilt_d), np.radians(pan_d), np.radians(roll_d)
                    cr, sr = np.cos(roll), np.sin(roll)
                    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
                    ct, st = np.cos(tilt), np.sin(tilt)
                    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
                    cp, sp = np.cos(pan), np.sin(pan)
                    Rz = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])

                    R = _R_BASE @ (Rz @ Ry @ Rx)
                    t_vec = arm.mean(0) - R @ cam.mean(0)
                    pred = (R @ cam.T).T + t_vec
                    rmse = float(np.mean(np.linalg.norm(pred - arm, axis=1)))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_R = R.copy()
                        best_t = t_vec.copy()
                        best_angles = (tilt_d, pan_d, roll_d)

    return best_R, best_t


def run_calibration(arm_controller, camera) -> CalibrationResult | None:
    """运行完整的 ArUco 标定流程（需要 arm + camera 已连接）。

    Args:
        arm_controller: ArmController 实例
        camera: RealSenseCamera 实例

    Returns:
        CalibrationResult 或 None（标定失败）
    """
    import cv2

    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    TIP_MARKER_ID = 1

    CALIB_POSES = [
        [-120.0, 76.9, -65.6, 0.1, 52.3, 4.8],
        [-110.0, 76.9, -65.6, 0.1, 52.3, 4.8],
        [-100.0, 76.9, -65.6, 0.1, 52.3, 4.8],
        [-90.0, 76.9, -65.6, 0.1, 52.3, 4.8],
        [-80.0, 76.9, -65.6, 0.1, 52.3, 4.8],
        [-70.0, 76.9, -65.6, 0.1, 52.3, 4.8],
        [-60.0, 76.9, -65.6, 0.1, 52.3, 4.8],
        [-90.0, 80.0, -65.6, 0.1, 49.2, 4.8],
        [-90.0, 73.0, -65.6, 0.1, 56.2, 4.8],
        [-90.0, 83.0, -65.6, 0.1, 46.2, 4.8],
        [-90.0, 70.0, -65.6, 0.1, 59.2, 4.8],
        [-90.0, 76.9, -60.0, 0.1, 46.7, 4.8],
        [-90.0, 76.9, -70.0, 0.1, 56.7, 4.8],
        [-90.0, 76.9, -55.0, 0.1, 41.7, 4.8],
        [-90.0, 76.9, -75.0, 0.1, 61.7, 4.8],
        [-110.0, 80.0, -65.6, 0.1, 49.2, 4.8],
        [-70.0, 73.0, -65.6, 0.1, 56.2, 4.8],
        [-110.0, 73.0, -65.6, 0.1, 56.2, 4.8],
        [-70.0, 80.0, -65.6, 0.1, 49.2, 4.8],
        [-110.0, 76.9, -60.0, 0.1, 46.7, 4.8],
        [-70.0, 76.9, -70.0, 0.1, 56.7, 4.8],
        [-110.0, 76.9, -70.0, 0.1, 56.7, 4.8],
        [-70.0, 76.9, -60.0, 0.1, 46.7, 4.8],
        [-90.0, 76.9, -65.6, 8.0, 52.3, 4.8],
        [-90.0, 76.9, -65.6, -8.0, 52.3, 4.8],
    ]

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    from algorithms.kinematics.real_d1_ik import D1Kinematics

    # 预检
    arm_controller.move_to(CALIB_POSES[0] + [0], mode=1, wait_time=4.0)
    time.sleep(1.0)
    for _ in range(5):
        camera.grab()

    color, _ = camera.grab()
    if color is not None:
        corners, ids, _ = detector.detectMarkers(color)
        if ids is not None and TIP_MARKER_ID in ids.flatten():
            print(f"  [标定] 预检通过：检测到 ArUco ID={TIP_MARKER_ID}")
        else:
            print(f"  [标定] 警告：预检未检测到标记，继续采集...")

    # 采集
    points_cam, points_arm = [], []

    for i, pose in enumerate(CALIB_POSES):
        print(f"  [标定] 姿态 {i+1}/{len(CALIB_POSES)}")
        arm_controller.move_to(pose + [0], mode=1, wait_time=3.0)
        time.sleep(1.5)

        for _ in range(15):
            camera.grab()

        cam_pts = []
        for _ in range(5):
            color, df = camera.grab()
            if color is None:
                continue
            corners, ids, _ = detector.detectMarkers(color)
            if ids is None:
                continue
            for j, mid in enumerate(ids.flatten()):
                if mid != TIP_MARKER_ID:
                    continue
                c = corners[j][0]
                cx, cy = float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))
                p3d = RealSenseCamera.deproject_pixel(cx, cy, df, camera.intrinsics)
                if p3d is not None:
                    cam_pts.append(p3d)

        if len(cam_pts) < 3:
            print(f"    ✗ 检测不稳定 ({len(cam_pts)}/5)，跳过")
            continue

        p3d_avg = np.mean(cam_pts, axis=0).tolist()
        fk_pos = list(D1Kinematics.get_position(pose[:6]))

        points_cam.append(p3d_avg)
        points_arm.append(fk_pos)

        print(f"    ✓ cam=({p3d_avg[0]:.4f},{p3d_avg[1]:.4f},{p3d_avg[2]:.4f})")
        print(f"      arm=({fk_pos[0]:.4f},{fk_pos[1]:.4f},{fk_pos[2]:.4f})")

    if len(points_cam) < 4:
        print("  [标定] 错误：有效点太少")
        return None

    # 求解
    R_svd, t_svd = solve_rigid_transform(points_cam, points_arm)
    cam_arr = np.array(points_cam)
    arm_arr = np.array(points_arm)
    rmse_svd = float(np.mean(np.linalg.norm((R_svd @ cam_arr.T).T + t_svd - arm_arr, axis=1)))

    R_phy, t_phy = solve_constrained_transform(points_cam, points_arm)
    rmse_phy = float(np.mean(np.linalg.norm((R_phy @ cam_arr.T).T + t_phy - arm_arr, axis=1)))

    print(f"  SVD RMSE={rmse_svd*1000:.1f}mm, 物理约束 RMSE={rmse_phy*1000:.1f}mm")

    if rmse_phy <= rmse_svd * 1.5:
        R, t, method = R_phy, t_phy, "constrained_physical"
    else:
        R, t, method = R_svd, t_svd, "svd"

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    result_path = os.path.join(OUT_DIR, "calibration_result.json")
    with open(result_path, "w") as f:
        json.dump({
            "T_cam_to_arm": T.tolist(),
            "R": R.tolist(),
            "t": t.tolist(),
            "rmse_m": float(np.mean(np.linalg.norm((R @ cam_arr.T).T + t - arm_arr, axis=1))),
            "num_points": len(points_cam),
            "method": method,
            "marker_id": TIP_MARKER_ID,
        }, f, indent=2)
    print(f"  标定结果已保存: {result_path}")

    return CalibrationResult(T_cam_to_arm=T, method=method, rmse_m=rmse_phy)


def load_calibration(calib_path: str | None = None) -> CalibrationResult:
    """加载已保存的标定结果，失败则返回几何估算结果。"""
    if calib_path and os.path.exists(calib_path):
        with open(calib_path) as f:
            data = json.load(f)
        T = np.array(data["T_cam_to_arm"])
        return CalibrationResult(
            T_cam_to_arm=T,
            method=data.get("method", "loaded"),
            rmse_m=data.get("rmse_m", 0.0),
        )

    # 几何估算（fallback）
    R = _R_BASE.copy()
    t = np.array([0.017, -0.120, 0.211])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    print("  使用几何估算变换")
    return CalibrationResult(T_cam_to_arm=T, method="geometric_estimate")


# 延迟导入
from algorithms.perception.real_perception import RealSenseCamera  # noqa: E402
