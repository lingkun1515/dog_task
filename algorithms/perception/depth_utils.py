"""共享深度工具：Sim/Real 共用的像素→3D 反投影和深度查询。

Sim 和 Real 的区别仅在于深度/颜色的来源不同：
  Sim:  MuJoCo mjr_readPixels → RGB + depth buffer → linear meters
  Real: D455 RealSense → wait_for_frames → BGR + depth frame → meters

一旦拿到 numpy RGB (H,W,3) + depth_meters (H,W) float32 后，
所有后续处理（depth_at_pixel / deproject_pixel）完全相同。
"""

import numpy as np


def depth_at_pixel(
    depth_meters: np.ndarray,
    cx: int,
    cy: int,
    roi_size: int = 5,
    min_depth: float = 0.1,
    max_depth: float = 3.0,
) -> float | None:
    """获取像素邻域中位数深度（米）。

    Args:
        depth_meters: (H, W) float32 深度图，单位米。
        cx, cy: 像素坐标。
        roi_size: 邻域半边长。
        min_depth, max_depth: 有效深度范围（米）。

    Returns:
        中位数深度（米），或 None。
    """
    h, w = depth_meters.shape
    y1, y2 = max(0, cy - roi_size), min(h, cy + roi_size)
    x1, x2 = max(0, cx - roi_size), min(w, cx + roi_size)
    roi = depth_meters[y1:y2, x1:x2].astype(np.float64)
    valid = roi[(roi > min_depth) & (roi < max_depth)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def deproject_pixel(
    cx: float,
    cy: float,
    depth_m: float,
    intrinsics: dict,
) -> tuple[float, float, float]:
    """像素坐标 → 相机坐标系 3D 点（与 rs2_deproject_pixel_to_point 一致）。

    Args:
        cx, cy: 像素坐标。
        depth_m: 深度（米）。
        intrinsics: {"fx", "fy", "cx", "cy"} 相机内参。

    Returns:
        (x, y, z) 相机坐标系 3D 坐标（米），z 向前。
    """
    x = (cx - intrinsics["cx"]) / intrinsics["fx"] * depth_m
    y = (cy - intrinsics["cy"]) / intrinsics["fy"] * depth_m
    return (float(x), float(y), float(depth_m))


def intrinsics_from_fov(fovy_deg: float, width: int, height: int) -> dict:
    """从垂直 FOV 和分辨率计算相机内参（方形像素）。

    Args:
        fovy_deg: 垂直视场角（度）。
        width, height: 图像分辨率。

    Returns:
        {"fx", "fy", "cx", "cy", "width", "height", "fovy"}
    """
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg / 2.0))
    return {
        "fx": float(fy),
        "fy": float(fy),
        "cx": float(width / 2.0),
        "cy": float(height / 2.0),
        "width": width,
        "height": height,
        "fovy": fovy_deg,
    }
