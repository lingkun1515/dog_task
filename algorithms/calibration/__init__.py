"""手眼标定模块。

标定文件约定：
  - Real: algorithms/calibration/output/calibration_result.json
  - Sim:  algorithms/calibration/output/sim_calibration_result.json

标定模块负责生成上述文件，抓取模块启动时从文件加载。
"""

import json
import os
from pathlib import Path

import numpy as np

from algorithms.calibration.base import CalibrationResult

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

REAL_CALIB_FILE = str(_OUTPUT_DIR / "calibration_result.json")
SIM_CALIB_FILE = str(_OUTPUT_DIR / "sim_calibration_result.json")


def load_calibration_file(calib_path: str) -> CalibrationResult:
    """从 JSON 文件加载标定结果（sim/real 通用）。

    文件格式与原仓 pick_up_trash/output/calibration_result.json 一致：
      {"T_cam_to_arm": [[...], ...], "method": "...", "rmse_m": ...}

    Args:
        calib_path: 标定 JSON 文件的绝对路径。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"标定文件不存在: {calib_path}")

    with open(calib_path) as f:
        data = json.load(f)

    T = np.array(data["T_cam_to_arm"])
    return CalibrationResult(
        T_cam_to_arm=T,
        method=data.get("method", "loaded"),
        rmse_m=data.get("rmse_m", 0.0),
    )


__all__ = ["CalibrationResult", "load_calibration_file", "REAL_CALIB_FILE", "SIM_CALIB_FILE"]
