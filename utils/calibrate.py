#!/usr/bin/env python3
"""ArUco 单标记手眼标定入口（实机）。

从原仓 pick_up_trash/calibrate.py 移植。
夹爪夹住 ArUco 标记 (ID=1)，移动 25 个姿态采集点对，SVD 求解 T_cam_to_arm。

用法:
    python scripts/calibrate.py [--arm-host 192.168.123.100] [--arm-port 8088]

输出:
    algorithms/calibration/output/calibration_result.json
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.calibration.real_calibration import run_calibration
from execution.real_robots.camera import RealSenseCamera


def main():
    parser = argparse.ArgumentParser(description="ArUco 手眼标定 (实机)")
    parser.add_argument("--arm-host", default="192.168.123.100")
    parser.add_argument("--arm-port", type=int, default=8088)
    args = parser.parse_args()

    print("=" * 60)
    print("  ArUco 单标记手眼标定")
    print("=" * 60)

    print("\n[1] 连接机械臂...")
    from algorithms.grasp.executor import RealArmExecutor
    arm = RealArmExecutor(host=args.arm_host, port=args.arm_port)

    print("[2] 启动 D455...")
    camera = RealSenseCamera()

    print("[3] 开始标定流程...")
    result = run_calibration(arm, camera)

    if result is None:
        print("\n标定失败！")
        sys.exit(1)

    from algorithms.calibration import REAL_CALIB_FILE
    print(f"\n标定完成！")
    print(f"  方法: {result.method}")
    print(f"  RMSE: {result.rmse_m * 1000:.1f} mm")
    print(f"  文件: {REAL_CALIB_FILE}")

    camera.stop()


if __name__ == "__main__":
    main()
