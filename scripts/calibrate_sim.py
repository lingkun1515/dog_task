#!/usr/bin/env python3
"""仿真手眼标定生成入口。

从 MJCF 中的 camera/arm_base 位姿真值直接计算 T_cam_to_arm。
不需要启动仿真窗口，仅加载模型静态数据。

用法:
    python scripts/calibrate_sim.py [--config execution/sim_mujoco/config_d1.yaml]

输出:
    algorithms/calibration/output/sim_calibration_result.json
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="仿真手眼标定 (从 MJCF 真值)")
    parser.add_argument(
        "--config",
        default="execution/sim_mujoco/config_d1.yaml",
        help="仿真配置文件路径",
    )
    args = parser.parse_args()

    import yaml
    import mujoco

    config_path = _PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    xml_path = str(_PROJECT_ROOT / cfg["xml_path"])
    algo_cfg = cfg.get("algorithms", {})
    camera_name = cfg.get("camera_name", "front_cam")
    arm_base_body = algo_cfg.get("arm_base_body", "d1_base_link")

    print("=" * 60)
    print("  仿真手眼标定 (MJCF 真值)")
    print("=" * 60)
    print(f"\n  模型: {xml_path}")
    print(f"  相机: {camera_name}")
    print(f"  臂基座: {arm_base_body}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    from algorithms.calibration import SIM_CALIB_FILE
    from algorithms.calibration.sim_calibration import create_sim_calibration

    result = create_sim_calibration(
        model=model,
        data=data,
        camera_name=camera_name,
        arm_base_body_name=arm_base_body,
        save_to_file=True,
    )

    print(f"\n标定完成！")
    print(f"  方法: {result.method}")
    print(f"  RMSE: {result.rmse_m * 1000:.1f} mm")
    print(f"  文件: {SIM_CALIB_FILE}")
    print(f"\n  T_cam_to_arm:")
    for row in result.T_cam_to_arm:
        print(f"    [{row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f} {row[3]:7.4f}]")


if __name__ == "__main__":
    main()
