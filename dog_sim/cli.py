"""命令行入口: 解析参数并分发到编排器."""

from __future__ import annotations

import argparse

from .config import load_config
from .app.factory import build_orchestrator


def main() -> int:
    """解析 ``python3 -m dog_sim`` 的命令行并执行对应子命令.

    子命令:
        healthcheck  — 检查 arm/camera/mobility 是否就绪.
        fixed-once   — 固定底座: 相机定位后直接抓取一次.
        mobile-once  — 移动闭环: Go2 靠近对准后抓取.
        walk-and-pick — Go2 前进指定距离后抓取.

    返回:
        0 成功, 1 业务失败, 2 配置/装配错误.
    """
    parser = argparse.ArgumentParser(description="DogTaskSim: Go2 + D1 + D455 MuJoCo simulation runner")
    parser.add_argument("--config", required=True, help="JSON configuration path")
    parser.add_argument(
        "command",
        choices=["healthcheck", "fixed-once", "mobile-once", "walk-and-pick"],
    )
    args = parser.parse_args()

    try:
        orchestrator = build_orchestrator(load_config(args.config))
    except (OSError, ValueError) as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    if args.command == "healthcheck":
        statuses = orchestrator.healthcheck()
        for name, status in statuses.items():
            print(f"{name}: {'READY' if status.ready else 'NOT READY'} - {status.message}")
        return 0 if all(status.ready for status in statuses.values()) else 1

    if args.command == "fixed-once":
        result = orchestrator.run_fixed_base_once()
    elif args.command == "walk-and-pick":
        result = orchestrator.run_walk_and_pick()
    else:
        result = orchestrator.run_mobile_single()

    for stream_name in ("stdout", "stderr"):
        output = result.details.get(stream_name)
        if output:
            print(f"\n--- D1 {stream_name} ---")
            print(output.rstrip())
    print(f"{'OK' if result.success else 'FAILED'}: {result.message}")
    return 0 if result.success else 1
