import argparse
import logging

from .actions import (
    go_docking, go_to_location, pick_and_put,
    go_docking_sim, go_to_location_sim, pick_and_put_sim,
)
from .config import load_robot_config
from .fsm import RobotTaskFSM

_ACTIONS_REAL = {
    "go_to_location": go_to_location,
    "pick_and_put": pick_and_put,
    "go_docking": go_docking,
}
_ACTIONS_SIM = {
    "go_to_location": go_to_location_sim,
    "pick_and_put": pick_and_put_sim,
    "go_docking": go_docking_sim,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DogTask 机器人任务状态机")
    parser.add_argument(
        "robot_id",
        help="机器人编号，对应 config/robots/<编号>.toml",
    )
    parser.add_argument(
        "--action",
        choices=tuple(_ACTIONS_REAL),
        default=None,
        help="仅执行指定动作；默认不指定则运行完整 FSM",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="最大重试次数（默认: 3）",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    config = load_robot_config(args.robot_id)
    logging.info(
        "已加载机器人 %s 配置: mode=%s execution_url=%s host=%s",
        config.robot_id, config.mode, config.execution_url, config.host,
    )

    fsm = RobotTaskFSM(
        max_retries=args.max_retries,
        config=config,
    )

    if args.action is None:
        fsm.run()
        return

    actions = _ACTIONS_SIM if config.mode == "sim" else _ACTIONS_REAL
    next_state = actions[args.action](fsm)
    logging.info("动作 %s 执行完成，下一状态: %s", args.action, next_state.name)


if __name__ == "__main__":
    main()
