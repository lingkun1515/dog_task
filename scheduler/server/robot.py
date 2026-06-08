"""Web 服务运行时的机器人配置加载。

服务默认使用仿真机器人，可通过环境变量 SCHEDULER_ROBOT_ID 指定。
"""

import os

from ..config import RobotConfig, load_robot_config

DEFAULT_ROBOT_ID = "sim_go2_piper"


def get_robot_id() -> str:
    return os.environ.get("SCHEDULER_ROBOT_ID", DEFAULT_ROBOT_ID)


def get_robot_config() -> RobotConfig:
    return load_robot_config(get_robot_id())


def list_robot_ids() -> list[str]:
    """列出 config/robots/*.toml 中的机器人编号。"""
    from ..config import robots_config_dir

    ids = [p.stem for p in robots_config_dir().glob("*.toml")]
    return sorted(ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
