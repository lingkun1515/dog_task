"""JSON 配置加载与路径解析工具."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


# 仓库根目录(dog_task 的上一级), 用于把配置里的相对路径解析为绝对路径.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    """从 JSON 文件加载完整配置字典.

    参数:
        path: 配置文件路径, 如 ``config/demo.unified-sim.json``.
    返回:
        顶层必须是 JSON 对象(字典).
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a JSON object")
    return data


def section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """读取配置中的某个命名段(如 ``arm`` / ``camera`` / ``workflow``).

    若缺失或不是对象, 抛出 ValueError, 便于启动时快速发现配置错误.
    """
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing configuration section: {name}")
    return value


def project_path(value: str | Path) -> Path:
    """把配置里的路径转为绝对路径.

    已是绝对路径则原样返回; 相对路径则相对于 PROJECT_ROOT(仓库根目录).
    """
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
