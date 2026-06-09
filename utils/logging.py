"""DogTask 统一日志配置。

用法：
    from utils.logging import setup_logging
    setup_logging("execution", "logs/execution.log")
    # 之后所有 logging.getLogger(__name__) 的输出同时写文件 + 终端
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGURED: set[str] = set()

LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str,
    log_file: str,
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """配置根 logger：同时输出到终端（INFO+）和文件（DEBUG+）。

    Args:
        name: 日志名称标识（用于首次配置去重）
        log_file: 日志文件路径（相对于项目根目录或绝对路径）
        console_level: 终端输出级别
        file_level: 文件输出级别
        max_bytes: 单文件最大字节数（默认 10MB）
        backup_count: 保留的历史日志文件数
    """
    if name in _CONFIGURED:
        return logging.getLogger()

    log_path = Path(log_file) if os.path.isabs(log_file) else _PROJECT_ROOT / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(console_level)
        console.setFormatter(formatter)
        root.addHandler(console)

    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED.add(name)

    logging.getLogger(name).info("日志系统已初始化 → %s", log_path)
    return root
