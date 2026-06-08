import logging
import time

from ..states import RobotState
from .connection import SocketConnection

logger = logging.getLogger(__name__)
RETRY_DELAY_SECONDS = 1.0


def execute(fsm):
    fsm.mark_metric("go_to_location_begin")
    print("\n[状态: GO_TO_LOCATION] 正在从充电桩移动到指定地点...")
    fsm.notify_timeline("go_to_B")

    for attempt in range(1, fsm.max_retries + 1):
        print(f"\n[状态: GO_TO_LOCATION] 移动到指定地点{attempt}...")
        try:
            with SocketConnection(
                host=fsm.config.host,
                port=fsm.config.port,
            ) as conn:
                fsm.mark_metric("nav_command_sent")
                if conn.go_to_location(run_timeout=10.0, finish_timeout=180.0):
                    fsm.mark_metric("nav_arrived")
                    print("[动作完成] 已到达指定地点。")
                    return RobotState.PICK_AND_PUT
        except Exception as e:
            logger.error(f"[状态: GO_TO_LOCATION] 移动到指定地点异常: {e}")
            if attempt < fsm.max_retries:
                time.sleep(RETRY_DELAY_SECONDS)
            continue
        logger.warning(f"[状态: GO_TO_LOCATION] 移动到指定地点失败{attempt}")
        if attempt < fsm.max_retries:
            time.sleep(RETRY_DELAY_SECONDS)

    print(
        f"[状态: GO_TO_LOCATION] 移动到指定地点失败。"
    )
    return RobotState.FAILED
