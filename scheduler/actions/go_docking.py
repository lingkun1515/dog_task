import logging
import time

from ..states import RobotState
from .connection import SocketConnection

logger = logging.getLogger(__name__)
RETRY_DELAY_SECONDS = 3.0


def execute(fsm):
    fsm.mark_metric("go_docking_begin")
    print("\n[状态: GO_DOCKING] 正在从指定地点返回充电桩...")
    fsm.notify_timeline("return_A")

    for attempt in range(1, fsm.max_retries + 1):
        print(f"\n[状态: GO_DOCKING] 正在返回充电桩{attempt}...")
        try:
            with SocketConnection(
                host=fsm.config.host,
                port=fsm.config.port,
            ) as conn:
                fsm.mark_metric("dock_command_sent")
                if conn.go_docking(run_timeout=10.0, finish_timeout=180.0):
                    fsm.mark_metric("dock_arrived")
                    print("[动作完成] 已回到充电桩。")
                    return RobotState.FINISHED
        except Exception as e:
            logger.error(f"[状态: GO_DOCKING] 返回充电桩异常: {e}")
            if attempt < fsm.max_retries:
                time.sleep(RETRY_DELAY_SECONDS)
            continue
        logger.warning(f"[状态: GO_DOCKING] 返回充电桩失败{attempt}")
        if attempt < fsm.max_retries:
            time.sleep(RETRY_DELAY_SECONDS)

    print(
        f"[状态: GO_DOCKING] 返回充电桩失败。"
    )
    return RobotState.FAILED
