import logging
import time

from ..states import RobotState
from .http_utils import http_post, poll_status

logger = logging.getLogger(__name__)
NAV_TIMEOUT = 180.0
RETRY_DELAY_SECONDS = 3.0


def execute(fsm):
    fsm.mark_metric("go_docking_begin")
    logger.info("GO_DOCKING: 返回充电桩 (%.2f, %.2f)", fsm.config.home_x, fsm.config.home_y)
    fsm.notify_timeline("return_A")

    http_post(f"{fsm.config.execution_url}/api/detect/disable")

    payload = {
        "x": fsm.config.home_x,
        "y": fsm.config.home_y,
        "require_heading": True,
        "goal_heading": 0.0,
    }

    for attempt in range(1, fsm.max_retries + 1):
        logger.info("GO_DOCKING: 第%d次尝试", attempt)
        resp = http_post(fsm.config.navigate_url(), payload)
        if resp is None:
            if attempt < fsm.max_retries:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

        fsm.mark_metric("dock_command_sent")
        result = poll_status(
            fsm.config.navigate_status_url(),
            success_values=["arrived"],
            failure_values=["error"],
            timeout=NAV_TIMEOUT,
            heartbeat_fn=fsm.emit_heartbeat,
        )
        if result == "arrived":
            fsm.mark_metric("dock_arrived")
            logger.info("GO_DOCKING: 已回到充电桩")
            http_post(f"{fsm.config.execution_url}/api/stop")
            return RobotState.FINISHED

        logger.warning("GO_DOCKING: 返回未完成 (%s)", result)
        if attempt < fsm.max_retries:
            time.sleep(RETRY_DELAY_SECONDS)

    logger.error("GO_DOCKING: 返回充电桩失败")
    return RobotState.FAILED
