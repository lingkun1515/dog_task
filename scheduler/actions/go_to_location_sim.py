import logging
import time

from ..states import RobotState
from .http_utils import http_post, poll_status

logger = logging.getLogger(__name__)
NAV_TIMEOUT = 180.0
RETRY_DELAY_SECONDS = 1.0


def execute(fsm):
    fsm.mark_metric("go_to_location_begin")
    logger.info("GO_TO_LOCATION: 导航到目标点 (%.2f, %.2f)", fsm.config.target_x, fsm.config.target_y)
    fsm.notify_timeline("go_to_B")

    threshold = getattr(fsm.config, "arrival_threshold", 0.5)
    payload = {
        "x": fsm.config.target_x,
        "y": fsm.config.target_y,
        "require_heading": False,
        "arrival_threshold": threshold,
    }

    for attempt in range(1, fsm.max_retries + 1):
        logger.info("GO_TO_LOCATION: 第%d次尝试", attempt)
        resp = http_post(fsm.config.navigate_url(), payload)
        if resp is None:
            if attempt < fsm.max_retries:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

        fsm.mark_metric("nav_command_sent")
        result = poll_status(
            fsm.config.navigate_status_url(),
            success_values=["arrived"],
            failure_values=["error"],
            timeout=NAV_TIMEOUT,
            heartbeat_fn=fsm.emit_heartbeat,
        )
        if result == "arrived":
            fsm.mark_metric("nav_arrived")
            logger.info("GO_TO_LOCATION: 已到达目标点")
            return RobotState.PICK_AND_PUT

        logger.warning("GO_TO_LOCATION: 导航未完成 (%s)", result)
        if attempt < fsm.max_retries:
            time.sleep(RETRY_DELAY_SECONDS)

    logger.error("GO_TO_LOCATION: 导航失败")
    return RobotState.FAILED
