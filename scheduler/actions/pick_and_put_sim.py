import logging

from ..states import RobotState
from .http_utils import http_post, poll_status

logger = logging.getLogger(__name__)
GRASP_TIMEOUT = 180.0


def execute(fsm):
    fsm.mark_metric("pick_and_put_begin")
    fsm.notify_timeline("arm_start")

    detect_url = f"{fsm.config.execution_url}/api/detect"
    http_post(f"{detect_url}/enable")

    for attempt in range(1, fsm.max_retries + 1):
        logger.info("PICK_AND_PUT: 第%d次尝试抓取", attempt)
        resp = http_post(fsm.config.grasp_url)
        if resp is None or resp.get("status") != "accepted":
            logger.warning("PICK_AND_PUT: 抓取请求未被接受")
            continue

        result = poll_status(
            fsm.config.grasp_status_url(),
            success_values=["success"],
            failure_values=["error"],
            timeout=GRASP_TIMEOUT,
            heartbeat_fn=fsm.emit_heartbeat,
            status_key="planner_state",
        )
        if result == "success":
            fsm.mark_metric("grasp_success")
            logger.info("PICK_AND_PUT: 抓取成功")
            break
        logger.warning("PICK_AND_PUT: 抓取失败 (%s)", result)
    else:
        logger.error("PICK_AND_PUT: 达到最大重试次数(%d)，任务失败", fsm.max_retries)
        http_post(f"{detect_url}/disable")
        return RobotState.FAILED

    http_post(f"{detect_url}/disable")
    fsm.mark_metric("put_begin")
    fsm.mark_metric("put_done")
    fsm.notify_timeline("arm_done")
    fsm.mark_metric("arm_home_done")
    return RobotState.GO_DOCKING
