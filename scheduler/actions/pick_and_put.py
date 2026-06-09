import logging
import threading

from ..states import RobotState
from .http_utils import http_post

logger = logging.getLogger(__name__)
GRASP_TIMEOUT = 3 * 60


def _parse_grasp_response(data: dict | None) -> bool:
    if data is None:
        return False
    if data.get("success") is True:
        target = data.get("target")
        if isinstance(target, dict):
            logger.info(
                "抓取成功: type=%s conf=%s depth_m=%s pixel=%s arm_xyz=%s",
                target.get("type"), target.get("conf"),
                target.get("depth_m"), target.get("pixel"), target.get("arm_xyz"),
            )
        else:
            logger.info("抓取成功")
        return True
    logger.warning("抓取失败，响应: %s", data)
    return False


def _call_grasp_with_heartbeat(fsm, grasp_url: str, interval_sec: float = 10.0) -> bool:
    done = threading.Event()
    result: list[bool] = [False]

    def worker():
        resp = http_post(grasp_url, timeout=GRASP_TIMEOUT)
        result[0] = _parse_grasp_response(resp)
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    while not done.wait(timeout=interval_sec):
        fsm.emit_heartbeat()
    return result[0]


def execute(fsm):
    fsm.mark_metric("pick_and_put_begin")
    fsm.notify_timeline("arm_start")

    for attempt in range(1, fsm.max_retries + 1):
        logger.info("PICK_AND_PUT: 第%d次尝试抓取", attempt)
        if _call_grasp_with_heartbeat(fsm, fsm.config.grasp_url):
            fsm.mark_metric("grasp_success")
            logger.info("PICK_AND_PUT: 抓取成功")
            break
        logger.warning("PICK_AND_PUT: 抓取失败")
    else:
        logger.error("PICK_AND_PUT: 达到最大重试次数(%d)，任务失败", fsm.max_retries)
        return RobotState.FAILED

    fsm.mark_metric("put_begin")
    fsm.mark_metric("put_done")
    fsm.notify_timeline("arm_done")
    fsm.mark_metric("arm_home_done")
    return RobotState.GO_DOCKING
