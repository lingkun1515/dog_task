import logging
import time

from ..states import RobotState
from .http_utils import http_get_json, http_post

logger = logging.getLogger(__name__)
GRASP_TIMEOUT = 240.0
POLL_INTERVAL = 1.0


def _is_scene_success(status_data: dict, fsm_scene: str) -> bool:
    """场景化的成功判定。

    - lawn_debris / golf_ball : planner_state == 'success' 或 task_result.outcome in {success, partial}
    - rain_inspect / material_drop : task_result.outcome in {success, partial}
    """
    task_result = status_data.get("task_result") or {}
    outcome = task_result.get("outcome")
    if outcome in ("success", "partial"):
        return True
    # 兜底：planner_state 成功也视为成功（兼容单目标抓取）
    planner_state = status_data.get("planner_state")
    if planner_state == "success":
        return True
    return False


def _wait_for_completion(fsm, timeout: float) -> dict:
    """轮询 grasp status 直到 busy=False（作业结束），返回最终状态。"""
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        data = http_get_json(fsm.config.grasp_status_url())
        if data is not None:
            last = data
            # error 状态立即返回
            if data.get("planner_state") == "error":
                return data
            # busy=False 表示作业线程已退出
            if not data.get("busy", True):
                return data
        fsm.emit_heartbeat()
        time.sleep(POLL_INTERVAL)
    return last


def execute(fsm):
    fsm.mark_metric("pick_and_put_begin")
    fsm.notify_timeline("arm_start")

    detect_url = f"{fsm.config.execution_url}/api/detect"
    http_post(f"{detect_url}/enable")

    scene = getattr(fsm, "scene", None) or "lawn_debris"
    logger.info("PICK_AND_PUT: scene=%s", scene)

    for attempt in range(1, fsm.max_retries + 1):
        logger.info("PICK_AND_PUT: 第%d次尝试 (scene=%s)", attempt, scene)
        resp = http_post(fsm.config.grasp_url)
        if resp is None or resp.get("status") != "accepted":
            logger.warning("PICK_AND_PUT: 作业请求未被接受")
            continue

        # 轮询直到作业线程结束（busy=False），场景无关
        final_status = _wait_for_completion(fsm, GRASP_TIMEOUT)
        scene_ok = _is_scene_success(final_status, scene)
        logger.info("PICK_AND_PUT: planner=%s task_result=%s scene_ok=%s",
                    final_status.get("planner_state"),
                    final_status.get("task_result"), scene_ok)

        if scene_ok:
            fsm.mark_metric("grasp_success")
            task_result = final_status.get("task_result") or {}
            msg = task_result.get("message") or "作业成功"
            logger.info("PICK_AND_PUT: %s", msg)
            break
        logger.warning("PICK_AND_PUT: 作业未完成 (planner=%s)",
                       final_status.get("planner_state"))
    else:
        logger.error("PICK_AND_PUT: 达到最大重试次数(%d)，任务失败 (scene=%s)",
                     fsm.max_retries, scene)
        http_post(f"{detect_url}/disable")
        return RobotState.FAILED

    http_post(f"{detect_url}/disable")
    fsm.mark_metric("put_begin")
    fsm.mark_metric("put_done")
    fsm.notify_timeline("arm_done")
    fsm.mark_metric("arm_home_done")
    return RobotState.GO_DOCKING
