import logging
import math
import time

from ..states import RobotState
from .http_utils import http_post, poll_status

logger = logging.getLogger(__name__)
NAV_TIMEOUT = 180.0
RETRY_DELAY_SECONDS = 1.0


def _compute_dwell_stop(
    target_x: float, target_y: float,
    home_x: float, home_y: float,
    dwell: float,
) -> tuple[float, float]:
    """计算导航停靠点：在 home→target 方向上，距 target 前方 dwell 米处停下。

    这样机器人不会直接压在目标上方（避免相机盲区/机械臂工作空间边界）。
    dwell<=0 时直接返回 target（保持原行为）。
    """
    if dwell <= 0:
        return target_x, target_y
    dx = target_x - home_x
    dy = target_y - home_y
    dist = math.hypot(dx, dy)
    if dist <= dwell:
        # 目标本身就在 dwell 范围内，无法再后撤 → 直接用 home 附近
        return home_x, home_y
    # 沿 home→target 方向，从 target 往 home 方向退 dwell 米
    ux, uy = dx / dist, dy / dist
    stop_x = target_x - ux * dwell
    stop_y = target_y - uy * dwell
    return stop_x, stop_y


def execute(fsm):
    fsm.mark_metric("go_to_location_begin")
    cfg = fsm.config
    scene = getattr(fsm, "scene", None) or cfg.task_scene
    logger.info("GO_TO_LOCATION: 导航到目标点 (%.2f, %.2f) scene=%s",
                cfg.target_x, cfg.target_y, scene)
    fsm.notify_timeline("go_to_B")

    http_post(f"{cfg.execution_url}/api/detect/enable")

    # 场景化导航停靠点：巡检场景默认 dwell 0.6m（机器人不压在目标上方）
    dwell = cfg.nav_dwell_distance
    if dwell == 0.0 and scene == "rain_inspect":
        dwell = 0.6  # 场景默认值（配置未显式指定时）
    stop_x, stop_y = _compute_dwell_stop(
        cfg.target_x, cfg.target_y, cfg.home_x, cfg.home_y, dwell,
    )
    if dwell > 0:
        logger.info("GO_TO_LOCATION: 停靠点 (%.2f, %.2f) 距目标 %.2fm (dwell=%.2f)",
                    stop_x, stop_y, math.hypot(cfg.target_x - stop_x, cfg.target_y - stop_y), dwell)

    threshold = getattr(cfg, "arrival_threshold", 0.5)
    payload = {
        "x": stop_x,
        "y": stop_y,
        "require_heading": False,
        "arrival_threshold": threshold,
    }

    for attempt in range(1, fsm.max_retries + 1):
        logger.info("GO_TO_LOCATION: 第%d次尝试", attempt)
        resp = http_post(cfg.navigate_url(), payload)
        if resp is None:
            if attempt < fsm.max_retries:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

        fsm.mark_metric("nav_command_sent")
        result = poll_status(
            cfg.navigate_status_url(),
            success_values=["arrived"],
            failure_values=["error"],
            timeout=NAV_TIMEOUT,
            heartbeat_fn=fsm.emit_heartbeat,
        )
        if result == "arrived":
            fsm.mark_metric("nav_arrived")
            logger.info("GO_TO_LOCATION: 已到达停靠点")
            return RobotState.PICK_AND_PUT

        logger.warning("GO_TO_LOCATION: 导航未完成 (%s)", result)
        if attempt < fsm.max_retries:
            time.sleep(RETRY_DELAY_SECONDS)

    logger.error("GO_TO_LOCATION: 导航失败")
    return RobotState.FAILED
