import json
import logging
import threading
import time
import urllib.error
import urllib.request

from ..states import RobotState

logger = logging.getLogger(__name__)
NAV_TIMEOUT = 180.0
STATUS_POLL_INTERVAL = 1.0
RETRY_DELAY_SECONDS = 1.0


def _call_navigate(navigate_url: str, target_x: float, target_y: float, require_heading: bool = False, arrival_threshold: float = 0.5) -> bool:
    """发送导航请求到执行侧服务。"""
    payload = json.dumps({
        "x": target_x, "y": target_y,
        "require_heading": require_heading,
        "arrival_threshold": arrival_threshold,
    }).encode("utf-8")
    request = urllib.request.Request(
        navigate_url, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return 200 <= response.status < 300
    except urllib.error.URLError as exc:
        logger.error("导航请求失败: %s", exc.reason)
        return False


def _poll_navigate_status(status_url: str, timeout: float, heartbeat_fn) -> str:
    """轮询导航状态，返回 'arrived' / 'error' / 'timeout'。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(status_url, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                status = data.get("status", "unknown")
                if status == "arrived":
                    return "arrived"
                if status == "error":
                    return "error"
        except Exception as exc:
            logger.warning("查询导航状态失败: %s", exc)
        heartbeat_fn()
        time.sleep(STATUS_POLL_INTERVAL)
    return "timeout"


def execute(fsm):
    fsm.mark_metric("go_to_location_begin")
    print("\n[状态: GO_TO_LOCATION] 正在导航到目标点...")
    fsm.notify_timeline("go_to_B")

    target_x = fsm.config.target_x
    target_y = fsm.config.target_y
    nav_url = fsm.config.navigate_url()
    status_url = fsm.config.navigate_status_url()

    for attempt in range(1, fsm.max_retries + 1):
        print(f"\n[状态: GO_TO_LOCATION] 导航到目标({target_x}, {target_y}) 第{attempt}次...")
        try:
            threshold = getattr(fsm.config, "arrival_threshold", 0.5)
            if not _call_navigate(nav_url, target_x, target_y, require_heading=False, arrival_threshold=threshold):
                logger.warning("导航请求发送失败")
                if attempt < fsm.max_retries:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue

            fsm.mark_metric("nav_command_sent")

            def heartbeat():
                fsm.emit_heartbeat()

            result = _poll_navigate_status(status_url, NAV_TIMEOUT, heartbeat)
            if result == "arrived":
                fsm.mark_metric("nav_arrived")
                print("[动作完成] 已到达目标点。")
                return RobotState.PICK_AND_PUT
            else:
                logger.warning(f"导航未完成: {result}")
        except Exception as e:
            logger.error(f"[状态: GO_TO_LOCATION] 导航异常: {e}")
        if attempt < fsm.max_retries:
            time.sleep(RETRY_DELAY_SECONDS)

    print("[状态: GO_TO_LOCATION] 导航失败。")
    return RobotState.FAILED
