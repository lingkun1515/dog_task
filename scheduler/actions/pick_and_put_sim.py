import json
import logging
import threading
import time
import urllib.error
import urllib.request

from ..states import RobotState

logger = logging.getLogger(__name__)
GRASP_TIMEOUT = 180.0
STATUS_POLL_INTERVAL = 1.0


def _call_grasp(grasp_url: str) -> bool:
    """调用抓取接口。"""
    request = urllib.request.Request(grasp_url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("status") == "accepted"
    except urllib.error.URLError as exc:
        logger.error("抓取请求失败: %s", exc.reason)
        return False


def _poll_grasp_status(status_url: str, timeout: float, heartbeat_fn) -> bool:
    """轮询抓取状态直到完成或超时。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(status_url, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                status = data.get("status", "unknown")
                if status == "success":
                    return True
                if status == "error":
                    return False
        except Exception as exc:
            logger.warning("查询抓取状态失败: %s", exc)
        heartbeat_fn()
        time.sleep(STATUS_POLL_INTERVAL)
    return False


def execute(fsm):
    fsm.mark_metric("pick_and_put_begin")
    fsm.notify_timeline("arm_start")

    for attempt in range(1, fsm.max_retries + 1):
        print(f"\n[状态: PICK_AND_PUT] 第{attempt}次尝试抓取...")
        if not _call_grasp(fsm.config.grasp_url):
            print("[结果] 抓取请求提交失败。")
            continue

        def heartbeat():
            fsm.emit_heartbeat()

        if _poll_grasp_status(fsm.config.grasp_status_url(), GRASP_TIMEOUT, heartbeat):
            fsm.mark_metric("grasp_success")
            print("[结果] 抓取成功！")
            break
        print("[结果] 抓取失败。")
    else:
        print(f"[状态: PICK_AND_PUT] 已达到最大重试次数({fsm.max_retries})，任务失败。")
        return RobotState.FAILED

    fsm.mark_metric("put_begin")
    print("[状态: PICK_AND_PUT] 目标已回收。")
    fsm.mark_metric("put_done")
    fsm.notify_timeline("arm_done")
    fsm.mark_metric("arm_home_done")
    return RobotState.GO_DOCKING
