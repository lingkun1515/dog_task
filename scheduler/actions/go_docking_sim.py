import json
import logging
import time
import urllib.error
import urllib.request

from ..states import RobotState

logger = logging.getLogger(__name__)
NAV_TIMEOUT = 180.0
STATUS_POLL_INTERVAL = 1.0
RETRY_DELAY_SECONDS = 3.0

# 回程目标：由配置文件 home_x / home_y 决定


def execute(fsm):
    fsm.mark_metric("go_docking_begin")
    print("\n[状态: GO_DOCKING] 正在返回充电桩(原点)...")
    fsm.notify_timeline("return_A")

    nav_url = fsm.config.navigate_url()
    status_url = fsm.config.navigate_status_url()

    for attempt in range(1, fsm.max_retries + 1):
        print(f"\n[状态: GO_DOCKING] 返回充电桩 第{attempt}次...")
        try:
            payload = json.dumps({"x": fsm.config.home_x, "y": fsm.config.home_y, "require_heading": True}).encode("utf-8")
            request = urllib.request.Request(
                nav_url, data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                if not (200 <= response.status < 300):
                    raise urllib.error.URLError(f"HTTP {response.status}")

            fsm.mark_metric("dock_command_sent")
            deadline = time.monotonic() + NAV_TIMEOUT
            arrived = False
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(status_url, timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        if data.get("status") == "arrived":
                            arrived = True
                            break
                        if data.get("status") == "error":
                            break
                except Exception:
                    pass
                fsm.emit_heartbeat()
                time.sleep(STATUS_POLL_INTERVAL)

            if arrived:
                fsm.mark_metric("dock_arrived")
                print("[动作完成] 已回到充电桩。")
                # Stop all motion on the sim side
                try:
                    urllib.request.urlopen(
                        urllib.request.Request(
                            f"{fsm.config.execution_url}/api/stop", data=b"", method="POST"
                        ),
                        timeout=5,
                    )
                except Exception:
                    pass
                return RobotState.FINISHED
        except Exception as e:
            logger.error(f"[状态: GO_DOCKING] 返回充电桩异常: {e}")
        if attempt < fsm.max_retries:
            time.sleep(RETRY_DELAY_SECONDS)

    print("[状态: GO_DOCKING] 返回充电桩失败。")
    return RobotState.FAILED
