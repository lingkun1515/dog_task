import json
import logging
import threading
import urllib.error
import urllib.request

from ..states import RobotState

logger = logging.getLogger(__name__)

GRASP_TIMEOUT = 3 * 60


def _parse_grasp_response(body: bytes) -> bool:
    """解析抓取接口 JSON 响应，根据 success 字段判断是否成功。"""
    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("抓取响应解析失败: %s", exc)
        return False

    if data.get("success") is True:
        target = data.get("target")
        if isinstance(target, dict):
            logger.info(
                "抓取成功: type=%s conf=%s depth_m=%s pixel=%s arm_xyz=%s",
                target.get("type"),
                target.get("conf"),
                target.get("depth_m"),
                target.get("pixel"),
                target.get("arm_xyz"),
            )
        else:
            logger.info("抓取成功")
        return True

    logger.warning("抓取失败，响应: %s", data)
    return False


def _call_grasp(grasp_url: str) -> bool:
    """调用抓取接口，等价于 curl -X POST <orin>/grasp。"""
    request = urllib.request.Request(grasp_url, data=b"", method="POST")
    try:
        logger.info("调用抓取接口: %s", grasp_url)
        with urllib.request.urlopen(request, timeout=GRASP_TIMEOUT) as response:
            if response.status < 200 or response.status >= 300:
                logger.error("抓取请求失败，HTTP %s", response.status)
                return False
            body = response.read()
            return _parse_grasp_response(body)
    except urllib.error.HTTPError as exc:
        logger.error("抓取请求失败，HTTP %s: %s", exc.code, exc.reason)
        try:
            return _parse_grasp_response(exc.read())
        except Exception:
            return False
    except urllib.error.URLError as exc:
        logger.error("抓取请求失败: %s", exc.reason)
        return False


def _call_grasp_with_heartbeat(fsm, grasp_url: str, interval_sec: float = 10.0) -> bool:
    """抓取 HTTP 阻塞期间定期推送心跳，避免 SSE 长时间无 data 导致断连。"""
    done = threading.Event()
    result: list[bool] = [False]

    def worker() -> None:
        try:
            result[0] = _call_grasp(grasp_url)
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    while not done.wait(timeout=interval_sec):
        fsm.emit_heartbeat()
    return result[0]


def execute(fsm):
    """
    合并任务：捡垃圾(失败重试) → 成功后放入垃圾框
    若捡垃圾重试耗尽，直接任务失败；否则完成两个动作后跳转到返回A
    """
    fsm.mark_metric("pick_and_put_begin")
    fsm.notify_timeline("arm_start")
    for attempt in range(1, fsm.max_retries + 1):
        print(f"\n[状态: PICK_AND_PUT] 第{attempt}次尝试捡起面前的垃圾...")
        if _call_grasp_with_heartbeat(fsm, fsm.config.grasp_url):
            fsm.mark_metric("grasp_success")
            print("[结果] 捡垃圾成功！")
            break
        print("[结果] 捡垃圾失败。")
    else:
        print(
            f"[状态: PICK_AND_PUT] 已达到最大重试次数({fsm.max_retries})，任务失败。"
        )
        return RobotState.FAILED

    fsm.mark_metric("put_begin")
    print("[状态: PICK_AND_PUT] 将垃圾放入后方的垃圾框中...")
    print("[动作完成] 垃圾已放入垃圾框。")
    fsm.mark_metric("put_done")
    fsm.notify_timeline("arm_done")
    fsm.mark_metric("arm_home_done")
    return RobotState.GO_DOCKING
