"""Shared HTTP utilities for scheduler action handlers."""

import json
import logging
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 180.0
POLL_INTERVAL = 1.0


def http_post(url: str, payload: dict | None = None, timeout: float = 10.0) -> dict | None:
    """POST JSON to URL, return parsed response or None on failure."""
    body = json.dumps(payload).encode("utf-8") if payload else b""
    headers = {"Content-Type": "application/json"} if payload else {}
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode("utf-8"))
            logger.error("POST %s → HTTP %d", url, resp.status)
            return None
    except urllib.error.HTTPError as exc:
        logger.error("POST %s → HTTP %d: %s", url, exc.code, exc.reason)
        try:
            return json.loads(exc.read().decode("utf-8"))
        except Exception:
            return None
    except urllib.error.URLError as exc:
        logger.error("POST %s 失败: %s", url, exc.reason)
        return None


def http_get_json(url: str, timeout: float = 5.0) -> dict | None:
    """GET JSON from URL, return parsed response or None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("GET %s 失败: %s", url, exc)
        return None


def poll_status(
    status_url: str,
    success_values: list[str],
    failure_values: list[str],
    timeout: float = DEFAULT_TIMEOUT,
    heartbeat_fn=None,
    status_key: str = "status",
) -> str:
    """Poll a status endpoint until a terminal value is reached.

    Returns the terminal status string, or 'timeout'.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = http_get_json(status_url)
        if data is not None:
            status = data.get(status_key, "unknown")
            if status in success_values:
                return status
            if status in failure_values:
                return status
        if heartbeat_fn:
            heartbeat_fn()
        time.sleep(POLL_INTERVAL)
    return "timeout"
