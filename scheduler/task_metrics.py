"""任务各阶段耗时采集与汇总。"""

from __future__ import annotations

import time
from typing import Any


# 汇总表头（与打印顺序一致）
SUMMARY_LABELS = (
    ("task_id", "任务ID"),
    ("dispatch_s", "任务下发耗时(s)"),
    ("depart_to_nav_s", "出桩至导航点耗时(s)"),
    ("grasp_success_s", "抓取成功耗时(s)"),
    ("put_s", "放球耗时(s)"),
    ("arm_home_s", "机械臂回零位耗时(s)"),
    ("dock_return_s", "回桩耗时(s)"),
)


class TaskTiming:
    """基于 monotonic 时钟的阶段埋点，用于任务结束后的耗时汇总。"""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._marks: dict[str, float] = {"task_start": time.monotonic()}

    def mark(self, name: str) -> None:
        self._marks[name] = time.monotonic()

    def duration_s(self, start: str, end: str) -> float | None:
        t0 = self._marks.get(start)
        t1 = self._marks.get(end)
        if t0 is None or t1 is None or t1 < t0:
            return None
        return round(t1 - t0, 3)

    def build_summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dispatch_s": self.duration_s("task_start", "nav_command_sent"),
            "depart_to_nav_s": self.duration_s("nav_command_sent", "nav_arrived"),
            "grasp_success_s": self.duration_s("pick_and_put_begin", "grasp_success"),
            "put_s": self.duration_s("put_begin", "put_done"),
            "arm_home_s": self.duration_s("put_done", "arm_home_done"),
            "dock_return_s": self.duration_s("dock_command_sent", "dock_arrived"),
        }

    def format_summary_lines(self) -> list[str]:
        summary = self.build_summary()
        lines = ["========== 任务耗时汇总 =========="]
        for key, label in SUMMARY_LABELS:
            value = summary.get(key)
            if key == "task_id":
                text = str(value) if value is not None else "-"
            else:
                text = f"{value:.3f}" if isinstance(value, (int, float)) else "-"
            lines.append(f"{label}: {text}")
        lines.append("==================================")
        return lines
