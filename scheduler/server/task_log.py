"""任务运行日志持久化：按次写入 JSONL，便于后续分析卡点。"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from ..task_metrics import TaskTiming


def _logs_root() -> Path:
    if log_dir := os.environ.get("MOWER_LOG_DIR"):
        return Path(log_dir)
    return Path(__file__).resolve().parent.parent.parent / "logs" / "runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class TaskRunRecorder:
    """单次任务运行的 JSONL 记录器（线程内使用，非线程安全）。"""

    def __init__(self, path: Path, run_id: str):
        self.run_id = run_id
        self.path = path
        self._file: TextIO = path.open("a", encoding="utf-8")
        self._started_at = datetime.now(timezone.utc)
        self.timing = TaskTiming(run_id)

    @classmethod
    def open(
        cls,
        *,
        robot_id: str,
        action: str | None,
        max_retries: int,
        host: str = "",
        orin: str = "",
        port: int = 0,
        grasp_port: int = 0,
        mode: str = "real",
        execution_url: str = "",
    ) -> TaskRunRecorder:
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        day_dir = _logs_root() / datetime.now().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"{run_id}.jsonl"
        recorder = cls(path, run_id)
        recorder.event(
            "task_start",
            robot_id=robot_id,
            action=action or "full_fsm",
            max_retries=max_retries,
            mode=mode,
            host=host,
            orin=orin,
            port=port,
            grasp_port=grasp_port,
            execution_url=execution_url,
            log_file=str(path),
        )
        return recorder

    def _elapsed_ms(self) -> int:
        return int(
            (datetime.now(timezone.utc) - self._started_at).total_seconds() * 1000
        )

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "ts": _now_iso(),
            "event": event,
            "run_id": self.run_id,
            "elapsed_ms": self._elapsed_ms(),
            **fields,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def log_line(self, msg: str) -> None:
        if msg:
            self.event("output", msg=msg)

    def mark(self, name: str) -> None:
        self.timing.mark(name)
        self.event("mark", name=name)

    def emit_timing_summary(self) -> list[str]:
        """写入 timing_summary 事件并返回可打印行。"""
        summary = self.timing.build_summary()
        lines = self.timing.format_summary_lines()
        self.event("timing_summary", **summary)
        return lines

    def finish(
        self,
        *,
        final_state: str | None = None,
        ok: bool | None = None,
        error: str | None = None,
        dispatch_ms: int | None = None,
        finished_at_ms: int | None = None,
        notify_ms: int | None = None,
    ) -> None:
        duration_ms = self._elapsed_ms()
        self.event(
            "task_end",
            final_state=final_state,
            ok=ok,
            error=error,
            duration_ms=duration_ms,
            dispatch_ms=dispatch_ms,
            finished_at_ms=finished_at_ms,
            notify_ms=notify_ms,
        )
        self._file.close()

    def close_on_error(self, error: str) -> None:
        self.finish(final_state=None, ok=False, error=error)


def log_task_rejected(robot_id: str, action: str | None, reason: str) -> None:
    """任务未启动（如并发冲突）时追加一条摘要到当日 rejected 日志。"""
    day_dir = _logs_root() / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / "_rejected.jsonl"
    record = {
        "ts": _now_iso(),
        "event": "task_rejected",
        "robot_id": robot_id,
        "action": action or "full_fsm",
        "reason": reason,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
