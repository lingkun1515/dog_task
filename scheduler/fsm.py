import logging
import time

from .actions import go_docking, go_to_location, pick_and_put
from .actions import go_docking_sim, go_to_location_sim, pick_and_put_sim
from .config import RobotConfig
from .states import RobotState

logger = logging.getLogger(__name__)

_HANDLERS_REAL = {
    RobotState.GO_TO_LOCATION: go_to_location,
    RobotState.PICK_AND_PUT: pick_and_put,
    RobotState.GO_DOCKING: go_docking,
}

_HANDLERS_SIM = {
    RobotState.GO_TO_LOCATION: go_to_location_sim,
    RobotState.PICK_AND_PUT: pick_and_put_sim,
    RobotState.GO_DOCKING: go_docking_sim,
}


class RobotTaskFSM:
    def __init__(
        self,
        config: RobotConfig,
        max_retries: int = 3,
        scene: str | None = None,
        on_state_change=None,
        on_timeline=None,
        on_mark=None,
        on_heartbeat=None,
    ):
        self.config = config
        self._state = RobotState.GO_TO_LOCATION
        self.max_retries = max_retries
        # 任务场景（lawn_debris/golf_ball/rain_inspect/material_drop）。
        # 可在派发时通过 scene 覆盖 config.task_scene；缺省沿用配置。
        self.scene = scene or config.task_scene
        self.on_state_change = on_state_change
        self.on_timeline = on_timeline
        self.on_mark = on_mark
        self.on_heartbeat = on_heartbeat
        self._handlers = _HANDLERS_SIM if config.mode == "sim" else _HANDLERS_REAL

    @property
    def state(self):
        return self._state

    @property
    def is_terminal(self):
        return self._state in (RobotState.FINISHED, RobotState.FAILED)

    def notify_timeline(self, step: str) -> None:
        if self.on_timeline is not None:
            self.on_timeline(step)

    def mark_metric(self, name: str) -> None:
        if self.on_mark is not None:
            self.on_mark(name)

    def emit_heartbeat(self) -> None:
        if self.on_heartbeat is not None:
            self.on_heartbeat()

    def step(self):
        if self.is_terminal:
            return self._state
        self._execute_current_state()
        return self._state

    def run(self):
        print("=" * 50)
        print("任务开始：充电桩 → 目标点 → 抓取 → 回充电桩")
        print("=" * 50)

        while not self.is_terminal:
            self.step()
            time.sleep(0.3)

        print("\n任务结束，最终状态:", self._state.name)

    def _set_state(self, new_state):
        old_state = self._state
        if old_state == new_state:
            return
        logger.info("FSM 状态转换: %s → %s", old_state.name, new_state.name)
        self._state = new_state
        if self.on_state_change is not None:
            self.on_state_change(old_state, new_state)

    def _execute_current_state(self):
        handler = self._handlers.get(self._state)
        if handler is None:
            print(f"错误：未知状态 {self._state}")
            self._set_state(RobotState.FAILED)
            return
        self._set_state(handler(self))
