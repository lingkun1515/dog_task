"""Mock 摄像头: 按配置脚本返回预设观测序列, 用于无硬件联调."""

from __future__ import annotations

from typing import Optional, Sequence

from ...core.models import HealthStatus, TargetObservation


class MockCamera:
    """从 JSON observations 列表按序返回 TargetObservation, 实现 CameraController."""

    def __init__(
        self,
        observations: Sequence[Optional[TargetObservation]],
        *,
        repeat_last: bool = True,
    ) -> None:
        self._observations = list(observations)
        self._repeat_last = bool(repeat_last)  # 序列用完后是否重复最后一帧
        self._index = 0

    def healthcheck(self) -> HealthStatus:
        return HealthStatus(True, "mock camera ready")

    def get_target(self, *, require_stable: bool = False) -> Optional[TargetObservation]:
        observation = self._next()
        if require_stable and (observation is None or not observation.stable):
            return None
        return observation

    def _next(self) -> Optional[TargetObservation]:
        """取下一条观测; 超出列表时按 repeat_last 决定返回 None 或最后一项."""
        if not self._observations:
            return None
        if self._index < len(self._observations):
            observation = self._observations[self._index]
            self._index += 1
            return observation
        return self._observations[-1] if self._repeat_last else None
