"""Mock 移动底盘: 记录速度/姿态指令, 用于 mock 联调与单测."""

from __future__ import annotations

from typing import Optional

from ...core.models import (
    ActionResult,
    HealthStatus,
    MobilityCapabilities,
    MobilityState,
    NavOptions,
    Posture,
    Pose2D,
    VelocityCommand,
)


class MockMobility:
    """测试用底盘占位, 实现 MobilityController 协议(含姿态/能力/状态)."""

    def __init__(self) -> None:
        self.commands: list[VelocityCommand] = []  # 历史速度指令, 供单测检查
        self.postures: list[str] = []              # 历史姿态切换, 供单测检查
        self.stop_count = 0
        self.emergency_stop_count = 0
        self._stopped = True
        self._posture = Posture.STAND.value

    def healthcheck(self) -> HealthStatus:
        return HealthStatus(True, "mock mobility ready")

    def capabilities(self) -> MobilityCapabilities:
        """mock 支持速度与三种姿态, 不支持全局导航与返航(便于覆盖能力判断分支)."""
        return MobilityCapabilities(
            supports_velocity=True,
            supports_strafe=True,
            supports_global_nav=False,
            supports_docking=False,
            supports_posture=(Posture.STAND.value, Posture.STAND_DOWN.value, Posture.SIT.value),
        )

    def get_state(self) -> MobilityState:
        return MobilityState(is_stopped=self._stopped, posture=self._posture)

    def set_velocity(self, command: VelocityCommand) -> ActionResult:
        self.commands.append(command)
        # 全零速度视为已停止.
        self._stopped = (
            command.linear_x_mps == 0.0
            and command.linear_y_mps == 0.0
            and command.angular_z_rps == 0.0
        )
        return ActionResult(True, "mock velocity accepted")

    def stop(self) -> ActionResult:
        self.stop_count += 1
        self._stopped = True
        return ActionResult(True, "mock mobility stopped")

    def is_stopped(self) -> bool:
        return self._stopped

    def set_posture(self, posture: str) -> ActionResult:
        supported = self.capabilities().supports_posture
        if posture not in supported:
            return ActionResult(False, f"mock mobility does not support posture: {posture}")
        self.postures.append(posture)
        self._posture = posture
        return ActionResult(True, f"mock mobility posture -> {posture}")

    def go_to(self, goal: Pose2D, options: NavOptions) -> ActionResult:
        return ActionResult(False, "mock mobility has no global navigation")

    def dock(self, dock_id: Optional[str] = None) -> ActionResult:
        return ActionResult(False, "mock mobility has no docking")

    def emergency_stop(self) -> ActionResult:
        self.emergency_stop_count += 1
        self._stopped = True
        return ActionResult(True, "mock mobility emergency stopped")
