"""Mock 机械臂: 联调时记录请求并始终返回成功, 不连真机."""

from __future__ import annotations

from ...core.models import (
    ActionResult,
    ArmCapabilities,
    HealthStatus,
    PickRequest,
    Reachability,
    TaughtPickRequest,
)


class MockArm:
    """测试/演示用机械臂, 实现 ArmController 协议."""

    def __init__(self) -> None:
        self.pick_requests: list[PickRequest] = []           # 记录每次 pick_and_place 入参
        self.taught_pick_requests: list[TaughtPickRequest] = []
        self.stop_count = 0
        self.safe_home_count = 0

    def healthcheck(self) -> HealthStatus:
        return HealthStatus(True, "mock arm ready")

    def capabilities(self) -> ArmCapabilities:
        return ArmCapabilities(dof=6, supports_taught_replay=True, grasp_policies=("auto",))

    def estimate_reachability(self, request: PickRequest) -> Reachability:
        """mock 永远可达, 真实适配器应做 IK 可达性判断."""
        return Reachability(True, "mock arm assumes reachable")

    def pick_and_place(self, request: PickRequest, context=None) -> ActionResult:
        """假装完成抓放, 并把请求存下来供单测断言."""
        self.pick_requests.append(request)
        return ActionResult(True, "mock arm completed pick and fixed-basket place")

    def replay_taught_pick(self, request: TaughtPickRequest) -> ActionResult:
        self.taught_pick_requests.append(request)
        return ActionResult(True, "mock arm completed taught pick replay")

    def safe_home(self) -> ActionResult:
        self.safe_home_count += 1
        return ActionResult(True, "mock arm at safe home")

    def stop(self) -> ActionResult:
        self.stop_count += 1
        return ActionResult(True, "mock arm stopped")
