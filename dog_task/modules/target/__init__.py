"""目标坐标来源适配器集合(摄像头 / 手动)."""

from .camera import CameraTargetProvider
from .manual import ManualTargetProvider

__all__ = ["CameraTargetProvider", "ManualTargetProvider"]
