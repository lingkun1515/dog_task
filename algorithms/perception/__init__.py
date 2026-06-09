from algorithms.perception.base import Detection, ObjectDetector
from algorithms.perception.detector import HSVDetector, YOLODetector

__all__ = [
    "Detection",
    "ObjectDetector",
    "YOLODetector",
    "HSVDetector",
    "SimObjectDetector",
]


def __getattr__(name):
    if name == "SimObjectDetector":
        from algorithms.perception.sim_perception import SimObjectDetector
        return SimObjectDetector
    if name == "RealSenseCamera":
        from algorithms.perception.real_perception import RealSenseCamera
        return RealSenseCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
