"""摄像头坐标来源适配器.

把 CameraController 的 camera_link 观测转换到 arm_base 坐标, 作为 TargetProvider
供重抓流程"重新确认坐标"使用. 转换逻辑与 orchestrator._target_in_arm_base 保持一致.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ...core.interfaces import CameraController
from ...core.models import TargetObservation, Vector3
from ...core.transforms import RigidTransform


class CameraTargetProvider:
    """基于摄像头的目标坐标来源.

    每次 next_target 都向摄像头请求一帧稳定观测, 经外参转换并叠加按类别的
    抓取偏置后, 返回 arm_base 坐标. 无稳定目标时返回 None(调用方沿用旧坐标).
    """

    def __init__(
        self,
        camera: CameraController,
        camera_to_arm_base: RigidTransform,
        grasp_offset_by_class: Optional[Mapping[str, Vector3]] = None,
        *,
        require_stable: bool = True,
    ) -> None:
        """注入摄像头、外参与抓取偏置.

        参数:
            camera:               摄像头控制器(满足 CameraController 协议).
            camera_to_arm_base:   camera_link -> arm_base 的刚体变换.
            grasp_offset_by_class:按目标类别的抓取点偏置(arm_base, 米), 可选.
            require_stable:       是否要求多帧稳定的观测, 默认 True.
        """
        self._camera = camera
        self._transform = camera_to_arm_base
        self._offsets = dict(grasp_offset_by_class or {})
        self._require_stable = bool(require_stable)

    def next_target(self, last: Optional[Vector3]) -> Optional[Vector3]:
        """请求一帧观测并转换到 arm_base; 无稳定目标返回 None."""
        observation = self._camera.get_target(require_stable=self._require_stable)
        if observation is None:
            return None
        return self._to_arm_base(observation)

    def _to_arm_base(self, observation: TargetObservation) -> Vector3:
        """把 camera_link 观测转换到 arm_base 并叠加类别抓取偏置."""
        if observation.frame_id != "camera_link":
            raise ValueError(
                f"expected camera_link observation, got {observation.frame_id}"
            )
        raw = self._transform.apply(observation.position_m)
        offset = self._offsets.get(observation.class_name, Vector3(0.0, 0.0, 0.0))
        return raw.shifted(offset)
