"""RLPolicy: ONNX 运动策略推理封装.

加载 Isaac-Lab 训练的 Go2 locomotion policy，输入观测向量，输出关节动作。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RLPolicy:
    """ONNX locomotion policy for Go2."""

    def __init__(self, model_path: str, obs_dim: int = 45) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"RL policy not found: {path}")

        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._obs_dim = obs_dim
        logger.info("RL policy loaded: %s (obs_dim=%d)", path.name, obs_dim)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """运行一次推理，返回 (12,) 动作向量."""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        obs_f32 = obs.astype(np.float32)
        outputs = self._session.run([self._output_name], {self._input_name: obs_f32})
        return outputs[0].flatten()
