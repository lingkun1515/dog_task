"""手动输入坐标来源适配器.

作为 TargetProvider, 让操作员在重抓时手动给出新坐标(或回车沿用旧坐标).
也支持预设坐标序列, 便于离线联调与单元测试(无需真人交互).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

from ...core.models import Vector3


class ManualTargetProvider:
    """手动/预设坐标来源.

    两种用法:
      1. 交互式: 不传 preset, 每次 next_target 调用 prompt_fn 读取一行 "x y z";
         空行表示无新坐标(返回 None, 沿用旧坐标).
      2. 预设序列: 传入 preset(可含 None 表示该次沿用旧坐标), 按顺序返回,
         耗尽后一律返回 None. 适合测试与脚本化注入.
    """

    def __init__(
        self,
        preset: Optional[Iterable[Optional[Vector3]]] = None,
        *,
        prompt_fn: Callable[[str], str] = input,
    ) -> None:
        """配置交互函数或预设序列.

        参数:
            preset:    预设坐标序列, 元素为 Vector3 或 None(该次沿用旧坐标);
                       为 None 时进入交互模式.
            prompt_fn: 交互模式下读取一行输入的函数, 默认 input, 便于测试替换.
        """
        self._preset = list(preset) if preset is not None else None
        self._prompt_fn = prompt_fn
        self._index = 0

    def next_target(self, last: Optional[Vector3]) -> Optional[Vector3]:
        """返回下一坐标: 预设模式按序取, 交互模式读取并解析输入."""
        if self._preset is not None:
            if self._index >= len(self._preset):
                return None
            value = self._preset[self._index]
            self._index += 1
            return value
        return self._read_from_prompt()

    def _read_from_prompt(self) -> Optional[Vector3]:
        """交互式读取一行 "x y z"; 空行或解析失败返回 None."""
        raw = self._prompt_fn("输入新坐标 x y z(米), 直接回车沿用旧坐标: ").strip()
        if not raw:
            return None
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> Optional[Vector3]:
        """把一行文本解析为 Vector3; 非三个数则返回 None."""
        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            return None
        try:
            x, y, z = (float(item) for item in parts)
        except ValueError:
            return None
        return Vector3(x, y, z)
