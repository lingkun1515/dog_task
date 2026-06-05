#!/usr/bin/env python3
"""D1 运动学自洽测试: 随机关节角 -> FK -> IK 还原, 验证求解器正确性.

运行: PYTHONPATH=./pylibs python3 test_kinematics.py
"""

import os
import sys

import numpy as np

from d1_kinematics import D1Chain

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.normpath(os.path.join(
    HERE, "../../../../urdf/D1-550 URDF/d1_550_description/urdf/d1_550_description.urdf"))


def main():
    ch = D1Chain(URDF)
    assert [j.name for j in ch.revolute] == [f"Joint{i}" for i in range(1, 7)]

    rng = np.random.default_rng(2024)
    n, ok_pos, ok_appr = 50, 0, 0
    for _ in range(n):
        q = rng.uniform(ch.lower * 0.85, ch.upper * 0.85)
        t = ch.fk(q)
        pos = t[:3, 3]
        appr = t[:3, ch.tool_axis]  # 用该姿态的工具轴(法兰X)作为接近方向目标
        # 位置-only
        _, i1 = ch.ik(pos, n_restarts=20)
        if i1["pos_err"] is not None and i1["pos_err"] < 1e-3:
            ok_pos += 1
        # 位置 + 接近方向
        _, i2 = ch.ik(pos, approach_dir=appr, n_restarts=20)
        if (i2["pos_err"] is not None and i2["pos_err"] < 1e-3
                and i2["ori_err"] is not None and i2["ori_err"] < 1e-2):
            ok_appr += 1

    print(f"位置-only IK:      {ok_pos}/{n} 通过(<1mm)")
    print(f"位置+接近方向 IK:  {ok_appr}/{n} 通过(<1mm 且 <0.57°)")
    passed = ok_pos >= n * 0.95 and ok_appr >= n * 0.9
    print("结果:", "PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
