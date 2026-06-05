#!/usr/bin/env python3
"""
D1 机械臂逆解命令行工具: 把目标坐标解算成机械臂可下发的关节参数.

两种用法:
  1) 单点求解(solve): 给一个目标位姿 -> 输出 6 个关节角(度) + DDS JSON 指令.
  2) 抓取规划(grasp): 给瓶子坐标 + 收纳框坐标 -> 输出整套抓放动作序列指令.

坐标系: base_link(机械臂底座), x 向前, z 向上, 单位米. 角度单位: 命令行用度.

示例:
  # 解算前方 0.3m、高 0.2m 处的目标(竖直向下抓取姿态)
  python3 solve_ik.py solve --xyz 0.3 0 0.2 --rpy 180 0 0

  # 规划: 瓶子在(0.3,0,0.05), 收纳框在(0.1,0.25,0.1)
  python3 solve_ik.py grasp --bottle 0.3 0 0.05 --basket 0.1 0.25 0.1 --send

  # 示教回放抓取: 抓取示教角 -89.8 61.9 6.3 9.8 6.9 -3.8, 放置示教角 91.4 74.2 -53.0 0.7 65.1 -6.0
  python3 solve_ik.py teach --grasp-joints -89.8 61.9 6.3 9.8 6.9 -3.8 --place-joints 91.4 74.2 -53.0 0.7 65.1 -6.0 --send-safe

"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from d1_kinematics import D1Chain, JointMapping

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.normpath(os.path.join(
    HERE, "../../../../urdf/D1-550 URDF/d1_550_description/urdf/d1_550_description.urdf"))
# 通用 DDS 发布器(由 C++ 编译): 接收一个 JSON 字符串并发到 rt/arm_Command
DEFAULT_SENDER = os.path.normpath(os.path.join(HERE, "../build_linux/arm_cmd_send"))
# 慢速安全运动原语: 从当前角小步插值到目标角
DEFAULT_SAFE_MOVE = os.path.normpath(os.path.join(HERE, "../build_linux/safe_move_to"))
# 连接 D1 的有线网卡: 默认本机实测网卡, 可用环境变量 D1_NET_IFACE 覆盖.
# 例: Orin 上走网桥时设 D1_NET_IFACE=br123, 无需改源码.
NET_IFACE = os.environ.get("D1_NET_IFACE", "enx6c1ff765a912")

# 夹爪标定值(舵机刻度, 已实测确认). angle6 对应 DDS 的夹爪通道, id=6.
# 实测: angle 越大越张开, 舵机有效范围 0~50(发>50 封顶在50).
#   angle=50 完全张开(~55mm), angle=0 接近闭合(~15mm缝). 近似 缝隙(mm)≈15+angle.
#   夹住值 ≈ 瓶径(mm) - 20, 例: 40mm 瓶 -> 约 20.
GRIPPER_OPEN = 50.0    # 完全张开(接近瓶子前)
GRIPPER_CLOSE = 20.0   # 夹住普通瓶(3-5cm), 按瓶径微调: CLOSE≈瓶径mm-20


def build_chain(args) -> D1Chain:
    """
    构建 D1Chain 机械臂运动链对象，根据输入参数自适应关节映射参数。

    参数:
        args: 包含 urdf、tcp、sign、offset 等属性的命名空间（通常为 argparse.Namespace）。

    返回:
        D1Chain 实例，包含机械臂连杆结构及关节限位、映射关系等。

    示例:
        # 假设 args 中包含 urdf 文件路径和 TCP 偏移距离
        args.urdf = "path/to/urdf"
        args.tcp = 0.10
        # 若不提供 sign/offset, 则使用默认映射
        chain = build_chain(args)

        # 带关节正负方向(sign)和零点偏置(offset)自定义
        args.sign = [1, -1, 1, 1, -1, 1]
        args.offset = [0, 0, -10, 0, 0, 0]
        chain = build_chain(args)

    说明:
        - D1 夹爪方向与法兰 X 轴同向（实测为 X 轴），末端 TCP 偏移与工具轴均取 X 轴。
        - 若提供 sign/offset 参数，则覆盖默认关节方向和零点偏置。
        - 关节正负与偏置会通过 JointMapping 映射关系处理。

    """
    mapping = JointMapping()
    if args.sign:
        mapping.sign = np.array([float(x) for x in args.sign], dtype=float)
    if args.offset:
        mapping.offset_deg = np.array([float(x) for x in args.offset], dtype=float)
    # D1 夹爪沿法兰 X 轴伸出(实测确认), TCP 偏移与接近轴均取 X 轴.
    return D1Chain(args.urdf, tcp_offset=(args.tcp, 0.0, 0.0), tool_axis=0, mapping=mapping)


def dds_multi_joint_cmd(angles_deg, seq=4, mode=1) -> str:
    """funcode 2: 全关节角度控制. angles_deg 为 angle0~angle6 共 7 个."""
    a = list(angles_deg) + [0.0] * (7 - len(angles_deg))
    data = {"mode": mode}
    for i in range(7):
        data[f"angle{i}"] = round(float(a[i]), 3)
    return json.dumps({"seq": seq, "address": 1, "funcode": 2, "data": data},
                      ensure_ascii=False, separators=(",", ":"))


def dds_single_joint_cmd(joint_id, angle_deg, seq=4, delay_ms=0) -> str:
    """funcode 1: 单关节角度控制(此处用于夹爪 id=6)."""
    return json.dumps({"seq": seq, "address": 1, "funcode": 1,
                       "data": {"id": int(joint_id), "angle": round(float(angle_deg), 3),
                                "delay_ms": int(delay_ms)}},
                      ensure_ascii=False, separators=(",", ":"))


def solve_pose(chain: D1Chain, xyz, rpy_deg=None, approach=None, seed=None):
    """求解单个位姿, 返回 (q_rad, dds_deg, info) 或在失败时打印并返回 None.

    approach: 夹爪接近方向(世界系向量), 优先于 rpy_deg; 二者皆 None 时只解位置.

    示例输入:
        # 只给定抓取位置，不约束姿态
        solve_pose(chain, [0.5, 0.1, 0.3])
        # 给定抓取位置 + 姿态
        solve_pose(chain, [0.5, 0.1, 0.3], rpy_deg=[0, 90, 0])
        # 给定抓取位置 + 末端接近方向
        solve_pose(chain, [0.5, 0.1, 0.3], approach=[0, 0, -1])
        # 带初值
        solve_pose(chain, [0.5, 0.1, 0.3], rpy_deg=[0, 0, 0], seed=[0, 0, 0, 0, 0, 0])

    示例输出:
        # 若可达
        (
            array([ 0.35, -0.23, 1.13, ...]),        # q_rad: 逆解出的关节弧度(6元素ndarray)
            array([ 20.1, -12.3, 64.7, ...]),        # dds_deg: 下发角度(度),6元素ndarray)
            {
                'pos_err': 0.0013,                  # 末端实际与目标距离误差(米)
                'ori_err': 0.017,                   # 姿态误差(弧度)
                'in_limits': True,                  # 逆解角度是否全部在机械臂限位范围内
                'reachable': True,                  # 是否同时满足可达、姿态、限位
                ...
            }
        )

        # 若不可达, 返回 None

    """
    rpy_rad = np.radians(rpy_deg) if rpy_deg is not None else None
    q, info = chain.ik(xyz, target_rpy=rpy_rad, approach_dir=approach, seed=seed)
    if q is None:
        return None
    dds_deg = chain.to_dds_deg(q)
    ok_pos = info["pos_err"] is not None and info["pos_err"] < 5e-3
    constrained = (rpy_deg is not None) or (approach is not None)
    ok_ori = (not constrained) or (info["ori_err"] is not None and info["ori_err"] < 5e-2)
    ok_lim = chain.in_limits(q)
    info.update(reachable=bool(ok_pos and ok_ori and ok_lim), in_limits=ok_lim)
    return q, dds_deg, info


def print_solution(tag, dds_deg, info):
    print(f"[{tag}]")
    print("  关节角(度) angle0~angle5 = " +
          ", ".join(f"{v:8.3f}" for v in dds_deg))
    pe = info.get("pos_err")
    oe = info.get("ori_err")
    print(f"  位置误差 = {pe*1000:.2f} mm" if pe is not None else "  位置误差 = N/A",
          end="")
    print(f", 姿态误差 = {np.degrees(oe):.2f}°" if oe else "")
    print(f"  关节在限位内: {info['in_limits']}, 可达: {info['reachable']}")
    if not info["reachable"]:
        print("  !! 目标不可达或超限, 请调整坐标/姿态或确认 TCP 偏移与符号映射")


def maybe_send(cmd_json, sender, do_send):
    if not do_send:
        return
    if not os.path.exists(sender):
        print(f"  [发送跳过] 未找到发布器 {sender}, 请先编译 build_linux/arm_cmd_send")
        return
    # 多网卡环境必须显式绑定网卡, 否则 DDS 组播走错网卡, 指令发不到机械臂.
    subprocess.run([sender, cmd_json, NET_IFACE], check=False)
    print("  [已发送]")


def cmd_solve(args):
    chain = build_chain(args)
    res = solve_pose(chain, args.xyz, rpy_deg=args.rpy, approach=args.approach_dir)
    if res is None:
        print("求解失败: 无解")
        sys.exit(1)
    _, dds_deg, info = res
    print_solution("solve", dds_deg, info)
    cmd = dds_multi_joint_cmd(dds_deg)
    print("  DDS 指令:", cmd)
    maybe_send(cmd, args.sender, args.send)


def cmd_grasp(args):
    """瓶子 -> 收纳框 的完整抓放序列."""
    chain = build_chain(args)
    bottle = np.array(args.bottle, dtype=float)
    basket = np.array(args.basket, dtype=float)
    approach = float(args.approach)        # 抓取点上方接近高度(米)
    appr_dir = np.array(args.approach_dir, dtype=float)  # 夹爪接近方向, 默认顶部向下
    t_mode = getattr(args, "transfer_mode", 1)           # 转运段 mode: 0=小平滑(快), 1=大平滑(慢)

    # 判断是否为纯竖直向下: 若是, IK 解完后强制 angle5=0(不转圈).
    # approach 模式只约束接近方向(2DOF), 绕工具轴的自旋自由, 导致 angle5 跳变.
    # 当 tool_axis=X 且 TCP 偏移纯沿 X 时, 竖直向下改变 angle5 不影响 TCP 位置,
    # 所以可以在 IK 解完后安全地将 angle5 覆盖为 0.
    fix_wrist = getattr(args, "fix_wrist", True)
    is_vertical_down = (abs(appr_dir[0]) < 1e-3 and abs(appr_dir[1]) < 1e-3 and appr_dir[2] < -0.9)
    use_fix_wrist = fix_wrist and is_vertical_down
    if use_fix_wrist:
        print("[姿态约束] 竖直向下, 强制 angle5=0 固定腕部不转(--no-fix-wrist 可关闭)")

    # 关键路点: (标签, 目标xyz, 夹爪角度 or None 表示沿用, mode or None)
    # LIFT 和 PLACE_PRE 属于转运段, 可通过 --transfer-mode 控制
    waypoints = [
        ("PRE_GRASP 抓取上方", bottle + [0, 0, approach], GRIPPER_OPEN, None),
        ("GRASP 抓取点",       bottle,                    None,         None),
        ("CLOSE 闭合夹爪",      None,                      GRIPPER_CLOSE, None),
        ("LIFT 提起",          bottle + [0, 0, approach],  None,         t_mode),
        ("PLACE_PRE 框上方",    basket + [0, 0, approach],  None,         t_mode),
        ("PLACE 放入框内",      basket,                    None,         None),
        ("RELEASE 松开夹爪",    None,                      GRIPPER_OPEN, None),
        ("RETREAT 抬起退回",    basket + [0, 0, approach],  None,         None),
    ]

    seq = 4
    seed = np.zeros(6)
    sequence_cmds = []   # (tag, json) 仅用于展示/落盘
    steps = []           # 结构化步骤, 用于安全执行: {tag,type,...}
    all_ok = True
    for tag, xyz, grip, mode in waypoints:
        if xyz is not None:
            res = solve_pose(chain, xyz, approach=appr_dir, seed=seed)
            if res is None:
                print(f"[{tag}] 求解失败"); all_ok = False; continue
            q, dds_deg, info = res
            # 竖直向下时强制 angle5=0: 工具 X 对齐 -Z 时自旋不影响 TCP 位置
            if use_fix_wrist:
                dds_deg = np.array(dds_deg)
                dds_deg[5] = 0.0
            seed = q  # 用上一路点作为下一点的初值, 保证轨迹连续
            print_solution(tag, dds_deg, info)
            all_ok = all_ok and info["reachable"]
            cmd = dds_multi_joint_cmd(dds_deg, seq=seq)
            print("  ->", cmd)
            sequence_cmds.append((tag, cmd))
            st = {"tag": tag, "type": "move", "angles": [round(float(v), 2) for v in dds_deg]}
            if mode is not None:
                st["mode"] = mode
            steps.append(st)
            seq += 1
        if grip is not None:
            gcmd = dds_single_joint_cmd(6, grip, seq=seq)
            print(f"[{tag}] 夹爪 -> {grip}")
            print("  ->", gcmd)
            sequence_cmds.append((tag, gcmd))
            steps.append({"tag": tag, "type": "grip", "value": float(grip), "json": gcmd})
            seq += 1

    print("\n==== 动作序列汇总 ====")
    for tag, cmd in sequence_cmds:
        print(f"{tag}: {cmd}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([c for _, c in sequence_cmds], f, ensure_ascii=False, indent=2)
        print(f"\n已写入命令序列: {args.out}")

    if not all_ok:
        print("\n!! 序列中存在不可达路点, 请调整坐标/接近高度/TCP/夹爪标定后重试")

    if args.send_safe or args.send:
        if not all_ok:
            print("存在不可达路点, 为安全起见取消自动执行")
            return
        # 抓取前先归零, 确保从已知姿态出发
        zero_before_grasp(args)
        execute_steps(steps, args, safe=args.send_safe)


def zero_before_grasp(args):
    """抓取前归零: 用 safe_move_to 将所有关节安全回到零位."""
    import time
    safe_move = getattr(args, "safe_move", DEFAULT_SAFE_MOVE)
    zero_angles = [0.0] * 6 + [50.0]  # 6 关节归零 + 夹爪张开
    print("\n[归零] 抓取前先回零位...")
    if os.path.exists(safe_move):
        cmd = [safe_move, *[str(a) for a in zero_angles],
               "--step", "3.0", "--dt", "80", "--iface", NET_IFACE,
               "--max-wait-ms", "12000"]
        subprocess.run(cmd, check=False)
        print("[归零] 完成, 开始执行抓取序列\n")
    else:
        # safe_move 不存在时退化为直接发送归零命令
        cmd = dds_multi_joint_cmd(zero_angles)
        maybe_send(cmd, getattr(args, "sender", DEFAULT_SENDER), True)
        time.sleep(getattr(args, "dwell", 2.5))
        print("[归零] 已发送归零命令, 开始执行抓取序列\n")


def execute_steps(steps, args, safe=True):
    """逐步执行抓取序列. safe=True 时手臂路点走 safe_move_to 慢速插值."""
    import time
    hold_grip = None
    for st in steps:
        if st["type"] == "grip":
            print(f"[执行] {st['tag']} 夹爪 -> {st['value']}")
            if os.path.exists(args.sender):
                # 直接跑 arm_cmd_send 下发夹爪角度
                subprocess.run([args.sender, st["json"], NET_IFACE], check=False)
            # 后续手臂移动必须继续下发最近一次夹爪目标, 不能用夹爪反馈角覆盖.
            # 夹紧时反馈角会因物体卡住而偏大; 松开时反馈角也可能尚未到 50.
            # 若移动命令改用反馈角, 会导致夹紧力被放松或释放不彻底.
            hold_grip = float(st["value"])
            
            # 夹爪等待: 张开时等 release_dwell(确保准备好), 闭合时等 grip_dwell(确保合拢)
            if st["value"] >= GRIPPER_OPEN:
                time.sleep(getattr(args, "release_dwell", 1.0))
            else:
                time.sleep(getattr(args, "grip_dwell", 1.5))
       
        else:  # move
            slow = st.get("slow", False)
            move_step = str(getattr(args, "slow_step", 2.0) if slow else getattr(args, "step", 2.0))
            move_dt = str(getattr(args, "slow_dt", 120) if slow else getattr(args, "dt", 120))
            spd = "慢" if slow else "快"
            move_angles = list(st["angles"])
            if hold_grip is not None:
                move_angles.append(hold_grip)
            grip_note = f", 夹爪保持{hold_grip}" if hold_grip is not None else ""
            print(f"[执行] {st['tag']} {spd}速移动(步长{move_step}/间隔{move_dt}ms{grip_note}) -> {move_angles}")
            if safe and os.path.exists(args.safe_move):
                # ['../build_linux/safe_move_to', '10', '20', '30', '40', '50', '60', '70', '--step', '5.0', '--dt', '60', '--iface', 'can0']
                cmd = [args.safe_move, *[str(a) for a in move_angles],
                       "--step", move_step, "--dt", move_dt, "--iface", NET_IFACE]
                # 如果包含 max_wait_ms，就加入对应的参数，teach模式才有
                if "max_wait_ms" in st:
                    cmd += ["--max-wait-ms", str(st["max_wait_ms"])]
                if "mode" in st:
                    cmd += ["--mode", str(st["mode"])]
                # 调用 safe_move_to 执行闭环插值运动
                subprocess.run(cmd, check=False)
            else:
                cmd = dds_multi_joint_cmd(move_angles)
                maybe_send(cmd, args.sender, True)
            time.sleep(args.dwell)


def cmd_teach(args):
    """示教回放抓取: 抓取/放置点用示教关节角(最可靠), 上方点用 IK 解(真正正上方),
    抓取段做"先到正上方 -> 垂直分段下降"且慢速, 转运段快速.

    示教角为 DDS 下发角(度), angle0~angle5. 适合臂展边缘等纯坐标 IK 不稳的场景.
    """
    chain = build_chain(args)
    gq = np.array(args.grasp_joints, dtype=float)    # 抓取示教角(度)
    pq = np.array(args.place_joints, dtype=float)     # 放置示教角(度)
    h = float(args.approach)                          # 正上方接近高度(米)
    t_mode = getattr(args, "transfer_mode", 1)        # 转运段 mode: 0=小平滑(快), 1=大平滑(慢)
    appr = np.array([0.0, 0.0, -1.0])                 # 夹爪垂直向下
    gq_u = chain.mapping.dds_deg_to_urdf(gq)
    pq_u = chain.mapping.dds_deg_to_urdf(pq)
    grasp_pos = chain.fk(gq_u)[:3, 3]
    place_pos = chain.fk(pq_u)[:3, 3]

    def above(pos, seed_u, fallback_deg):
        """求某点正上方 h 处的关节角(度); IK 失败或超限则回退到抬肩近似."""
        q, info = chain.ik(pos + [0, 0, h], approach_dir=appr, seed=seed_u, n_restarts=3)
        if q is not None and chain.in_limits(q) and info["pos_err"] < 0.02:
            return np.round(chain.to_dds_deg(q), 2)
        r = fallback_deg.copy(); r[1] -= 12.0
        return np.round(r, 2)

    def vstep(pos, frac, seed_u, fallback_deg):
        """求某点正上方 h*frac 处的垂直下降中间点(度)."""
        q, info = chain.ik(pos + [0, 0, h * frac], approach_dir=appr, seed=seed_u, n_restarts=3)
        if q is not None and chain.in_limits(q) and info["pos_err"] < 0.02:
            return np.round(chain.to_dds_deg(q), 2)
        r = fallback_deg.copy(); r[1] -= 12.0 * frac
        return np.round(r, 2)

    above_g = above(grasp_pos, gq_u, gq)
    g_mid = vstep(grasp_pos, 0.5, gq_u, gq)          # 抓取下降中间点(高 5cm)
    g_low = vstep(grasp_pos, 0.25, gq_u, gq)         # 抓取下降低位点(高 2.5cm), 减少末端突然下沉
    # 抓取点也使用同一条垂直向下的 IK 解, 保证下降段理论上只改变 z.
    # 原示教角只作为初值/fallback, 避免最后一步切回示教姿态造成腕部突变。
    g_grasp = vstep(grasp_pos, 0.0, gq_u, gq)
    # 放置点在臂展边缘, 正上方 IK 会出扭腕解(J5 大幅翻转).
    # 继续使用示教姿态抬肩: J1 减 12 度, 实测约抬高 10cm, 稳定且贴近示教姿态.
    above_p = np.round(pq.copy(), 2); above_p[1] -= 12.0

    for tag, q in [("抓取点", gq), ("放置点", pq)]:
        p = chain.fk(chain.mapping.dds_deg_to_urdf(q))[:3, 3]
        print(f"[{tag}] 关节角={np.round(q,1)}  坐标=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")

    steps = [
        {"tag": "OPEN_START 抓取前先开到最大", "type": "grip", "value": GRIPPER_OPEN},
        {"tag": "PRE_GRASP 移到瓶正上方", "type": "move", "angles": list(above_g), "slow": False,
         "max_wait_ms": 10000},
        {"tag": "OPEN_CONFIRM 下降前确认全开", "type": "grip", "value": GRIPPER_OPEN},
        {"tag": "DESCEND 垂直下降到中间点", "type": "move", "angles": list(g_mid), "slow": True},
        {"tag": "DESCEND 垂直下降到低位点", "type": "move", "angles": list(g_low), "slow": True},
        {"tag": "GRASP 抓取点",           "type": "move", "angles": list(g_grasp), "slow": True},
        {"tag": "CLOSE 闭合夹爪",         "type": "grip", "value": GRIPPER_CLOSE},
        {"tag": "LIFT 垂直提起",          "type": "move", "angles": list(above_g), "slow": True,
         "max_wait_ms": 5000, "mode": t_mode},
        {"tag": "PLACE_PRE 移到框正上方", "type": "move", "angles": list(above_p), "slow": False,
         "max_wait_ms": 10000, "mode": t_mode},
        {"tag": "PLACE 垂直放入框内",      "type": "move", "angles": list(np.round(pq, 2)), "slow": True},
        {"tag": "RELEASE 松开夹爪",       "type": "grip", "value": GRIPPER_OPEN},
        {"tag": "RETREAT 垂直退回",       "type": "move", "angles": list(above_p), "slow": False,
         "max_wait_ms": 5000},
    ]
    for st in steps:
        if st["type"] == "grip":
            st["json"] = dds_single_joint_cmd(6, st["value"])
        elif not chain.in_limits(chain.mapping.dds_deg_to_urdf(np.array(st["angles"]))):
            print(f"!! [{st['tag']}] 超出关节限位: {st['angles']}")

    print("\n==== 示教抓取序列(先正上方->垂直下降, 抓取段慢) ====")
    for st in steps:
        if st["type"] == "move":
            print(f"{st['tag']} [{'慢' if st.get('slow') else '快'}]: {st['angles']}")
        else:
            print(f"{st['tag']}: 夹爪 -> {st['value']}")

    if args.send_safe or args.send:
        execute_steps(steps, args, safe=args.send_safe)


def add_common(p):
    p.add_argument("--urdf", default=DEFAULT_URDF, help="URDF 文件路径")
    p.add_argument("--tcp", type=float, default=0.12, help="腕部到夹爪夹持中心的偏移(米, 沿末端 z)")
    p.add_argument("--sign", nargs=6, help="6 关节 URDF->DDS 符号(默认全 1)")
    p.add_argument("--offset", nargs=6, help="6 关节零位偏移(度, 默认全 0)")
    p.add_argument("--sender", default=DEFAULT_SENDER, help="C++ DDS 发布器路径")
    p.add_argument("--send", action="store_true", help="解算后直接通过 DDS 下发")


def main():
    ap = argparse.ArgumentParser(description="D1 机械臂逆解/抓取规划工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("solve", help="单点逆解")
    ps.add_argument("--xyz", nargs=3, type=float, required=True, help="目标位置 x y z(米)")
    ps.add_argument("--rpy", nargs=3, type=float, default=None, help="完整目标姿态 roll pitch yaw(度), 约束 6 自由度")
    ps.add_argument("--approach-dir", nargs=3, type=float, default=None,
                    help="夹爪接近方向向量(世界系), 如 0 0 -1 表示顶部向下; 优先于 --rpy")
    add_common(ps)
    ps.set_defaults(func=cmd_solve)

    pg = sub.add_parser("grasp", help="抓取-放置序列规划")
    pg.add_argument("--bottle", nargs=3, type=float, required=True, help="瓶子位置 x y z(米)")
    pg.add_argument("--basket", nargs=3, type=float, required=True, help="收纳框放置位置 x y z(米)")
    pg.add_argument("--approach-dir", nargs=3, type=float, default=[0.0, 0.0, -1.0],
                    help="夹爪接近方向向量(世界系), 默认 0 0 -1 顶部向下; 水平抓取用 1 0 0")
    pg.add_argument("--approach", type=float, default=0.15, help="抓取/放置点上方接近高度(米)")
    pg.add_argument("--dwell", type=float, default=1.0, help="执行时每步间隔(秒), --send 模式下需足够长让机械臂到位")
    pg.add_argument("--grip-dwell", type=float, default=1.5, help="夹爪闭合后等待时间(秒), 确保合拢后再抬起")
    pg.add_argument("--step", type=float, default=5.0, help="运动步长(度/步, 越大越快)")
    pg.add_argument("--dt", type=int, default=60, help="步间间隔(毫秒, 越小越快)")
    pg.add_argument("--out", default=None, help="把命令序列写入 JSON 文件")
    pg.add_argument("--transfer-mode", type=int, default=1, choices=[0, 1],
                    help="转运段(LIFT->PLACE_PRE)的 funcode2 mode: 0=小平滑(快), 1=大平滑(慢, 默认)")
    pg.add_argument("--send-safe", action="store_true",
                    help="慢速安全执行: 手臂路点走 safe_move_to 小步插值(推荐真机用)")
    pg.add_argument("--no-fix-wrist", dest="fix_wrist", action="store_false", default=True,
                    help="关闭腕部固定: 竖直向下时不约束 angle5, 允许 IK 自由选择(默认开启固定)")
    pg.add_argument("--safe-move", default=DEFAULT_SAFE_MOVE, help="safe_move_to 可执行路径")
    add_common(pg)
    pg.set_defaults(func=cmd_grasp)

    pt = sub.add_parser("teach", help="示教回放抓取(直接用示教关节角, 最稳)")
    pt.add_argument("--grasp-joints", nargs=6, type=float, required=True, help="抓取示教角(度) angle0~angle5")
    pt.add_argument("--place-joints", nargs=6, type=float, required=True, help="放置示教角(度) angle0~angle5")
    pt.add_argument("--approach", type=float, default=0.15, help="抓取/释放点上方接近高度(米, 默认0.10)")
    pt.add_argument("--dwell", type=float, default=0.5, help="执行时每步间隔(秒)")
    pt.add_argument("--release-dwell", type=float, default=1.0,
                    help="松开夹爪后等待时间(秒), 确保 angle6 开到 50 再退回")
    pt.add_argument("--step", type=float, default=6.0, help="转运段步长(度/步, 越大越快, 默认6)")
    pt.add_argument("--dt", type=int, default=50, help="转运段步间间隔(毫秒, 默认50)")
    pt.add_argument("--slow-step", type=float, default=1.5, help="抓取段步长(度/步, 默认1.5, 慢且稳)")
    pt.add_argument("--slow-dt", type=int, default=120, help="抓取段步间间隔(毫秒, 默认120)")
    pt.add_argument("--transfer-mode", type=int, default=1, choices=[0, 1],
                    help="转运段(LIFT->PLACE_PRE)的 funcode2 mode: 0=小平滑(快), 1=大平滑(慢, 默认)")
    pt.add_argument("--send-safe", action="store_true", help="慢速安全执行(推荐)")
    pt.add_argument("--safe-move", default=DEFAULT_SAFE_MOVE, help="safe_move_to 可执行路径")
    add_common(pt)
    pt.set_defaults(func=cmd_teach)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
