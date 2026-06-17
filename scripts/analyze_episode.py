"""分层分析器：对评估 episode 进行多维度数值分析。

用法：
    python -m scripts.analyze_episode --run-dir logs/eval_episodes/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EpisodeAnalyzer:
    """评估 Episode 分析器。

    运行五层分析：
    1. FSM 流程分析
    2. 导航质量分析
    3. 抓取质量分析
    4. 关节跟踪分析
    5. 稳定性分析
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.results: dict[str, Any] = {}

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载 episode 数据文件。"""
        # 元数据
        meta_path = self.run_dir / "episode_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        # FSM 时间线
        fsm_path = self.run_dir / "fsm_timeline.json"
        if fsm_path.exists():
            with open(fsm_path) as f:
                self.fsm_timeline = json.load(f)
        else:
            self.fsm_timeline = []

        # 导航轨迹
        nav_path = self.run_dir / "nav_trajectory.json"
        if nav_path.exists():
            with open(nav_path) as f:
                self.nav_trajectory = json.load(f)
        else:
            self.nav_trajectory = []

        # 抓取日志
        grasp_path = self.run_dir / "grasp_log.json"
        if grasp_path.exists():
            with open(grasp_path) as f:
                self.grasp_log = json.load(f)
        else:
            self.grasp_log = []

        # 关节状态
        joint_path = self.run_dir / "joint_states.json"
        if joint_path.exists():
            with open(joint_path) as f:
                self.joint_states = json.load(f)
        else:
            self.joint_states = []

    def analyze_fsm_flow(self) -> dict:
        """分析 FSM 状态转换流程。

        Returns:
            dict: FSM 分析结果
        """
        result = {
            "total_phases": len(self.fsm_timeline),
            "phase_sequence": [],
            "phase_durations": {},
            "state_transitions": [],
            "errors": [],
        }

        if not self.fsm_timeline:
            result["errors"].append("无 FSM 时间线数据")
            return result

        # 提取阶段序列
        prev_time = 0
        for entry in self.fsm_timeline:
            phase = entry["phase"]
            timestamp = entry["timestamp"]

            result["phase_sequence"].append(phase)

            # 计算阶段持续时间
            if phase not in result["phase_durations"]:
                result["phase_durations"][phase] = 0
            result["phase_durations"][phase] += timestamp - prev_time

            # 记录状态转换
            if result["phase_sequence"] and len(result["phase_sequence"]) > 1:
                prev_phase = result["phase_sequence"][-2]
                result["state_transitions"].append({
                    "from": prev_phase,
                    "to": phase,
                    "timestamp": timestamp,
                })

            prev_time = timestamp

        # 检查是否有错误状态
        error_phases = [p for p in result["phase_sequence"] if "error" in p.lower()]
        if error_phases:
            result["errors"].append(f"检测到错误状态: {error_phases}")

        # 计算总时长
        if self.fsm_timeline:
            result["total_duration_s"] = self.fsm_timeline[-1]["timestamp"]

        logger.info("FSM 分析完成: %d 个阶段, 总时长 %.1fs",
                    result["total_phases"], result.get("total_duration_s", 0))
        return result

    def analyze_navigation(self) -> dict:
        """分析导航质量。

        Returns:
            dict: 导航分析结果
        """
        result = {
            "trajectory_points": len(self.nav_trajectory),
            "forward_arrival_error_m": None,
            "return_arrival_error_m": None,
            "heading_error_deg": None,
            "path_length_m": 0,
            "forward_path_m": 0,
            "return_path_m": 0,
            "path_efficiency": None,
            "max_heading_error_deg": 0,
            "smoothness": None,
            "errors": [],
        }

        if not self.nav_trajectory:
            result["errors"].append("无导航轨迹数据")
            return result

        # 计算路径长度
        positions = [np.array(p["position"]) for p in self.nav_trajectory]
        for i in range(1, len(positions)):
            result["path_length_m"] += float(np.linalg.norm(positions[i] - positions[i-1]))

        # Split trajectory into forward and return segments
        return_start_idx = None
        for entry in self.fsm_timeline:
            if entry["phase"] == "return_start":
                ts = entry["timestamp"]
                for j, p in enumerate(self.nav_trajectory):
                    if p["timestamp"] >= ts:
                        return_start_idx = j
                        break
                break

        if return_start_idx is not None and return_start_idx > 0:
            for i in range(1, return_start_idx):
                result["forward_path_m"] += float(np.linalg.norm(positions[i] - positions[i-1]))
            for i in range(return_start_idx + 1, len(positions)):
                result["return_path_m"] += float(np.linalg.norm(positions[i] - positions[i-1]))
        else:
            result["forward_path_m"] = result["path_length_m"]

        # 计算到达误差
        if self.meta.get("target"):
            target = np.array([self.meta["target"]["x"], self.meta["target"]["y"]])
            if return_start_idx is not None:
                forward_final = positions[return_start_idx]
            else:
                forward_final = positions[-1]
            result["forward_arrival_error_m"] = float(np.linalg.norm(forward_final[:2] - target))

            straight_line = float(np.linalg.norm(target - positions[0][:2]))
            if straight_line > 0:
                result["path_efficiency"] = straight_line / max(result["forward_path_m"], 0.01)

        # Return arrival error
        if self.meta.get("home"):
            home = np.array([self.meta["home"]["x"], self.meta["home"]["y"]])
            result["return_arrival_error_m"] = float(np.linalg.norm(positions[-1][:2] - home))
        elif self.meta.get("final_state", {}).get("nav_state") == "arrived":
            result["return_arrival_error_m"] = 0.0

        # 计算朝向误差（在去程到达点计算，非最终位置）
        headings = [p["yaw"] for p in self.nav_trajectory]
        if self.meta.get("target"):
            # 使用去程到达点（return_start_idx）而非轨迹末尾（home位置）
            if return_start_idx is not None and return_start_idx < len(positions):
                heading_pos = positions[return_start_idx]
                heading_yaw = headings[return_start_idx]
            else:
                # 无返程数据时，找离目标最近的轨迹点
                target_2d = np.array([self.meta["target"]["x"], self.meta["target"]["y"]])
                dists = [np.linalg.norm(p[:2] - target_2d) for p in positions]
                closest_idx = int(np.argmin(dists))
                heading_pos = positions[closest_idx]
                heading_yaw = headings[closest_idx]
            target_heading = math.atan2(
                self.meta["target"]["y"] - heading_pos[1],
                self.meta["target"]["x"] - heading_pos[0]
            )
            heading_error = abs(heading_yaw - target_heading)
            heading_error = min(heading_error, 2 * math.pi - heading_error)
            result["heading_error_deg"] = math.degrees(heading_error)

        # 计算最大朝向误差变化
        for i in range(1, len(headings)):
            delta = abs(headings[i] - headings[i-1])
            delta = min(delta, 2 * math.pi - delta)
            result["max_heading_error_deg"] = max(result["max_heading_error_deg"], math.degrees(delta))

        # 计算路径平滑性（朝向变化的标准差）
        if len(headings) > 1:
            heading_deltas = []
            for i in range(1, len(headings)):
                delta = abs(headings[i] - headings[i-1])
                delta = min(delta, 2 * math.pi - delta)
                heading_deltas.append(delta)
            result["smoothness"] = float(np.std(heading_deltas))

        logger.info("导航分析完成: 路径长度 %.2fm, 到达误差 %.3fm, 朝向误差 %.1f°",
                    result["path_length_m"],
                    result.get("forward_arrival_error_m") or 0,
                    result.get("heading_error_deg", 0))
        return result

    def analyze_grasp(self) -> dict:
        """分析抓取质量。

        Returns:
            dict: 抓取分析结果
        """
        result = {
            "grasp_phases": len(self.grasp_log),
            "ik_success": False,
            "grasp_success": False,
            "lift_success": False,
            "arm_motion_smoothness": None,
            "grasp_duration_s": None,
            "errors": [],
        }

        if not self.grasp_log:
            result["errors"].append("无抓取日志数据")
            return result

        # 检查抓取阶段
        phases = [entry["phase"] for entry in self.grasp_log]

        # IK 成功：有 grasping 阶段
        result["ik_success"] = "grasping" in phases

        # 抓取成功：有 grasp_success 阶段
        result["grasp_success"] = "grasp_success" in phases

        # 提起成功：planner execute_full_cycle 在 success 前必经 lift → park
        if result["grasp_success"]:
            result["lift_success"] = True

        # 计算抓取持续时间（从首次 'grasping' 到 'grasp_success'）
        if "grasping" in phases and "grasp_success" in phases:
            start_idx = phases.index("grasping")
            success_idx = phases.index("grasp_success")
            result["grasp_duration_s"] = (
                self.grasp_log[success_idx]["timestamp"] -
                self.grasp_log[start_idx]["timestamp"]
            )

        # 分析机械臂运动平滑性
        if len(self.grasp_log) > 1:
            arm_positions = [np.array(entry["arm_positions"]) for entry in self.grasp_log]
            velocities = []
            for i in range(1, len(arm_positions)):
                dt = self.grasp_log[i]["timestamp"] - self.grasp_log[i-1]["timestamp"]
                if dt > 0:
                    vel = np.linalg.norm(arm_positions[i] - arm_positions[i-1]) / dt
                    velocities.append(vel)
            if velocities:
                result["arm_motion_smoothness"] = float(np.std(velocities))

        logger.info("抓取分析完成: IK=%s, 抓取=%s, 提起=%s, 耗时=%.1fs",
                    result["ik_success"], result["grasp_success"],
                    result["lift_success"], result.get("grasp_duration_s") or 0)
        return result

    def analyze_joint_tracking(self) -> dict:
        """分析关节跟踪质量。

        Returns:
            dict: 关节跟踪分析结果
        """
        result = {
            "total_samples": len(self.joint_states),
            "joint_velocities_std": None,
            "max_joint_velocity": None,
            "joint_smoothness": None,
            "errors": [],
        }

        if len(self.joint_states) < 2:
            result["errors"].append("关节状态样本不足")
            return result

        # 计算关节速度
        joint_positions = [np.array(entry["joint_positions"]) for entry in self.joint_states]
        timestamps = [entry["timestamp"] for entry in self.joint_states]

        velocities = []
        for i in range(1, len(joint_positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                vel = (joint_positions[i] - joint_positions[i-1]) / dt
                velocities.append(vel)

        if velocities:
            velocities = np.array(velocities)
            result["joint_velocities_std"] = float(np.std(velocities))
            result["max_joint_velocity"] = float(np.max(np.abs(velocities)))

            # 计算平滑性（加速度的标准差）
            if len(velocities) > 1:
                accelerations = np.diff(velocities, axis=0)
                result["joint_smoothness"] = float(np.std(accelerations))

        logger.info("关节跟踪分析完成: %d 样本, 最大速度 %.3f rad/s",
                    result["total_samples"], result.get("max_joint_velocity", 0))
        return result

    def analyze_stability(self) -> dict:
        """分析机器人稳定性。

        Returns:
            dict: 稳定性分析结果
        """
        result = {
            "base_height_min": None,
            "base_height_max": None,
            "base_height_std": None,
            "tilt_angle_max": None,
            "fall_detected": False,
            "oscillation_detected": False,
            "errors": [],
        }

        if not self.fsm_timeline:
            result["errors"].append("无 FSM 时间线数据")
            return result

        # 提取基座高度
        base_positions = [np.array(entry["base_pos"]) for entry in self.fsm_timeline]
        heights = [pos[2] if len(pos) > 2 else 0 for pos in base_positions]

        if heights:
            result["base_height_min"] = float(min(heights))
            result["base_height_max"] = float(max(heights))
            result["base_height_std"] = float(np.std(heights))

            # 检测摔倒（高度突然下降）
            if result["base_height_min"] < 0.1:  # 高度低于 0.1m 可能是摔倒
                result["fall_detected"] = True

        # 检测振荡（朝向变化过快）
        headings = [entry["base_yaw"] for entry in self.fsm_timeline]
        if len(headings) > 10:
            heading_deltas = np.diff(headings)
            # 处理角度环绕
            heading_deltas = np.abs(heading_deltas)
            heading_deltas = np.minimum(heading_deltas, 2 * np.pi - heading_deltas)
            if np.max(heading_deltas) > 0.5:  # 单步变化超过 0.5 rad
                result["oscillation_detected"] = True

        logger.info("稳定性分析完成: 高度范围 [%.3f, %.3f]m, 摔倒=%s, 振荡=%s",
                    result.get("base_height_min", 0),
                    result.get("base_height_max", 0),
                    result["fall_detected"],
                    result["oscillation_detected"])
        return result

    def run_all(self) -> dict:
        """运行所有分析并保存结果。

        Returns:
            dict: 所有分析结果
        """
        logger.info("=" * 60)
        logger.info("开始分析 episode: %s", self.run_dir)
        logger.info("=" * 60)

        # 运行各层分析
        self.results["fsm_flow"] = self.analyze_fsm_flow()
        self.results["navigation"] = self.analyze_navigation()
        self.results["grasp"] = self.analyze_grasp()
        self.results["joint_tracking"] = self.analyze_joint_tracking()
        self.results["stability"] = self.analyze_stability()

        # 计算综合得分
        self.results["overall_score"] = self._calculate_overall_score()

        # 添加元数据
        self.results["run_dir"] = str(self.run_dir)
        self.results["meta"] = self.meta

        # 保存结果
        output_path = self.run_dir / "analysis_summary.json"
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info("=" * 60)
        logger.info("分析完成，综合得分: %.1f/100", self.results["overall_score"])
        logger.info("结果已保存: %s", output_path)
        logger.info("=" * 60)

        return self.results

    def _calculate_overall_score(self) -> float:
        """计算综合得分（0-100）。"""
        score = 100.0

        # FSM 流程得分 (20分)
        fsm = self.results.get("fsm_flow", {})
        if fsm.get("errors"):
            score -= 20
        elif fsm.get("total_duration_s", 0) > 180:
            score -= 10  # 超时扣分

        # 导航得分 (30分)
        nav = self.results.get("navigation", {})
        if nav.get("errors"):
            score -= 30
        else:
            # 到达误差
            fwd_err = nav.get("forward_arrival_error_m")
            ret_err = nav.get("return_arrival_error_m")
            arrival_err = max(fwd_err or 0, ret_err or 0)
            if arrival_err > 0.5:
                score -= 15
            elif arrival_err > 0.3:
                score -= 10
            elif arrival_err > 0.1:
                score -= 5

            # 朝向误差
            if nav.get("heading_error_deg") is not None:
                if nav["heading_error_deg"] > 30:
                    score -= 10
                elif nav["heading_error_deg"] > 15:
                    score -= 5

        # 抓取得分 (40分)
        grasp = self.results.get("grasp", {})
        if grasp.get("errors"):
            score -= 40
        else:
            if not grasp.get("ik_success"):
                score -= 20
            if not grasp.get("grasp_success"):
                score -= 15
            if not grasp.get("lift_success"):
                score -= 5

        # 稳定性得分 (10分)
        stability = self.results.get("stability", {})
        if stability.get("fall_detected"):
            score -= 50  # 摔倒是严重问题
        if stability.get("oscillation_detected"):
            score -= 10

        return max(0, min(100, score))


def main():
    parser = argparse.ArgumentParser(description="DogTaskSim Episode 分析器")
    parser.add_argument("--run-dir", required=True, help="Episode 运行目录")
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error("运行目录不存在: %s", run_dir)
        return 1

    analyzer = EpisodeAnalyzer(run_dir)
    results = analyzer.run_all()

    # 打印摘要
    print("\n" + "=" * 60)
    print("分析摘要")
    print("=" * 60)
    print(f"综合得分: {results['overall_score']:.1f}/100")
    print()
    print("FSM 流程:")
    print(f"  总阶段数: {results['fsm_flow'].get('total_phases', 0)}")
    print(f"  总时长: {results['fsm_flow'].get('total_duration_s', 0):.1f}s")
    print()
    print("导航:")
    print(f"  去程误差: {results['navigation'].get('forward_arrival_error_m', 'N/A')}m")
    print(f"  回程误差: {results['navigation'].get('return_arrival_error_m', 'N/A')}m")
    print(f"  朝向误差: {results['navigation'].get('heading_error_deg', 'N/A')}°")
    print(f"  路径长度: {results['navigation'].get('path_length_m', 0):.2f}m")
    print()
    print("抓取:")
    print(f"  IK 成功: {results['grasp'].get('ik_success', False)}")
    print(f"  抓取成功: {results['grasp'].get('grasp_success', False)}")
    print(f"  提起成功: {results['grasp'].get('lift_success', False)}")
    print()
    print("稳定性:")
    print(f"  摔倒检测: {results['stability'].get('fall_detected', False)}")
    print(f"  振荡检测: {results['stability'].get('oscillation_detected', False)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
