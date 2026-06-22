"""一键运行评估 episode：启动仿真 → 跑任务 → 录制视频 → 采集指标。

用法：
    python -m scripts.run_eval_episode --config sim_go2_d1 --record-video
    python -m scripts.run_eval_episode --config sim_go2_d1 --target-x 8 --target-y 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import setup_logging

setup_logging("eval_episode", "logs/eval_episode.log")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DogTaskSim 评估 Episode 运行器")
    parser.add_argument("--config", default="sim_go2_d1", help="机器人配置 ID 或 .toml 路径")
    parser.add_argument("--target-x", type=float, default=None, help="导航目标 X 坐标")
    parser.add_argument("--target-y", type=float, default=None, help="导航目标 Y 坐标")
    parser.add_argument(
        "--scene", default=None,
        help="任务场景: lawn_debris / golf_ball / rain_inspect / material_drop",
    )
    parser.add_argument("--record-video", action="store_true", help="录制视频")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认 logs/eval_episodes/<timestamp>/）")
    parser.add_argument("--max-duration", type=float, default=180.0, help="最大运行时间（秒）")
    parser.add_argument("--headless", action="store_true", default=True, help="无头模式（默认）")
    parser.add_argument("--gui", action="store_true", help="GUI 模式（需要显示器）")
    return parser.parse_args()


class EpisodeRunner:
    """评估 Episode 运行器。

    负责：
    1. 初始化仿真场景
    2. 运行 FSM 任务
    3. 录制视频（可选）
    4. 采集指标
    5. 保存结果
    """

    def __init__(self, config_path: str, output_dir: Path, record_video: bool = True):
        self.config_path = config_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_video = record_video

        # 数据采集
        self.episode_meta: dict[str, Any] = {}
        self.fsm_timeline: list[dict] = []
        self.nav_trajectory: list[dict] = []
        self.grasp_log: list[dict] = []
        self.joint_states: list[dict] = []

        # 组件
        self.scene = None
        self.recorder = None
        self.start_time = None

    def setup(self):
        """初始化仿真场景和录制器。"""
        import os
        os.environ["MUJOCO_GL"] = "egl" if not args.gui else "glfw"

        from execution.sim_mujoco.scene import SimulationScene

        render_mode = "gui" if args.gui else "headless"
        self.scene = SimulationScene(self.config_path, render_mode=render_mode)

        if self.record_video:
            from scripts.record_video import MultiViewRecorder
            # 复用仿真相机的 GL context（headless EGL 下必须，否则重复创建失败）
            shared_ctx = None
            try:
                shared_ctx = self.scene.camera._context
            except AttributeError:
                pass
            self.recorder = MultiViewRecorder(
                self.scene.robot.model,
                self.scene.robot.data,
                self.output_dir,
                context=shared_ctx,
            )
            logger.info("视频录制已启用 (shared GL context=%s)", shared_ctx is not None)

        # 记录 episode 元数据
        self.episode_meta = {
            "config": self.config_path,
            "start_time": datetime.now().isoformat(),
            "git_hash": self._get_git_hash(),
            "record_video": self.record_video,
        }

    def _get_git_hash(self) -> str:
        """获取当前 git commit hash。"""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def run_task(self, target_x: float, target_y: float, max_duration: float = 180.0,
                 home_x: float = 0.0, home_y: float = -10.0, scene: str | None = None):
        """运行完整的 FSM 任务。

        Args:
            target_x: 导航目标 X
            target_y: 导航目标 Y
            max_duration: 最大运行时间（秒）
            home_x/home_y: 返航目标
            scene: 任务场景（lawn_debris/golf_ball/rain_inspect/material_drop）
        """
        import threading

        self.start_time = time.time()
        self.episode_meta["target"] = {"x": target_x, "y": target_y}
        self.episode_meta["home"] = {"x": home_x, "y": home_y}
        if scene:
            self.episode_meta["scene"] = scene

        # 启动仿真
        self.scene.start()
        time.sleep(0.5)  # 等待仿真稳定

        # 激活任务场景（重定位 body + 切换感知器）
        if scene:
            try:
                self.scene.configure_scene(scene, target_x, target_y, home_x, home_y)
                logger.info("场景已激活: %s", scene)
            except Exception as e:
                logger.error("场景激活失败: %s", e)

        # 采集初始状态
        self._capture_state("task_start")

        # 启动导航（场景化停靠点：巡检场景后撤避免压在目标上方）
        nav_dwell = 0.0
        if scene == "rain_inspect":
            nav_dwell = 0.6
        if nav_dwell > 0:
            import math as _math
            dx = target_x - home_x
            dy = target_y - home_y
            dist = _math.hypot(dx, dy)
            if dist > nav_dwell:
                ux, uy = dx / dist, dy / dist
                stop_x = target_x - ux * nav_dwell
                stop_y = target_y - uy * nav_dwell
                logger.info("开始导航: 停靠点 (%.2f, %.2f) 距目标 %.2fm (dwell=%.2f)",
                            stop_x, stop_y, nav_dwell, nav_dwell)
                self.scene.navigate_to(stop_x, stop_y)
            else:
                logger.info("开始导航: target=(%.2f, %.2f) (dwell 太小，直达)", target_x, target_y)
                self.scene.navigate_to(target_x, target_y)
        else:
            logger.info("开始导航: target=(%.2f, %.2f)", target_x, target_y)
            self.scene.navigate_to(target_x, target_y)
        self._capture_state("nav_start")

        # 等待到达或超时
        nav_start = time.time()
        while time.time() - nav_start < max_duration:
            state = self.scene.state
            nav_state = state["nav_state"]

            # 采集状态
            self._capture_state("navigating")

            # 录制帧
            if self.recorder:
                self.recorder.capture_frame()

            # 检查是否到达
            if nav_state.value == "arrived":
                logger.info("导航到达: pos=(%.2f, %.2f)", state["base_pos"][0], state["base_pos"][1])
                self._capture_state("nav_arrived")
                break

            # 检查是否出错
            if nav_state.value == "error":
                logger.error("导航出错")
                self._capture_state("nav_error")
                break

            time.sleep(0.033)  # ~30fps

        # 到达后朝向对齐（纯原地旋转，不前进，避免过近遮挡球体）
        if self.scene.state["nav_state"].value == "arrived":
            import math as _math
            robot_pos = self.scene.state["base_pos"]
            goal_heading = _math.atan2(target_y - robot_pos[1], target_x - robot_pos[0])
            logger.info("到达后朝向对齐: goal_heading=%.2f°", _math.degrees(goal_heading))
            self.scene.start_heading_align(goal_heading)
            align_start = time.time()
            ALIGN_TIMEOUT = 8.0
            while time.time() - align_start < ALIGN_TIMEOUT:
                align_state = self.scene.state
                self._capture_state("heading_align")
                if self.recorder:
                    self.recorder.capture_frame()
                if align_state["nav_state"].value == "arrived":
                    h_err = abs(align_state["base_yaw"] - goal_heading)
                    h_err = min(h_err, 2 * _math.pi - h_err)
                    logger.info("朝向对齐完成: heading_error=%.1f°", _math.degrees(h_err))
                    break
                time.sleep(0.033)
            else:
                h_err = abs(self.scene.state["base_yaw"] - goal_heading)
                h_err = min(h_err, 2 * _math.pi - h_err)
                logger.warning("朝向对齐超时 (%.1fs): heading_error=%.1f°, 继续抓取", ALIGN_TIMEOUT, _math.degrees(h_err))

        # 到达后启动抓取/作业
        if self.scene.state["nav_state"].value == "arrived":
            logger.info("启动作业 (scene=%s)", scene or "lawn_debris")
            self.scene.enable_detect()
            self.scene.start_grasp()
            self._capture_state("grasp_start")

            # 等待作业完成（超时给多目标场景更长时间）
            task_timeout = 240.0 if scene == "golf_ball" else 60.0
            grasp_start = time.time()
            task_timed_out = False
            while time.time() - grasp_start < task_timeout:
                state = self.scene.state

                # 采集状态
                self._capture_state("grasping")

                # 录制帧
                if self.recorder:
                    self.recorder.capture_frame()

                # 检查作业是否结束（algo 线程退出）
                if not self.scene._algo_running:
                    time.sleep(0.2)  # 等待 task_result 更新
                    state = self.scene.state
                    task_result = state.get("task_result")
                    grasp_state = state["grasp_state"]
                    # 场景化成功判定
                    scene_ok = False
                    if task_result:
                        scene_ok = task_result.get("outcome") in ("success", "partial")
                    elif grasp_state == "success":
                        scene_ok = True
                    if scene_ok:
                        msg = (task_result or {}).get("message", "作业成功")
                        logger.info("作业成功: %s", msg)
                        self._capture_state("grasp_success")
                    else:
                        logger.warning("作业失败: grasp_state=%s task_result=%s",
                                      grasp_state, task_result)
                        self._capture_state("grasp_failed")
                    break

                time.sleep(0.033)
            else:
                # 作业超时：等待 algo 线程最多 30s 自然完成（避免截断 task_result）
                task_timed_out = True
                logger.warning("作业超时 (>%ss)，等待 algo 线程收尾...", task_timeout)
                drain_start = time.time()
                while time.time() - drain_start < 30.0:
                    if not self.scene._algo_running:
                        time.sleep(0.3)
                        break
                    time.sleep(0.5)
                if self.scene._algo_running:
                    logger.warning("algo 线程仍在运行，强制停止")
                    self.scene.stop_all()
                    time.sleep(0.5)
                self._capture_state("grasp_failed")

        # 判断作业是否成功（场景化）
        cur_state = self.scene.state
        task_result = cur_state.get("task_result")
        task_succeeded = False
        if task_result:
            outcome = task_result.get("outcome")
            # success / partial 都算作业成功（partial = golf_ball 部分回收）
            task_succeeded = outcome in ("success", "partial")
        elif cur_state["grasp_state"] == "success":
            task_succeeded = True
        if task_timed_out and not task_succeeded:
            logger.warning("作业超时且无 task_result，标记为失败")

        # 作业成功后返回起点
        if task_succeeded:
            # 释放夹爪 + 收回手臂，避免约束干扰返航
            logger.info("准备返航: 释放夹爪, 收回手臂")
            self.scene.robot._gripper_closed = False
            self.scene.robot._algo_arm_target = None
            time.sleep(3.0)  # 等待手臂归位 + 球落地稳定 + 站立稳定

            logger.info("开始返回起点: (%.2f, %.2f)", home_x, home_y)
            self.scene.navigate_to(home_x, home_y)
            self._capture_state("return_start")

            # 等待返回
            return_start = time.time()
            while time.time() - return_start < max_duration:
                state = self.scene.state
                nav_state = state["nav_state"]

                # 采集状态
                self._capture_state("returning")

                # 录制帧
                if self.recorder:
                    self.recorder.capture_frame()

                # 检查是否到达
                if nav_state.value == "arrived":
                    logger.info("返回到达")
                    self._capture_state("return_arrived")
                    break

                time.sleep(0.033)

        # 记录结束时间
        self.episode_meta["end_time"] = datetime.now().isoformat()
        self.episode_meta["duration_s"] = time.time() - self.start_time

        # 计算最终状态
        final_state = self.scene.state
        self.episode_meta["final_state"] = {
            "nav_state": final_state["nav_state"].value,
            "grasp_state": final_state["grasp_state"],
            "base_pos": final_state["base_pos"],
            "task_result": final_state.get("task_result"),
        }

        # 判断任务是否成功（场景化）
        self.episode_meta["success"] = (
            task_succeeded and
            final_state["nav_state"].value == "arrived"
        )
        self.episode_meta["task_succeeded"] = task_succeeded

        logger.info("Episode 结束: success=%s, task_succeeded=%s, duration=%.1fs",
                    self.episode_meta["success"], task_succeeded,
                    self.episode_meta["duration_s"])

    def _capture_state(self, phase: str):
        """采集当前状态。"""
        state = self.scene.state
        timestamp = time.time() - self.start_time if self.start_time else 0

        # FSM 时间线
        self.fsm_timeline.append({
            "timestamp": timestamp,
            "phase": phase,
            "nav_state": state["nav_state"].value,
            "grasp_state": state["grasp_state"],
            "base_pos": state["base_pos"],
            "base_yaw": state["base_yaw"],
        })

        # 导航轨迹
        if phase in ("navigating", "nav_arrived", "heading_align", "returning", "return_arrived"):
            self.nav_trajectory.append({
                "timestamp": timestamp,
                "position": state["base_pos"],
                "yaw": state["base_yaw"],
                "target": state["nav_target"],
            })

        # 抓取日志
        if phase in ("grasping", "grasp_success", "grasp_failed"):
            self.grasp_log.append({
                "timestamp": timestamp,
                "phase": phase,
                "grasp_state": state["grasp_state"],
                "arm_positions": state["arm_positions"],
            })

        # 关节状态
        self.joint_states.append({
            "timestamp": timestamp,
            "phase": phase,
            "joint_positions": state["joint_positions"],
            "arm_positions": state["arm_positions"],
        })

    def save_results(self):
        """保存所有结果到文件。"""
        # 保存视频
        video_paths = {}
        if self.recorder:
            video_paths = self.recorder.save()
            logger.info("视频已保存: %s", list(video_paths.values()))

        # 保存元数据
        self.episode_meta["video_paths"] = {k: str(v) for k, v in video_paths.items()}
        with open(self.output_dir / "episode_meta.json", "w") as f:
            json.dump(self.episode_meta, f, indent=2, ensure_ascii=False)

        # 保存 FSM 时间线
        with open(self.output_dir / "fsm_timeline.json", "w") as f:
            json.dump(self.fsm_timeline, f, indent=2, ensure_ascii=False)

        # 保存导航轨迹
        with open(self.output_dir / "nav_trajectory.json", "w") as f:
            json.dump(self.nav_trajectory, f, indent=2, ensure_ascii=False)

        # 保存抓取日志
        with open(self.output_dir / "grasp_log.json", "w") as f:
            json.dump(self.grasp_log, f, indent=2, ensure_ascii=False)

        # 保存关节状态
        with open(self.output_dir / "joint_states.json", "w") as f:
            json.dump(self.joint_states, f, indent=2, ensure_ascii=False)

        logger.info("所有结果已保存到: %s", self.output_dir)

    def cleanup(self):
        """清理资源。"""
        if self.scene:
            self.scene.stop()


def main():
    global args
    args = parse_args()

    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = PROJECT_ROOT / "logs" / "eval_episodes" / timestamp

    logger.info("=" * 60)
    logger.info("DogTaskSim 评估 Episode")
    logger.info("配置: %s", args.config)
    logger.info("目标: (%s, %s)", args.target_x, args.target_y)
    logger.info("场景: %s", args.scene or "(默认 lawn_debris)")
    logger.info("录制视频: %s", args.record_video)
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 60)

    # 解析目标坐标
    target_x = args.target_x
    target_y = args.target_y

    # 如果未指定目标，从配置文件读取
    if target_x is None or target_y is None:
        from scheduler.config import load_robot_config
        config = load_robot_config(args.config)
        target_x = target_x or config.target_x
        target_y = target_y or config.target_y
        home_x = config.home_x
        home_y = config.home_y
        logger.info("从配置读取目标: (%.2f, %.2f), 返航: (%.2f, %.2f)", target_x, target_y, home_x, home_y)

    # 场景：命令行 > 配置 > 默认
    scene = args.scene
    if scene is None:
        from scheduler.config import load_robot_config
        _cfg_for_scene = load_robot_config(args.config)
        scene = _cfg_for_scene.task_scene
    logger.info("使用场景: %s", scene)

    # 运行 episode
    runner = EpisodeRunner(args.config, output_dir, args.record_video)
    try:
        runner.setup()
        # 从配置读取 home 坐标
        from scheduler.config import load_robot_config
        _cfg = load_robot_config(args.config)
        runner.run_task(target_x, target_y, args.max_duration,
                        home_x=_cfg.home_x, home_y=_cfg.home_y, scene=scene)
        runner.save_results()

        # 运行分析器
        logger.info("运行分析器...")
        from scripts.analyze_episode import EpisodeAnalyzer
        analyzer = EpisodeAnalyzer(output_dir)
        analyzer.run_all()

        # 输出总结
        logger.info("=" * 60)
        logger.info("Episode 评估完成")
        logger.info("成功: %s", runner.episode_meta.get("success"))
        logger.info("耗时: %.1fs", runner.episode_meta.get("duration_s", 0))
        logger.info("结果目录: %s", output_dir)
        logger.info("=" * 60)

        return 0 if runner.episode_meta.get("success") else 1

    except Exception as e:
        logger.error("Episode 运行失败: %s", e, exc_info=True)
        return 2
    finally:
        runner.cleanup()


if __name__ == "__main__":
    sys.exit(main())
