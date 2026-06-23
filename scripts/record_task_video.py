"""自主任务视频录制器：跑完整 FSM 任务 + 录制第三视角视频。

设计动机：
- AGENT.md 机制 3「视频第一性原则」要求每轮评估产出视频供人工/AI 审查。
- 之前 --record-video 在 headless EGL 环境失败（gladLoadGL error）。
- 本模块用 GLFW 隐藏窗口 + 自有 GL context，独立于主 sim server，可在
  无显示器的服务器上稳定运行（只要 GPU 驱动 + X server 可用）。

输出：
    logs/task_videos/<scene>_<YYYYMMDDTHHMMSS>.mp4   # 第三视角主视频
    logs/task_videos/<scene>_<YYYYMMDDTHHMMSS>_front.mp4  # 第一视角（可选）

用法：
    # 单场景
    python -m scripts.record_task_video --config sim_go2_d1 --scene rain_inspect
    # 全部 4 个场景
    python -m scripts.record_task_video --config sim_go2_d1 --all-scenes
    # 自定义目标点
    python -m scripts.record_task_video --scene golf_ball --target-x 12 --target-y 0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 必须在 import mujoco 前设 MUJOCO_GL=glfw（egl 在本机失败）
# 若用户已显式设置则尊重其选择
os.environ.setdefault("MUJOCO_GL", "glfw")

from utils.logging import setup_logging

setup_logging("record_task_video", "logs/record_task_video.log")
logger = logging.getLogger(__name__)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

VIDEO_OUTPUT_DIR = PROJECT_ROOT / "logs" / "task_videos"
ALL_SCENES = ("lawn_debris", "golf_ball", "rain_inspect", "material_drop", "mixed_debris")
DEFAULT_FPS = 30
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540


class TaskVideoRecorder:
    """独立的任务视频录制器：自建 GLFW 隐藏窗口 + GL context，捕获 sim 帧。

    与 scripts/record_video.py 的区别：
    - 本类完全自建 GL 上下文（GLFW 隐藏窗口），不依赖外部 viewer/camera。
    - 适合在 headless 服务器（无桌面）上独立跑评估 + 录视频。
    - 同时录制第三视角（跟踪相机）+ 第一视角（前置相机）。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        front_cam_name: str = "front_cam",
        track_body_name: str = "base_link",
        own_window: bool = False,
        existing_context: "mujoco.MjrContext | None" = None,
        annotate_fn: "callable | None" = None,
    ):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.fps = fps
        self._glfw_window = None
        self._glfw = None
        self._owns_context = False
        # 检测标注回调：annotate_fn(rgb) -> annotated_rgb。
        # 由调用方传入，封装 SimObjectDetector.annotate_frame + detect_enabled 逻辑，
        # 使 PiP 显示与 web /api/video_feed 完全一致（含检测框 + 深度 + 置信度）。
        self._annotate_fn = annotate_fn

        # GL context 三选一：
        # 1) existing_context: 直接复用（推荐，避免多 MjrContext segfault）
        # 2) own_window=True: 自建 GLFW 隐藏窗口 + MjrContext
        # 3) 默认: 假设调用方已 make_context_current，自建 MjrContext
        if existing_context is not None:
            self._context = existing_context
            logger.info("[TaskVideoRecorder] 复用现有 MjrContext")
        elif own_window:
            self._init_gl_context()
            self._scene = mujoco.MjvScene(model, maxgeom=10000)
            self._opt = mujoco.MjvOption()
            self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._owns_context = True
        else:
            import glfw as _glfw
            self._glfw = _glfw
            self._scene = mujoco.MjvScene(model, maxgeom=10000)
            self._opt = mujoco.MjvOption()
            self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._owns_context = True

        # MuJoCo 渲染对象（如果 existing_context 路径已建则跳过）
        if not hasattr(self, "_scene"):
            self._scene = mujoco.MjvScene(model, maxgeom=10000)
            self._opt = mujoco.MjvOption()

        # 第三视角（跟踪机器人）
        self._third_cam = mujoco.MjvCamera()
        self._third_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body_name)
        self._third_cam.trackbodyid = bid if bid >= 0 else 0
        self._third_cam.distance = 3.5
        self._third_cam.elevation = -20
        self._third_cam.azimuth = 130

        # 第一视角（前置相机）— 用相机原生分辨率渲染（640×480），
        # 这样 SimObjectDetector.annotate_frame 的像素坐标（基于相机内参）
        # 才能正确对齐，PiP 显示与 web /api/video_feed 完全一致。
        self._front_native_w = 640
        self._front_native_h = 480
        self._front_cam = mujoco.MjvCamera()
        front_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, front_cam_name)
        if front_cam_id >= 0:
            self._front_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._front_cam.fixedcamid = front_cam_id
        else:
            # 没有前置相机时退化为跟踪相机
            self._front_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._front_cam.trackbodyid = self._third_cam.trackbodyid
            self._front_cam.distance = 1.0
        # 第一视角的独立 MjvScene（不同分辨率）
        self._front_scene = mujoco.MjvScene(model, maxgeom=10000)

        # 帧缓冲：第三视角 + 第一视角成对存储（用于 PiP 合成）
        self._third_frames: list[np.ndarray] = []
        self._front_frames: list[np.ndarray] = []
        self._phase_labels: list[tuple[int, str]] = []  # (frame_idx, label)

        # PiP（画中画）配置：front cam 缩略图嵌入第三视角左上角
        self._pip_w = max(160, self.width // 4)   # PiP 宽度（默认主视频的 1/4）
        self._pip_h = max(90, self.height // 4)
        self._pip_x = 12                           # 左上角边距
        self._pip_y = 44                           # 留出顶部阶段标签的空间

        logger.info(
            "[TaskVideoRecorder] GL context ready, %dx%d @ %dfps",
            width, height, fps,
        )

    def _init_gl_context(self) -> None:
        """创建 GLFW 隐藏窗口并设为当前 GL context。"""
        import glfw

        if not glfw.init():
            raise RuntimeError("GLFW 初始化失败（无 X server？检查 DISPLAY 环境变量）")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(self.width, self.height, "offscreen", None, None)
        if not window:
            raise RuntimeError("GLFW 隐藏窗口创建失败")
        glfw.make_context_current(window)
        self._glfw_window = window
        self._glfw = glfw
        logger.info("[TaskVideoRecorder] GLFW 隐藏窗口已创建（offscreen render）")

    def _render_view(
        self,
        cam: mujoco.MjvCamera,
        *,
        scene: mujoco.MjvScene | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray | None:
        """渲染指定相机视图，返回 RGB (H, W, 3) uint8。

        默认用主 scene + 录制分辨率；front cam 传 scene=self._front_scene +
        原生分辨率（640×480）以对齐检测器内参。
        """
        scn = scene if scene is not None else self._scene
        w = width if width is not None else self.width
        h = height if height is not None else self.height
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(
            self.model, self.data, self._opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, scn,
        )
        err = mujoco.mjr_render(viewport, scn, self._context)
        if err:
            return None
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, viewport, self._context)
        return np.flipud(rgb)

    def _render_front_annotated(self) -> np.ndarray | None:
        """渲染第一视角（原生 640×480）并叠加检测框（与 web 端一致）。"""
        front = self._render_view(
            self._front_cam,
            scene=self._front_scene,
            width=self._front_native_w,
            height=self._front_native_h,
        )
        if front is None:
            return None
        # 叠加检测框（如果调用方提供了 annotate_fn）
        if self._annotate_fn is not None:
            try:
                front = self._annotate_fn(front)
            except Exception as e:
                logger.debug("[front] 标注异常: %s", e)
        return front

    def capture_frame(self, phase_label: str | None = None) -> None:
        """同步捕获第三视角 + 第一视角一帧，可选附加阶段标签。

        两视角成对存储，保存时合成 PiP 单视频。
        """
        third = self._render_view(self._third_cam)
        front = self._render_front_annotated()
        if third is not None:
            self._third_frames.append(third)
            # 第一视角缺失时用黑帧占位，保持成对（注意尺寸用原生 front 分辨率）
            if front is not None:
                self._front_frames.append(front)
            else:
                self._front_frames.append(
                    np.zeros((self._front_native_h, self._front_native_w, 3), dtype=np.uint8)
                )
            if phase_label:
                self._phase_labels.append((len(self._third_frames) - 1, phase_label))

    def _draw_phase_overlay(self, frame: np.ndarray, idx: int) -> np.ndarray:
        """在帧上叠加阶段标签 + 时间戳 + 帧号。"""
        ts = idx / self.fps if self.fps > 0 else 0
        # 当前阶段
        current_phase = ""
        for fidx, label in self._phase_labels:
            if fidx <= idx:
                current_phase = label
            else:
                break
        # 左上角：阶段 + 时间
        text1 = f"[{current_phase or 'running'}]  t={ts:.1f}s"
        cv2.rectangle(frame, (0, 0), (470, 36), (0, 0, 0), -1)
        cv2.putText(frame, text1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        # 右下角：帧号
        cv2.rectangle(frame, (self.width - 130, self.height - 30),
                      (self.width, self.height), (0, 0, 0), -1)
        cv2.putText(frame, f"#{idx:05d}", (self.width - 120, self.height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    def _compose_pip(self, third: np.ndarray, front: np.ndarray) -> np.ndarray:
        """把 front cam 缩略图嵌入 third 的左上角（画中画）。

        third/front 均为 RGB (H, W, 3) uint8。原地修改 third 并返回。
        """
        # 缩放 front 到 PiP 尺寸
        pip = cv2.resize(front, (self._pip_w, self._pip_h), interpolation=cv2.INTER_AREA)
        # PiP 边框（白色 2px + 黑色外描边，让任何背景下都可见）
        cv2.rectangle(third,
                      (self._pip_x - 2, self._pip_y - 2),
                      (self._pip_x + self._pip_w + 2, self._pip_y + self._pip_h + 2),
                      (0, 0, 0), -1)
        cv2.rectangle(third,
                      (self._pip_x, self._pip_y),
                      (self._pip_x + self._pip_w, self._pip_y + self._pip_h),
                      (255, 255, 255), 2)
        # 贴入
        third[self._pip_y:self._pip_y + self._pip_h,
              self._pip_x:self._pip_x + self._pip_w] = pip
        # "Front Cam" 标签
        cv2.rectangle(third,
                      (self._pip_x, self._pip_y + self._pip_h),
                      (self._pip_x + 100, self._pip_y + self._pip_h + 18),
                      (0, 0, 0), -1)
        cv2.putText(third, "Front Cam", (self._pip_x + 4, self._pip_y + self._pip_h + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return third

    def save(self, out_path: Path, *, include_front: bool = True) -> dict[str, Path]:
        """保存合成视频（第三视角 + front cam PiP 左上角）。

        返回 {"video": Path}。include_front=False 时退化为只保存第三视角。
        """
        result: dict[str, Path] = {}
        if not self._third_frames:
            logger.warning("无帧可保存")
            return result

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (self.width, self.height))

        n = len(self._third_frames)
        has_front = include_front and len(self._front_frames) == n
        for i in range(n):
            frame = self._third_frames[i].copy()
            if has_front:
                frame = self._compose_pip(frame, self._front_frames[i])
            frame = self._draw_phase_overlay(frame, i)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        result["video"] = out_path
        logger.info("[save] 合成视频: %s (%d 帧, PiP=%s)", out_path, n, has_front)

        # 尝试转 H.264 提升兼容性
        self._try_h264_encode(out_path)
        return result

    def _try_h264_encode(self, path: Path) -> None:
        """如果系统有 ffmpeg + libx264，转码为 H.264 提升兼容性。"""
        import subprocess
        tmp = path.with_suffix(".h264.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(path),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-loglevel", "error",
                    str(tmp),
                ],
                capture_output=True, check=True,
            )
            path.unlink()
            tmp.rename(path)
            logger.info("[h264] %s 已转码为 H.264", path.name)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug("[h264] 转码跳过（%s），保留 mp4v", e)
            if tmp.exists():
                tmp.unlink()

    def close(self) -> None:
        """释放 GL 资源（仅释放自己拥有的，shared context 由 owner 释放）。"""
        if self._owns_context:
            try:
                if self._context is not None:
                    self._context.free()
            except Exception:
                pass
        try:
            if self._glfw_window is not None and self._glfw is not None:
                self._glfw.destroy_window(self._glfw_window)
                self._glfw.terminate()
        except Exception:
            pass

    @property
    def frame_count(self) -> int:
        return len(self._third_frames)


# ---------------------------------------------------------------------------
# 任务运行器：复用 SimulationScene + 手动 FSM，捕获每个阶段
# ---------------------------------------------------------------------------
def _run_task_with_recording(
    *,
    config_path: str,
    scene: str,
    target_x: float,
    target_y: float,
    home_x: float,
    home_y: float,
    out_path: Path,
    max_duration: float = 200.0,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    include_front: bool = True,
) -> dict[str, Any]:
    """跑完整任务并录制视频。返回 {success, duration_s, video_paths}。

    关键：必须**先创建 GLFW 窗口**确立 GL context，**再** import + 构造
    SimulationScene。否则 SimRGBDCamera 构造时 MjrContext 失败会污染 GL
    状态，后续渲染全部 gladLoadGL error。
    """
    import math
    import threading

    # 1. 先建 GLFW 隐藏窗口（确立当前线程的 GL context）
    import glfw
    if not glfw.init():
        raise RuntimeError("GLFW 初始化失败")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    # 不指定 CONTEXT_VERSION — 让 GLFW 选择默认（兼容性最好，
    # ARB_framebuffer_object 在大多数驱动上默认可用）
    glfw_window = glfw.create_window(width, height, "offscreen", None, None)
    if not glfw_window:
        raise RuntimeError("GLFW 隐藏窗口创建失败")
    glfw.make_context_current(glfw_window)
    logger.info("[record] GLFW 隐藏窗口已创建（offscreen render）")

    # 2. 现在才 import + 构造场景。
    # 此时 GLFW window 已 make_current，SimRGBDCamera 构造时 MjrContext 会
    # 成功创建（不再是 _ok=False 的哑对象）—— 感知管线也跟着恢复。
    from execution.sim_mujoco.scene import SimulationScene, SCENE_GOLF_BALL, SCENE_RAIN_INSPECT

    scene_obj = SimulationScene(config_path, render_mode="headless")

    # 3. 构建检测标注回调：与 web /api/video_feed 完全一致。
    # 仅当 scene._detect_enabled 为 True 时叠加检测框（与 scene loop 逻辑对齐）。
    def _annotate_fn(rgb):
        if not scene_obj._detect_enabled:
            return rgb
        detector = scene_obj._sim_detector
        if detector is None:
            return rgb
        return detector.annotate_frame(rgb)

    # 4. 复用场景相机的 MjrContext（不要建第二个！否则同一 window 上
    # 两个 MjrContext 会导致 segfault）。
    cam_ctx = scene_obj.camera._context
    if cam_ctx is None:
        # 极端情况：相机 context 仍失败，recorder 自建兜底
        recorder = TaskVideoRecorder(
            scene_obj.robot.model, scene_obj.robot.data,
            width=width, height=height, fps=fps, own_window=False,
            annotate_fn=_annotate_fn,
        )
    else:
        recorder = TaskVideoRecorder(
            scene_obj.robot.model, scene_obj.robot.data,
            width=width, height=height, fps=fps,
            own_window=False, existing_context=cam_ctx,
            annotate_fn=_annotate_fn,
        )
    recorder._glfw_window = glfw_window  # 让 close() 能正确清理

    result: dict[str, Any] = {
        "scene": scene, "config": config_path,
        "target": [target_x, target_y], "home": [home_x, home_y],
        "success": False, "duration_s": 0.0, "video_paths": {},
    }
    start = time.time()

    try:
        # 关键：禁用 scene loop 内的相机渲染，避免主线程录制 + 后台线程渲染的
        # GL 冲突。检测器在录制时走 xpos ground-truth 兜底（sim 仍准确）。
        scene_obj._render_in_loop = False
        # 启动仿真
        scene_obj.start()
        time.sleep(0.5)
        # 激活场景几何
        scene_obj.configure_scene(scene, target_x, target_y, home_x, home_y)
        recorder.capture_frame(phase_label=f"scene={scene}")

        # 场景化导航停靠点（巡检后撤）
        nav_dwell = 0.6 if scene == SCENE_RAIN_INSPECT else 0.0
        nav_x, nav_y = target_x, target_y
        if nav_dwell > 0:
            dx, dy = target_x - home_x, target_y - home_y
            dist = math.hypot(dx, dy)
            if dist > nav_dwell:
                nav_x = target_x - (dx / dist) * nav_dwell
                nav_y = target_y - (dy / dist) * nav_dwell

        # === 阶段 1：导航去程 ===
        logger.info("[%s] 导航去程 → (%.2f, %.2f)", scene, nav_x, nav_y)
        scene_obj.enable_detect()
        scene_obj.navigate_to(nav_x, nav_y)
        nav_start = time.time()
        arrived = False
        while time.time() - nav_start < max_duration:
            st = scene_obj.state
            recorder.capture_frame(phase_label="navigate_to_target")
            if st["nav_state"].value == "arrived":
                arrived = True
                break
            if st["nav_state"].value == "error":
                break
            time.sleep(1.0 / fps)
        recorder.capture_frame(phase_label="arrived_B" if arrived else "nav_failed")
        result["arrived"] = arrived
        if not arrived:
            logger.warning("[%s] 去程导航未到达", scene)

        # === 阶段 2：作业（抓取/巡检/投放） ===
        task_ok = False
        if arrived:
            logger.info("[%s] 启动作业", scene)
            scene_obj.start_grasp()
            time.sleep(0.5)
            # 等待作业完成
            task_timeout = 300.0 if scene in (SCENE_GOLF_BALL, "mixed_debris") else 90.0
            task_start = time.time()
            while time.time() - task_start < task_timeout:
                recorder.capture_frame(phase_label=f"task:{scene_obj.state['grasp_state']}")
                if not scene_obj._algo_running:
                    time.sleep(0.3)
                    break
                time.sleep(1.0 / fps)
            # 判定结果
            st = scene_obj.state
            tr = st.get("task_result")
            if tr and tr.get("outcome") in ("success", "partial"):
                task_ok = True
                label = f"task_done:{tr.get('outcome')}"
            elif st["grasp_state"] == "success":
                task_ok = True
                label = "task_done:success"
            else:
                label = f"task_failed:{st['grasp_state']}"
            recorder.capture_frame(phase_label=label)
            result["task_outcome"] = (tr or {}).get("outcome")
            result["task_message"] = (tr or {}).get("message")

        # === 阶段 3：返航 ===
        if task_ok:
            logger.info("[%s] 返航 → (%.2f, %.2f)", scene, home_x, home_y)
            scene_obj.robot._gripper_closed = False
            scene_obj.robot._algo_arm_target = None
            time.sleep(2.0)
            scene_obj.navigate_to(home_x, home_y, require_heading=True, goal_heading=0.0)
            ret_start = time.time()
            returned = False
            while time.time() - ret_start < max_duration:
                recorder.capture_frame(phase_label="return_to_dock")
                if scene_obj.state["nav_state"].value == "arrived":
                    returned = True
                    break
                time.sleep(1.0 / fps)
            recorder.capture_frame(phase_label="docked" if returned else "dock_failed")
            result["returned"] = returned
            result["success"] = returned

        # 收尾帧
        recorder.capture_frame(phase_label="FINISHED" if result["success"] else "END")
        result["duration_s"] = time.time() - start

        # 保存视频
        video_paths = recorder.save(out_path, include_front=include_front)
        result["video_paths"] = {k: str(v) for k, v in video_paths.items()}
        result["frame_count"] = recorder.frame_count
        logger.info(
            "[%s] 完成: success=%s duration=%.1fs frames=%d videos=%s",
            scene, result["success"], result["duration_s"],
            recorder.frame_count, list(video_paths.keys()),
        )
    finally:
        # 清理顺序：recorder（不释放 shared ctx）→ scene（释放 camera 的 ctx）→ glfw window
        recorder.close()
        scene_obj.stop()
        try:
            glfw.destroy_window(glfw_window)
        except Exception:
            pass
        try:
            glfw.terminate()
        except Exception:
            pass

    return result


def _gen_video_path(scene: str, out_dir: Path | None = None) -> Path:
    """生成视频文件路径：<out_dir>/<scene>_<YYYYMMDDTHHMMSS>.mp4。"""
    base = out_dir or VIDEO_OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return base / f"{scene}_{ts}.mp4"


def parse_args():
    p = argparse.ArgumentParser(description="DogTaskSim 自主任务视频录制")
    p.add_argument("--config", default="sim_go2_d1", help="机器人配置 ID")
    p.add_argument("--scene", default=None, help="单个场景（与 --all-scenes 互斥）")
    p.add_argument("--all-scenes", action="store_true", help="录制全部 4 个场景")
    p.add_argument("--target-x", type=float, default=12.0)
    p.add_argument("--target-y", type=float, default=0.0)
    p.add_argument("--home-x", type=float, default=None, help="默认从配置读取")
    p.add_argument("--home-y", type=float, default=None, help="默认从配置读取")
    p.add_argument("--max-duration", type=float, default=200.0)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--no-front", action="store_true", help="不录制第一视角")
    p.add_argument("--out-dir", default=None, help="输出目录（默认 logs/task_videos/）")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.scene and not args.all_scenes:
        logger.error("必须指定 --scene <name> 或 --all-scenes")
        return 1
    scenes = ALL_SCENES if args.all_scenes else [args.scene]
    for s in scenes:
        if s not in ALL_SCENES:
            logger.error("未知场景: %s（支持: %s）", s, ALL_SCENES)
            return 1

    # 从配置读取 home（如未显式指定）
    home_x = args.home_x
    home_y = args.home_y
    if home_x is None or home_y is None:
        from scheduler.config import load_robot_config
        cfg = load_robot_config(args.config)
        home_x = home_x if home_x is not None else cfg.home_x
        home_y = home_y if home_y is not None else cfg.home_y

    out_dir = Path(args.out_dir) if args.out_dir else VIDEO_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    overall_start = time.time()
    for i, scene in enumerate(scenes, 1):
        logger.info("")
        logger.info("#" * 60)
        logger.info("# [%d/%d] 录制场景: %s", i, len(scenes), scene)
        logger.info("#" * 60)
        video_path = _gen_video_path(scene, out_dir)
        try:
            r = _run_task_with_recording(
                config_path=args.config,
                scene=scene,
                target_x=args.target_x,
                target_y=args.target_y,
                home_x=home_x,
                home_y=home_y,
                out_path=video_path,
                max_duration=args.max_duration,
                width=args.width,
                height=args.height,
                fps=args.fps,
                include_front=not args.no_front,
            )
            results.append(r)
        except Exception as e:
            logger.error("[%s] 录制失败: %s", scene, e, exc_info=True)
            results.append({"scene": scene, "success": False, "error": str(e)})

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("录制完成汇总")
    logger.info("=" * 60)
    for r in results:
        ok = "✓" if r.get("success") else "✗"
        scene = r.get("scene", "?")
        dur = r.get("duration_s", 0)
        vp = r.get("video_paths", {})
        logger.info("  %s %-16s %.1fs  videos=%s",
                    ok, scene, dur, list(vp.values()) or "无")
    logger.info("总耗时: %.1fs, 输出目录: %s", time.time() - overall_start, out_dir)

    # 写 manifest
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "target": [args.target_x, args.target_y],
        "home": [home_x, home_y],
        "results": results,
    }
    manifest_path = out_dir / f"manifest_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    with open(manifest_path, "w") as f:
        import json
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("manifest: %s", manifest_path)

    return 0 if all(r.get("success") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
