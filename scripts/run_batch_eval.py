"""批量评估：连续跑多个场景，汇总对比报告。

用法：
    python -m scripts.run_batch_eval --config sim_go2_d1
    python -m scripts.run_batch_eval --config sim_go2_d1 --scenes golf_ball,rain_inspect
    python -m scripts.run_batch_eval --config sim_go2_d1 --record-video

输出：
    logs/eval_batch/<timestamp>/
        <scene1>/                 # 单场景 episode 结果（同 run_eval_episode）
        <scene2>/
        ...
        batch_summary.json        # 跨场景汇总
        batch_summary.md          # 人类可读 markdown 报告
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging import setup_logging

setup_logging("eval_batch", "logs/eval_batch.log")
logger = logging.getLogger(__name__)

ALL_SCENES = ("lawn_debris", "golf_ball", "rain_inspect", "material_drop", "mixed_debris")


def parse_args():
    p = argparse.ArgumentParser(description="DogTaskSim 批量场景评估")
    p.add_argument("--config", default="sim_go2_d1", help="机器人配置 ID")
    p.add_argument(
        "--scenes",
        default=",".join(ALL_SCENES),
        help=f"逗号分隔的场景列表（默认全部: {','.join(ALL_SCENES)}）",
    )
    p.add_argument("--target-x", type=float, default=12.0)
    p.add_argument("--target-y", type=float, default=0.0)
    p.add_argument(
        "--record-video", action="store_true", default=True,
        help="录制任务视频（默认开启；--no-record-video 关闭）",
    )
    p.add_argument(
        "--no-record-video", dest="record_video", action="store_false",
        help="禁用视频录制（只跑评估指标）",
    )
    p.add_argument("--max-duration", type=float, default=200.0, help="单场景最大运行时间")
    p.add_argument("--output-dir", default=None, help="输出目录（默认 logs/eval_batch/<timestamp>/）")
    return p.parse_args()


def _read_episode_summary(scene_dir: Path) -> dict:
    """读取单场景 episode 的关键指标。"""
    meta_path = scene_dir / "episode_meta.json"
    analysis_path = scene_dir / "analysis_summary.json"
    result: dict = {"scene_dir": str(scene_dir)}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        result["success"] = meta.get("success")
        result["task_succeeded"] = meta.get("task_succeeded")
        result["duration_s"] = meta.get("duration_s")
        result["scene"] = meta.get("scene")
        tr = (meta.get("final_state") or {}).get("task_result") or {}
        result["task_outcome"] = tr.get("outcome")
        result["task_message"] = tr.get("message")
        result["task_details"] = tr.get("details")
    if analysis_path.exists():
        with open(analysis_path) as f:
            analysis = json.load(f)
        result["overall_score"] = analysis.get("overall_score")
        nav = analysis.get("navigation", {}) or {}
        result["forward_arrival_error_m"] = nav.get("forward_arrival_error_m")
        result["return_arrival_error_m"] = nav.get("return_arrival_error_m")
        result["heading_error_deg"] = nav.get("heading_error_deg")
        result["path_length_m"] = nav.get("path_length_m")
        result["path_efficiency"] = nav.get("path_efficiency")
        stab = analysis.get("stability", {}) or {}
        result["fall_detected"] = stab.get("fall_detected")
    return result


def _write_markdown_report(summaries: list[dict], batch_dir: Path, args) -> None:
    """生成人类可读的 markdown 报告。"""
    lines = [
        "# DogTaskSim 批量场景评估报告",
        "",
        f"- **配置**: `{args.config}`",
        f"- **目标**: ({args.target_x}, {args.target_y})",
        f"- **生成时间**: {datetime.now().isoformat(timespec='seconds')}",
        f"- **场景数**: {len(summaries)}",
        "",
        "## 汇总",
        "",
        "| 场景 | 结果 | outcome | 耗时(s) | 得分 | 到达误差(m) | 朝向误差(°) | 路径(m) | 路径效率 | 摔倒 |",
        "|------|------|---------|---------|------|-------------|-------------|---------|----------|------|",
    ]
    pass_count = 0
    total_duration = 0.0
    for s in summaries:
        scene = s.get("scene", "?")
        ok = "✓" if s.get("success") else "✗"
        if s.get("success"):
            pass_count += 1
        dur = s.get("duration_s")
        total_duration += dur or 0
        outcome = s.get("task_outcome") or "-"
        msg = s.get("task_message") or ""
        score = s.get("overall_score")
        fwd = s.get("forward_arrival_error_m")
        hdg = s.get("heading_error_deg")
        path = s.get("path_length_m")
        eff = s.get("path_efficiency")
        fall = "是" if s.get("fall_detected") else "否"
        lines.append(
            f"| {scene} | {ok} | {outcome} | {dur:.1f} | {score:.0f} | "
            f"{fwd:.3f} | {hdg:.1f} | {path:.2f} | "
            f"{(eff*100):.0f}% | {fall} |"
            if isinstance(fwd, (int, float)) and isinstance(score, (int, float))
            else f"| {scene} | {ok} | {outcome} | - | - | - | - | - | - | {fall} |"
        )
    lines += [
        "",
        f"**通过率**: {pass_count}/{len(summaries)}",
        f"**总耗时**: {total_duration:.1f}s（平均 {total_duration/len(summaries):.1f}s/场景）",
        "",
        "## 各场景详情",
        "",
    ]
    for s in summaries:
        lines.append(f"### {s.get('scene', '?')}")
        lines.append(f"- 结果目录: `{s.get('scene_dir')}`")
        if s.get("video_path"):
            lines.append(f"- 任务视频: `{s['video_path']}`")
        if s.get("task_message"):
            lines.append(f"- 作业消息: {s['task_message']}")
        details = s.get("task_details") or {}
        if details:
            lines.append(f"- 作业详情: `{json.dumps(details, ensure_ascii=False)}`")
        lines.append("")

    (batch_dir / "batch_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown 报告已生成: %s", batch_dir / "batch_summary.md")


def main():
    args = parse_args()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    for s in scenes:
        if s not in ALL_SCENES:
            logger.error("未知场景: %s（支持: %s）", s, ALL_SCENES)
            return 1

    if args.output_dir:
        batch_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        batch_dir = PROJECT_ROOT / "logs" / "eval_batch" / timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DogTaskSim 批量场景评估")
    logger.info("配置: %s", args.config)
    logger.info("场景: %s", scenes)
    logger.info("输出: %s", batch_dir)
    logger.info("=" * 60)

    summaries: list[dict] = []
    overall_start = time.time()

    for i, scene in enumerate(scenes, 1):
        scene_dir = batch_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        logger.info("")
        logger.info("########## [%d/%d] 场景 %s ##########", i, len(scenes), scene)

        # 调用单场景评估（复用 run_eval_episode）
        import subprocess
        cmd = [
            sys.executable, "-m", "scripts.run_eval_episode",
            "--config", args.config,
            "--scene", scene,
            "--target-x", str(args.target_x),
            "--target-y", str(args.target_y),
            "--max-duration", str(args.max_duration),
            "--output-dir", str(scene_dir),
        ]
        if args.record_video:
            cmd.append("--record-video")
        logger.info("运行: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        logger.info("场景 %s 退出码: %d", scene, result.returncode)

        # 汇总该场景结果
        summary = _read_episode_summary(scene_dir)
        summary["scene"] = scene
        summary["exit_code"] = result.returncode
        summaries.append(summary)

        # 视频录制（默认开启）：调用 record_task_video 产出
        # logs/task_videos/<scene>_<timestamp>.mp4（PiP + 检测框）
        if args.record_video:
            logger.info("----- 场景 %s 视频录制 -----", scene)
            vid_cmd = [
                sys.executable, "-m", "scripts.record_task_video",
                "--config", args.config,
                "--scene", scene,
                "--target-x", str(args.target_x),
                "--target-y", str(args.target_y),
                "--max-duration", str(args.max_duration),
            ]
            logger.info("运行: %s", " ".join(vid_cmd))
            vid_result = subprocess.run(vid_cmd, cwd=str(PROJECT_ROOT))
            summary["video_exit_code"] = vid_result.returncode
            # 找最新的该场景视频
            import glob
            vids = sorted(glob.glob(str(PROJECT_ROOT / "logs" / "task_videos" / f"{scene}_*.mp4")))
            if vids:
                summary["video_path"] = vids[-1]
                logger.info("场景 %s 视频已保存: %s", scene, vids[-1])

        # 增量保存（中断时仍可用）
        with open(batch_dir / "batch_summary.json", "w") as f:
            json.dump({
                "config": args.config,
                "target": {"x": args.target_x, "y": args.target_y},
                "scenes": scenes,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "total_elapsed_s": time.time() - overall_start,
                "results": summaries,
            }, f, indent=2, ensure_ascii=False)

    # 生成 markdown 报告
    _write_markdown_report(summaries, batch_dir, args)

    pass_count = sum(1 for s in summaries if s.get("success"))
    logger.info("")
    logger.info("=" * 60)
    logger.info("批量评估完成: %d/%d 通过, 总耗时 %.1fs",
                pass_count, len(summaries), time.time() - overall_start)
    logger.info("报告: %s", batch_dir / "batch_summary.md")
    logger.info("=" * 60)
    return 0 if pass_count == len(summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
