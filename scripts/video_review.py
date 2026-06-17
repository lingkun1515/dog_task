"""多模态 AI 视频审查脚本（子 Agent 隔离环境）。

作为独立子 Agent 运行，调用多模态大模型 API 审查仿真视频，
输出结构化审查报告和打分。

默认使用 Codex 内置的 mimo-v2.5 多模态模型（通过本地代理 127.0.0.1:15721）。

用法：
    python -m scripts.video_review --video logs/eval_episodes/<ts>/third_person.mp4
    python -m scripts.video_review --video <path> --meta <path>/episode_meta.json
    python -m scripts.video_review --video <path> --api openai --model gpt-4o  # 回退到 OpenAI
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# mimo-v2.5 默认配置（Codex 本地代理）
DEFAULT_MIMO_BASE_URL = "http://127.0.0.1:15721/v1"
DEFAULT_MIMO_MODEL = "mimo-v2.5"

# 打分准则（详见 scoring_rubric.md）
SCORING_RUBRIC = """
# DogTaskSim 仿真视频打分准则

## 总体判定
- **PASS**: 任务成功完成，机器人行为稳定合理
- **HIGH_RISK**: 任务可能成功，但存在明显问题需要关注
- **FAIL**: 任务失败或存在严重问题

## 10 项结构化检查

### 1. 行走稳定性 (20分)
- 四足步态是否自然？是否有拖步、滑步？
- 机身是否保持水平？倾斜角是否过大？
- 行进方向是否平滑？是否有突然转向？

### 2. 导航精度 (15分)
- 是否到达目标位置附近？
- 到达后是否对齐朝向？
- 路径是否合理？是否有不必要的绕路？

### 3. 目标准确性 (10分)
- 到达目标位置后，机械臂是否能检测到目标？
- 目标是否在相机视野内？

### 4. 机械臂运动 (15分)
- IK 求解是否成功？
- 机械臂运动是否平滑？
- 是否有奇异姿态或关节限位问题？

### 5. 抓取执行 (20分)
- 夹爪是否正确闭合？
- 是否成功抓起目标物体？
- 提起过程中物体是否掉落？

### 6. 返航能力 (10分)
- 抓取后是否能返回起点？
- 返航路径是否合理？

### 7. 碰撞检测 (5分)
- 是否有明显的穿模现象？
- 机器人是否与障碍物碰撞？

### 8. 整体协调性 (5分)
- 底盘和机械臂运动是否协调？
- 各阶段切换是否平滑？

### 9. 异常情况 (扣分项)
- 是否摔倒？(-50分)
- 是否卡死？(-30分)
- 是否有异常抖动？(-10分)

### 10. 改进建议
- 列出 3-5 项优先改进方向
"""

REVIEW_PROMPT_TEMPLATE = """你是一个机器人仿真视频审查专家。请仔细观看以下视频，并根据打分准则进行评估。

## 系统说明
这是 Go2 四足机器人 + D1 机械臂的移动抓取仿真任务。
任务流程：导航到目标位置 -> 检测目标 -> 抓取 -> 返回起点。

## 任务上下文
{meta_context}

## 打分准则
{scoring_rubric}

## 输出要求
请以 JSON 格式输出审查结果，包含以下字段：
{{
    "overall_verdict": "PASS" | "HIGH_RISK" | "FAIL",
    "total_score": 0-100,
    "scores": {{
        "walking_stability": 0-20,
        "navigation_accuracy": 0-15,
        "target_detection": 0-10,
        "arm_motion": 0-15,
        "grasp_execution": 0-20,
        "return_capability": 0-10,
        "collision_check": 0-5,
        "overall_coordination": 0-5
    }},
    "deductions": [
        {{"reason": "摔倒", "points": -50}},
        ...
    ],
    "issues_found": [
        "问题1描述",
        "问题2描述",
        ...
    ],
    "improvement_suggestions": [
        "建议1",
        "建议2",
        ...
    ],
    "detailed_analysis": "详细分析文本..."
}}

请仔细观看视频中的每一帧，特别关注：
1. 机器人的行走步态是否自然
2. 机械臂运动是否平滑
3. 抓取动作是否成功
4. 是否有碰撞或穿模
5. 整体任务流程是否顺畅
"""


def encode_video_base64(video_path: Path) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def format_meta_context(meta: dict) -> str:
    lines = []
    if "target" in meta:
        lines.append(f"- 导航目标: ({meta['target']['x']}, {meta['target']['y']})")
    if "duration_s" in meta:
        lines.append(f"- 运行时长: {meta['duration_s']:.1f}秒")
    if "success" in meta:
        lines.append(f"- 任务成功: {meta['success']}")
    if "final_state" in meta:
        fs = meta["final_state"]
        lines.append(f"- 最终状态: 导航={fs.get('nav_state')}, 抓取={fs.get('grasp_state')}")
    if "config" in meta:
        lines.append(f"- 机器人配置: {meta['config']}")
    return "\n".join(lines) if lines else "无额外上下文"


def _extract_keyframes(video_path: Path, n_frames: int = 5) -> list[str]:
    """从视频中提取 n 帧关键帧，返回 base64 JPEG 列表。"""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        frame_idx = int(total_frames * (i + 1) / (n_frames + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
    cap.release()
    return frames


def call_multimodal_api(
    video_path: Path,
    prompt: str,
    api_provider: str = "mimo",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MIMO_MODEL,
    base_url: Optional[str] = None,
) -> dict:
    """调用多模态大模型 API 审查视频。

    支持三种 provider：
    - mimo: 通过 Codex 本地代理调用 mimo-v2.5（默认）
    - openai: 直接调用 OpenAI API
    - gemini: 调用 Google Gemini API
    """
    if api_provider == "mimo":
        import openai

        effective_base_url = base_url or os.getenv("MIMO_BASE_URL", DEFAULT_MIMO_BASE_URL)
        effective_key = api_key or os.getenv("MIMO_API_KEY")
        if not effective_key:
            raise RuntimeError("MIMO_API_KEY 未设置，请在环境变量中配置")

        client = openai.OpenAI(api_key=effective_key, base_url=effective_base_url)
        frames = _extract_keyframes(video_path)

        content: list[dict] = [{"type": "text", "text": prompt}]
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"},
            })

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4000,
        )
        return {"response": response.choices[0].message.content}

    elif api_provider == "openai":
        import openai

        client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        frames = _extract_keyframes(video_path)

        content = [{"type": "text", "text": prompt}]
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"},
            })

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4000,
        )
        return {"response": response.choices[0].message.content}

    elif api_provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        gmodel = genai.GenerativeModel(model)
        video_file = genai.upload_file(path=str(video_path))
        response = gmodel.generate_content([prompt, video_file])
        return {"response": response.text}

    else:
        raise ValueError(f"不支持的 API 提供商: {api_provider}")


def parse_review_response(response_text: str) -> dict:
    import re

    try:
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            return json.loads(json_match.group())
        return {
            "overall_verdict": "UNKNOWN",
            "total_score": 0,
            "raw_response": response_text,
            "parse_error": "无法解析 JSON",
        }
    except json.JSONDecodeError as e:
        return {
            "overall_verdict": "UNKNOWN",
            "total_score": 0,
            "raw_response": response_text,
            "parse_error": str(e),
        }


def review_video(
    video_path: Path,
    meta_path: Optional[Path] = None,
    api_provider: str = "mimo",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MIMO_MODEL,
    base_url: Optional[str] = None,
) -> dict:
    meta = {}
    if meta_path and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    meta_context = format_meta_context(meta)
    prompt = REVIEW_PROMPT_TEMPLATE.format(
        meta_context=meta_context,
        scoring_rubric=SCORING_RUBRIC,
    )

    logger.info("调用 %s API (model=%s) 审查视频...", api_provider, model)
    result = call_multimodal_api(video_path, prompt, api_provider, api_key, model, base_url)

    review = parse_review_response(result.get("response", ""))
    review["video_path"] = str(video_path)
    review["meta_path"] = str(meta_path) if meta_path else None
    review["api_provider"] = api_provider
    review["model"] = model
    return review


def main():
    parser = argparse.ArgumentParser(description="DogTaskSim 视频审查")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--meta", default=None, help="元数据文件路径")
    parser.add_argument("--api", default="mimo", choices=["mimo", "openai", "gemini"], help="API 提供商")
    parser.add_argument("--api-key", default=None, help="API key (默认从环境变量读取)")
    parser.add_argument("--model", default=DEFAULT_MIMO_MODEL, help="模型名称")
    parser.add_argument("--base-url", default=None, help="API base URL (mimo 默认本地代理)")
    parser.add_argument("--output", default=None, help="输出文件路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    video_path = Path(args.video)
    meta_path = Path(args.meta) if args.meta else None

    if not video_path.exists():
        logger.error("视频文件不存在: %s", video_path)
        return 1

    review = review_video(video_path, meta_path, args.api, args.api_key, args.model, args.base_url)

    output_path = Path(args.output) if args.output else video_path.parent / "video_review.json"
    with open(output_path, "w") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)

    logger.info("审查结果已保存: %s", output_path)
    logger.info("总体判定: %s (得分: %d)", review.get("overall_verdict"), review.get("total_score", 0))

    if review.get("issues_found"):
        logger.info("发现问题:")
        for issue in review["issues_found"]:
            logger.info("  - %s", issue)

    if review.get("improvement_suggestions"):
        logger.info("改进建议:")
        for suggestion in review["improvement_suggestions"][:3]:
            logger.info("  - %s", suggestion)

    return 0


if __name__ == "__main__":
    sys.exit(main())
