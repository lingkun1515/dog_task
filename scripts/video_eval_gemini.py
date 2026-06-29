"""Multimodal video evaluation using Google Gemini API.

Extracts key frames from an eval episode video, sends them to Gemini for
visual analysis, and returns structured evaluation feedback.

Usage:
    python -m scripts.video_eval_gemini --video path/to/third_person.mp4 --scene golf_ball

Requires:
    - Google Gemini API key (passed via --api-key or GEMINI_API_KEY env var)
    - Network proxy if in restricted region (HTTP_PROXY/HTTPS_PROXY env vars)
    - ffmpeg on PATH
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Scene-specific evaluation prompts
SCENE_PROMPTS = {
    "golf_ball": (
        "You are evaluating a quadruped robot (Go2 + D1 arm) performing "
        "a multi-ball grasping task in MuJoCo simulation.\n\n"
        "Analyze these video frames from a third-person camera view. "
        "The robot should:\n"
        "1. Navigate to the target area where 5 white golf balls are on the ground\n"
        "2. Sit down (lower its body)\n"
        "3. Use its arm to grasp each ball one by one\n"
        "4. Lift each ball up\n\n"
        "Return JSON:\n"
        "{\n"
        '  "robot_posture": "standing/walking/fallen/tipping?",\n'
        '  "navigation_quality": "smooth/erratic/stuck?",\n'
        '  "arm_movement": "reaching balls? smooth/jerky?",\n'
        '  "grasp_visible": "arm touching/grasping any balls?",\n'
        '  "balls_status": "how many visible? on ground or held?",\n'
        '  "key_events": ["event1", "event2"],\n'
        '  "overall_assessment": "brief summary",\n'
        '  "severity": "none|minor|major|critical"\n'
        "}"
    ),
    "lawn_debris": (
        "Evaluate a quadruped robot performing debris grasping. "
        "The robot should navigate to a brown branch on the ground, "
        "sit down, and grasp it.\n\n"
        "Return JSON: robot_posture, navigation_quality, arm_movement, "
        "grasp_visible, key_events, overall_assessment, severity"
    ),
    "material_drop": (
        "Evaluate a robot performing material drop. The robot should pick "
        "up a red payload box and carry/drop it at a target location.\n\n"
        "Return JSON: robot_posture, navigation_quality, arm_movement, "
        "payload_visible, drop_visible, key_events, overall_assessment, severity"
    ),
    "rain_inspect": (
        "Evaluate a robot performing rain inspection. It should navigate to "
        "puddle areas and inspect them (no grasping).\n\n"
        "Return JSON: robot_posture, navigation_quality, inspection_visible, "
        "key_events, overall_assessment, severity"
    ),
    "mixed_debris": (
        "Evaluate a robot performing mixed debris cleanup. Multiple object "
        "types (ball, boxes, bottle, bag) need grasping.\n\n"
        "Return JSON: robot_posture, navigation_quality, arm_movement, "
        "grasp_visible, objects_visible, key_events, overall_assessment, severity"
    ),
    "_default": (
        "Evaluate a quadruped robot task in MuJoCo simulation.\n\n"
        "Return JSON: robot_posture, navigation_quality, arm_movement, "
        "key_events, overall_assessment, severity"
    ),
}


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY not set. Use --api-key or export GEMINI_API_KEY=...")
    return key


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    num_frames: int = 8,
) -> list[str]:
    video_path = str(video_path)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        total_frames = int(result.stdout.strip())
    except Exception:
        total_frames = 300

    if total_frames < num_frames:
        num_frames = max(1, total_frames)

    step = max(1, total_frames // num_frames)
    frame_indices = [str(i * step) for i in range(num_frames)]
    select_expr = "+".join("eq(n\\," + idx + ")" for idx in frame_indices)

    output_pattern = os.path.join(output_dir, "frame_%03d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "select='" + select_expr + "',scale=1280:-1",
        "-vsync", "vfr",
        "-q:v", "2",
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        logger.error("ffmpeg failed: %s", result.stderr[-500:])

    frames = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".jpg")
    )
    logger.info("Extracted %d frames from %s", len(frames), video_path)
    return frames


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_gemini(
    api_key: str,
    prompt: str,
    image_paths: list[str],
    model: str = DEFAULT_MODEL,
    timeout: int = 180,
) -> dict:
    parts = [{"text": prompt}]

    for img_path in image_paths:
        b64 = encode_image(img_path)
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64,
            }
        })

    url = API_BASE + "/models/" + model + ":generateContent?key=" + api_key

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    resp = urllib.request.urlopen(req, timeout=timeout)
    data = json.loads(resp.read().decode())

    candidates = data.get("candidates", [])
    if not candidates:
        return {"error": "No candidates in response", "raw": data}

    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "text": text}


def evaluate_episode(
    video_path: str | Path,
    scene: str,
    api_key: str | None = None,
    num_frames: int = 8,
    model: str = DEFAULT_MODEL,
) -> dict:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError("Video not found: " + str(video_path))

    if api_key is None:
        api_key = get_api_key()

    prompt = SCENE_PROMPTS.get(scene, SCENE_PROMPTS["_default"])

    with tempfile.TemporaryDirectory(prefix="gemini_eval_") as tmpdir:
        frames = extract_frames(video_path, tmpdir, num_frames=num_frames)

        if not frames:
            return {"error": "No frames extracted", "video": str(video_path)}

        logger.info("Calling Gemini %s with %d frames for scene=%s", model, len(frames), scene)

        result = call_gemini(api_key, prompt, frames, model=model)

    return {
        "video": str(video_path),
        "scene": scene,
        "model": model,
        "frames_analyzed": len(frames),
        "gemini_eval": result,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Gemini multimodal video evaluation")
    parser.add_argument("--video", required=True, help="Path to episode video")
    parser.add_argument("--scene", default="lawn_debris", help="Scene name")
    parser.add_argument("--api-key", default=None, help="Gemini API key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames to extract")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = evaluate_episode(
        args.video, args.scene,
        api_key=args.api_key,
        num_frames=args.num_frames,
        model=args.model,
    )

    output_path = args.output or str(
        Path(args.video).parent / "gemini_eval.json"
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nSaved to: " + output_path)


if __name__ == "__main__":
    main()
