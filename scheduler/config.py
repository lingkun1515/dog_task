from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path

DEFAULT_ROBOT_PORT = 9002
DEFAULT_GRASP_PORT = 5000
DEFAULT_GRASP_PATH = "/grasp"
DEFAULT_EXECUTION_PORT = 8100


@dataclass(frozen=True)
class RobotConfig:
    """机器人连接配置，由 config/robots/<编号>.toml 加载。"""

    robot_id: str
    mode: str = "real"          # "real" | "sim"
    host: str = ""              # 实机底盘 TCP 地址
    port: int = DEFAULT_ROBOT_PORT
    orin: str = ""              # 实机机械臂/抓取服务地址
    grasp_port: int = DEFAULT_GRASP_PORT
    grasp_path: str = DEFAULT_GRASP_PATH
    execution_url: str = ""     # 仿真/执行侧 HTTP 服务地址
    target_x: float = 5.0       # 目标点 X（仿真导航用）
    target_y: float = 0.0       # 目标点 Y
    home_x: float = 0.0         # 回程目标 X（充电桩/原点）
    home_y: float = -10.0       # 回程目标 Y

    # 机械臂类型（sim: piper/arx5/wx250s, real: d1）
    arm_type: str = ""
    # 实机 D1 机械臂网络地址
    arm_host: str = ""
    arm_port: int = 8088
    # 实机标定文件路径
    calibration_path: str = ""
    # 仿真 MJCF body 名称
    tcp_body: str = ""
    arm_base_body: str = ""
    # 目标物体
    target_bodies: str = ""   # 逗号分隔
    target_labels: str = ""   # 逗号分隔
    target_classes: str = ""  # YOLO 检测类别（逗号分隔）

    @property
    def grasp_url(self) -> str:
        if self.mode == "sim":
            return f"{self.execution_url}/api/grasp"
        return f"http://{self.orin}:{self.grasp_port}{self.grasp_path}"

    def video_feed_url(self, *, detect: bool = True) -> str:
        """MJPEG 视频流。"""
        if self.mode == "sim":
            return f"{self.execution_url}/api/video_feed"
        url = f"http://{self.orin}:{self.grasp_port}/video_feed"
        return f"{url}?detect=1" if detect else url

    def navigate_url(self) -> str:
        """导航接口地址。"""
        if self.mode == "sim":
            return f"{self.execution_url}/api/navigate"
        return ""

    def navigate_status_url(self) -> str:
        """导航状态查询地址。"""
        if self.mode == "sim":
            return f"{self.execution_url}/api/navigate/status"
        return ""

    def grasp_status_url(self) -> str:
        """抓取状态查询地址。"""
        if self.mode == "sim":
            return f"{self.execution_url}/api/grasp/status"
        return ""


def robots_config_dir() -> Path:
    if config_dir := os.environ.get("SCHEDULER_CONFIG_DIR"):
        return Path(config_dir)
    return Path(__file__).resolve().parent.parent / "config" / "robots"


def load_robot_config(robot_id: str) -> RobotConfig:
    """根据机器编号加载 config/robots/<robot_id>.toml。"""
    config_path = robots_config_dir() / f"{robot_id}.toml"
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到机器人配置: {config_path}")

    with config_path.open("rb") as config_file:
        data = tomllib.load(config_file)

    mode = str(data.get("mode", "real"))

    arm_type = str(data.get("arm_type", ""))
    arm_host = str(data.get("arm_host", ""))
    arm_port = int(data.get("arm_port", 8088))
    calibration_path = str(data.get("calibration_path", ""))
    tcp_body = str(data.get("tcp_body", ""))
    arm_base_body = str(data.get("arm_base_body", ""))
    target_bodies = str(data.get("target_bodies", ""))
    target_labels = str(data.get("target_labels", ""))
    target_classes = str(data.get("target_classes", ""))

    if mode == "sim":
        execution_url = data.get("execution_url")
        if not execution_url:
            raise ValueError(f"仿真模式下缺少 execution_url 字段: {config_path}")
        return RobotConfig(
            robot_id=robot_id,
            mode="sim",
            execution_url=str(execution_url),
            target_x=float(data.get("target_x", 5.0)),
            target_y=float(data.get("target_y", 0.0)),
            home_x=float(data.get("home_x", 0.0)),
            home_y=float(data.get("home_y", -10.0)),
            arm_type=arm_type,
            tcp_body=tcp_body,
            arm_base_body=arm_base_body,
            target_bodies=target_bodies,
            target_labels=target_labels,
        )

    host = data.get("host")
    orin = data.get("orin")
    if not host or not orin:
        raise ValueError(f"实机模式缺少 host 或 orin 字段: {config_path}")

    return RobotConfig(
        robot_id=robot_id,
        mode="real",
        host=str(host),
        orin=str(orin),
        port=int(data.get("port", DEFAULT_ROBOT_PORT)),
        grasp_port=int(data.get("grasp_port", DEFAULT_GRASP_PORT)),
        grasp_path=str(data.get("grasp_path", DEFAULT_GRASP_PATH)),
        arm_type=arm_type,
        arm_host=arm_host,
        arm_port=arm_port,
        calibration_path=calibration_path,
        target_classes=target_classes,
    )
