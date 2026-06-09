# DogTaskSim — 机器人移动抓取任务调度与仿真系统

Go2 / Mower 底盘 + D1 / Piper 机械臂的移动抓取任务系统。提供**调度后台 → 执行侧服务**两层架构，支持**仿真（MuJoCo）与实机任务演示**。

- 调度后台集成自 https://github.com/topsun-bot/Mower
- 机械臂检测与抓取集成自 https://github.com/topsun-bot/pick_up_trash
- Go2 运控 RL policy 集成自 https://github.com/zzzJie-Robot/LeggedManip_Lab

## 快速开始

### 1. 启动仿真执行服务

```bash
# Go2 + D1
python -m execution.sim_mujoco.sim_task_server --config sim_go2_d1 --render

# 无头模式（不开 GUI 窗口）
python -m execution.sim_mujoco.sim_task_server --config sim_go2_d1 --port 8100
```

### 2. 启动调度后台

```bash
python -m scheduler.server.app
# 浏览器打开 http://localhost:8200
```

### 3. 手动测试 API

```bash
curl http://localhost:8100/health
curl -X POST http://localhost:8100/api/navigate -H "Content-Type: application/json" -d '{"x": 5.0, "y": 0}'
curl http://localhost:8100/api/navigate/status
curl -X POST http://localhost:8100/api/grasp
curl http://localhost:8100/api/detect
curl http://localhost:8100/api/state
```

### 4. 辅助工具

```bash
# 仿真标定（从 MJCF 真值计算 T_cam_to_arm）
python -m utils.calibrate_sim --config sim_go2_d1

# 实机标定（ArUco 手眼标定）
python -m utils.calibrate --arm-host 192.168.123.100

# 端到端接口测试
python -m utils.test_pipeline --url http://localhost:8100
```

## 配置说明

**所有配置集中在 `config/robots/<robot_id>.toml`**，执行端和调度端统一读取同一文件：

```
config/robots/
├── sim_go2_piper.toml   # 仿真: Go2 + Piper
├── sim_go2_d1.toml      # 仿真: Go2 + D1
└── real_mower_d1.toml   # 实机: Mower + D1
```

### Sim 配置示例 (sim_go2_d1.toml)

```toml
mode = "sim"
execution_url = "http://localhost:8100"

# 调度参数
target_x = 12.0             # 导航目标点
target_y = 0.0
home_x = 0.0                # 回程点（充电桩）
home_y = -10.0
arrival_threshold = 0.45    # 到达判定距离(m)，需 ≤ 机械臂最大可达

# 仿真参数
xml_path = "assets/go2_d1/scene.xml"
simulation_dt = 0.005
control_decimation = 4
default_angles = [...]      # 18 DOF 默认关节角(rad)
arm_rl_enabled = false      # false=手臂保持收缩, true=RL控制手臂

# 算法抓取
[algorithms]
arm_kinematics = "d1"
arm_base_body = "d1_base_link"
target_bodies = ["target_sphere"]   # MuJoCo body 名称
target_labels = ["sphere"]          # 检测标签（显示在视频帧上）
approach_height = 0.12
move_wait = 0.6                     # 仿真中每步运动等待(秒)
gripper_wait = 0.2                  # 夹爪动作等待(秒)
safe_park_angles = [0.0, 30.0, -10.0, 0.0, 0.0, 0.0]
```

### Real 配置示例 (real_mower_d1.toml)

```toml
mode = "real"
host = "10.10.170.239"      # 底盘 TCP
orin = "10.10.170.191"      # Orin (视频流/抓取服务)
port = 9002
grasp_port = 5000
arm_host = "192.168.123.100"
arm_port = 8088
calibration_path = "output/calibration_result.json"

[algorithms]
arm_kinematics = "d1"
target_classes = ["bottle"]         # YOLO 检测类别
approach_height = 0.10
descend_step = 0.03
safe_park_angles = [0.0, 30.0, -10.0, 0.0, 0.0, 0.0]
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `arrival_threshold` | 导航到达判定距离(m)，需 ≤ 机械臂最大可达距离 |
| `arm_rl_enabled` | `false`=手臂保持 default_angles 收缩，抓取时由 GraspPlanner 接管 |
| `[algorithms].target_labels` | 检测标签，影响视频流叠加标注文字和检测管道 |
| `[algorithms].arm_kinematics` | `"d1"`=URDF 解析式 IK(sim2real对齐)，`"mujoco"`=数值 IK(通用) |
| `[algorithms].move_wait` | 仿真机械臂每步运动等待时间，调小可加快抓取速度 |

## 环境变量

| 变量 | 说明 |
|------|------|
| `SCHEDULER_ROBOT_ID` | 调度服务默认机器人 ID（默认: `sim_go2_piper`） |
| `SCHEDULER_CONFIG_DIR` | 机器人配置目录（默认: `config/robots/`） |
| `MUJOCO_GL` | MuJoCo 渲染后端（默认: `egl`；`--render` 时自动切为 `glfw`） |
| `MOWER_LOG_DIR` | 任务运行日志目录（默认: `logs/runs/`） |

## 架构概览

```
浏览器 (localhost:8200)
    │ HTTP + SSE
    ▼
scheduler/server/app.py ─── FSM 状态机 ─── actions/ (sim/real)
    │                                           │
    │ 读取配置                                   │ HTTP (sim) / TCP (real)
    ▼                                           ▼
config/robots/<id>.toml  ◄────────────  execution/sim_mujoco/
                                        execution/real_robots/
```

### 目录结构

```
DogTaskSim/
├── config/robots/            # 统一配置（调度+执行共读）
├── scheduler/                # 调度后台
│   ├── server/app.py         # FastAPI Web 服务
│   ├── fsm.py                # 状态机
│   ├── config.py             # RobotConfig 加载
│   └── actions/              # 动作实现 (sim + real)
├── execution/                # 执行侧
│   ├── sim_mujoco/           # MuJoCo 仿真
│   │   ├── sim_task_server.py  # HTTP 服务入口
│   │   ├── scene.py          # 仿真场景 + 控制循环
│   │   ├── robot_loader.py   # 模型加载 + PD 控制
│   │   ├── policy_runner.py  # RL locomotion policy
│   │   ├── grasp.py          # 开环抓取 (fallback)
│   │   └── camera.py         # 离屏 RGB-D 相机
│   └── real_robots/          # 实机执行
│       ├── real_task_server.py  # HTTP 服务入口
│       └── camera.py         # RealSense D455
├── algorithms/               # Sim/Real 共用算法
│   ├── perception/           # YOLO + HSV + depth
│   ├── kinematics/           # IK/FK (D1, MuJoCo)
│   ├── calibration/          # 手眼标定
│   ├── grasp/                # GraspPlanner + ArmExecutor
│   └── navigation/           # NavigationController
├── assets/                   # MJCF 模型 + RL policy
├── utils/                    # 工具 + 辅助脚本
│   ├── logging.py            # 统一日志配置
│   ├── calibrate_sim.py      # 仿真标定
│   ├── calibrate.py          # 实机 ArUco 标定
│   └── test_pipeline.py      # 端到端接口测试
└── logs/                     # 运行日志
    ├── execution.log
    ├── scheduler.log
    └── runs/                 # 任务 JSONL 记录
```

## 日志

执行侧和调度侧各自输出到独立日志文件，同时打印到终端：

- `logs/execution.log` — 仿真/实机执行层日志
- `logs/scheduler.log` — 调度 FSM / Web 服务日志

## 任务流程

```
派发 → GO_TO_LOCATION (导航+检测) → PICK_AND_PUT (抓取) → GO_DOCKING (返航+对齐) → FINISHED
```

## TODO

- [ ] 调试仿真实现同现实完全对齐的抓取流程，减少 sim2real gap
- [ ] 集成宇树官方 MuJoCo 仿真运控（替代当前第三方 RL policy）
- [ ] 定位支持外部输入、导航 stack 支持外部服务（如 nav2）
- [ ] 更新算法模块（感知、IK、抓取策略）

## 运行环境

```bash
pip install fastapi uvicorn numpy mujoco torch ultralytics opencv-python Pillow
```

桌面可视化需要 GLFW：`sudo apt install libglfw3`
