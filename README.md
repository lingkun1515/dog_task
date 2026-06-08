# DogTaskSim — 机器人移动抓取任务调度与仿真系统

Go2 四足底盘 + Piper 六轴机械臂的移动抓取任务系统。提供**调度后台 → 执行侧服务**两层架构，支持**仿真（MuJoCo）与实机统一调度**，目标是为 sim2real 快速开发验证提供闭环基础设施。

## 设计意图

机器人算法从仿真到真机部署通常需要大量的适配工作——通信协议不同、坐标系不同、状态反馈格式不同。DogTaskSim 在调度层抽象了"仿真"和"实机"两种模式，上层 FSM 状态机、SSE 推送、前端 UI 完全复用，底层仅切换 action handler：

- **仿真模式**：调度侧通过 HTTP API 驱动 MuJoCo 执行侧服务（导航 + 抓取 + 视频流）
- **实机模式**：调度侧通过 TCP Socket + HTTP 驱动真实机器人（沿用 Mower 项目的底盘通信协议）

这样算法同学可以在仿真中完成全流程验证，确认无误后只需修改一行 TOML 配置（`mode = "sim"` → `mode = "real"`）即可切换到实机测试。

当前阶段先完成前后端和仿真侧 demo 执行闭环，后续逐步集成 RL 策略、视觉感知等模块。

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│  浏览器 (localhost:8000)                              │
│  ├─ 任务类型选择 + 一键派发                           │
│  ├─ SSE 实时时间线                                    │
│  ├─ MJPEG 视频流                                      │
│  └─ 结果回执弹窗                                      │
└──────────────┬──────────────────────────────────────┘
               │ HTTP + SSE
┌──────────────▼──────────────────────────────────────┐
│  scheduler/  — 调度后台                               │
│  ├─ server/app.py   FastAPI Web 服务 + 内联前端       │
│  ├─ fsm.py          RobotTaskFSM 状态机               │
│  ├─ config.py       RobotConfig 配置加载（real/sim）   │
│  ├─ states.py       RobotState 枚举                   │
│  ├─ actions/        动作实现（实机 + 仿真各一套）      │
│  └─ server/robot.py 机器人列表 + task_log.py 日志      │
└──────────────┬──────────────────────────────────────┘
               │ HTTP (仿真) 或 TCP (实机)
┌──────────────▼──────────────────────────────────────┐
│  execution/  — 执行侧服务                              │
│  ├─ sim_mujoco/    MuJoCo 仿真执行服务                 │
│  │   ├─ server.py      FastAPI HTTP 接口              │
│  │   ├─ scene.py       仿真场景 + 控制循环             │
│  │   ├─ robot_loader.py 模型加载 + PD 控制             │
│  │   ├─ navigation.py  导航控制器（航向对齐+滑动）     │
│  │   ├─ grasp.py       抓取控制器（开环位姿序列）      │
│  │   ├─ camera.py      离屏渲染→JPEG                   │
│  │   ├─ viewer.py      桌面可视化（可选 --render）     │
│  │   └─ config.yaml    仿真参数                        │
│  └─ robots/        实机部署预留（空壳）                 │
└──────────────────────────────────────────────────────┘
```

## 目录结构

```
DogTaskSim/
├── scheduler/                    # 调度后台
│   ├── main.py                   # CLI 入口
│   ├── states.py                 # RobotState 枚举
│   ├── fsm.py                    # RobotTaskFSM 状态机
│   ├── config.py                 # RobotConfig + 配置加载
│   ├── task_metrics.py           # TaskTiming
│   ├── actions/                  # 动作实现
│   │   ├── connection.py         # 实机：TCP Socket 连接
│   │   ├── go_to_location.py     # 实机：TCP 导航
│   │   ├── pick_and_put.py       # 实机：HTTP 抓取
│   │   ├── go_docking.py         # 实机：TCP 返航
│   │   ├── go_to_location_sim.py # 仿真：HTTP 导航
│   │   ├── pick_and_put_sim.py   # 仿真：HTTP 抓取
│   │   └── go_docking_sim.py     # 仿真：HTTP 返航
│   └── server/                   # Web 服务
│       ├── app.py                # FastAPI + 内联前端
│       ├── robot.py              # 机器人列表
│       ├── task_log.py           # JSONL 任务日志
│       └── static/               # 静态资源
├── execution/                    # 执行侧服务
│   ├── sim_mujoco/               # MuJoCo 仿真执行
│   │   ├── server.py             # HTTP 服务入口
│   │   ├── scene.py              # 场景 + 仿真循环
│   │   ├── robot_loader.py       # 模型加载 + PD
│   │   ├── navigation.py         # 导航控制器
│   │   ├── grasp.py              # 抓取控制器
│   │   ├── camera.py             # 离屏渲染相机
│   │   ├── viewer.py             # 桌面可视化窗口
│   │   └── config.yaml           # 默认仿真参数
│   └── robots/                   # 实机部署预留
├── assets/                       # 模型与网格资源
│   ├── go2/meshes/               # Go2 底盘 STL
│   ├── piper/meshes/             # Piper 机械臂 STL
│   ├── go2_piper/                # Go2+Piper 组合场景
│   │   ├── go2piper.xml          # 机器人 MJCF 模型
│   │   └── scene.xml             # 场景定义（地面+目标球）
│   └── d1/                       # D1 机械臂（预留）
├── config/
│   └── robots/                   # 机器人配置 TOML
│       ├── sim_go2_piper.toml    # 仿真配置示例
│       └── 239.toml              # 实机配置示例
├── scripts/                      # 测试脚本
├── logs/runs/                    # 任务运行日志
├── pyproject.toml
└── README.md
```

## 运行环境

### 调度侧 (scheduler)

- Python >= 3.10
- fastapi >= 0.136.3
- uvicorn[standard] >= 0.48.0
- numpy >= 2.2.6

```bash
pip install fastapi uvicorn numpy
```

### 仿真执行侧 (execution/sim_mujoco)

额外需要 MuJoCo：

- mujoco >= 3.0.0, < 4.0.0
- pyyaml >= 6.0

```bash
pip install mujoco pyyaml
```

**桌面可视化**需要 X11 或 Wayland 显示环境，且安装 GLFW 库：

```bash
# Ubuntu/Debian
sudo apt install libglfw3

# 或通过 conda
conda install -c conda-forge glfw
```

当前项目依赖 conda 环境 `mower`（用户已有）。

## 快速开始

### 1. 启动仿真执行服务

```bash
# 无头模式（仅 HTTP，默认 EGL 离屏渲染）
python -m execution.sim_mujoco.server --port 8100

# 桌面可视化模式（打开 MuJoCo 窗口，可鼠标旋转/缩放观察）
python -m execution.sim_mujoco.server --port 8100 --render
```

仿真服务启动后，MuJoCo 加载 Go2+Piper 机器人模型，场景中在 (12, 0, 0.15) 处有一个红色目标球体。

API 端点：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/navigate` | 导航到目标 `{"x": 12.0, "y": 0}` |
| GET | `/api/navigate/status` | 导航状态 `{"status": "moving"/"arrived"}` |
| POST | `/api/navigate/cancel` | 取消导航 |
| POST | `/api/grasp` | 执行抓取序列 |
| GET | `/api/grasp/status` | 抓取状态 `{"status": "idle"/"success"/"error"}` |
| GET | `/api/state` | 完整机器人状态快照 |
| GET | `/api/video_feed` | MJPEG 视频流 |
| POST | `/api/reset` | 重置仿真 |

### 2. 启动调度后台

```bash
# 默认监听 8000 端口
python -m scheduler.server.app
```

打开浏览器访问 `http://localhost:8000`：

- 左侧：任务类型选择 → 点击"一键派发清理任务"
- 右侧：仿真视频画面
- 底部时间线：实时显示 8 步任务进度
- 任务完成后弹出结果回执

### 3. 手动测试仿真 API

```bash
# 健康检查
curl http://localhost:8100/health

# 发起导航
curl -X POST http://localhost:8100/api/navigate \
  -H "Content-Type: application/json" \
  -d '{"x": 5.0, "y": 0}'

# 查询导航状态
curl http://localhost:8100/api/navigate/status

# 执行抓取
curl -X POST http://localhost:8100/api/grasp

# 查询机器人状态
curl http://localhost:8100/api/state
```

### 4. CLI 单步测试

```bash
# 完整 FSM 运行（仿真模式）
python -m scheduler sim_go2_piper

# 只执行导航
python -m scheduler sim_go2_piper --action go_to_location

# 只执行抓取
python -m scheduler sim_go2_piper --action pick_and_put
```

## 任务流程

```
前端点击派发
  → SSE: ack（任务受理）
  → GO_TO_LOCATION → POST /api/navigate {"x": 12, "y": 0}
  → SSE: go_to_B → arrived_B_confirmed（到点确认）
  → PICK_AND_PUT  → POST /api/grasp
  → SSE: arm_start → arm_done（抓取完成）
  → GO_DOCKING    → POST /api/navigate {"x": 0, "y": 0}
  → SSE: return_A → done（闭环结束）
  → 前端弹出结果回执
```

## 配置说明

机器人配置位于 `config/robots/<id>.toml`：

```toml
# 仿真模式
mode = "sim"
execution_url = "http://localhost:8100"
target_x = 12.0
target_y = 0.0
```

```toml
# 实机模式
mode = "real"
host = "10.10.170.239"
orin = "10.10.170.191"
port = 9002
grasp_port = 5000
```

调度后台会自动扫描 `config/robots/*.toml` 并在前端下拉列表中展示。

环境变量：

- `SCHEDULER_ROBOT_ID` — 调度服务默认机器人 ID（默认: `sim_go2_piper`）
- `SCHEDULER_CONFIG_DIR` — 机器人配置目录（默认: `config/robots/`）
- `SIM_CONFIG` — 仿真配置文件路径（默认: `execution/sim_mujoco/config.yaml`）
- `MUJOCO_GL` — MuJoCo 渲染后端（默认: `egl`；`--render` 时自动切换为 `glfw`）

## 仿真参数

`execution/sim_mujoco/config.yaml` 中的关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `simulation_dt` | 0.005 | 物理步长 (200Hz) |
| `control_decimation` | 4 | 控制降采样 (50Hz) |
| `nav_linear_speed` | 0.5 | 导航线速度 (m/s) |
| `nav_angular_speed` | 0.8 | 导航角速度 (rad/s) |
| `nav_arrival_threshold` | 0.5 | 到达判定距离 (m) |
| `grasp_arm_angles` | [0.3, 1.5, -2.2, ...] | 抓取时机械臂目标角度 |

## 后续计划

- [ ] 集成 RL 腿足运动策略（当前为滑动模式）
- [ ] 仿真侧夹爪闭合/物体抓取物理
- [ ] 视觉感知模块（YOLO 目标检测）集成
- [ ] 实机模式端到端联调
- [ ] 多机器人并发调度
