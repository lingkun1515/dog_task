# DogTaskSim

基于 MuJoCo 的 Go2 四足机器狗 + D1 机械臂 + D455 深度相机联合仿真平台。用于户外清理任务的端到端算法验证与 UI 演示。

## 系统要求

- Ubuntu 20.04+ / macOS
- conda 环境 `mower`（Python 3.10+）
- 桌面环境（MuJoCo 3D 可视化窗口需要显示器）

## 快速开始

```bash
# 1. 激活环境
conda activate mower

# 2. 安装依赖（首次运行）
pip install -r requirements-sim.txt

# 3. 小白一键全流程演示（UI + 3D 窗口）
python3 scripts/run_full_demo.py
```

打开浏览器访问 `http://127.0.0.1:8765`，点击 **"一键派发清理任务"** 即可观看完整演示。

## 项目结构

```
DogTaskSim/
├── dog_task/                     # 核心仿真库
│   ├── cli.py                    # CLI 入口（python3 -m dog_task）
│   ├── config.py                 # 路径解析
│   ├── app/
│   │   ├── factory.py            # 模块工厂（根据 driver 字段实例化）
│   │   └── demo_orchestrator.py  # 任务编排器（fixed-once / walk-and-pick）
│   ├── core/
│   │   └── models.py             # 数据模型（PickRequest, VelocityCommand 等）
│   └── modules/
│       ├── sim/
│       │   ├── scene_builder.py  # MuJoCo 场景组装（Go2 + D1 + Camera）
│       │   └── world.py          # 共享仿真世界（线程安全）
│       ├── mobility/
│       │   └── go2_mujoco.py     # Go2 底盘控制（kinematic 模式）
│       ├── arm/
│       │   ├── mujoco_unified.py # D1 机械臂控制（IK 规划 + 轨迹执行）
│       │   └── ik/
│       │       └── d1_kinematics.py  # D1 六轴运动学链 + IK 求解
│       └── camera/
│           └── mujoco_rgbd.py    # 仿真相机（ground_truth / YOLO 检测 + 渲染）
├── UI/
│   ├── server.py                 # UI HTTP 服务器（端口 8765）
│   └── static/                   # 前端页面
├── assets/                       # 模型资源
│   ├── go2/                      # Go2 MJCF
│   ├── d1/                       # D1 URDF
│   └── rl_models/                # RL 策略 ONNX 模型（可选）
├── config/                       # 仿真配置文件
└── scripts/                      # 测试与演示脚本
    ├── run_full_demo.py          # 一键全流程演示
    ├── run_sim_with_ui.sh        # 双进程 UI+仿真启动
    ├── test_sim_world.py         # SimWorld 单元测试
    ├── test_mobility.py          # Go2 运动学单元测试
    ├── test_arm.py               # D1 机械臂单元测试
    ├── test_camera.py            # 仿真相机单元测试
    ├── test_ui_server.py         # UI 服务器单元测试
    └── test_walk_and_pick.py     # 端到端行走抓取测试
```

## 运行测试

```bash
conda activate mower

# 单元测试（每个可独立运行）
python3 scripts/test_sim_world.py
python3 scripts/test_mobility.py
python3 scripts/test_arm.py
python3 scripts/test_camera.py
python3 scripts/test_ui_server.py

# 模块健康检查
python3 -m dog_task healthcheck

# 单次抓取（固定底座）
python3 -m dog_task fixed-once

# 行走 + 抓取
python3 scripts/test_walk_and_pick.py
```

## 架构要点

- **协议驱动**：Mobility / Arm / Camera 三类模块通过 protocol 接口解耦，`factory.py` 根据 config 的 `driver` 字段选择仿真或真机实现
- **线程安全**：`SimWorld` 使用 `threading.RLock` 保护所有 `qpos` / `qvel` / `mocap` 写操作，相机渲染线程与主控线程安全并发
- **跨进程通信**：仿真进程通过 HTTP POST 推送相机帧（`/api/camera/frame`）和任务状态（`/api/task/status`）到 UI 服务器进程
- **运动学模式**：Go2 采用 kinematic 控制（直接操作 freejoint qpos/qvel），无需 GPU 推理，适合快速原型验证
