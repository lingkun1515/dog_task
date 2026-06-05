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

---

## 小白全流程操作指南

以下为零基础用户操作步骤，全程无需命令行交互。

### 第一步：启动仿真

在终端中执行（只需这一条命令）：

```bash
conda activate mower
cd /home/lenovo/Projects/MowerProject/DogTaskSim
python3 scripts/run_full_demo.py
```

终端会依次打印启动日志，随后桌面弹出 **MuJoCo 3D 可视化窗口**（狗站在地面上，等待命令）。

### 第二步：打开浏览器

浏览器地址栏输入 **`http://127.0.0.1:8765`**，看到 Husqvarna 概念体验界面。

### 第三步：点击"一键派发清理任务"

点击页面中央蓝色主按钮。接下来无需任何操作，自动执行以下 7 个步骤：

| 步骤 | 3D 窗口中 | UI 界面中 |
|------|-----------|-----------|
| ① 初始姿态 | Go2 站立，D1 臂收拢 | 状态栏："设备就绪，等待派发任务" |
| ② 前进 | Go2 向前行走 0.6 米 | 时间线："设备正在前往 B 点目标区域" |
| ③ 趴下 | Go2 蹲下，稳定底盘 | 状态栏："已到达 B 点，目标确认完成" |
| ④ 相机检测 | 视角转向目标区域 | 相机画面从占位图自动切换为 LIVE 实时流 |
| ⑤ 机械臂抓取 | D1 臂伸出→抓瓶→提篮→放入→缩回 | 时间线："D1 正在执行安全抓取动作" |
| ⑥ 完成 | Go2 保持蹲姿 | 弹出结果面板：任务编号 / 处理对象 / 耗时 |
| ⑦ 自由观看 | 可旋转缩放 3D 视角 | 点击"返回待命界面"关闭结果面板 |

### UI 按钮说明

| 按钮 | 位置 | 作用 |
|------|------|------|
| **一键派发清理任务** | 页面顶部 | 启动全流程演示 |
| **暂停任务 / 继续任务** | 右侧运营保障区 | 任务执行中可随时暂停 |
| **人工接管** | 右侧运营保障区 | 紧急停止，切换为人工控制 |
| **开启雨天模式** | 右侧运营保障区 | 切换室内演示预案（场景不变，UI 状态切换） |
| **返回待命界面** | 结果弹窗底部 | 关闭结果面板，设备回到待命状态 |

### 3D 窗口鼠标操作

| 操作 | 效果 |
|------|------|
| 鼠标左键拖拽 | 旋转视角 |
| 鼠标滚轮 | 缩放画面 |
| 鼠标右键拖拽 | 平移视角 |
| 关闭窗口 | 退出仿真程序 |

### 完整流程图

```
终端                                  浏览器                             3D 窗口
───                                   ───                                ───
python3 scripts/run_full_demo.py
  │                                   
  ├─ UI 服务器启动 ─────────────→ 打开 http://127.0.0.1:8765
  ├─ 仿真场景构建                                       ←──────────→ 窗口弹出，Go2 站立
  └─ 等待任务派发                      点击"一键派发清理任务"
                                        │
                                  UI 时间线推进           Go2 向前行走 0.6m
                                  "前往目标点"            
                                        │
                                  "到点确认"              Go2 趴下
                                  LIVE 画面激活           D455 相机视角
                                        │
                                  "机械臂作业"            D1 伸出→抓取→入篮→缩回
                                        │
                                  "任务完成"              保持蹲姿
                                  结果弹窗弹出
                                        │
                                  关闭弹窗 / 关闭浏览器    关闭 3D 窗口 → 程序退出
```

---

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
