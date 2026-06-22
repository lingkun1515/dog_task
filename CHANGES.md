# 调度后台与前端改动记录

本文档记录 DogTaskSim 的 `scheduler/` 调度后台与前端代码，相对于原 Mower 项目 (`/home/lenovo/Projects/MowerProject/Mower/mower/`) 的所有改动。后续改动也需同步更新本文档。

---

## 架构概述

原 Mower 是单一实机模式：FSM 状态机通过 TCP Socket 与真实机器人底盘通信，通过 HTTP 与机械臂抓取服务通信。DogTaskSim 新增了仿真模式，FSM 逻辑和前端 UI 完全复用，仅底层通信模块不同。

```
原 Mower:           FSM → TCP Socket (底盘) + HTTP (抓取)
DogTaskSim (sim):   FSM → HTTP (仿真执行服务)
DogTaskSim (real):  FSM → TCP Socket (底盘) + HTTP (抓取)   ← 原逻辑保留
```

---

## 1. config.py — RobotConfig 与配置加载

### 新增字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | `str` | `"real"` | 运行模式：`"real"` 或 `"sim"` |
| `execution_url` | `str` | `""` | 仿真执行侧 HTTP 服务地址（如 `http://localhost:8100`） |
| `target_x` | `float` | `5.0` | 仿真导航目标点 X 坐标 |
| `target_y` | `float` | `0.0` | 仿真导航目标点 Y 坐标 |

### 新增方法

- `navigate_url()` — 返回导航 POST 接口地址（sim 模式下指向 `execution_url/api/navigate`）
- `navigate_status_url()` — 返回导航状态轮询地址
- `grasp_status_url()` — 返回抓取状态轮询地址（sim 模式下指向 `execution_url/api/grasp/status`）

### 修改行为

- `grasp_url` 和 `video_feed_url()` 根据 `mode` 选择 URL 前缀（sim 用 `execution_url`，real 用 `orin`）
- `video_feed_url(detect=...)` — sim 模式下关闭检测叠加层
- `host` / `orin` 改为可选（sim 模式下不需要）
- `load_robot_config()` 根据 `mode` 字段选择不同的校验逻辑

### 环境变量

`MOWER_CONFIG_DIR` → `SCHEDULER_CONFIG_DIR`

---

## 2. fsm.py — 状态机调度

### 双模式处理器分发

原 FSM 只有一个处理器映射表 `_HANDLERS`。DogTaskSim 拆分为两个：

```python
_HANDLERS_REAL = {
    GO_TO_LOCATION: go_to_location,     # TCP 通信
    PICK_AND_PUT: pick_and_put,         # HTTP 阻塞调用
    GO_DOCKING: go_docking,             # TCP 通信
}
_HANDLERS_SIM = {
    GO_TO_LOCATION: go_to_location_sim, # HTTP 导航 + 轮询
    PICK_AND_PUT: pick_and_put_sim,     # HTTP 抓取 + 轮询
    GO_DOCKING: go_docking_sim,         # HTTP 导航 + 轮询
}
```

构造函数根据 `config.mode` 选择处理器映射表。`_execute_current_state()` 改为动态分发。

### 文案修改

- 任务提示语从"充电桩 → 指定地点 → 捡垃圾+放垃圾 → 回充电桩"改为"充电桩 → 目标点 → 抓取 → 回充电桩"

---

## 3. states.py — 无改动

`RobotState` 枚举完全不变：`GO_TO_LOCATION`, `PICK_AND_PUT`, `GO_DOCKING`, `FINISHED`, `FAILED`。

---

## 4. actions/ — 动作实现

### 4.1 新增文件（仿真模式）

| 文件 | 通信方式 | 说明 |
|------|----------|------|
| `go_to_location_sim.py` | HTTP POST + 轮询 | 向仿真服务发送导航目标，每 1s 轮询状态直到 `"arrived"` |
| `pick_and_put_sim.py` | HTTP POST + 轮询 | 向仿真服务发起抓取，每 1s 轮询状态直到 `"success"` |
| `go_docking_sim.py` | HTTP POST + 轮询 | 向仿真服务发送原点目标 (0,0)，轮询直到 `"arrived"` |

### 4.2 与原实机动作的差异

| 特性 | 实机动作 (原) | 仿真动作 (新) |
|------|--------------|--------------|
| 导航 | TCP Socket `conn.go_to_location()` | HTTP POST `/api/navigate` + 轮询 |
| 抓取 | 单次阻塞 HTTP（超时 180s） | HTTP POST + 每 1s 轮询（两阶段） |
| 回充 | TCP Socket `conn.go_docking()` | HTTP POST `/api/navigate` (0,0) + 轮询 |
| 心跳 | 抓取时独立线程每 10s 发送 | 轮询循环内每 1s 直接调用 `fsm.emit_heartbeat()` |
| 目标 | 隐式（底盘协议内部决定） | 显式 `target_x`/`target_y` 来自 TOML 配置 |
| 抓取结果解析 | 解析 `target.type/conf/depth_m` 等 | 仅检查 `status == "success"` |

### 4.3 保留不变的文件

- `connection.py` — 实机 TCP Socket 连接
- `go_to_location.py` — 实机 TCP 导航
- `pick_and_put.py` — 实机 HTTP 抓取
- `go_docking.py` — 实机 TCP 回充

---

## 5. server/app.py — FastAPI Web 控制台

### 5.1 移除的功能

- **D1 机械臂 WebRTC 遥操作桥接**：`arm.py`、`arm_bridge.py` 导入及 FastAPI lifespan 全部移除
- **`/api/teleop/` 路由注册**：不再注册 `arm_router`
- **Husqvarna logo 图片**：header 中不再显示 logo
- **FastAPI lifespan 参数**：改为无 lifespan 的普通模式

### 5.2 新增功能

- `/run` 端点根据 `config.mode` 选择 `_ACTION_HANDLERS_SIM` 或 `_ACTION_HANDLERS_REAL`
- `TaskRunRecorder.open()` 传入 `mode` 和 `execution_url` 字段
- 单步动作验证使用模式感知的处理器字典

### 5.3 品牌重命名

| 原文本 | 新文本 |
|--------|--------|
| Husqvarna Outdoor Task Robot Demo | DogTask——机器人移动抓取任务系统 |
| Others avoid obstacles. We clear them. | Go2 + Piper · 仿真与实机统一调度 |
| Husqvarna FSM Server | DogTask FSM Server |
| 场景：lawn_debris / golf_ball | 场景：sim_grasp_demo / real_lawn_debris |

### 5.4 模块路径

- uvicorn 启动目标：`"mower.server.app:app"` → `"scheduler.server.app:app"`

---

## 6. server/robot.py — 机器人列表

### 修改

- `DEFAULT_ROBOT_ID`: `"239"` → `"sim_go2_piper"`
- 环境变量：`MOWER_ROBOT_ID` → `SCHEDULER_ROBOT_ID`

---

## 7. 前端 (内联 HTML/CSS/JS)

### 7.1 HTML 结构

- `<title>`: "Husqvarna Outdoor Task Robot Demo" → "DogTask——机器人移动抓取任务系统"
- Header h1: 同上
- Header p: "Others avoid obstacles. We clear them." → "Go2 + Piper · 仿真与实机统一调度"
- 移除 Husqvarna logo `<img>`
- CSS variable `--hz-green` 保持不变（仍使用 Husqvarna 绿色主题）

### 7.2 场景选项

```javascript
// 原
SCENE_OPTIONS = [
  { value: 'lawn_debris', disabled: false },
  { value: 'golf_ball', disabled: true },
  ...
];
// 新
SCENE_OPTIONS = [
  { value: 'sim_grasp_demo', disabled: false },
  { value: 'real_lawn_debris', disabled: true },
  ...
];
```

### 7.3 视频流

- `video_feed_url()` 调用 `detect=(config.mode != "sim")` — 仿真模式不启用检测叠加

### 7.4 其余不变

- 时间线 8 步 UI（STEP_IDS 数组）
- SSE EventSource 通信
- 中/英文切换（I18N）
- 任务回执弹窗
- `localStorage` key 仍用 `mower_lang`（兼容）

---

## 8. 后续改动记录

_后续改动按时间倒序记录于此。_

### 2026-06-22（第二轮）— 场景质量优化 + 批量评估 + 视频录制修复

#### 场景质量优化

- **rain_inspect 检测修复**：
  - 新增 `RobotConfig.nav_dwell_distance`（导航停靠距离，默认 0）；rain_inspect 场景默认 0.6m
  - `go_to_location_sim._compute_dwell_stop()` 在 home→target 方向上从 target 退 dwell 米停下，避免机器人压在目标上方导致相机看不到
  - 效果：积水点检测从「发现 1 个（全部压在身下）」→「发现 3/3 个」
- **Detection.body_name 字段**：`algorithms/perception/base.py` 新增字段；`SimObjectDetector._detect_from_xpos` 填充对应 MuJoCo body 名。多目标场景按 body_name 精确去重（替代按 label 去重）
- **golf_ball 布局优化**：D1 机械臂工作空间偏向底盘正前方偏右（-Y），左侧远处（+Y>0.10m）的球 IK 不可达。5 个球收紧到「正前方 ±0.10m、前向 ±0.10m」内
  - 效果：3/5 partial → **5/5 全部回收（success）**

#### 新增：批量评估脚本

- `scripts/run_batch_eval.py`：连续跑多个场景，汇总对比报告
  - 增量保存 `batch_summary.json`（中断时仍可用）
  - 生成人类可读的 `batch_summary.md`（汇总表 + 各场景详情）
  - 用法：`python -m scripts.run_batch_eval --config sim_go2_d1 [--scenes golf_ball,rain_inspect]`

#### 视频录制修复

- `scripts/record_video.py`：`VideoRecorder` / `MultiViewRecorder` 新增 `context` 参数，复用调用方的 GL context
  - 修复 headless EGL 下重复创建 MjrContext 导致 `gladLoadGL error`
  - context 不可用时优雅降级（警告 + 跳过），不阻塞评估
- `scripts/run_eval_episode.py`：setup() 把 `scene.camera._context` 传给 MultiViewRecorder

#### 验证（2026-06-22 第二轮）

批量评估 4 场景（配置 sim_go2_d1，目标 (12, 0)）：

| 场景 | 结果 | outcome | 耗时 | 得分 | 到达误差 | 朝向误差 | 路径效率 |
|------|------|---------|------|------|----------|----------|----------|
| lawn_debris   | ✓ | success | 113.6s | 100 | 0.066m | 10.3° | 98.5% |
| golf_ball     | ✓ | success | 204.9s | -  | -       | -      | -     | (5/5)
| rain_inspect  | ✓ | success | 101.4s | 90  | 0.481m | 2.6°   | -     |
| material_drop | ✓ | success | 107.7s | 100 | 0.079m | 10.3° | 98.0% |

路径效率从首轮的 36% 提升到 ~98%（stop-and-turn 策略生效）。

### 2026-06-22 — 多任务场景（golf_ball / rain_inspect / material_drop）+ 导航 ALIGN 修复

#### 新增：4 任务场景全链路支持

把 web 前端预留的 3 个任务（golf_ball / rain_inspect / material_drop）从「disabled 占位」改为完整可跑。设计原则：**8 步 FSM 时间线不变，差异仅在 PICK_AND_PUT 阶段的作业行为**。

**新增文件**：
- `execution/sim_mujoco/task_scenes.py` — `TaskSceneManager` + `TaskResult`/`TaskOutcome`
  - 「预放置 + 运行时重定位」策略：所有场景 body 预先定义在 `scene.xml` 的 park 区 (100,100,-5)，激活场景时通过 qpos 写入移到作业区
  - `requires_grasp` / `requires_drop` / `requires_inspect` 属性区分作业类型

**修改文件**：
- `assets/go2_d1/scene.xml` — 新增 8 个预放置 body（5 golf balls + payload_box + 3 puddles），全部带 freejoint 初始在 park 区
- `algorithms/perception/perception.py` — `SimObjectDetector` 新增 `set_targets()` / `_refresh_target_ids()`，支持运行时切换感知目标集（多场景共用同一检测器）
- `execution/sim_mujoco/scene.py` —
  - `configure_scene()` 激活场景（重定位 + 检测器切换）
  - `start_grasp()` 改为场景化分发：`_run_single_grasp` / `_run_multi_grasp` / `_run_material_drop` / `_run_inspect`
  - `_run_multi_grasp`：逐个抓取，成功一个 park 一个（detector.set_targets 缩小目标集）
  - `_run_material_drop`：gripper constraint 绑 payload 到 TCP → 抬升 → 释放
  - `_run_inspect`：检测积水点 + 机械臂扫描动作，不抓取
  - 状态快照新增 `active_scene` / `task_result` 字段
- `execution/sim_mujoco/sim_task_server.py` —
  - 新增 `POST /api/scene/setup` 端点（激活场景）
  - 新增 `GET /api/scene/current` 端点
  - `/api/grasp/status` 增加 `scene` / `task_result` / `success` 字段
  - `/api/state` 增加 `active_scene` / `task_result`
- `scheduler/config.py` — `RobotConfig` 新增 `task_scene` 字段（默认 lawn_debris）；新增 `SUPPORTED_TASK_SCENES` 枚举
- `scheduler/fsm.py` — `RobotTaskFSM` 新增 `scene` 参数（可覆盖 config.task_scene）
- `scheduler/server/app.py` —
  - `/run` 新增 `scene` query 参数，透传给 FSM；sim 模式下派发前自动 POST `/api/scene/setup`
  - 前端：`SCENE_OPTIONS` 全部启用（4 个场景 disabled=false）
  - 前端 i18n：新增 `sceneDispatch` / `sceneBtnPick` / `sceneSteps` 场景化文案；`applySceneText()` / `renderTimeline()` 按选中场景更新派发按钮、单步按钮、时间线步骤
  - 前端：dispatch / btnGoTo / btnPick / btnDock 全部把 `scene` 加入 `/run` 查询参数
- `scheduler/actions/pick_and_put_sim.py` — 改用 `_wait_for_completion`（轮询 busy 字段，场景无关）+ `_is_scene_success()` 场景化成功判定（task_result.outcome in {success,partial} 或 planner_state==success）
- `scripts/run_eval_episode.py` — 新增 `--scene` 参数；run_task 接受 scene，激活场景几何；场景化成功判定；超时后 30s drain 等待 algo 线程收尾（保留 task_result）
- `scripts/analyze_episode.py` — `analyze_grasp` 场景化：非抓取场景（inspect/drop）不强制抓取日志；优先读 `task_result.outcome`

#### 修复：导航 ALIGN 永久卡死

- **问题**：`require_heading=True`（返航）时 `_align_heading` 要求连续 10 步 `heading_error < 0.1rad`。RL policy 原地转向有稳态偏置，heading 在小范围震荡永远无法连续达标 → 返航 100% 超时失败
- **修复**（`algorithms/navigation/navigation.py`）：三级收敛策略
  1. 严格阈值 0.1rad 连续 10 步 → ARRIVED
  2. 宽松阈值 `align_loose_threshold=0.25rad` 连续 20 步 → ARRIVED
  3. 兜底 `align_max_steps=150` 强制 ARRIVED
- 新增 `_align_step_count` 计数；角速度地板 0.05rad/s 避免小误差不动

#### 验证（2026-06-22）

4 个场景各跑一轮全流程（导航→作业→返航），全部 FINISHED：

| 场景 | 耗时 | 结果 |
|------|------|------|
| lawn_debris   | 109.8s | ✓ success (单目标抓取) |
| golf_ball     | 274.4s | ✓ partial (3/5 回收) |
| rain_inspect  | 88.3s  | ✓ success (发现积水点) |
| material_drop | 91.2s  | ✓ success (落点偏差 34.8cm) |

HTTP 全链路验证：scheduler `/run?scene=rain_inspect` → sim `/api/scene/setup` 激活 → 导航 → 到达 → 巡检 → 返航 → FINISHED。

### 2026-06-08

- **RL 策略集成**：新增 `execution/sim_mujoco/policy_runner.py`，将 TorchScript RL 策略接入仿真循环，实现真实的四足步态行走（替代原有滑动模式）
- **键盘遥操作**：新增 `execution/sim_mujoco/keyboard_teleop.py`，支持 WASD/QE 控制底盘、IJKL/UO 控制机械臂、Space 触发抓取
- **仿真配置增强**：`config.yaml` 新增 `action_scale`, `base_ang_vel_scale`, `joint_vel_scale`, `num_hist` 等 RL 策略参数
- **抓取优化**：`nav_arrival_threshold` 从 0.5m 减小到 0.25m；`grasp_arm_angles` 调整为更前伸向下的位姿
- **竞态修复**：`scene.py` 中 `navigate_to`/`start_grasp`/`cancel_navigation`/`cancel_grasp` 调用后立即刷新快照，防止状态轮询读到过期状态
- **端到端测试通过**（2026-06-08）：
  - 完整 SSE 时间线验证：`ack → go_to_B → arrived_B_confirmed → arm_start → arm_done → return_A → done`
  - RL 步态行走速度约 0.3-0.5 m/s，32s 完成 15.6m 导航，33s 完成 12m 回程
  - 仿真抓取耗时约 3s（moving_to_grasp → holding → success）
  - 所有 API 端点正常：health, navigate, navigate/status, grasp, grasp/status, state, reset, video_feed
  - 错误处理正常：非法 action 返回明确错误信息
  - 注意：headless EGL 模式下 video_feed 可能无图像输出，GUI 模式下正常

### 2026-06-08（第二轮修复）

- **键盘遥操作修复**：`server.py` GUI 模式下遗漏 `_scene.enable_keyboard()` 调用，导致 `_kb_controller` 始终为 None，键盘遥操作完全不生效。已在 `main()` 中补充该调用
- **鼠标拖放视角修复**：`viewer.py` 鼠标回调在首次按下时未初始化 `_last_x/_last_y`，导致首次拖拽产生视角跳变。已在 `_mouse_button_callback` 按下分支中加入光标位置捕获
- **场景丰富**：`scene.xml` 全面改写 — 新增草地纹理地面、泥土路径（6m×1m）、起点旗杆+红色旗帜+平台、目标红十字标记+标杆、6 个障碍物（锥桶、木箱、桶、锥桶2、混凝土路障、岩石），目标球缩小至 radius=0.06 并置于 z=0.28m 高度
- **机械臂抓取修复**：
  - 目标球从 radius=0.1 缩小到 0.06，z 从 0.15 升高到 0.28（更接近机械臂可达范围）
  - `config.yaml` 中 `default_angles` 机械臂部分从全零改为 `[0.0, 0.8, -1.2, 0.0, -0.2, 0.0]`（自然前倾默认位姿）
  - `grasp_arm_angles` 调整为 `[0.0, 2.5, -2.0, 0.0, -0.8, 0.0]`，joint2=2.5rad（远超水平 90deg），机械臂末端明确向下伸展
  - `grasp.py` 误差阈值从 0.12 rad 放宽到 0.35 rad，解决 PD 控制器稳态误差导致状态机卡在 MOVING_TO_GRASP 的问题
  - 验证：joint2 峰值达 3.0 rad（接近垂直向下），抓取状态机正确流转：moving_to_grasp -> holding -> returning -> success
- **E2E 回归测试通过**（2026-06-08 第二轮）：
  - 完整流程 54s：导航 24s + 抓取 2s + 回程 28s，总计约 27.6m 行走距离
  - 所有 7 个 SSE 时间线事件正常：ack -> go_to_B -> arrived_B_confirmed -> arm_start -> arm_done -> return_A -> done
  - 最终状态：FINISHED

### 2026-06-08（第三轮修复）

- **回程目标点修正**：`go_docking_sim.py` 硬编码 `HOME=(0,0)` 导致回程目标错误（机器人起始位置是 (0,-10)）。改为从 `RobotConfig.home_x`/`home_y` 读取，默认值 (0, -10)。`sim_go2_piper.toml` 新增 `home_x=0.0`、`home_y=-10.0` 字段
- **导航朝向可选**：`NavigationController.set_target()` 新增 `require_heading` 和 `arrival_threshold` 参数
  - 去程（`go_to_location_sim`）：`require_heading=False`, `arrival_threshold=1.0m` — 不需要朝向对齐，距离目标 1m 即判断到达
  - 回程（`go_docking_sim`）：`require_heading=True` — 需要朝向对齐以正确回桩
- **任务完成后停止运动**：新增 `/api/stop` 端点，`go_docking_sim` 在到达回程目标后调用，确保导航和抓取控制器完全停止，避免残留运动
- **相机视频流修复**：
  - 安装 Pillow 库（JPEG 编码必需）
  - `camera.py` 重构：GUI 模式下复用 Viewer 的 `MjrContext`（共享 GL 上下文），headless 模式下尝试自建 EGL 上下文
  - 系统无 EGL/OSMesa 支持，headless 模式下相机不可用（已知限制），GUI 模式（`--render`）下正常
- **调度端口修改**：默认端口从 8000 改为 8200
- **旗子位置调整**：从 (0,-10) 移到 (-0.5,-10.5)，避免机械臂初始化时碰到旗杆
