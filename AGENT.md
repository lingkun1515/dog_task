# DogTaskSim 闭环研发方法论 — AGENT.md

> 基于 sonic_agent 项目的 AI Coding 闭环调试方法论迁移
> 适配：Go2 四足底盘 + D1 机械臂 + MuJoCo 仿真 + 调度后台
> 目标：自主闭环迭代研发，最小化 sim2real gap

---

## 一、核心原则

本文件是 **AI agent 的长期记忆库**，随着实验迭代不断演化。只写**可复用的原则和判断逻辑**，不写流水账。

**写入规则**：
- ✅ 抖动/失败的根因归因原则
- ✅ 经验证有效的参数范围和阈值
- ✅ sim2real 对齐的关键发现
- ✅ 调试过程中提炼的判断逻辑
- ❌ 每次 run 的详细数值（存 logs/）
- ❌ 临时猜测（等验证后再写）
- ❌ 大段日志或 API key

---

## 二、闭环七大机制

### 机制 1：PR 驱动的渐进式开发

将长期研发目标分解为有严格依赖关系的 PR，每个 PR 只解决一个核心问题：

| PR | 目标 | 验收标准 | 核心验证手段 |
|----|------|----------|-------------|
| **PR0** | 基础仿真跑通：MuJoCo 加载 + RL 行走 + 相机渲染 | 机器人在场景中行走不摔倒，相机输出正常帧 | 视频录制 + 目视 |
| **PR1** | 导航精度：边走边转 + 目标到达 + 朝向对齐 | 到达误差 < 0.3m，朝向误差 < 15° | 导航轨迹分析 + 视频 |
| **PR2** | 感知检测：相机内参 + YOLO/HSV 检测 + 坐标变换 | 检测到目标，pixel→arm 坐标变换误差 < 5cm | 检测日志 + 标定验证 |
| **PR3** | 抓取全流程：IK 求解 + 下降 + 夹取 + 提起 | IK 成功率 > 80%，抓取动作完整执行 | 抓取视频 + 关节轨迹 |
| **PR4** | 闭环任务：导航→检测→抓取→返航完整流程 | FSM 全流程 FINISHED，无 FAILED | 端到端视频 + 耗时统计 |
| **PR5** | Sim2Real 对齐：实机部署验证 | 仿真行为与实机一致 | 实机视频对比 |
| **PR6** | 场景泛化：多目标/多场景/动态障碍 | 不同场景下任务成功率 > 70% | 批量评估 dashboard |
| **PR7** | 鲁棒性：失败恢复 + 重试 + 异常处理 | 重试后成功率提升，无崩溃 | 异常注入测试 |

**设计原则**：每个 PR 验收时必须确认前一个 PR 的产物仍然可用。

### 机制 2：四层信号分离 — 精确诊断根因

将问题空间按信号链拆为四层，每层有独立的分析工具。先定位层，再深入：

```
第一层：调度/FSM 层
  数据源：scheduler.log, task_run.jsonl, SSE 事件
  判断：FSM 状态转换是否正确？超时？重试？
  若失败 → 查 FSM handler / HTTP 通信 / 超时配置
            │
            │ FSM 正常但执行异常
            ▼
第二层：规划/算法层（导航 + IK + 抓取规划）
  数据源：nav_trajectory, ik_solutions, grasp_plan
  判断：导航路径是否合理？IK 是否可解？抓取规划是否正确？
  若失败 → 查 NavigationController / IK solver / GraspPlanner
            │
            │ 规划正常但执行异常
            ▼
第三层：PD 控制/执行层
  数据源：joint_positions, joint_velocities, PD targets
  判断：PD 跟踪是否到位？关节速度/力矩是否超限？
  若抖 → 查 PD gain (kps/kds) / action_scale / 关节限位
            │
            │ 控制正常但物理异常
            ▼
第四层：MuJoCo 物理层
  数据源：qpos, qvel, contact forces, body positions
  判断：接触力是否异常？是否有穿模？摩擦是否足够？
  若抖 → 查 MuJoCo XML 场景参数 / 摩擦系数 / solver 设置
```

**核心原则**：如果关节目标 (target_dof_pos) 已经异常，优先查规划层；如果目标正常但实际关节位置抖动，查 PD 控制层；如果控制命令正常但物理行为异常，查 MuJoCo 场景参数。

### 机制 3：视频第一性原则

**AI agent 不能只查代码日志，必须直接"看"视频。**

#### 视频录制方案

MuJoCo viewer 自主录制：
1. `SimRGBDCamera` 已支持离屏 RGB-D 渲染（EGL headless 模式）
2. 每个评估 episode 自动录制多视角视频：
   - **第三视角**（tracking camera）：观察机器人整体行为
   - **第一视角**（front_cam）：观察感知和抓取细节
3. 视频参数：640×480, 30fps, H.264 / yuv420p

#### 产出结构

```
logs/eval_episodes/<timestamp>/
    third_person.mp4      # 第三视角视频
    front_camera.mp4      # 第一视角视频
    episode_meta.json     # 运行元数据
    fsm_timeline.json     # FSM 状态时间线
    nav_trajectory.csv    # 导航轨迹
    grasp_log.json        # 抓取过程日志
    joint_states.csv      # 关节状态序列
    analysis_summary.json # 分析器汇总
```

### 机制 4：多模态 AI 视频审查（子 Agent 隔离）

用多模态大模型作为"第二双眼睛"审查视频，但**不是最终裁判**。

#### 子 Agent 架构

```
主 Agent（负责关键路径 & 最终整合）
    │
    ├── 视频审查子 Agent（独立环境，客观评估）
    │   - 输入：视频文件 + 任务元数据 + 指标摘要
    │   - 调用：多模态大模型 API（mimo-v2.5，Codex 本地代理）
    │   - 输出：结构化审查报告 + 打分
    │
    └── 分析器子 Agent（可选，并行运行）
        - 输入：episode 数据文件
        - 运行：各层分析脚本
        - 输出：数值分析报告
```

#### 审查约束

- 独立环境，不共享主 Agent 的上下文
- 只接收视频和元数据，不接收主 Agent 的判断
- 输出结构化 JSON，包含 PASS/FAIL/HIGH_RISK 判定和详细理由
- 若子 Agent 判定与主 Agent 数值分析冲突，以数值和视频为准

### 机制 5：分层分析器矩阵

| 分析脚本 | 诊断维度 | 输出文件 | 判断阈值 |
|---------|---------|---------|---------|
| `analyze_navigation.py` | 导航质量 | `nav_summary.json` | 到达误差、路径效率、朝向误差 |
| `analyze_grasp.py` | 抓取质量 | `grasp_summary.json` | IK 成功率、抓取成功率、提起高度 |
| `analyze_joint_tracking.py` | 关节跟踪 | `joint_summary.json` | 跟踪误差、速度平滑性 |
| `analyze_fsm_flow.py` | 任务流程 | `fsm_summary.json` | 状态转换正确性、耗时分布 |
| `analyze_stability.py` | 机器人稳定性 | `stability_summary.json` | 基座高度、倾斜角、是否摔倒 |

### 机制 6：AGENTS.md 长期记忆

本文件即为长期记忆。每轮实验后提炼原则写入，避免重复犯错。

### 机制 7：Git 精确追溯

每个实验 run 必须记录：
```
git rev-parse HEAD           # 精确的 commit
git status --short           # 脏文件数量
关键控制模式                  # config 参数
场景 XML 路径                # 场景配置
机器人配置路径                # TOML 配置
```

---

## 三、闭环数据流

```
用户给出长期研发目标
    │
    ▼
AI Agent 自主执行：
    a. 写代码 / 改配置
    b. 运行评估 episode（一键脚本）
       → 启动 MuJoCo sim（headless EGL）
       → 启动调度后台
       → 派发任务
       → 录制视频（第三视角 + 第一视角）
       → 采集指标
    c. 运行分析器
       → 导航分析 / 抓取分析 / 关节分析 / 稳定性分析
    d. 视频审查（子 Agent）
       → 多模态 AI 看视频 + 打分
    e. 综合判断
       → 视频目视 + 分析器数值 + 子 Agent 审查
       → 输出：PASS / FAIL / HIGH_RISK
       → 定位问题在第几层
    f. 更新 AGENTS.md
       → 将发现转化为可复用原则
    g. 向用户报告 + 准备下一轮
    │
    ▼
用户确认方向 → 继续下一轮
```

---

## 四、当前控制诊断

### 导航

- 导航控制器使用边走边转策略，仅到达目标圈内后原地 ALIGN
- 行走速度由 RL policy 控制，导航只提供速度命令
- 到达判定距离 (arrival_threshold) 需 <= 机械臂最大可达距离 (D1 约 0.55m)

### 抓取

- 抓取流程：检测x3 → 选最近 → SAFE_PARK → IK above → 逐步下降 → 闭合 → 提起 → 归位
- IK 使用解析式 (D1) 或数值式 (MuJoCo Jacobian)
- 坐标变换：Detection.position_cam → calibration.cam_to_arm() → arm base frame
- 仿真标定从 MJCF 真值计算，实机标定用 ArUco

### RL Policy

- 控制 18 DOF（12 腿 + 6 臂），但 arm_rl_enabled=false 时强制覆盖手臂为默认收缩
- 抓取时 GraspPlanner 接管手臂（优先级最高）
- action_scale 控制关节运动幅度

---

## 五、Sim2Real 对齐要点

### 必须对齐的维度

1. **关节映射**：仿真与实机的关节顺序、正方向、限位必须一致
2. **相机内参**：仿真相机的 FOV、分辨率需匹配实机 RealSense D455
3. **标定矩阵**：仿真从 MJCF 推算，实机用 ArUco，两者需在同一坐标系下可比
4. **控制频率**：仿真 control_decimation 需匹配实机控制周期
5. **PD 增益**：仿真 kps/kds 需逼近实机实际值
6. **夹爪语义**：仿真 gripper_open/close 角度需与实机一致

### 不需要对齐的

- 物理参数精确匹配（摩擦、质量分布）— 先保证行为一致，再精调物理
- 渲染质量 — 评估用低分辨率即可，交付用高分辨率

### 关键发现：朝向对齐

- **纯原地旋转**：`start_heading_align()` 不可使用 `navigate_to(require_heading=True)`，后者 min_steps=20 期间机器人会向前蠕动，导致坐在球上方、相机无法检测。
- **实现要点**：`_step_count=999` 跳过 min_steps，`arrival_threshold+=1.0` 阻止距离到达判定，`_state=MOVE` 确保 update() 不 early-return。
- **对齐精度**：heading 误差 <2.5°（实测 0.02°、2.11°、0.65°），满足 PR1 验收标准（<15°）。

---

## 六、子 Agent 并行分工

```
主 Agent（负责关键路径 & 最终整合）
    │
    ├── Agent A：视频审查（只读，多模态 AI 打分）
    │
    ├── Agent B：指标分析（只读，运行分析器脚本）
    │
    ├── Agent C：代码修改（可写，实现具体功能）
    │
    └── Agent D：测试验证（只读，运行测试和 smoke test）
```

---

## 七、评估门控（Gate）

| PR | 门控条件 |
|----|---------|
| PR0 | 视频中机器人行走不摔倒，相机输出正常 |
| PR1 | 到达误差 < 0.3m，朝向误差 < 15度，路径无碰撞 |
| PR2 | 检测到目标，坐标变换误差 < 5cm |
| PR3 | IK 成功率 > 80%，抓取动作完整执行 |
| PR4 | FSM FINISHED，视频确认抓取成功，总耗时 < 120s |
| PR5 | 仿真与实机行为目视一致 |
| PR6 | 多场景成功率 > 70% |
| PR7 | 异常注入后能恢复，无崩溃 |

---

## 八、快速启动指令

新 session 启动时，读取本文件即可了解项目全貌。然后执行：

```bash
# 0. 激活 conda 环境（必须）
conda activate mower

# 1. 安装视频审查依赖（首次）
# mimo-v2.5 通过 Codex 本地代理访问，无需额外安装

# 2. 运行最新评估
python -m scripts.run_eval_episode --config sim_go2_d1 --record-video

# 3. 运行分析器
python -m scripts.analyze_episode --run-dir logs/eval_episodes/<latest>

# 4. 视频审查（子 Agent，自动使用 mimo-v2.5）
python -m scripts.video_review --video logs/eval_episodes/<latest>/third_person.mp4

# 5. 综合判断并更新 AGENTS.md
```

**环境说明**：本项目使用 conda `mower` 环境，已预装 mujoco 3.9.0、opencv 4.13.0、numpy 2.2.6。视频审查使用 mimo-v2.5 多模态模型（通过 Codex 本地代理 127.0.0.1:15721）。

---

## 九、已知问题与待解决

- **IK 下降步失败**：每轮都出现（步6-8失败）。球体在 arm frame z≈-0.34m，above 位置 z≈-0.16m (IK 成功)，但下降到 z≈-0.25m 时 IK 误差 40-60mm 超过容限。根因：球在 arm 工作空间边界外。当前靠 gripper constraint 兜底（抓取仍成功）。已实现 reposition_fn：IK above 失败时站起→坐下→重新定位→重试。
- **视频录制未启用**：`--record-video` 未使用，不影响迭代。

---

### 关键发现：蠕动靠近（Creep Phase）

- **问题**：arrival_threshold=0.3m 时到达误差 ~0.289m，过大。
- **方案**：到 arrival_threshold 后进入蠕动阶段，以 min(0.15, dist*0.5) 速度 + 比例偏航继续靠近，直到 fine_threshold（默认 0.08m）或超过 60 步。
- **效果**：到达误差从 0.289m 降至 0.049-0.144m（平均 ~0.082m），5/5 成功。
- **副作用**：朝向误差稳定在 ~9.5°（蠕动阶段的比例偏航修正不完全）。仍在 15° 阈值内。
- **xpos 深度修复**：蠕动后距离太近，坐下后相机越过球体（depth<=0.01），检测失败。修复：depth<=0.01 时仍返回 detection（position_cam 有效），pixel 用 (-1,-1)。

### 关键发现：导航路径长度

- 当前路径 ~33m，机器人从 (0,0) 到 (12,0) 直线距离 12m，路径效率 ~36%。
- 原因：边走边转策略在微小航向偏差时频繁转向，RL policy 每步损失行进距离。
- 当前不影响成功率但影响效率，可作为后续优化方向。

### 关键发现：多任务场景架构（task_scene）

- **设计**：4 个场景（lawn_debris/golf_ball/rain_inspect/material_drop）共用同一 8 步 FSM 时间线，差异仅在 PICK_AND_PUT 阶段的行为。
- **几何管理**：MuJoCo 编译后的 MjModel 不能动态增删 body。采用「预放置 + 运行时重定位」——所有场景 body 预先放在 scene.xml 的 park 区（100,100,-5），激活时通过 qpos 写入移到作业区。与 `apply_gripper_constraint` 写 ball qpos 同一思路。
- **场景切换链路**：前端 sceneSelect → `/run?scene=xxx` → scheduler `/api/scene/setup` POST → `TaskSceneManager.setup_scene()` 重定位 body + `SimObjectDetector.set_targets()` 切换感知目标。
- **场景化作业分发**：`SimulationScene.start_grasp()` 按 `_active_scene` 选择 worker：
  - lawn_debris → `_run_single_grasp`（原 execute_full_cycle）
  - golf_ball → `_run_multi_grasp`（逐个抓取，成功一个 park 一个）
  - material_drop → `_run_material_drop`（gripper constraint 绑 payload 到 TCP → 释放）
  - rain_inspect → `_run_inspect`（仅检测积水点，机械臂扫描动作）
- **结果语义**：`TaskResult.outcome` ∈ {success/partial/failed/skipped}。PICK_AND_PUT 用 `_is_scene_success()` 判定：success/partial 都算作业成功。golf_ball 3/5 即 partial → success。
- **验证（2026-06-22）**：4 个场景各跑一轮全流程（导航→作业→返航），全部 FINISHED：
  - lawn_debris 109.8s / golf_ball 274.4s (3/5 partial) / rain_inspect 88.3s / material_drop 91.2s

### 关键发现：ALIGN 阶段收敛（导航 bug 修复）

- **问题**：`require_heading=True` 时，`_align_heading` 要求连续 10 步 heading_error < 0.1rad (5.7°)。RL policy 原地旋转有稳态偏置，heading 在小范围震荡，永远无法连续 10 步达标 → 永久卡在 ALIGN，返航 100% 失败。
- **修复**：三级收敛策略
  1. 严格阈值 `heading_threshold` (0.1rad) 连续 10 步 → ARRIVED
  2. 宽松阈值 `align_loose_threshold` (0.25rad ≈ 14°) 连续 20 步 → ARRIVED
  3. 兜底：`align_max_steps` (150 步 ≈ 3s) 强制 ARRIVED
- **副作用**：goal_heading 不再严格保证，但 docking 功能恢复。sim2real 时需注意：实机原地转向若更精确，可禁用宽松阈值。

### 关键发现：场景化导航停靠点（nav_dwell_distance）

- **问题**：rain_inspect 场景机器人导航**到达**目标点 (12,0)，正好压在积水点上方，相机深度为负（在相机后方），检测不到。
- **方案**：导航到「目标点 - 朝向方向 × nav_dwell_distance」处停下，留出观察距离。
  - `RobotConfig.nav_dwell_distance` 字段（默认 0）；rain_inspect 场景默认 0.6m。
  - `go_to_location_sim._compute_dwell_stop()` 在 home→target 方向上从 target 退 dwell 米。
- **效果**：rain_inspect 从「发现 1 个积水点（全部压在身下）」→「发现 3/3 个积水点」。

### 关键发现：Detection.body_name 字段（sim 去重）

- **新增**：`Detection.body_name` 字段（sim 从 xpos 兜底填充对应的 MuJoCo body 名）。
- **用途**：多个同 label 目标（如 3 个 puddle 都是 label="puddle"）按 body_name 去重，否则按 label 去重只算 1 个。
- **通用价值**：sim 端任何多目标场景（multi-ball、multi-debris）现在能精确统计「检测到几个不同的物体」。

### 关键发现：golf_ball 球体布局

- **D1 工作空间偏向**：机械臂可达范围偏底盘正前方偏右（-Y）。左侧远处（+Y > 0.10m）的球 IK 不可达。
- **布局规则**：5 个球散布在「正前方 ±0.10m、前向 ±0.10m」内，避免左侧远处。
- **效果**：3/5 → 5/5 全部回收。

### 关键发现：路径效率（stop-and-turn）

- 导航 stop-and-turn 策略（heading_error > stop_turn_threshold 时停车原地转）把路径效率从 36% 提升到 **98.5%**。
- 去程 12m 直线，实际路径 12.2m；返航 15.6m 直线（含对齐），实际 32m（一来一回）。

### 关键发现：MuJoCo 渲染后端（GLFW 隐藏窗口）

- **EGL 失败**：本机虽然有 NVIDIA GPU + libEGL，但 `MUJOCO_GL=egl` 创建 MjrContext 时报 `gladLoadGL error`（原因未定位，可能是 nvidia EGL 设备权限/驱动问题）。
- **GLFW 隐藏窗口可用**：`MUJOCO_GL=glfw` + 显式 `glfw.create_window(VISIBLE=FALSE)` + `make_context_current` → MjrContext 创建成功，`mjr_render` 正常。
- **顺序敏感**：必须**先创建 GLFW 窗口 + make_current**，**再** import/构造 SimulationScene。否则 SimRGBDCamera 构造时的 MjrContext 失败会污染 GL 状态。
- **共享 MjrContext**：同一 GLFW window 上不能有两个 MjrContext（segfault）。视频录制器复用 scene.camera._context，不另建。

### 关键发现：scene loop 线程并发安全（_physics_paused）

- **问题**：scene.start() 启动后台 loop 线程跑 mj_step；主线程的 configure_scene() 写 qpos + mj_forward 与之并发 → segfault。
- **修复**：SimulationScene 新增 `_physics_paused` 标志，loop 内 `if not self._physics_paused: self.robot.step()`。configure_scene 暂停期间完成几何重定位，再恢复。
- **通用价值**：任何需要在主线程修改 sim 状态（reset、scene switch、qpos 注入）的操作都应先置 `_physics_paused=True`。

### 关键发现：抓取真实性改造（L1 weld + L2 物理夹爪）

**三层抓取架构（L1→L2→L1兜底 混合）：**

- **L0（旧，已弃用）**：`apply_gripper_constraint` 硬设球 qpos = TCP 位置 + 清零速度。「上帝绑定」，球完全无物理。
- **L1（weld 软约束）**：MuJoCo `weld` equality 把球绑定到 d1_link6。球通过物理约束跟随 TCP，松手停用 weld 后球自由下落。eq_data 布局 `[anchor(3), relquat(4)=(0,0,0,1), relpos(3)]`。
- **L2（物理手指）**：两根手指加 slide joint（沿 link6 局部 X 轴开合）+ box 碰撞几何（高摩擦）+ motor。闭合时手指 motor 施加内向力矩，靠摩擦接触夹球。
- **L2+L1 混合（当前生产）**：闭合时优先物理接触；若 300 步内接触未发生（IK 没到位），激活 weld 兜底。张开时停用 weld + 手指外扩，球下落。

**L2 关键参数：**
- 手指 body pos：link6 局部 `(-0.00562, ±0.034, 0.0706)`（TCP 两侧）
- slide joint：range ±0.034m，指间距 0-68mm（D1 实机 66.8mm）
- 接触面 friction=1.8，solref=0.01
- 手指 mass=0.01kg（超过 0.05 会破坏手臂平衡导致摔倒）
- motor ctrlrange=±3 N·m

### 关键发现：L3 多目标类型 + pin 机制

**多类型目标体（mixed_debris 场景）：**
- 球(sphere) + 方块(box×2) + 圆柱(cylinder/瓶) + 袋装(capsule) 共 5 种形状
- 感知按 `body_name` 分类标签（ball/box/bottle/bag），`Detection.body_name` 字段去重
- multi-grasp 逐个抓取，每个目标回收后 park + 下一个

**pin 机制（核心修复）：**
- **问题**：freejoint body 有 mass 时受重力下落。即使 `contype=0` 不碰撞地面，body 仍自由下落（qpos 持续变化）。导航 40s 期间 debris 全部掉到地下。
- **修复**：`TaskSceneManager.pin_body(name, pos, quat)` 记录钉住位置，`apply_pins()` 每步重置 qpos + 清零 qvel。scene loop 在 physics step 后调用。
- **抓取时**：unpin 当前目标（让 weld 接管），完成后该 body 已 park（不需要重新 pin）。
- **通用**：任何 freejoint 静态摆放的 body 都需要 pin，否则会受重力影响。

**单 weld 动态重绑定（避免多 weld segfault）：**
- **问题**：scene.xml 定义 8 个 weld equality（每个 debris 一个），即使全 inactive，MuJoCo 在 mj_forward 预计算 weld Jacobian 时，body1 在 park 区（100,100,-5）距离 body2(d1_link6) 极远 → Jacobian 奇异 → segfault。
- **修复**：只保留 1 个 weld，运行时 `model.eq_obj1id[weld_id] = 当前目标 body_id` 动态重绑定。
- **教训**：MuJoCo equality 数量要最小化，inactive 的 weld 仍消耗 solver 计算且可能引入数值不稳定。

**contype/conaffinity 碰撞开关：**
- debris body 初始 `contype=0 conaffinity=0`（不碰撞，避免 park 区干扰主场景物理）
- 激活时 `_relocate_body(enable_collision=True)` 设 contype=1（参与碰撞 + 抓取接触）

**验证（2026-06-22 L3）：**
- mixed_debris 165.9s，5/5 全回收（球+方块×2+瓶+袋），路径 32.67m，到达误差 0.165m
- lawn_debris 基线正常（90s success）

### 关键发现：IK 到位改进（加深 sit_pose）

**根因定位（量化）：**
- 球在地面 z=0.04m，坐下后 arm_base z=0.279m → 球相对 arm z=-0.239m
- IK 工作空间边界：z ≥ -0.20m 可达，z ≤ -0.25m 不可达
- 球的 z=-0.239 刚好在边界外（差 0.039m）→ IK 下降失败 → TCP 离球 279mm → 物理接触不发生

**修复（加深 sit_pose）：**
- `scene.py`: 后腿 thigh 1.5→1.8, calf -2.2→-2.5（更深折叠）
- arm_base 从 z=0.279 降到 z=0.247（降 3.2cm）
- 球相对 arm z 从 -0.239 改善到 -0.207（进入 IK 可达范围）
- base_z 从 ~0.22 降到 0.192（仍稳定，未摔倒）

**multi-grasp 加速：**
- `planner.py`: move_wait 2.0→1.0s, gripper_wait 0.5→0.3s
- 单次抓取 ~30s → ~18s
- mixed_debris 全流程 165.9s → 126.2s

**验证（2026-06-23）：**
- lawn_debris: TCP 到达球的位置（视觉确认球在夹爪手指间），result=success
- mixed_debris: 126.2s（原 165.9s），5/5 全回收

- **设计**：`scripts/record_task_video.py` 独立跑完整 FSM 任务 + 录制。
- **合成视频**：第三视角（跟踪相机）为主画面，第一视角（front_cam 640×480 原生分辨率）作 PiP 嵌入左上角，叠加与 web `/api/video_feed` 一致的检测框。
- **检测框对齐**：front cam 必须用原生 640×480 渲染（匹配 SimObjectDetector 内参 fx/fy/cx/cy），否则像素坐标错位。PiP 缩放时整体缩放，框仍正确。
- **命名**：`logs/task_videos/<scene>_<YYYYMMDDTHHMMSS>.mp4`，每个场景一个文件。
- **阶段叠加**：每帧叠加 `[phase] t=X.Xs` 标签 + 帧号，便于人工/AI 审查定位。
- **H.264 转码**：用 ffmpeg + libx264 转码提升兼容性；失败则保留 mp4v。
- **验证（2026-06-22）**：4 场景全部录制成功（lawn_debris 90s / rain_inspect 69s / material_drop 81s / golf_ball 163s）。




## 十、参数经验库

### 导航参数

| 参数 | 当前值 | 经验范围 | 说明 |
|------|--------|---------|------|
| arrival_threshold | 0.3m(sched)/0.5m(nav) | 0.3-0.8m | 调度器 0.3m，导航内部默认 0.5m |
| nav_linear_speed | 0.5 m/s | 0.3-0.8 | 过快会导致过冲 |
| nav_angular_speed | 0.8 rad/s | 0.5-1.2 | 转弯速度 |
| return require_heading | **禁用** | — | RL policy 无法收敛 <0.1rad，返航不要求朝向 |
| return arrival_threshold | 0.5m (默认) | 0.3-1.0 | 不要设 1.5m，会导致提前停止 |
| heading_align_timeout | 8s | 5-12 | 到达后原地旋转对齐朝向的超时时间 |

### 球体参数

| 参数 | 当前值 | 经验范围 | 说明 |
|------|--------|---------|------|
| ball_radius | 0.025m | 0.02-0.03 | D1 夹爪指间距 66.8mm，50mm 球体每侧余量 8.4mm |
| ball_height | 0.04m | 0.03-0.06 | 球心高度（含半径后触地） |
| ball_mass | 0.005kg | 0.003-0.01 | 轻球更容易夹取 |

### 坐下姿态参数

| 参数 | 当前值 | 经验范围 | 说明 |
|------|--------|---------|------|
| sit_rear_thigh | 1.5 | 1.3-1.8 | 后腿大腿折叠角度（站立=1.0） |
| sit_rear_calf | -2.2 | -2.5 - -2.0 | 后腿小腿角度（站立=-1.5） |
| sit_front_thigh | 1.0 | 0.8-1.2 | 前腿大腿角度（站立=0.8） |
| sit_front_calf | -1.7 | -1.9 - -1.5 | 前腿小腿角度（站立=-1.5） |
| sit_settle_time | 2.5s | 2.0-3.0 | PD 收敛等待，<2s 会抖动 |
| pre_return_wait | 3.0s | 2.5-4.0 | 返航前等待手臂归位 + 站立恢复 |

### 抓取参数

| 参数 | 当前值 | 经验范围 | 说明 |
|------|--------|---------|------|
| approach_height | 0.18m | 0.15-0.25 | 过小会导致 above 位置也在 IK 边界 |
| descend_step | 0.015m | 0.01-0.03 | 每次下降步长 |
| move_wait | 0.6s | 0.3-1.0 | 机械臂每步等待时间 |
| IK tol | 3mm (default) | 3-8mm | 下降阶段应放宽到 5-8mm，边界处 3mm 太严格 |
| max_attempts | 3 | 2-4 | IK 失败时坐下重新检测再抓取 |

### PD 控制参数

| 参数 | 当前值 | 经验范围 | 说明 |
|------|--------|---------|------|
| kps (腿) | 35.0 | 20-50 | 位置增益 |
| kds (腿) | 0.8 | 0.5-1.5 | 速度增益 |
| kps (臂) | 40.0 | 25-60 | 手臂位置增益 |
| action_scale | 0.25 | 0.15-0.4 | RL 动作缩放 |
| kps (臂) | 40.0 | 25-60 | 手臂位置增益 |
| action_scale | 0.25 | 0.15-0.4 | RL 动作缩放 |
