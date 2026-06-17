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

_随实验迭代更新_

---

## 十、参数经验库

_随实验迭代更新_

### 导航参数

| 参数 | 当前值 | 经验范围 | 说明 |
|------|--------|---------|------|
| arrival_threshold | 0.6m | 0.3-0.8m | 需 <= 机械臂最大可达 |
| nav_linear_speed | 0.5 m/s | 0.3-0.8 | 过快会导致过冲 |
| nav_angular_speed | 0.8 rad/s | 0.5-1.2 | 转弯速度 |

### 抓取参数

| 参数 | 当前值 | 经验范围 | 说明 |
|------|--------|---------|------|
| approach_height | 0.12m | 0.08-0.20 | 从上方接近的高度偏移 |
| descend_step | 0.015m | 0.01-0.03 | 每次下降步长 |
| move_wait | 0.6s | 0.3-1.0 | 机械臂每步等待时间 |

### PD 控制参数

| 参数 | 当前值 | 经验范围 | 说明 |
|------|--------|---------|------|
| kps (腿) | 35.0 | 20-50 | 位置增益 |
| kds (腿) | 0.8 | 0.5-1.5 | 速度增益 |
| kps (臂) | 40.0 | 25-60 | 手臂位置增益 |
| action_scale | 0.25 | 0.15-0.4 | RL 动作缩放 |
