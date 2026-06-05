# Husqvarna Outdoor Task Robot Demo UI

面向现场演示的轻量产品界面。UI 不直接控制真机，只展示派单、执行、结果回传和异常兜底状态。

## 启动

仓库根目录执行：

```bash
# 正常启动(自动检测相机)
python3 UI/server.py --port 8765

# 不使用相机
python3 UI/server.py --port 8765 --no-camera
```

浏览器打开：

```text
http://127.0.0.1:8765
```

默认使用 mock 流程。点击“一键派发清理任务”后，页面会自动演示：

```text
待命 -> 已接单 -> 前往 B 点 -> 到点确认 -> 机械臂作业
     -> 已入篮 -> 返回 A 点 -> 完成
```

## Orin 联调

联调时禁用自动 mock 流转：

```bash
python3 UI/server.py --host 0.0.0.0 --port 8765 --no-mock
```

Orin 编排器在状态变化时调用：

```bash
curl -X POST http://127.0.0.1:8765/api/task/status \
  -H 'Content-Type: application/json' \
  -d '{"status":"go_to_B"}'
```

支持的主链路状态：

| 状态 | 页面显示 |
|---|---|
| `ack` | 已接单 |
| `go_to_B` | 前往 B 点 |
| `arrived_B_confirmed` | 到点确认 |
| `arm_start` | 机械臂作业 |
| `arm_done` | 已入篮 |
| `return_A` | 返回 A 点 |
| `done` | 完成并弹出结果回传卡片 |
| `failed` | 产品化失败提示 |

可选字段：

```json
{
  "status": "arm_done",
  "object_name": "网球",
  "evidence_image": "/static/assets/site-preview.svg",
  "message": "异物已安全放入回收篮"
}
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 服务健康检查 |
| `GET` | `/api/task` | 当前任务快照 |
| `POST` | `/api/task/dispatch` | 创建任务 |
| `POST` | `/api/task/status` | Orin 回传产品状态 |
| `POST` | `/api/task/pause` | 暂停任务 |
| `POST` | `/api/task/resume` | 继续任务 |
| `POST` | `/api/task/takeover` | 切换人工接管 |
| `POST` | `/api/task/reset` | 返回待命 |
| `POST` | `/api/mode/rain` | 开关雨天室内演示预案 |

## 设计边界

- 不展示 IP 地址、脚本路径、日志堆栈和 DDS 指令。
- 不从浏览器直接触发机械臂或 Go2 原始控制命令。
- Logo 当前为品牌预留位。获得正式素材授权后替换即可。
