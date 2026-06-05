#!/bin/bash
# 启动 UI 服务器 + 统一仿真平台演示
# 用法: conda activate mower && bash scripts/run_sim_with_ui.sh
#
# 浏览器打开 http://127.0.0.1:8765 查看 UI
# "B 点目标确认画面" 会显示仿真相机的实时 RGBD 渲染

set -e
cd "$(dirname "$0")/.."

export MUJOCO_GL=egl

echo "[1/2] 启动 UI 服务器 (port 8765)..."
python3 UI/server.py --port 8765 &
UI_PID=$!
sleep 2

if curl -s http://127.0.0.1:8765/ > /dev/null 2>&1; then
    echo "  UI 已启动: http://127.0.0.1:8765"
else
    echo "  UI 启动失败"
    exit 1
fi

echo "[2/2] 运行统一仿真 (Go2 + D1 + D455 sim)..."
echo "  配置: config/demo.unified-sim-ui.json"
echo ""

python3 -m dog_task --config config/demo.unified-sim-ui.json fixed-once
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== 仿真抓取完成 ==="
else
    echo "=== 仿真失败 (exit=$EXIT_CODE) ==="
fi

kill $UI_PID 2>/dev/null
exit $EXIT_CODE
