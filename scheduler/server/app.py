"""DogTask — 机器人移动抓取任务调度控制台。"""

import json
import logging
import queue
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..actions import (
    go_docking, go_to_location, pick_and_put,
    go_docking_sim, go_to_location_sim, pick_and_put_sim,
)
from ..config import load_robot_config
from ..fsm import RobotTaskFSM
from ..states import RobotState
from .robot import DEFAULT_ROBOT_ID, list_robot_ids
from .task_log import TaskRunRecorder, log_task_rejected

_ACTION_HANDLERS_REAL = {
    "go_to_location": go_to_location,
    "pick_and_put": pick_and_put,
    "go_docking": go_docking,
}
_ACTION_HANDLERS_SIM = {
    "go_to_location": go_to_location_sim,
    "pick_and_put": pick_and_put_sim,
    "go_docking": go_docking_sim,
}
_ACTION_SUCCESS_FINAL = {
    "go_to_location": {RobotState.PICK_AND_PUT.name},
    "pick_and_put": {RobotState.GO_DOCKING.name},
    "go_docking": {RobotState.FINISHED.name},
}

app = FastAPI(title="DogTask FSM Server")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# 进程级锁：fsm.run() 阻塞且 redirect_stdout 是进程级的，禁止并发运行
_run_lock = threading.Lock()
# 抓取等长耗时步骤期间 worker 无输出，需定时推送 SSE data 心跳（注释对 Nginx/浏览器常无效）
_SSE_KEEPALIVE_SEC = 10
_task_running = False


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DogTask — 机器人移动抓取任务系统</title>
  <style>
    :root {
      --hz-green: #008542;
      --hz-green-dark: #006b35;
      --hz-black: #1a1a1a;
      --hz-gray: #f4f5f6;
      --hz-border: #e2e4e8;
      --hz-text: #2d2d2d;
      --hz-muted: #6b7280;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: "Segoe UI", "Helvetica Neue", "Microsoft YaHei", sans-serif;
      background: var(--hz-gray); color: var(--hz-text); min-height: 100vh;
    }
    .header {
      background: var(--hz-black); color: #fff; padding: 20px 32px;
      display: flex; align-items: center; gap: 20px;
    }
    .logo {
      height: 44px; width: auto; object-fit: contain; flex-shrink: 0;
      background: #fff; padding: 6px 12px; border-radius: 6px;
    }
    .header-text { flex: 1; min-width: 0; }
    .header-text h1 { margin: 0; font-size: 22px; font-weight: 600; letter-spacing: .02em; }
    .header-text p { margin: 4px 0 0; font-size: 13px; color: rgba(255,255,255,.65); }
    .lang-toggle {
      margin-left: auto; padding: 8px 16px; font-size: 13px; font-weight: 600;
      background: rgba(255,255,255,.12); color: #fff; border: 1px solid rgba(255,255,255,.35);
      border-radius: 8px; cursor: pointer; flex-shrink: 0;
    }
    .lang-toggle:hover { background: rgba(255,255,255,.22); }
    body.lang-en .action-row,
    body.lang-en .log-section { display: none !important; }
    .main { max-width: 1440px; margin: 0 auto; padding: 24px 20px 40px; }
    .grid {
      display: grid; grid-template-columns: minmax(240px, 280px) minmax(0, 1fr);
      gap: 20px; align-items: start;
    }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: #fff; border-radius: 12px; border: 1px solid var(--hz-border);
      padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    .card-task { padding: 18px 16px; }
    .card-video { padding: 18px 20px; min-width: 0; }
    .card-title { font-size: 15px; font-weight: 600; margin: 0 0 16px; color: var(--hz-black); }
    .card-video .card-title { margin-bottom: 8px; }
    .card-video .card-subtitle { margin: 0 0 12px; font-size: 12px; color: var(--hz-muted); }
    .scene-row { margin-bottom: 16px; }
    .scene-row label { display: block; font-size: 12px; color: var(--hz-muted); margin-bottom: 6px; }
    .scene-row select {
      width: 100%; padding: 10px 12px; border: 1px solid var(--hz-border);
      border-radius: 8px; font-size: 14px; background: #fff;
    }
    .dispatch-btn {
      width: 100%; padding: 14px 24px; font-size: 16px; font-weight: 600;
      background: var(--hz-green); color: #fff; border: none; border-radius: 10px;
      cursor: pointer; transition: background .15s;
    }
    .dispatch-btn:hover:not(:disabled) { background: var(--hz-green-dark); }
    .dispatch-btn:disabled { background: #9ca3af; cursor: not-allowed; }
    .alert {
      display: none; margin-top: 16px; padding: 12px 14px; border-radius: 8px;
      font-size: 13px; line-height: 1.5;
    }
    .alert.show { display: block; }
    .alert-warn { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .alert-fail { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .timeline { list-style: none; margin: 0; padding: 0; }
    .card-task .card-title { font-size: 14px; margin-bottom: 12px; }
    .card-task .card-title.section { margin-top: 20px; }
    .timeline li {
      display: flex; align-items: flex-start; gap: 10px; padding: 6px 0;
      position: relative;
    }
    .timeline li:not(:last-child)::after {
      content: ""; position: absolute; left: 9px; top: 26px; bottom: -2px;
      width: 2px; background: var(--hz-border);
    }
    .timeline li.done:not(:last-child)::after { background: var(--hz-green); }
    .dot {
      width: 20px; height: 20px; border-radius: 50%; border: 2px solid var(--hz-border);
      background: #fff; flex-shrink: 0; display: flex; align-items: center; justify-content: center;
      font-size: 10px; color: var(--hz-muted); z-index: 1;
    }
    .timeline li.active .dot {
      border-color: var(--hz-green); background: var(--hz-green); color: #fff;
      box-shadow: 0 0 0 4px rgba(0,133,66,.2);
    }
    .timeline li.done .dot { border-color: var(--hz-green); background: var(--hz-green); color: #fff; }
    .timeline li.fail .dot { border-color: #dc2626; background: #dc2626; color: #fff; }
    .step-label { font-size: 13px; font-weight: 500; padding-top: 1px; line-height: 1.35; }
    .step-label small { display: block; font-weight: 400; font-size: 11px; color: var(--hz-muted); margin-top: 1px; }
    .timeline li.active .step-label { color: var(--hz-green-dark); }
    .timeline li.done .step-label { color: var(--hz-green-dark); }
    .view-placeholder {
      width: 100%; aspect-ratio: 16/9; min-height: min(72vh, 640px);
      background: #1a1a1a; border-radius: 8px;
      overflow: hidden; position: relative;
    }
    @media (max-width: 900px) {
      .view-placeholder { min-height: 280px; aspect-ratio: 4/3; }
    }
    .live-feed {
      width: 100%; height: 100%; display: block; object-fit: contain;
    }
    .live-feed-error {
      position: absolute; inset: 0; display: flex; flex-direction: column;
      align-items: center; justify-content: center; padding: 16px;
      color: var(--hz-muted); font-size: 13px; text-align: center;
      background: linear-gradient(135deg, #e8eaed 0%, #d1d5db 100%);
    }
    .live-feed-error[hidden] { display: none; }
    .live-feed-status {
      position: absolute; left: 8px; bottom: 8px; padding: 4px 8px;
      font-size: 11px; font-weight: 600; color: #fff; background: rgba(0,133,66,.85);
      border-radius: 4px; opacity: 0; transition: opacity .3s;
      pointer-events: none;
    }
    .live-feed-status.show { opacity: 1; }
    .overlay {
      display: none; position: fixed; inset: 0; background: rgba(0,0,0,.45);
      align-items: center; justify-content: center; z-index: 100; padding: 24px;
    }
    .overlay.show { display: flex; }
    .result-card {
      background: #fff; border-radius: 16px; max-width: 440px; width: 100%;
      padding: 28px; box-shadow: 0 20px 60px rgba(0,0,0,.2);
    }
    .result-card h2 { margin: 0 0 4px; font-size: 18px; }
    .result-card .subtitle { font-size: 13px; color: var(--hz-muted); margin-bottom: 20px; }
    .result-row { display: flex; justify-content: space-between; padding: 8px 0;
      border-bottom: 1px solid var(--hz-border); font-size: 14px; }
    .result-row:last-of-type { border-bottom: none; }
    .result-row span:first-child { color: var(--hz-muted); }
    .result-status { font-weight: 600; }
    .result-status.ok { color: var(--hz-green); }
    .result-status.fail { color: #dc2626; }
    .result-photo {
      margin-top: 16px; height: 140px; background: var(--hz-gray); border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; color: var(--hz-muted); border: 1px dashed var(--hz-border);
    }
    .result-close {
      margin-top: 20px; width: 100%; padding: 12px; background: var(--hz-black);
      color: #fff; border: none; border-radius: 8px; font-size: 14px; cursor: pointer;
    }
    .log-toggle {
      padding: 6px 12px; font-size: 12px; font-weight: 500;
      background: #fff; color: var(--hz-text); border: 1px solid var(--hz-border);
      border-radius: 6px; cursor: pointer; transition: background .15s, border-color .15s;
      white-space: nowrap;
    }
    .log-toggle:hover { background: var(--hz-gray); border-color: #cbd5e1; }
    .log-toggle.active {
      background: var(--hz-black); color: #fff; border-color: var(--hz-black);
    }
    .log-panel {
      display: none; margin-bottom: 20px;
    }
    .log-panel.show { display: block; }
    .log-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 8px; gap: 8px;
    }
    .log-actions { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
    .log-clear {
      padding: 6px 12px; font-size: 12px; background: transparent;
      color: var(--hz-muted); border: 1px solid var(--hz-border);
      border-radius: 6px; cursor: pointer;
    }
    .log-clear:hover { color: var(--hz-text); border-color: #cbd5e1; }
    .log-box {
      height: 180px; overflow-y: auto; padding: 12px 14px;
      background: #1a1a1a; color: #d4d4d4; border-radius: 8px;
      font-family: "SF Mono", "Consolas", "Menlo", monospace;
      font-size: 11px; line-height: 1.6;
    }
    .log-box:empty::before {
      content: var(--log-empty-hint, "暂无日志，派发任务后将在此显示运行输出");
      color: #6b7280;
    }
    .log-line { word-break: break-all; }
    .log-line + .log-line { margin-top: 2px; }
    .action-row {
      display: flex; flex-direction: column; gap: 8px; margin-top: 16px;
    }
    .action-btn {
      width: 100%; padding: 10px 14px; font-size: 13px; font-weight: 500;
      background: #fff; color: var(--hz-text); border: 1px solid var(--hz-border);
      border-radius: 8px; cursor: pointer; transition: background .15s, border-color .15s;
    }
    .action-btn:hover:not(:disabled) { background: var(--hz-gray); border-color: #cbd5e1; }
    .action-btn:disabled { color: #9ca3af; cursor: not-allowed; background: #f9fafb; }
    body.lang-en .result-timing-zh { display: none; }
  </style>
</head>
<body>
  <header class="header">
    <div class="header-text">
      <h1>DogTask — 机器人移动抓取任务调度系统</h1>
      <p data-i18n="headerSubtitle">Go2 + Piper · 仿真与实机统一调度</p>
    </div>
    <button id="langToggle" class="lang-toggle" type="button">EN</button>
  </header>

  <main class="main">
    <div class="grid">
      <section class="card card-task">
        <div class="scene-row">
          <label for="sceneSelect" data-i18n="taskType">任务类型</label>
          <select id="sceneSelect"></select>
        </div>
        <button id="dispatchBtn" class="dispatch-btn" type="button" data-i18n="dispatch">一键派发清理任务</button>
        <div id="alertBox" class="alert"></div>

        <h2 class="card-title section" data-i18n="taskProgress">任务进度</h2>
        <ol class="timeline" id="timeline"></ol>

        <div class="scene-row" style="margin-top:20px;margin-bottom:0">
          <label for="robotSelect" data-i18n="robotConfig">机器人配置</label>
          <select id="robotSelect">__ROBOT_OPTIONS__</select>
        </div>
        <div id="actionRow" class="action-row">
          <button id="btnGoTo" class="action-btn" type="button" data-action="go_to_location" data-i18n="btnGoTo">前往目标位置</button>
          <button id="btnPick" class="action-btn" type="button" data-action="pick_and_put" data-i18n="btnPick">夹垃圾</button>
          <button id="btnDock" class="action-btn" type="button" data-action="go_docking" data-i18n="btnDock">回充电桩</button>
        </div>
      </section>

      <aside class="card card-video">
        <div class="log-section">
          <div class="log-header">
            <h2 class="card-title" style="margin:0" data-i18n="runLog">运行日志</h2>
            <div class="log-actions">
              <button id="logToggle" class="log-toggle" type="button" data-i18n="showLog">显示日志</button>
              <button id="logClear" class="log-clear" type="button" data-i18n="clearLog">清空</button>
            </div>
          </div>
          <div id="logPanel" class="log-panel">
            <div id="logBox" class="log-box"></div>
          </div>
        </div>
        <h2 class="card-title" data-i18n="targetView">目标位置确认画面</h2>
        <p class="card-subtitle" data-i18n="targetViewSub">实时 MJPEG · 叠加检测框</p>
        <div class="view-placeholder" id="liveView">
          <img id="liveFeed" class="live-feed" alt="现场视频流" data-i18n-alt="liveFeedAlt" />
          <div id="liveFeedError" class="live-feed-error" hidden>
            <span data-i18n="videoError">视频流连接失败</span>
            <small style="margin-top:8px" data-i18n="videoErrorHint">请确认 video_feed 服务已启动，且本机可访问配置中的 orin 地址</small>
          </div>
          <span id="liveFeedStatus" class="live-feed-status" data-i18n="targetConfirmed">✓ 目标位置已确认</span>
        </div>
      </aside>
    </div>
  </main>

  <div class="overlay" id="resultOverlay">
    <div class="result-card">
      <h2 data-i18n="resultTitle">作业回执</h2>
      <p class="subtitle" data-i18n="resultSub">Fleet 工单式结果回传</p>
      <div class="result-row"><span data-i18n="rTaskId">任务编号</span><span id="rTaskIdVal">—</span></div>
      <div class="result-row"><span data-i18n="rTarget">处理对象</span><span id="rTargetVal">—</span></div>
      <div class="result-row"><span data-i18n="rResult">处理结果</span><span class="result-status" id="rResultVal">—</span></div>
      <div class="result-row"><span data-i18n="rDuration">耗时</span><span id="rDurationVal">—</span></div>
      <div class="result-row result-timing-zh"><span>任务下发耗时</span><span id="rDispatchMs">—</span></div>
      <div class="result-row result-timing-zh"><span>任务全程耗时</span><span id="rE2eMs">—</span></div>
      <div class="result-row result-timing-zh"><span>完成回传耗时</span><span id="rNotifyMs">—</span></div>
      <div class="result-row"><span data-i18n="rDevice">设备状态</span><span id="rDeviceVal">—</span></div>
      <div class="result-photo" id="rPhoto" data-i18n="rPhoto">现场照片 / 截图（演示占位）</div>
      <button type="button" class="result-close" id="resultClose" data-i18n="close">关闭</button>
    </div>
  </div>

  <script>
    const STEP_IDS = [
      'idle', 'ack', 'go_to_B', 'arrived_B_confirmed', 'arm_start', 'arm_done', 'return_A', 'done',
    ];

    const I18N = {
      zh: {
        headerSubtitle: 'Go2 + Piper · 仿真与实机统一调度',
        taskType: '任务类型',
        dispatch: '一键派发清理任务',
        taskProgress: '任务进度',
        robotConfig: '机器人配置',
        btnGoTo: '前往目标位置',
        btnPick: '夹垃圾',
        btnDock: '回充电桩',
        runLog: '运行日志',
        showLog: '显示日志',
        hideLog: '隐藏日志',
        clearLog: '清空',
        targetView: '目标位置确认画面',
        targetViewSub: '实时 MJPEG · 叠加检测框',
        liveFeedAlt: '现场视频流',
        videoError: '视频流连接失败',
        videoErrorHint: '请确认 video_feed 服务已启动，且本机可访问配置中的 orin 地址',
        targetConfirmed: '✓ 目标位置已确认',
        resultTitle: '作业回执',
        resultSub: 'Fleet 工单式结果回传',
        rTaskId: '任务编号',
        rTarget: '处理对象',
        rResult: '处理结果',
        rDuration: '耗时',
        rDevice: '设备状态',
        rPhoto: '现场照片 / 截图（演示占位）',
        close: '关闭',
        logEmpty: '暂无日志，派发任务后将在此显示运行输出',
        scenes: {
          sim_grasp_demo: '仿真抓取演示 (Go2+Piper)',
          real_lawn_debris: '草坪异物清理（预留）',
          rain_inspect: '雨后场地巡检（预留）',
          material_drop: '养护物料投放（预留）',
        },
        steps: [
          { label: '待命', hint: '等待派单' },
          { label: '已接单', hint: '任务已受理' },
          { label: '前往目标位置', hint: '自主导航中' },
          { label: '到点确认', hint: '目标点已到达' },
          { label: '机械臂作业', hint: '识别与抓取' },
          { label: '已入篮', hint: '目标已回收' },
          { label: '返回充电桩', hint: '回充导航中' },
          { label: '完成', hint: '闭环结束' },
        ],
        failHint: {
          GO_TO_LOCATION: '前往作业点失败，已进入人工接管',
          PICK_AND_PUT: '到点确认或机械臂作业失败，已进入人工接管',
          GO_DOCKING: '返回充电区失败，已进入人工接管',
        },
        failDefault: '任务异常终止，已进入人工接管',
        errBusy: '系统正在执行其他任务，请稍后再试',
        errFailed: '任务下发失败，请稍后重试',
        errDisconnect: '连接中断，请检查设备后重试',
        errSseDropped: '连接已断开，任务仍在后台执行，请观察机器人或查看运行日志',
        taskEndedBackground: '后台任务已结束，请查看运行日志确认结果',
        resultOk: '已完成',
        resultFail: '未完成',
        deviceOk: '在线 · 待命',
        deviceFail: '需人工接管',
        durationMin: ' 分 ',
        durationSec: ' 秒',
        langBtn: 'EN',
      },
      en: {
        headerSubtitle: 'Go2 + Piper · Simulation & Real Robot Unified Dispatch',
        taskType: 'Task type',
        dispatch: 'Dispatch cleanup task',
        taskProgress: 'Task progress',
        robotConfig: 'Robot',
        btnGoTo: 'Go to target',
        btnPick: 'Pick debris',
        btnDock: 'Return to dock',
        runLog: 'Run log',
        showLog: 'Show log',
        hideLog: 'Hide log',
        clearLog: 'Clear',
        targetView: 'Target confirmation view',
        targetViewSub: 'Live MJPEG · Detection overlay',
        liveFeedAlt: 'Live video feed',
        videoError: 'Video stream connection failed',
        videoErrorHint: 'Ensure video_feed is running and orin address in config is reachable',
        targetConfirmed: '✓ Target position confirmed',
        resultTitle: 'Job receipt',
        resultSub: 'Fleet-style result report',
        rTaskId: 'Task ID',
        rTarget: 'Target',
        rResult: 'Result',
        rDuration: 'Duration',
        rDevice: 'Device status',
        rPhoto: 'Site photo / screenshot (demo placeholder)',
        close: 'Close',
        logEmpty: 'No logs yet. Output appears here after dispatch.',
        scenes: {
          sim_grasp_demo: 'Simulation Grasp Demo (Go2+Piper)',
          real_lawn_debris: 'Lawn debris cleanup (reserved)',
          rain_inspect: 'Post-rain inspection (reserved)',
          material_drop: 'Material drop (reserved)',
        },
        steps: [
          { label: 'Standby', hint: 'Awaiting dispatch' },
          { label: 'Accepted', hint: 'Task accepted' },
          { label: 'Go to target', hint: 'Autonomous navigation' },
          { label: 'Arrived', hint: 'Target reached' },
          { label: 'Arm operation', hint: 'Detect & grasp' },
          { label: 'In bin', hint: 'Target recovered' },
          { label: 'Return to dock', hint: 'Docking navigation' },
          { label: 'Done', hint: 'Cycle complete' },
        ],
        failHint: {
          GO_TO_LOCATION: 'Failed to reach work site. Manual takeover required.',
          PICK_AND_PUT: 'Arrival or arm operation failed. Manual takeover required.',
          GO_DOCKING: 'Failed to return to charging area. Manual takeover required.',
        },
        failDefault: 'Task aborted. Manual takeover required.',
        errBusy: 'Another task is running. Please try again later.',
        errFailed: 'Task dispatch failed. Please retry.',
        errDisconnect: 'Connection lost. Check device and retry.',
        errSseDropped: 'Connection lost. Task still running in background; watch the robot or check logs.',
        taskEndedBackground: 'Background task finished. Check run logs for the result.',
        resultOk: 'Completed',
        resultFail: 'Incomplete',
        deviceOk: 'Online · Standby',
        deviceFail: 'Manual takeover required',
        durationMin: ' min ',
        durationSec: ' sec',
        langBtn: '中文',
      },
    };

    const STEP_INDEX = {
      ack: 1,
      go_to_B: 2,
      arrived_B_confirmed: 3,
      arm_start: 4,
      arm_done: 5,
      return_A: 6,
      done: 7,
    };

    const SCENE_OPTIONS = [
      { value: 'sim_grasp_demo', disabled: false },
      { value: 'real_lawn_debris', disabled: true },
      { value: 'rain_inspect', disabled: true },
      { value: 'material_drop', disabled: true },
    ];

    let lang = localStorage.getItem('mower_lang') || 'zh';
    function t(key) { return (I18N[lang] && I18N[lang][key]) || (I18N.zh[key]) || key; }

    let stepIdx = 0;
    let taskStart = 0;
    let taskId = '';
    let lastFsm = '';
    let lastDispatchMs = null;
    let lastE2eMs = null;
    let lastNotifyMs = null;
    let es = null;
    let sseDropped = false;
    let taskPollTimer = null;

    const timelineEl = document.getElementById('timeline');
    const dispatchBtn = document.getElementById('dispatchBtn');
    const robotSelect = document.getElementById('robotSelect');
    const actionBtns = document.querySelectorAll('.action-btn');
    const alertBox = document.getElementById('alertBox');
    const liveFeed = document.getElementById('liveFeed');
    const liveFeedError = document.getElementById('liveFeedError');
    const liveFeedStatus = document.getElementById('liveFeedStatus');
    const resultOverlay = document.getElementById('resultOverlay');
    const logToggle = document.getElementById('logToggle');
    const logPanel = document.getElementById('logPanel');
    const logBox = document.getElementById('logBox');
    const logClear = document.getElementById('logClear');
    const langToggle = document.getElementById('langToggle');
    const sceneSelect = document.getElementById('sceneSelect');

    function applyLang() {
      const pack = I18N[lang];
      document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
      document.body.classList.toggle('lang-en', lang === 'en');
      document.body.classList.toggle('lang-zh', lang === 'zh');
      langToggle.textContent = pack.langBtn;
      document.querySelectorAll('[data-i18n]').forEach((el) => {
        const key = el.getAttribute('data-i18n');
        if (pack[key] !== undefined) el.textContent = pack[key];
      });
      document.querySelectorAll('[data-i18n-alt]').forEach((el) => {
        const key = el.getAttribute('data-i18n-alt');
        if (pack[key] !== undefined) el.alt = pack[key];
      });
      logBox.style.setProperty('--log-empty-hint', JSON.stringify(pack.logEmpty));
      const prev = sceneSelect.value || 'sim_grasp_demo';
      sceneSelect.innerHTML = SCENE_OPTIONS.map((o) =>
        '<option value="' + o.value + '"' + (o.disabled ? ' disabled' : '')
        + (o.value === prev ? ' selected' : '') + '>'
        + pack.scenes[o.value] + '</option>'
      ).join('');
      sceneSelect.value = prev;
      if (!sceneSelect.value) sceneSelect.value = 'sim_grasp_demo';
      setLogVisible(logPanel.classList.contains('show'));
      renderTimeline();
    }

    function renderTimeline() {
      const steps = I18N[lang].steps;
      timelineEl.innerHTML = STEP_IDS.map((id, i) => {
        const s = steps[i];
        let cls = '';
        if (i < stepIdx) cls = 'done';
        else if (i === stepIdx) cls = 'active';
        return '<li class="' + cls + '" data-step="' + id + '">'
          + '<span class="dot">' + (i < stepIdx ? '✓' : (i + 1)) + '</span>'
          + '<div class="step-label">' + s.label
          + '<small>' + s.hint + '</small></div></li>';
      }).join('');
    }

    function onLiveAtB() {
      liveFeedStatus.classList.add('show');
      setTimeout(() => liveFeedStatus.classList.remove('show'), 4000);
    }

    liveFeed.addEventListener('error', () => {
      liveFeed.style.display = 'none';
      liveFeedError.hidden = false;
    });
    liveFeed.addEventListener('load', () => {
      if (liveFeed.naturalWidth > 0) {
        liveFeed.style.display = 'block';
        liveFeedError.hidden = true;
      }
    });

    function appendLog(msg) {
      if (!msg) return;
      const line = document.createElement('div');
      line.className = 'log-line';
      line.textContent = msg;
      logBox.appendChild(line);
      logBox.scrollTop = logBox.scrollHeight;
    }

    function clearLog() {
      logBox.innerHTML = '';
    }

    function setLogVisible(show) {
      logPanel.classList.toggle('show', show);
      logToggle.classList.toggle('active', show);
      logToggle.textContent = show ? t('hideLog') : t('showLog');
    }

    function setStep(idx) {
      if (idx > stepIdx) stepIdx = idx;
      renderTimeline();
    }

    function handleTimeline(step) {
      const idx = STEP_INDEX[step];
      if (idx === undefined) return;
      setStep(idx);
      if (step === 'arrived_B_confirmed') onLiveAtB();
    }

    function showAlert(text, kind) {
      alertBox.textContent = text;
      alertBox.className = 'alert show alert-' + kind;
    }

    function hideAlert() {
      alertBox.className = 'alert';
    }

    function genTaskId() {
      const d = new Date();
      const p = (n, w) => String(n).padStart(w, '0');
      return 'HZ-' + d.getFullYear() + p(d.getMonth()+1,2) + p(d.getDate(),2)
        + '-' + p(d.getHours(),2) + p(d.getMinutes(),2) + p(d.getSeconds(),2);
    }

    function formatDuration(ms) {
      const s = Math.floor(ms / 1000);
      const m = Math.floor(s / 60);
      const r = s % 60;
      return m > 0 ? m + t('durationMin') + r + t('durationSec') : r + t('durationSec');
    }

    function formatTimingSec(ms) {
      if (ms == null || Number.isNaN(ms)) return '—';
      return (Math.max(0, ms) / 1000).toFixed(3) + ' 秒';
    }

    function sceneLabel(value) {
      return (I18N[lang].scenes[value] || I18N.zh.scenes.sim_grasp_demo);
    }

    function showResult(ok) {
      const elapsed = Date.now() - taskStart;
      const scene = sceneSelect.value;
      document.getElementById('rTaskIdVal').textContent = taskId;
      document.getElementById('rTargetVal').textContent = sceneLabel(scene);
      const rEl = document.getElementById('rResultVal');
      rEl.textContent = ok ? t('resultOk') : t('resultFail');
      rEl.className = 'result-status ' + (ok ? 'ok' : 'fail');
      document.getElementById('rDurationVal').textContent = formatDuration(elapsed);
      document.getElementById('rDispatchMs').textContent = formatTimingSec(lastDispatchMs);
      document.getElementById('rE2eMs').textContent = formatTimingSec(lastE2eMs);
      document.getElementById('rNotifyMs').textContent = formatTimingSec(lastNotifyMs);
      document.getElementById('rDeviceVal').textContent = ok ? t('deviceOk') : t('deviceFail');
      resultOverlay.classList.add('show');
      if (ok && stepIdx < 7) setStep(7);
    }

    function setTaskRunning(running) {
      dispatchBtn.disabled = running;
      robotSelect.disabled = running;
      actionBtns.forEach((btn) => { btn.disabled = running; });
    }

    function buildRunUrl(params) {
      const q = new URLSearchParams({ robot_id: robotSelect.value });
      Object.entries(params || {}).forEach(([k, v]) => {
        if (v !== undefined && v !== null) q.set(k, String(v));
      });
      return '/run?' + q.toString();
    }

    async function refreshVideoFeed() {
      try {
        const res = await fetch('/api/video_feed?robot_id=' + encodeURIComponent(robotSelect.value));
        const data = await res.json();
        if (data.url) {
          liveFeed.src = data.url;
          liveFeed.style.display = 'block';
          liveFeedError.hidden = true;
        }
      } catch (_) { /* 保持当前画面 */ }
    }

    function stopTaskPoll() {
      if (taskPollTimer) { clearInterval(taskPollTimer); taskPollTimer = null; }
    }

    function pollTaskUntilIdle() {
      stopTaskPoll();
      taskPollTimer = setInterval(async () => {
        try {
          const res = await fetch('/api/task/active');
          const data = await res.json();
          if (!data.active) {
            stopTaskPoll();
            showAlert(t('taskEndedBackground'), 'warn');
            setTaskRunning(false);
          }
        } catch (_) { /* 忽略轮询错误 */ }
      }, 5000);
    }

    function handleSseDropped() {
      if (sseDropped) return;
      sseDropped = true;
      if (es) { es.close(); es = null; }
      showAlert(t('errSseDropped'), 'warn');
      pollTaskUntilIdle();
    }

    function finishTask(ok, failMsg) {
      stopTaskPoll();
      sseDropped = false;
      if (es) { es.close(); es = null; }
      setTaskRunning(false);
      if (ok) {
        hideAlert();
        showResult(true);
      } else {
        const hints = I18N[lang].failHint;
        const hint = hints[lastFsm] || t('failDefault');
        showAlert(failMsg || hint, 'fail');
        const li = timelineEl.querySelector('li.active');
        if (li) li.classList.add('fail');
        showResult(false);
      }
    }

    let currentAction = null;

    function startTask(runParams, opts) {
      if (es) es.close();
      stopTaskPoll();
      sseDropped = false;
      currentAction = (opts && opts.action) || null;
      stepIdx = 0;
      lastFsm = currentAction === 'go_docking' ? 'GO_DOCKING'
        : currentAction === 'pick_and_put' ? 'PICK_AND_PUT' : 'GO_TO_LOCATION';
      taskStart = Date.now();
      lastDispatchMs = null;
      lastE2eMs = null;
      lastNotifyMs = null;
      taskId = genTaskId();
      hideAlert();
      clearLog();
      liveFeedStatus.classList.remove('show');
      refreshVideoFeed();
      renderTimeline();
      setTaskRunning(true);

      const runQuery = Object.assign({ client_start_ms: taskStart }, runParams || {});
      es = new EventSource(buildRunUrl(runQuery));
      es.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.type === 'ping') return;
        if (d.type === 'timeline') handleTimeline(d.step);
        else if (d.type === 'state') lastFsm = d.to;
        else if (d.type === 'log') appendLog(d.msg);
        else if (d.type === 'done') {
          const receivedAt = Date.now();
          if (d.dispatch_ms != null) lastDispatchMs = d.dispatch_ms;
          if (d.finished_at_ms != null && taskStart) {
            lastE2eMs = d.finished_at_ms - taskStart;
            if (lastE2eMs < 0) lastE2eMs = receivedAt - taskStart;
            lastNotifyMs = receivedAt - d.finished_at_ms;
            if (lastNotifyMs < 0) lastNotifyMs = 0;
          }
          const ok = d.ok !== undefined ? d.ok : d.final === 'FINISHED';
          finishTask(ok);
        }
        else if (d.type === 'error') {
          if (sseDropped) return;
          const raw = d.msg || '';
          const msg = raw.includes('已有任务') || raw.includes('already running')
            ? t('errBusy') : t('errFailed');
          finishTask(false, msg);
        }
      };
      es.onerror = () => {
        if (!dispatchBtn.disabled || sseDropped) return;
        handleSseDropped();
      };
    }

    function dispatch() {
      startTask({ max_retries: 3 });
    }

    dispatchBtn.addEventListener('click', dispatch);
    robotSelect.addEventListener('change', refreshVideoFeed);
    document.getElementById('btnGoTo').addEventListener('click', () => {
      startTask({ action: 'go_to_location', max_retries: 3 }, { action: 'go_to_location' });
    });
    document.getElementById('btnPick').addEventListener('click', () => {
      startTask({ action: 'pick_and_put', max_retries: 1 }, { action: 'pick_and_put' });
    });
    document.getElementById('btnDock').addEventListener('click', () => {
      startTask({ action: 'go_docking', max_retries: 3 }, { action: 'go_docking' });
    });
    langToggle.addEventListener('click', () => {
      lang = lang === 'zh' ? 'en' : 'zh';
      localStorage.setItem('mower_lang', lang);
      applyLang();
    });
    logToggle.addEventListener('click', () => setLogVisible(!logPanel.classList.contains('show')));
    logClear.addEventListener('click', clearLog);
    document.getElementById('resultClose').addEventListener('click', () => {
      resultOverlay.classList.remove('show');
    });

    applyLang();
    refreshVideoFeed();
  </script>
</body>
</html>
"""


class _QueueWriter:
    """把写入的文本按行推入队列，作为 redirect_stdout / redirect_stderr 的目标。"""

    def __init__(self, q: "queue.Queue", recorder: TaskRunRecorder | None = None):
        self._q = q
        self._recorder = recorder
        self._buf = ""

    def _emit_line(self, line: str) -> None:
        if not line:
            return
        self._q.put({"type": "log", "msg": line})
        if self._recorder is not None:
            self._recorder.log_line(line)

    def write(self, text: str):
        if not text:
            return
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit_line(line)

    def flush(self):
        if self._buf:
            self._emit_line(self._buf)
            self._buf = ""


class _QueueLogHandler(logging.Handler):
    """把 logging 模块输出推入 SSE 队列并写入任务日志。"""

    def __init__(self, q: "queue.Queue", recorder: TaskRunRecorder | None = None):
        super().__init__()
        self._q = q
        self._recorder = recorder
        self.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # 页面仅展示 INFO 及以上；DEBUG 仍写入磁盘任务日志便于排查
            if record.levelno >= logging.INFO:
                self._q.put({"type": "log", "msg": msg})
            if self._recorder is not None:
                self._recorder.log_line(msg)
        except Exception:  # noqa: BLE001
            self.handleError(record)


@contextmanager
def _capture_task_output(q: "queue.Queue", recorder: TaskRunRecorder | None = None):
    """任务运行期间捕获 print、stderr 与 logging 输出。"""
    writer = _QueueWriter(q, recorder)
    handler = _QueueLogHandler(q, recorder)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    try:
        with redirect_stdout(writer), redirect_stderr(writer):
            yield writer
    finally:
        writer.flush()
        root.removeHandler(handler)
        root.setLevel(old_level)


def _robot_options_html(selected: str = DEFAULT_ROBOT_ID) -> str:
    options = []
    for robot_id in list_robot_ids():
        sel = ' selected' if robot_id == selected else ''
        options.append(f'<option value="{robot_id}"{sel}>{robot_id}</option>')
    return '\n            '.join(options)


def _task_done_payload(
    action: str | None,
    final_state: str,
    *,
    client_start_ms: int | None = None,
    fsm_run_ms: int | None = None,
    finished_at_ms: int | None = None,
) -> dict:
    if action is None:
        ok = final_state == RobotState.FINISHED.name
    else:
        ok = final_state in _ACTION_SUCCESS_FINAL.get(action, set())
    dispatch_ms = None
    if client_start_ms is not None and fsm_run_ms is not None:
        dispatch_ms = max(0, fsm_run_ms - client_start_ms)
    return {
        "type": "done",
        "final": final_state,
        "ok": ok,
        "dispatch_ms": dispatch_ms,
        "finished_at_ms": finished_at_ms,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML.replace("__ROBOT_OPTIONS__", _robot_options_html(DEFAULT_ROBOT_ID))


@app.get("/api/task/active")
def api_task_active() -> JSONResponse:
    return JSONResponse({"active": _task_running})


@app.get("/api/video_feed")
def api_video_feed(robot_id: str = DEFAULT_ROBOT_ID) -> JSONResponse:
    try:
        config = load_robot_config(robot_id)
    except (FileNotFoundError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return JSONResponse({"url": config.video_feed_url(detect=(config.mode != "sim"))})


@app.get("/run")
def run(
    robot_id: str = Query(DEFAULT_ROBOT_ID, description="机器人编号"),
    action: str | None = Query(None, description="单步动作：go_to_location / pick_and_put / go_docking"),
    max_retries: int = Query(3, ge=1, le=10, description="动作最大重试次数"),
    client_start_ms: int | None = Query(None, description="前端点击派发时的 Date.now() 毫秒时间戳"),
) -> StreamingResponse:
    try:
        config = load_robot_config(robot_id)
    except (FileNotFoundError, ValueError) as exc:
        return StreamingResponse(
            iter([f'data: {json.dumps({"type": "error", "msg": str(exc)}, ensure_ascii=False)}\n\n']),
            media_type="text/event-stream",
        )

    action_handlers = _ACTION_HANDLERS_SIM if config.mode == "sim" else _ACTION_HANDLERS_REAL
    if action is not None and action not in action_handlers:
        return StreamingResponse(
            iter([f'data: {json.dumps({"type": "error", "msg": f"未知动作: {action}"}, ensure_ascii=False)}\n\n']),
            media_type="text/event-stream",
        )

    def event_stream():
        global _task_running
        if not _run_lock.acquire(blocking=False):
            reason = "已有任务正在运行，请稍后再试"
            log_task_rejected(robot_id, action, reason)
            payload = {"type": "error", "msg": reason}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return
        _task_running = True
        recorder: TaskRunRecorder | None = None
        try:
            q: "queue.Queue" = queue.Queue()
            recorder = TaskRunRecorder.open(
                robot_id=robot_id,
                action=action,
                max_retries=max_retries,
                mode=config.mode,
                host=config.host,
                orin=config.orin,
                port=config.port,
                grasp_port=config.grasp_port,
                execution_url=config.execution_url,
            )

            def on_timeline(step: str):
                q.put({"type": "timeline", "step": step})
                recorder.event("timeline", step=step)

            state_timeline = {
                ("GO_TO_LOCATION", "PICK_AND_PUT"): "arrived_B_confirmed",
                ("GO_DOCKING", "FINISHED"): "done",
            }

            timing_info: dict[str, int | None] = {
                "fsm_run_ms": None,
                "finished_at_ms": None,
            }

            def on_change(old, new):
                q.put({"type": "state", "from": old.name, "to": new.name})
                recorder.event("state", from_state=old.name, to_state=new.name)
                if new.name == RobotState.FINISHED.name:
                    timing_info["finished_at_ms"] = int(time.time() * 1000)
                step = state_timeline.get((old.name, new.name))
                if step:
                    q.put({"type": "timeline", "step": step})
                    recorder.event("timeline", step=step)

            def on_heartbeat():
                q.put({"type": "ping"})

            def worker():
                final_state: str | None = None
                task_ok: bool | None = None
                task_error: str | None = None
                done_payload: dict | None = None
                try:
                    q.put({"type": "timeline", "step": "ack"})
                    recorder.mark("task_ack")
                    recorder.event("timeline", step="ack")
                    with _capture_task_output(q, recorder):
                        fsm = RobotTaskFSM(
                            config=config,
                            max_retries=max_retries,
                            on_state_change=on_change,
                            on_timeline=on_timeline,
                            on_mark=recorder.mark,
                            on_heartbeat=on_heartbeat,
                        )
                        timing_info["fsm_run_ms"] = int(time.time() * 1000)
                        if action is None:
                            fsm.run()
                            final_state = fsm.state.name
                        else:
                            handler = action_handlers[action]
                            next_state = handler(fsm)
                            fsm._set_state(next_state)
                            final_state = fsm.state.name
                        if (
                            timing_info["finished_at_ms"] is None
                            and final_state == RobotState.FINISHED.name
                        ):
                            timing_info["finished_at_ms"] = int(time.time() * 1000)
                    done_payload = _task_done_payload(
                        action,
                        final_state,
                        client_start_ms=client_start_ms,
                        fsm_run_ms=timing_info["fsm_run_ms"],
                        finished_at_ms=timing_info["finished_at_ms"],
                    )
                    task_ok = done_payload["ok"]
                    q.put(done_payload)
                except Exception as exc:  # noqa: BLE001
                    task_error = str(exc)
                    q.put({"type": "error", "msg": task_error})
                finally:
                    if recorder is not None:
                        for line in recorder.emit_timing_summary():
                            q.put({"type": "log", "msg": line})
                        recorder.finish(
                            final_state=final_state,
                            ok=task_ok,
                            error=task_error,
                            dispatch_ms=(
                                done_payload.get("dispatch_ms") if done_payload else None
                            ),
                            finished_at_ms=timing_info.get("finished_at_ms"),
                        )
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()

            while True:
                try:
                    item = q.get(timeout=_SSE_KEEPALIVE_SEC)
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'ping'}, ensure_ascii=False)}\n\n"
                    continue
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            _task_running = False
            _run_lock.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def run_server():
    import uvicorn

    uvicorn.run("scheduler.server.app:app", host="0.0.0.0", port=8200)


if __name__ == "__main__":
    run_server()
