const TIMELINE = [
  ["ack", "已接单", "任务已同步至户外作业设备"],
  ["go_to_B", "前往目标点", "设备正在低速前往目标区域"],
  ["arrived_B_confirmed", "到点确认", "D455 已完成目标复核"],
  ["arm_start", "机械臂作业", "D1 正在执行安全抓取动作"],
  ["arm_done", "已入篮", "异物已放入固定回收篮"],
  ["return_A", "返回 A 点", "设备正在返回待命区域"],
  ["done", "任务完成", "结果与现场证据已回传"],
];

const ACTIVE_STATES = new Set([
  "ack",
  "go_to_B",
  "arrived_B_confirmed",
  "arm_start",
  "arm_done",
  "return_A",
  "paused",
]);

const stateOrder = Object.fromEntries(TIMELINE.map(([state], index) => [state, index]));
let lastStatus = "idle";
let resultShownFor = null;
let toastTimer = null;

const $ = (id) => document.getElementById(id);

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || "服务暂时不可用");
  return data;
}

function formatElapsed(seconds = 0) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60).toString().padStart(2, "0");
  const remainder = Math.floor(value % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  return new Date(timestamp * 1000).toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function eventTime(task, state) {
  const event = [...task.history].reverse().find((item) => item.status === state);
  return event ? formatTime(event.timestamp_s) : "";
}

function renderTimeline(task) {
  const currentIndex = stateOrder[task.status] ?? -1;
  const terminal = ["done", "failed", "manual_takeover"].includes(task.status);
  $("timeline").innerHTML = TIMELINE.map(([state, title, description], index) => {
    const done = task.status === "done" || index < currentIndex;
    const current = task.status === state;
    const failed = terminal && task.status !== "done" && index === Math.max(0, currentIndex);
    const classes = ["timeline-step", done ? "done" : "", current ? "current" : "", failed ? "failed" : ""]
      .filter(Boolean)
      .join(" ");
    const node = done ? "✓" : String(index + 1);
    return `
      <div class="${classes}">
        <span class="timeline-node">${node}</span>
        <div>
          <p class="timeline-title">${title}</p>
          <p class="timeline-description">${description}</p>
        </div>
        <time class="timeline-time">${eventTime(task, state)}</time>
      </div>
    `;
  }).join("");
}

function render(task) {
  const active = ACTIVE_STATES.has(task.status);
  const takeover = task.status === "manual_takeover";
  $("device-status").textContent = task.device_status;
  $("task-id").textContent = task.task_id || "尚未派单";
  $("task-scene").textContent = task.scene;
  $("elapsed-time").textContent = formatElapsed(task.elapsed_s);
  $("result-state").textContent = task.result || (active ? "执行中" : "等待任务");
  $("dispatch-button").disabled = active || takeover;
  $("dispatch-label").textContent = takeover ? "人工接管中" : active ? "任务执行中" : "一键派发清理任务";
  $("dispatch-hint").textContent = task.status_message;
  $("task-status-pill").textContent = task.status_message;
  $("evidence-image").src = task.evidence_image;
  $("result-image").src = task.evidence_image;
  $("camera-status").textContent = cameraLabel(task.status);
  $("rain-banner").classList.toggle("hidden", !task.rain_mode);
  $("rain-button").textContent = task.rain_mode ? "关闭雨天模式" : "开启雨天模式";
  $("pause-button").textContent = task.status === "paused" ? "继续任务" : "暂停任务";
  $("pause-button").disabled = !active;
  $("takeover-button").textContent = takeover ? "返回待命" : "人工接管";
  renderTimeline(task);

  if (task.status === "done" && resultShownFor !== task.task_id) {
    resultShownFor = task.task_id;
    showResult(task);
  }
  if (task.status === "failed" || task.status === "manual_takeover") {
    showToast(task.status_message, true);
  }
  lastStatus = task.status;
}

function cameraLabel(status) {
  if (["arrived_B_confirmed", "arm_start", "arm_done", "done"].includes(status)) {
    return "目标已确认 · 证据已留存";
  }
  if (status === "go_to_B") return "接近目标区域";
  return "等待现场确认";
}

function showResult(task) {
  $("result-task-id").textContent = task.task_id;
  $("result-object").textContent = task.object_name;
  $("result-message").textContent = task.result;
  $("result-elapsed").textContent = formatElapsed(task.elapsed_s);
  $("result-device").textContent = task.device_status;
  $("result-modal").classList.remove("hidden");
}

function showToast(message, error = false) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").classList.remove("hidden");
  toastTimer = setTimeout(() => $("toast").classList.add("hidden"), 3600);
}

async function dispatchTask() {
  try {
    const task = await request("/api/task/dispatch", {
      method: "POST",
      body: JSON.stringify({ scene: "草坪异物清理" }),
    });
    resultShownFor = null;
    render(task);
    showToast("清理任务已派发，设备已接单");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function pauseOrResume() {
  try {
    const endpoint = lastStatus === "paused" ? "/api/task/resume" : "/api/task/pause";
    const task = await request(endpoint, { method: "POST", body: "{}" });
    render(task);
    showToast(task.status_message);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function takeover() {
  try {
    const endpoint = lastStatus === "manual_takeover" ? "/api/task/reset" : "/api/task/takeover";
    const task = await request(endpoint, { method: "POST", body: "{}" });
    render(task);
    if (endpoint.endsWith("reset")) showToast("设备已返回待命状态");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function toggleRainMode() {
  const enabled = $("rain-banner").classList.contains("hidden");
  try {
    const task = await request("/api/mode/rain", {
      method: "POST",
      body: JSON.stringify({ enabled }),
    });
    render(task);
    showToast(enabled ? "已切换至雨天室内演示预案" : "已恢复标准户外演示模式");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function reset() {
  const task = await request("/api/task/reset", { method: "POST", body: "{}" });
  $("result-modal").classList.add("hidden");
  resultShownFor = null;
  render(task);
}

async function refresh() {
  try {
    render(await request("/api/task"));
  } catch (error) {
    showToast("暂时无法连接任务服务", true);
  }
}

function updateClock() {
  $("current-time").textContent = new Date().toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

$("dispatch-button").addEventListener("click", dispatchTask);
$("pause-button").addEventListener("click", pauseOrResume);
$("takeover-button").addEventListener("click", takeover);
$("rain-button").addEventListener("click", toggleRainMode);
$("reset-button").addEventListener("click", reset);
$("close-result").addEventListener("click", () => $("result-modal").classList.add("hidden"));

// --- 相机流管理 ---
let cameraConnected = false;
let cameraDisconnectCount = 0;

async function checkCamera() {
  try {
    const res = await request("/api/camera/status");
    if (res.connected && !cameraConnected) {
      cameraConnected = true;
      cameraDisconnectCount = 0;
      activateCameraStream();
    } else if (!res.connected && cameraConnected) {
      // 连续多次确认断连才真正切走, 避免瞬时波动导致画面闪断
      cameraDisconnectCount++;
      if (cameraDisconnectCount >= 3) {
        cameraConnected = false;
        cameraDisconnectCount = 0;
        deactivateCameraStream();
      }
    } else if (res.connected) {
      cameraDisconnectCount = 0;
    }
  } catch (_) {
    // 网络错误不切断画面
  }
}

function activateCameraStream() {
  const streamImg = $("camera-stream");
  const placeholderImg = $("evidence-image");
  const focusBox = $("focus-box");
  const badge = $("camera-live-badge");

  streamImg.src = "/api/camera/stream";
  streamImg.classList.remove("hidden");
  placeholderImg.classList.add("hidden");
  focusBox.classList.add("hidden");
  badge.classList.add("active");
}

function deactivateCameraStream() {
  const streamImg = $("camera-stream");
  const placeholderImg = $("evidence-image");
  const focusBox = $("focus-box");
  const badge = $("camera-live-badge");

  streamImg.src = "";
  streamImg.classList.add("hidden");
  placeholderImg.classList.remove("hidden");
  focusBox.classList.remove("hidden");
  badge.classList.remove("active");
}

updateClock();
setInterval(updateClock, 1000);
refresh();
setInterval(refresh, 700);
checkCamera();
setInterval(checkCamera, 3000);
