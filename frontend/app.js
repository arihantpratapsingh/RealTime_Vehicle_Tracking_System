// --- START OF FILE app.js ---

// Simple state management
const state = {
  file: null,
  fileUrl: null,
  isPlaying: false,
  isProcessing: false,
  showOverlay: true,
  stats: { cars: 0, trucks: 0, buses: 0, bikes: 0 },
  fps: 0,
  logs: [],
  confidenceThreshold: 0.5,
  iouThreshold: 0.45,
  frameCount: 0,
  lastFpsTime: performance.now(),
  analytics: { total: 0, passedUp: 0, passedDown: 0 },
  linePos: 0.5,
};

const WS_URL = 'ws://localhost:8000/ws/detect';
let ws = null;
let wsConnected = false;
let wsBusy = false; 

// This canvas is used to compress the image before sending to backend
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

// Elements
const videoEl = document.getElementById('video');
const canvasEl = document.getElementById('canvas');
const emptyStateEl = document.getElementById('empty-state');
const fileInputEl = document.getElementById('file-input');
const playBtn = document.getElementById('play-btn');
const playIcon = document.getElementById('play-icon');
const pauseIcon = document.getElementById('pause-icon');
const fileNameEl = document.getElementById('file-name');
const overlayToggle = document.getElementById('overlay-toggle');
const eyeOn = document.getElementById('eye-on');
const eyeOff = document.getElementById('eye-off');
const clearBtn = document.getElementById('clear-btn');
const confSlider = document.getElementById('conf-slider');
const confLabel = document.getElementById('conf-label');
const iouSlider = document.getElementById('iou-slider');
const iouLabel = document.getElementById('iou-label');
const processingDot = document.getElementById('processing-dot');
const processingText = document.getElementById('processing-text');
const statCars = document.getElementById('stat-cars');
const statTrucks = document.getElementById('stat-trucks');
const statBikes = document.getElementById('stat-bikes');
const statBuses = document.getElementById('stat-buses');
const logList = document.getElementById('log-list');
const eventsCount = document.getElementById('events-count');
const inferenceTimeEl = document.getElementById('inference-time');
const exportBtn = document.getElementById('export-btn');
const gpuUsageText = document.getElementById('gpu-usage-text');
const gpuUsageBar = document.getElementById('gpu-usage-bar');
const navLive = document.getElementById('nav-live');
const navAnalytics = document.getElementById('nav-analytics');
const navSettings = document.getElementById('nav-settings');
const sectionLive = document.getElementById('section-live');
const sectionAnalytics = document.getElementById('section-analytics');
const sectionSettings = document.getElementById('section-settings');
const analyticsTotal = document.getElementById('analytics-total');
const analyticsUp = document.getElementById('analytics-up');
const analyticsDown = document.getElementById('analytics-down');
const analyticsFps = document.getElementById('analytics-fps');
const settingOverlay = document.getElementById('setting-overlay');
const linePosSlider = document.getElementById('line-pos-slider');
const linePosLabel = document.getElementById('line-pos-label');
const settingsConfSlider = document.getElementById('settings-conf-slider');
const settingsIouSlider = document.getElementById('settings-iou-slider');
const settingsConfLabel = document.getElementById('settings-conf-label');
const settingsIouLabel = document.getElementById('settings-iou-label');

// --- HELPER FUNCTIONS ---

function setProcessing(isProcessing) {
  state.isProcessing = isProcessing;
  processingDot.className = `w-2 h-2 rounded-full ${
    isProcessing ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
  }`;
  processingText.textContent = isProcessing ? `PROCESSING ${state.fps} FPS` : "IDLE";
}

function updateStatsDisplay() {
  statCars.textContent = state.stats.cars;
  statTrucks.textContent = state.stats.trucks;
  statBikes.textContent = state.stats.bikes;
  statBuses.textContent = state.stats.buses;
}

function addLogItem({ type, time, confidence }) {
  const container = document.createElement("div");
  container.className =
    "flex items-center justify-between p-3 border-b border-slate-800 hover:bg-slate-800/50 transition-colors text-sm";
  container.dataset.type = type;
  container.dataset.time = time;
  container.dataset.confidence = (confidence * 100).toFixed(0);

  const left = document.createElement("div");
  left.className = "flex items-center gap-3";

  const iconWrap = document.createElement("span");
  iconWrap.className = "p-1.5 rounded-md bg-slate-800";
  const color =
    type === "truck"
      ? "text-orange-400"
      : type === "motorbike"
      ? "text-purple-400"
      : "text-blue-400";

  iconWrap.classList.add(color);
  iconWrap.innerHTML =
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>';

  const textWrap = document.createElement("div");
  const title = document.createElement("p");
  title.className = "font-medium text-slate-200 capitalize";
  title.textContent = type;

  const sub = document.createElement("p");
  sub.className = "text-xs text-slate-500";
  sub.textContent = time;

  textWrap.appendChild(title);
  textWrap.appendChild(sub);

  left.appendChild(iconWrap);
  left.appendChild(textWrap);

  const right = document.createElement("div");
  right.className = "flex items-center gap-2";

  const bar = document.createElement("div");
  bar.className = "h-1.5 w-16 bg-slate-800 rounded-full overflow-hidden";

  const fill = document.createElement("div");
  fill.className = "h-full bg-emerald-500 rounded-full";
  fill.style.width = `${confidence * 100}%`;

  bar.appendChild(fill);

  const pct = document.createElement("span");
  pct.className = "text-xs font-mono text-slate-400";
  pct.textContent = `${(confidence * 100).toFixed(0)}%`;

  right.appendChild(bar);
  right.appendChild(pct);

  container.appendChild(left);
  container.appendChild(right);

  logList.prepend(container);
  while (logList.children.length > 50) logList.removeChild(logList.lastChild);

  eventsCount.textContent = `${logList.children.length} Events`;
}

// Simple tracking for counting line crossings
const tracks = [];
function updateTracks(scaledDetections) {
  const maxDist = 50;
  const lineY = canvasEl.height * state.linePos;
  scaledDetections.forEach((d) => {
    const cx = d.x + d.w / 2;
    const cy = d.y + d.h / 2;
    let bestIdx = -1;
    let bestDist = Infinity;
    for (let i = 0; i < tracks.length; i++) {
      const t = tracks[i];
      if (t.class !== d.class) continue;
      const dist = Math.hypot(t.cx - cx, t.cy - cy);
      if (dist < maxDist && dist < bestDist) { bestDist = dist; bestIdx = i; }
    }
    if (bestIdx >= 0) {
      const t = tracks[bestIdx];
      const prevCy = t.cy;
      t.cx = cx; t.cy = cy; t.lastSeen = performance.now();
      if (prevCy < lineY && cy >= lineY && !t.markedDown) { state.analytics.passedDown++; t.markedDown = true; t.markedUp = false; }
      if (prevCy > lineY && cy <= lineY && !t.markedUp) { state.analytics.passedUp++; t.markedUp = true; t.markedDown = false; }
    } else {
      tracks.push({ class: d.class, cx, cy, lastSeen: performance.now(), markedUp: false, markedDown: false });
    }
  });
  for (let i = tracks.length - 1; i >= 0; i--) {
    if (performance.now() - tracks[i].lastSeen > 2000) tracks.splice(i, 1);
  }
}

function handleFileUpload(file) {
  if (!file) return;
  state.file = file;
  state.fileUrl = URL.createObjectURL(file);
  
  // Reset
  videoEl.src = state.fileUrl;
  videoEl.currentTime = 0;
  state.stats = { cars: 0, trucks: 0, buses: 0, bikes: 0 };
  state.analytics = { total: 0, passedUp: 0, passedDown: 0 };
  tracks.length = 0;
  logList.innerHTML = "";
  
  fileNameEl.textContent = file.name || "Unknown source";
  emptyStateEl.style.display = "none";
  updateStatsDisplay();

  // Initialize canvas size once metadata is loaded
  videoEl.onloadedmetadata = () => {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    // Draw the first frame static
    requestAnimationFrame(() => {
        const ctx = canvasEl.getContext("2d");
        ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    });
  };
}

// --- CORE LOGIC: LOCK-STEP PROCESSING ---

function togglePlayback() {
  if (!videoEl.src) return;

  if (state.isPlaying) {
    // Stop
    state.isPlaying = false;
    setProcessing(false);
    videoEl.pause(); // Ensure paused
  } else {
    // Start
    state.isPlaying = true;
    setProcessing(true);
    // Don't call videoEl.play()! We drive it manually.
    processCurrentFrame(); 
  }

  playIcon.classList.toggle("hidden", state.isPlaying);
  pauseIcon.classList.toggle("hidden", !state.isPlaying);
}

// 1. Capture Frame & Send
// Optimized: Send Binary Blob instead of Base64 String
function processCurrentFrame() {
  if (!state.isPlaying) return;
  if (!wsConnected || wsBusy) return;

  // Measure FPS
  const now = performance.now();
  state.frameCount++;
  if (now - state.lastFpsTime >= 1000) {
    state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFpsTime));
    state.frameCount = 0;
    state.lastFpsTime = now;
    processingText.textContent = `PROCESSING ${state.fps} FPS`;
  }

  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;

  if (vw && vh) {
    // 1. Resize logic (Keep it small for transfer speed)
    // 640 width is standard YOLO input size.
    const targetW = 640;
    const targetH = Math.round(vh * (targetW / vw));

    captureCanvas.width = targetW;
    captureCanvas.height = targetH;
    
    // Draw current video frame to small canvas
    captureCtx.drawImage(videoEl, 0, 0, targetW, targetH);

    wsBusy = true;
    state.lastRequestTime = performance.now();

    // 2. Convert to Blob (Binary) - JPEG quality 0.5 is usually fine for detection
    // .toBlob is asynchronous and sends raw bytes
    captureCanvas.toBlob((blob) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(blob); // <--- SENDING BYTES HERE
      } else {
        wsBusy = false;
      }
    }, 'image/jpeg', 0.5); 
  }
}

// 2. Render Results & Advance Frame
function renderAndAdvance(detections) {
  const ctx = canvasEl.getContext("2d");
  
  // OPTIMIZATION: Only resize canvas if dimensions actually change.
  // Setting .width/.height clears the canvas, causing black flicker if done every frame.
  if (canvasEl.width !== videoEl.videoWidth || canvasEl.height !== videoEl.videoHeight) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
  }

  // A. Draw the actual video frame first (This acts as the player view)
  // We force the drawImage to happen before we process detections
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

  // B. Draw Overlay
  if (state.showOverlay && detections) {
    const containerW = canvasEl.width;
    const containerH = canvasEl.height;

    // Detection coordinates come from 640xN, scale to full res
    const yoloW = 640;
    // We calculate yoloH based on aspect ratio to ensure boxes align perfectly
    const yoloH = Math.round(videoEl.videoHeight * (640 / videoEl.videoWidth));
    
    const scaleX = containerW / yoloW;
    const scaleY = containerH / yoloH;

    const scaled = [];
    detections.forEach((det) => {
      if (det.confidence < state.confidenceThreshold) return;

      const x = det.x * scaleX;
      const y = det.y * scaleY;
      const w = det.w * scaleX;
      const h = det.h * scaleY;

      let color = "#3b82f6";
      if (det.class === "truck") color = "#f97316";
      if (det.class === "bus") color = "#eab308";
      if (det.class === "motorbike") color = "#a855f7";

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      const label = `${det.class} ${Math.round(det.confidence * 100)}%`;
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.fillRect(x, y - 22, textWidth + 10, 22);

      ctx.globalAlpha = 1;
      ctx.fillStyle = "#fff";
      ctx.font = "bold 14px sans-serif";
      ctx.fillText(label, x + 5, y - 6);
      
      scaled.push({ class: det.class, x, y, w, h });
    });

    updateTracks(scaled);

    // Draw Line
    const lineYDraw = containerH * state.linePos;
    ctx.strokeStyle = 'rgba(99,102,241,0.9)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6,6]);
    ctx.beginPath();
    ctx.moveTo(0, lineYDraw);
    ctx.lineTo(containerW, lineYDraw);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // C. Calculate Inference Time
  const t1 = performance.now();
  inferenceTimeEl.textContent = `${(t1 - state.lastRequestTime).toFixed(1)}ms`;

  // D. Advance Video to next frame
  if (state.isPlaying && !videoEl.ended) {
    // Determine step based on video FPS (default to 30fps / 0.033s if unknown)
    const step = 1 / 30; 
    videoEl.currentTime = Math.min(videoEl.duration, videoEl.currentTime + step);
  } else if (videoEl.ended) {
    state.isPlaying = false;
    setProcessing(false);
    playIcon.classList.remove("hidden");
    pauseIcon.classList.add("hidden");
  }
}

// 3. Listener: When video has finished seeking to the new time, process again
videoEl.addEventListener('seeked', () => {
  if (state.isPlaying) {
    // Small timeout prevents browser choking if seeking is instant
    // requestAnimationFrame ensures we paint the DOM before grabbing image
    requestAnimationFrame(() => processCurrentFrame());
  }
});

// --- WEBSOCKET SETUP ---

function connectWS() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    wsConnected = true;
    console.log("WS Connected");
  };

  ws.onmessage = (event) => {
    wsBusy = false;

    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      data = null;
    }

    if (!data) return;

    // Log logic
    const newDets = data.detections || [];
    state.stats = data.stats || {};
    updateStatsDisplay();
    
    // Analytics
    state.analytics.total += newDets.length; // Note: this is sum of dets per frame, not unique.
    // Logic for logs (throttle slightly or filter)
    const nowStr = new Date().toLocaleTimeString();
    // Only log high confidence events or new IDs (simplified here)
    // For specific log logic, you might want to filter by Track ID if available
    
    analyticsTotal.textContent = state.analytics.total; // This metric might grow huge, usually you count unique IDs
    analyticsUp.textContent = state.analytics.passedUp;
    analyticsDown.textContent = state.analytics.passedDown;
    analyticsFps.textContent = state.fps;

    // Trigger rendering and next frame
    renderAndAdvance(newDets);
  };

  ws.onerror = () => {
    wsConnected = false;
    wsBusy = false;
  };

  ws.onclose = () => {
    wsConnected = false;
    wsBusy = false;
    setTimeout(connectWS, 1000);
  };
}

// --- EVENT LISTENERS ---

fileInputEl.addEventListener("change", (e) =>
  handleFileUpload(e.target.files[0])
);

playBtn.addEventListener("click", togglePlayback);

// Remove default video events like 'play'/'pause' controlling logic
// because we control currentTime manually.

overlayToggle.addEventListener("click", () => {
  state.showOverlay = !state.showOverlay;
  overlayToggle.className = `p-2 rounded-lg border ${
    state.showOverlay
      ? "bg-indigo-500/20 border-indigo-500 text-indigo-400"
      : "border-slate-600 text-slate-400"
  }`;
  eyeOn.classList.toggle("hidden", !state.showOverlay);
  eyeOff.classList.toggle("hidden", state.showOverlay);
});

clearBtn.addEventListener("click", () => {
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = null;
  state.fileUrl = null;
  state.isPlaying = false;
  
  setProcessing(false);
  videoEl.pause();
  videoEl.removeAttribute("src");
  videoEl.load();

  const ctx = canvasEl.getContext('2d');
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  emptyStateEl.style.display = "";
  playIcon.classList.remove("hidden");
  pauseIcon.classList.add("hidden");
});

confSlider.addEventListener("input", (e) => {
  state.confidenceThreshold = parseFloat(e.target.value);
  confLabel.textContent = `${(state.confidenceThreshold * 100).toFixed(0)}%`;
});

iouSlider.addEventListener("input", (e) => {
  state.iouThreshold = parseFloat(e.target.value);
  iouLabel.textContent = `${(state.iouThreshold * 100).toFixed(0)}%`;
});

// Export CSV
exportBtn && exportBtn.addEventListener('click', () => {
  const rows = ['time,type,confidence'];
  const children = Array.from(logList.children);
  children.forEach(child => {
    rows.push(`${child.dataset.time},${child.dataset.type},${child.dataset.confidence}`);
  });
  const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'report.csv';
  a.click();
  URL.revokeObjectURL(url);
});

// Settings / Tabs logic
function setTab(tab) {
  const showLive = tab === 'live';
  const showAnalytics = tab === 'analytics';
  const showSettings = tab === 'settings';
  sectionLive.classList.toggle('hidden', !showLive);
  sectionAnalytics.classList.toggle('hidden', !showAnalytics);
  sectionSettings.classList.toggle('hidden', !showSettings);
  
  if(navLive) {
    navLive.classList.toggle('bg-indigo-600/10', showLive);
    navLive.classList.toggle('text-indigo-400', showLive);
    navAnalytics.classList.toggle('bg-indigo-600/10', showAnalytics);
    navAnalytics.classList.toggle('text-indigo-400', showAnalytics);
    navSettings.classList.toggle('bg-indigo-600/10', showSettings);
    navSettings.classList.toggle('text-indigo-400', showSettings);
  }
}
navLive && navLive.addEventListener('click', () => setTab('live'));
navAnalytics && navAnalytics.addEventListener('click', () => setTab('analytics'));
navSettings && navSettings.addEventListener('click', () => setTab('settings'));
setTab('live');

// Settings bindings
settingOverlay && settingOverlay.addEventListener('change', (e) => { state.showOverlay = e.target.checked; });
linePosSlider && linePosSlider.addEventListener('input', (e) => { state.linePos = parseFloat(e.target.value); linePosLabel.textContent = `${Math.round(state.linePos*100)}%`; });
settingsConfSlider && settingsConfSlider.addEventListener('input', (e) => { state.confidenceThreshold = parseFloat(e.target.value); settingsConfLabel.textContent = `${(state.confidenceThreshold*100).toFixed(0)}%`; confSlider.value = state.confidenceThreshold; confLabel.textContent = settingsConfLabel.textContent; });
settingsIouSlider && settingsIouSlider.addEventListener('input', (e) => { state.iouThreshold = parseFloat(e.target.value); settingsIouLabel.textContent = `${(state.iouThreshold*100).toFixed(0)}%`; iouSlider.value = state.iouThreshold; iouLabel.textContent = settingsIouLabel.textContent; });

// Init
updateStatsDisplay();
connectWS();

// Poll system stats
async function fetchSystemStats() {
  try {
    const res = await fetch('http://localhost:8000/system/stats');
    const data = await res.json();
    const pct = data.gpu_memory_used_percent ?? data.gpu_utilization_percent;
    if (pct !== null) {
      gpuUsageText.textContent = `${pct}%`; gpuUsageBar.style.width = `${pct}%`;
    } else {
      gpuUsageText.textContent = '—'; gpuUsageBar.style.width = '0%';
    }
  } catch {}
}
setInterval(fetchSystemStats, 2000);
fetchSystemStats();