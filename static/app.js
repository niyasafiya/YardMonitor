/* Yard Monitor — dashboard client */

const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  token: localStorage.getItem("ym_token") || null,
  ws: null,
  reconnectMs: 1000,
  events: [],
  whitelist: [],
  assets: [],
  cameras: [],
  gateOpen: false,
  gateAutoCloseTimer: null,
  gateCloseSec: 6,
};


// ============================ Login flow ============================
function showLogin(err) {
  $("#login").style.display = "flex";
  if (err) { $("#login-err").textContent = err; $("#login-err").hidden = false; }
  setTimeout(() => $("#login-pw").focus(), 50);
}
function hideLogin() { $("#login").style.display = "none"; $("#login-err").hidden = true; }

async function login() {
  const pw = $("#login-pw").value.trim();
  if (!pw) { showLogin("Please enter the password."); return; }
  $("#login-btn").disabled = true;
  $("#login-btn").innerHTML = '<span class="spinner"></span> Signing in…';
  try {
    const r = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: pw }),
    });
    if (!r.ok) throw new Error("Wrong password");
    const data = await r.json();
    state.token = data.token;
    localStorage.setItem("ym_token", state.token);
    hideLogin();
    init();
  } catch (e) {
    showLogin(e.message);
  } finally {
    $("#login-btn").disabled = false;
    $("#login-btn").textContent = "Sign in";
  }
}

$("#login-btn").addEventListener("click", login);
$("#login-pw").addEventListener("keydown", (e) => { if (e.key === "Enter") login(); });
$("#btn-logout").addEventListener("click", () => {
  localStorage.removeItem("ym_token");
  location.reload();
});

// ============================ API helpers ============================
async function api(path, opts = {}) {
  const headers = opts.headers || {};
  if (state.token) headers["x-admin-token"] = state.token;
  if (opts.body && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
  const r = await fetch(path, { ...opts, headers });
  if (r.status === 401) {
    localStorage.removeItem("ym_token");
    closeModal();
    showLogin("Session expired — please sign in again.");
    throw new Error("unauthorized");
  }
  if (!r.ok) {
    let msg = `Request failed (${r.status})`;
    try { const j = await r.json(); if (j.detail) msg = j.detail; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

// ============================ WebSocket ============================
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;

  ws.onopen = () => {
    setWsStatus(true);
    state.reconnectMs = 1000;
  };
  ws.onclose = () => {
    setWsStatus(false);
    setTimeout(connectWS, state.reconnectMs);
    state.reconnectMs = Math.min(state.reconnectMs * 2, 15000);
  };
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    handleMessage(msg);
  };
}

function setWsStatus(ok) {
  const el = $("#ws-status");
  el.innerHTML = `<span class="led"></span>${ok ? "Live" : "Disconnected"}`;
  el.classList.toggle("ok", ok);
  el.classList.toggle("bad", !ok);
}

function handleMessage(msg) {
  if (msg.type === "snapshot") {
    const d = msg.data;
    updateStats(d.stats);
    state.events  = d.events  || [];  renderEvents();
    state.assets  = d.assets  || [];  renderAssets();
    state.cameras = d.cameras || [];  renderCameras();
    setGateStatus(d.gate_open);
    refreshWhitelist();
    refreshStats();
    return;
  }
  if (msg.type === "event") {
    const ev = msg.data;
    state.events.unshift(ev);
    state.events = state.events.slice(0, 100);
    renderEvents();           // instant UI update
    toastEvent(ev);           // popup notification
    refreshStats();           // update KPI counters immediately
    // Update gate status instantly from the event
    if (ev.gate_opened || ev.gate_is_open) setGateStatus(true);
    else if (ev.authorized === false)       setGateStatus(false);
    return;
  }
  if (msg.type === "camera_source") {
    camSources[msg.data?.id] = msg.data?.uri;
    renderCameras();
    return;
  }
  if (msg.type === "event_deleted") {
    state.events = state.events.filter(ev => ev.id !== msg.data?.id);
    renderEvents();
    return;
  }
  if (msg.type === "whitelist_update") { refreshWhitelist(); refreshStats(); return; }
  if (msg.type === "gate")             { refreshGate(); return; }
  if (msg.type === "system")           { toast({type:"warn", title:"System", body: msg.data?.msg || ""}); return; }
}

// ============================ Stats ============================
function updateStats(s) {
  if (!s) return;
  $("#kpi-entries").textContent   = s.entries_today   ?? 0;
  $("#kpi-exits").textContent     = s.exits_today     ?? 0;
  $("#kpi-denied").textContent    = s.denied_today    ?? 0;
  $("#kpi-whitelist").textContent = s.whitelist_size  ?? 0;
  $("#kpi-assets").textContent    = s.assets_present  ?? 0;
}
async function refreshStats() {
  try { updateStats(await api("/api/stats")); } catch {}
}

// ============================ Cameras ============================
// Track which source is active per camera  { camera_id: uri (0 or 1) }
const camSources = {};

function renderCameras() {
  const root = $("#camera-list");
  const cams = state.cameras;
  $("#cam-meta").textContent = cams.length ? `${cams.length} online` : "no cameras";
  if (!cams || cams.length === 0) {
    root.innerHTML = `<div class="placeholder">No cameras configured.<br>
      <span class="muted small">Edit <code>config.yaml</code> and restart.</span></div>`;
    return;
  }

  root.innerHTML = cams.map(c => {
    const active       = camSources[c.id] !== undefined ? camSources[c.id] : c.uri;
    const laptopActive = (String(active) === "0");
    const ivcamActive  = (String(active) === "1");
    return `
    <div class="camera" data-cam="${escapeHtml(c.id)}">
      <div class="cam-head">
        <span><strong>${escapeHtml(c.name)}</strong><span class="muted"> · ${escapeHtml(c.id)}</span></span>
        <span style="display:flex;align-items:center;gap:6px;">
          <span class="live">LIVE</span>
          <span class="role">${escapeHtml(c.role)}</span>
        </span>
      </div>

      <div class="cam-src-bar">
        <span class="cam-src-label">Camera source</span>
        <div class="src-toggle">
          <button class="src-btn${laptopActive ? " active" : ""}"
                  data-cam="${escapeHtml(c.id)}" data-uri="0">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
            Laptop Cam
          </button>
          <button class="src-btn${ivcamActive ? " active" : ""}"
                  data-cam="${escapeHtml(c.id)}" data-uri="1">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="2" width="14" height="20" rx="2"/><circle cx="12" cy="17" r="1" fill="currentColor"/></svg>
            iVCam (Phone)
          </button>
        </div>
      </div>

      <img src="/stream/${encodeURIComponent(c.id)}" alt="${escapeHtml(c.name)}"
           onerror="this.outerHTML='<div class=&quot;cam-placeholder&quot;>⏳ Stream loading… Models warming up (~20s on CPU).</div>'" />
    </div>`;
  }).join("");

  $$(".src-btn").forEach(btn => btn.addEventListener("click", async () => {
    const camId = btn.dataset.cam;
    const uri   = parseInt(btn.dataset.uri, 10);
    if (btn.classList.contains("active")) return;  // already selected
    $$(`[data-cam="${camId}"].src-btn`).forEach(b => b.disabled = true);
    try {
      await api(`/api/camera/${encodeURIComponent(camId)}/source`, {
        method: "POST",
        body: JSON.stringify({ uri }),
      });
      camSources[camId] = uri;
      renderCameras();
      toast({ type: "ok", title: "Camera switched",
              body: uri === 0 ? "Now using laptop built-in camera." : "Now using iVCam (phone)." });
    } catch (e) {
      $$(`[data-cam="${camId}"].src-btn`).forEach(b => b.disabled = false);
      toast({ type: "deny", title: "Switch failed", body: escapeHtml(e.message) });
    }
  }));
}

// ============================ Events ============================
function renderEvents() {
  const root = $("#events");
  $("#events-count").textContent = `${state.events.length} shown`;
  if (state.events.length === 0) {
    root.innerHTML = `
      <div class="empty">
        <div class="empty-ic">🚙</div>
        <div>Waiting for the first detection…</div>
        <div class="muted small">Drive a vehicle past a gate camera.</div>
      </div>`;
    return;
  }
  root.innerHTML = state.events.map(eventHtml).join("");
  $$(".ev-del").forEach(btn => btn.addEventListener("click", async (e) => {
    e.stopPropagation();
    const id = parseInt(btn.dataset.id, 10);
    btn.disabled = true;
    btn.textContent = "…";
    try {
      await api(`/api/events/${id}`, { method: "DELETE" });
      state.events = state.events.filter(ev => ev.id !== id);
      renderEvents();
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "✕";
      toast({ type: "deny", title: "Delete failed", body: escapeHtml(err.message) });
    }
  }));
}

function eventHtml(ev) {
  let klass = "unknown", badge = "unknown", badgeText = "UNKNOWN";
  if (ev.authorized)      { klass = "auth";    badge = "ok";   badgeText = "AUTHORIZED"; }
  else if (ev.plate)      { klass = "denied";  badge = "deny"; badgeText = "DENIED"; }

  const dt = new Date(ev.timestamp + (ev.timestamp && ev.timestamp.endsWith("Z") ? "" : "Z"));
  const time = dt.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  const date = dt.toLocaleDateString([], {month:'short', day:'numeric'});

  const dirIcon = ev.direction === "entry" ? "→" : ev.direction === "exit" ? "←" : "·";
  const dirLabel = ev.direction === "entry" ? "Entry" : ev.direction === "exit" ? "Exit" : "Sighting";

  const snapPath = ev.snapshot_path
    ? ev.snapshot_path.replace(/\\/g, "/").split("/").pop()
    : null;
  const img = snapPath
    ? `<img class="snap" src="/snapshots/${encodeURIComponent(snapPath)}" alt="" onerror="this.outerHTML='<div class=&quot;snap-placeholder&quot;>📷</div>'" />`
    : `<div class="snap-placeholder">📷</div>`;
  const conf = (ev.plate_confidence != null)
    ? `<span class="conf-pill">${(ev.plate_confidence * 100).toFixed(0)}%</span>`
    : "";
  return `
    <div class="event ${klass}" data-id="${ev.id}">
      ${img}
      <div class="ev-body">
        <div class="ev-top">
          <span class="plate">${escapeHtml(ev.plate || "— no plate —")}</span>
          ${conf}
        </div>
        <div class="meta">
          <span class="dir-chip ${ev.direction || 'unknown'}">${dirIcon} ${dirLabel}</span>
          <span class="sep">·</span>${escapeHtml(ev.camera_id || "")}
          ${ev.vehicle_type ? `<span class="sep">·</span>${escapeHtml(ev.vehicle_type)}` : ""}
          <span class="sep">·</span><span class="ev-time">${date}, ${time}</span>
        </div>
      </div>
      <div class="ev-right">
        <span class="badge ${badge}">${badgeText}</span>
        <button class="ev-del" data-id="${ev.id}" title="Delete this event">✕</button>
      </div>
    </div>`;
}

// ============================ Toast stack ============================
function toast({ type = "", title = "", body = "", icon = null, ttl = 4500 }) {
  const stack = $("#toasts");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `
    <div class="t-ic">${icon || (type === "ok" ? "✓" : type === "deny" ? "✕" : type === "warn" ? "⚠" : "•")}</div>
    <div class="t-body">
      <div class="t-title">${escapeHtml(title)}</div>
      <div class="t-meta">${body}</div>
    </div>`;
  stack.appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity .3s, transform .3s";
    el.style.opacity = "0"; el.style.transform = "translateX(20px)";
    setTimeout(() => el.remove(), 300);
  }, ttl);
}

function toastEvent(ev) {
  const dir = ev.direction === "entry" ? "Entry" : ev.direction === "exit" ? "Exit" : "Sighting";
  let type, title, body;
  if (ev.authorized) {
    type = "ok";
    title = `${dir} authorized`;
    body = `Plate <code>${escapeHtml(ev.plate)}</code> — gate opened.`;
  } else if (ev.plate) {
    type = "deny";
    title = `${dir} denied`;
    body = `Plate <code>${escapeHtml(ev.plate)}</code> is not whitelisted.`;
  } else {
    type = "warn";
    title = `${dir} — unknown vehicle`;
    body = `No plate could be read.`;
  }
  toast({ type, title, body });
}

// ============================ Whitelist ============================
async function refreshWhitelist() {
  try {
    state.whitelist = await api("/api/whitelist");
    renderWhitelist();
  } catch {}
}

function renderWhitelist() {
  const root = $("#whitelist");
  const active = state.whitelist.filter(v => v.active);
  if (active.length === 0) {
    root.innerHTML = `
      <div class="empty small">
        <div class="empty-ic">📋</div>
        <div>No vehicles authorized yet.</div>
        <div class="muted">Click <b>+ Add vehicle</b> to register one.</div>
      </div>`;
    return;
  }
  root.innerHTML = `
    <table class="wl">
      <thead><tr>
        <th>Plate</th><th>Owner</th><th>Type</th><th>Company</th><th></th>
      </tr></thead>
      <tbody>
        ${active.map(v => `
          <tr>
            <td class="plate-cell">${escapeHtml(v.plate)}</td>
            <td>${escapeHtml(v.owner_name || "")}<br><span class="muted small">${escapeHtml(v.owner_phone || "")}</span></td>
            <td>${escapeHtml(v.vehicle_type || "—")}</td>
            <td>${escapeHtml(v.company || "—")}</td>
            <td><button class="btn ghost small" data-rm="${escapeHtml(v.plate)}">Remove</button></td>
          </tr>`).join("")}
      </tbody>
    </table>`;
  $$("button[data-rm]").forEach(b => b.addEventListener("click", async () => {
    if (!confirm(`Remove ${b.dataset.rm} from whitelist?`)) return;
    try {
      await api(`/api/whitelist/${encodeURIComponent(b.dataset.rm)}`, { method: "DELETE" });
      toast({type:"ok", title:"Removed", body:`<code>${escapeHtml(b.dataset.rm)}</code> is no longer authorized.`});
      refreshWhitelist();
    } catch (e) {
      toast({type:"deny", title:"Couldn't remove", body: escapeHtml(e.message)});
    }
  }));
}

// ============================ Add-whitelist modal ============================
function openModal() {
  $("#wl-modal").hidden = false;
  $("#wl-err").hidden = true;
  setTimeout(() => $("#wl-plate").focus(), 50);
}
function closeModal() {
  $("#wl-modal").hidden = true;
  $("#wl-err").hidden = true;
  $$("#wl-modal input, #wl-modal textarea, #wl-modal select").forEach(i => i.value = "");
  $("#wl-save").disabled = false;
  $("#wl-save").textContent = "Save vehicle";
}

$("#btn-add-whitelist").addEventListener("click", openModal);
$("#wl-cancel").addEventListener("click", closeModal);
$("#wl-x").addEventListener("click", closeModal);

// Click on backdrop (the dark area, NOT inside the card) closes the modal.
$("#wl-modal").addEventListener("click", (e) => {
  if (e.target === $("#wl-modal")) closeModal();
});

// Escape key closes any modal or shows login if needed.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#wl-modal").hidden) closeModal();
});

// Enter inside the plate input submits.
$("#wl-plate").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#wl-save").click(); });

$("#wl-save").addEventListener("click", async () => {
  const plate = $("#wl-plate").value.trim().toUpperCase();
  if (!plate) {
    $("#wl-err").textContent = "License plate is required.";
    $("#wl-err").hidden = false;
    $("#wl-plate").focus();
    return;
  }
  const saveBtn = $("#wl-save");
  saveBtn.disabled = true;
  saveBtn.innerHTML = '<span class="spinner"></span> Saving…';
  $("#wl-err").hidden = true;
  try {
    await api("/api/whitelist", {
      method: "POST",
      body: JSON.stringify({
        plate,
        owner_name:   $("#wl-owner").value.trim()   || null,
        owner_phone:  $("#wl-phone").value.trim()   || null,
        vehicle_type: $("#wl-type").value           || null,
        company:      $("#wl-company").value.trim() || null,
        notes:        $("#wl-notes").value.trim()   || null,
      }),
    });
    closeModal();
    toast({type:"ok", title:"Vehicle authorized", body:`<code>${escapeHtml(plate)}</code> can now open the gate.`});
    refreshWhitelist();
    refreshStats();
  } catch (e) {
    $("#wl-err").textContent = e.message || "Save failed.";
    $("#wl-err").hidden = false;
    saveBtn.disabled = false;
    saveBtn.textContent = "Save vehicle";
  }
});

// ============================ Assets ============================
function renderAssets() {
  const root = $("#assets");
  if (state.assets.length === 0) {
    root.innerHTML = `
      <div class="empty small">
        <div class="empty-ic">📦</div>
        <div>No assets detected yet.</div>
        <div class="muted">Configure a yard camera to start tracking.</div>
      </div>`;
    return;
  }
  root.innerHTML = state.assets.map(a => `
    <div class="asset-card ${a.present ? "" : "absent"}">
      <img src="/snapshots/assets/${encodeURIComponent(a.asset_code)}.jpg"
           onerror="this.style.display='none'" alt="" />
      <div class="asset-body">
        <div class="asset-code">${escapeHtml(a.asset_code)}</div>
        <div class="asset-meta">
          ${escapeHtml(a.asset_type || "?")}${a.plate ? " · " + escapeHtml(a.plate) : ""}<br>
          ${a.present ? "✓ Present" : "✕ Absent"} · cam <code>${escapeHtml(a.last_camera || "?")}</code>
        </div>
      </div>
    </div>`).join("");
}

// ============================ Gate ============================
async function refreshGate() {
  try {
    const { is_open } = await api("/api/gate/status");
    setGateStatus(is_open);
  } catch {}
}

function setGateStatus(open) {
  state.gateOpen = open;
  const el = $("#gate-status");
  el.innerHTML = `<span class="led"></span>Gate: ${open ? "OPEN" : "CLOSED"}`;
  el.classList.toggle("ok", open);
  el.classList.toggle("warn", !open);
  el.classList.toggle("bad", false);
  // The server auto-closes after gateCloseSec — refresh status just after.
  if (open) {
    clearTimeout(state.gateAutoCloseTimer);
    state.gateAutoCloseTimer = setTimeout(refreshGate, (state.gateCloseSec + 1) * 1000);
  }
}

$("#btn-gate-open").addEventListener("click", async () => {
  try {
    await api("/api/gate/open", { method: "POST" });
    setGateStatus(true);
    toast({type:"ok", title:"Gate opened", body:"Manual override. Auto-closes in ~6 seconds."});
  } catch (e) { toast({type:"deny", title:"Couldn't open gate", body: escapeHtml(e.message)}); }
});
$("#btn-gate-close").addEventListener("click", async () => {
  try {
    await api("/api/gate/close", { method: "POST" });
    setGateStatus(false);
    toast({type:"warn", title:"Gate closed", body:"Forced closed by operator."});
  } catch (e) { toast({type:"deny", title:"Couldn't close gate", body: escapeHtml(e.message)}); }
});

// ============================ Util ============================
function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// ============================ Init ============================
function init() {
  connectWS();
  refreshGate();
  refreshStats();
  refreshWhitelist();
  setInterval(refreshStats,     5000);   // refresh KPIs every 5s
  setInterval(refreshWhitelist, 30000);  // refresh whitelist every 30s
}

// Boot
if (state.token) { hideLogin(); init(); }
else { showLogin(); }
