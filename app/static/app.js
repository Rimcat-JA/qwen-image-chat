"use strict";
const $ = (s) => document.querySelector(s);
const state = {
  convId: null,
  conv: null,
  attachments: [],      // {file?: File, reuse?: string, url: string}
  pollers: new Map(),   // message id -> interval
  settings: {},
};

const STAGES = {
  planning: "準備中…", queued: "順番待ち…", enhancing: "プロンプトを最適化中…", waiting: "順番待ち…",
  loading: "モデルを読み込み中…", preparing: "準備中…", conditioning: "指示と画像を解析中…",
  sampling: "生成中", vae_encode: "参照画像を読み込み中…", vae_decode: "仕上げ中…", encoding: "保存中…",
};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.json();
}

function esc(s) {
  return (s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ------------------------------------------------------------ status / settings
async function refreshStatus() {
  const box = $("#statusBox");
  try {
    const s = await api("/api/status");
    state.settings = s.settings;
    const ok = s.sd.ok;
    box.className = "status " + (ok ? "ok" : "bad");
    box.querySelector(".status-text").textContent = !ok
      ? "画像エンジン起動中…（数十秒かかります）"
      : s.warm === "running"
        ? "モデルを読み込み中…（起動直後だけ1〜2分）"
        : "画像エンジン稼働中" + (s.ollama ? " · LLM利用可" : "");
    $("#llmToggle").disabled = !s.ollama;
    $("#llmToggle").parentElement.title = s.ollama
      ? "ローカルLLMで指示文を詳しいプロンプトに書き換えます"
      : "Ollama が見つからないため利用できません";
    $("#setInfo").innerHTML = `画像エンジン: ${ok ? "接続OK" : "未接続"}<br>Ollama(プロンプト強化): ${s.ollama ? "利用可" : "未検出"}<br>` +
      `モデル: Turbo ${s.models.turbo ? "✓" : "✗"} / Base ${s.models.base ? "✓" : "✗"}`;
    return s;
  } catch {
    box.className = "status bad";
    box.querySelector(".status-text").textContent = "サーバーに接続できません";
  }
}

function applySettingsToUI() {
  const s = state.settings;
  if (s.model) $("#modelSelect").value = s.model;
  if (s.quality) $("#qualitySelect").value = s.quality;
  $("#llmToggle").checked = !!s.use_llm;
  $("#setPreview").checked = s.preview !== false;
}

async function saveSetting(patch) {
  state.settings = await api("/api/settings", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch),
  });
}

// ------------------------------------------------------------ conversations
async function loadConvList() {
  const list = await api("/api/conversations");
  const nav = $("#convList");
  nav.innerHTML = "";
  for (const c of list) {
    const el = document.createElement("div");
    el.className = "conv" + (c.id === state.convId ? " active" : "");
    el.innerHTML = `<span class="t">${c.source === "telegram" ? "✈️ " : ""}${esc(c.title)}</span><button class="del" title="削除">🗑</button>`;
    el.onclick = () => openConv(c.id);
    el.ondblclick = async () => {
      const t = prompt("チャット名を変更", c.title);
      if (t !== null) { await api(`/api/conversations/${c.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: t }) }); loadConvList(); }
    };
    el.querySelector(".del").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`「${c.title}」を削除しますか？`)) return;
      await api(`/api/conversations/${c.id}`, { method: "DELETE" });
      if (c.id === state.convId) newChat();
      loadConvList();
    };
    nav.appendChild(el);
  }
}

function newChat() {
  stopAllPollers();
  state.convId = null;
  state.conv = null;
  history.replaceState(null, "", "/");
  render();
  loadConvList();
  $("#input").focus();
}

async function openConv(id) {
  stopAllPollers();
  state.convId = id;
  try {
    state.conv = await api(`/api/conversations/${id}`);
  } catch {
    return newChat();
  }
  history.replaceState(null, "", `/?c=${id}`);
  render(true);
  loadConvList();
}

// ------------------------------------------------------------ rendering
function render(scroll = false) {
  const msgs = $("#messages");
  const conv = state.conv;
  $("#empty").style.display = conv && conv.messages.length ? "none" : "";
  $("#convTitle").textContent = conv?.title || "";
  msgs.innerHTML = "";
  if (!conv) return;
  for (const m of conv.messages) msgs.appendChild(renderMsg(m));
  if (scroll) scrollBottom();
}

function scrollBottom() {
  const c = $("#chat");
  requestAnimationFrame(() => (c.scrollTop = c.scrollHeight));
}

function renderMsg(m) {
  const el = document.createElement("div");
  el.className = "msg " + m.role;
  el.dataset.id = m.id;
  if (m.role === "user") {
    const imgs = (m.images || []).map((n) => `<img src="/images/${n}" data-full="/images/${n}" alt="">`).join("");
    el.innerHTML = `<div class="col">${imgs ? `<div class="uimgs">${imgs}</div>` : ""}${m.text ? `<div class="bubble">${esc(m.text)}</div>` : ""}</div>`;
  } else {
    el.innerHTML = `<div class="avatar">Q</div><div class="body"></div>`;
    fillAssistant(el.querySelector(".body"), m);
  }
  el.querySelectorAll("img[data-full]").forEach((i) => (i.onclick = () => openLightbox(i.dataset.full)));
  return el;
}

function metaLine(meta) {
  if (!meta || !meta.width) return "";
  const tags = [
    meta.mode === "edit" ? `編集（参照${(meta.refs || []).length}枚）` : "新規生成",
    `${meta.width}×${meta.height}`,
    meta.model === "base" ? "Base" : "Turbo",
    `seed ${meta.seed}`,
  ];
  if (meta.total_seconds) tags.push(`${meta.total_seconds}秒`);
  if (meta.enhancer) tags.push("プロンプト強化");
  return `<div class="meta">${tags.map((t) => `<span class="tag">${esc(String(t))}</span>`).join("")}</div>`;
}

function fillAssistant(body, m) {
  if (m.status === "pending") {
    body.innerHTML = `<div class="live">
        <div class="stage"><div class="spinner"></div><span class="stxt">${m.meta?.mode === "edit" ? "画像を編集しています…" : "画像を生成しています…"}</span></div>
        <div class="bar"><div></div></div>
        <div class="ph"><div class="shimmer"></div></div>
        <div class="actions cancel"><button class="b-cancel">■ 停止</button></div>
      </div>`;
    body.querySelector(".b-cancel").onclick = () => api(`/api/live/${m.id}/cancel`, { method: "POST" }).catch(() => {});
    if (m.live) updateLive(body, m.live);
    startPoller(m.id, body);
    return;
  }
  const err = m.status === "error" || m.status === "cancelled";
  const imgs = (m.images || []).map((n) =>
    `<div class="gen-wrap"><img class="gen-img" src="/images/${n}" data-full="/images/${n}" alt=""></div>`).join("");
  const meta = m.meta || {};
  const promptInfo = meta.prompt
    ? `<div class="prompt-box"><b>使用したプロンプト</b>\n${esc(meta.prompt)}${meta.user_text && meta.user_text !== meta.prompt ? `\n\n<b>元の指示</b>\n${esc(meta.user_text)}` : ""}</div>` : "";
  body.innerHTML = `${m.text ? `<div class="text ${err ? "err" : ""}">${esc(m.text)}</div>` : ""}${imgs}${metaLine(meta)}
    <div class="actions">
      ${m.images?.length ? `<button class="b-edit" title="この画像を添付して編集">✎ この画像を編集</button>
      <button class="b-dl" title="ダウンロード">⤓ 保存</button>` : ""}
      ${m.request ? `<button class="b-regen" title="同じ指示で別のバリエーションを生成">↻ 再生成</button>` : ""}
      ${meta.prompt ? `<button class="b-prompt">プロンプト</button>` : ""}
    </div>${promptInfo}`;
  body.querySelectorAll("img[data-full]").forEach((i) => (i.onclick = () => openLightbox(i.dataset.full)));
  const last = m.images?.[m.images.length - 1];
  body.querySelector(".b-edit")?.addEventListener("click", () => {
    addReuseAttachment(last);
    $("#input").focus();
  });
  body.querySelector(".b-dl")?.addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = `/images/${last}`;
    a.download = `qwen-image-${meta.seed ?? Date.now()}.png`;
    a.click();
  });
  body.querySelector(".b-regen")?.addEventListener("click", () => regenerate(m.id));
  body.querySelector(".b-prompt")?.addEventListener("click", () => body.querySelector(".prompt-box").classList.toggle("show"));
}

function updateLive(body, live) {
  const stxt = body.querySelector(".stxt");
  const bar = body.querySelector(".bar > div");
  const ph = body.querySelector(".ph");
  if (!stxt) return;
  let label = STAGES[live.stage] || STAGES[live.status] || "処理中…";
  if (live.stage === "sampling" && live.total) label = `生成中 ${live.completed ?? 0} / ${live.total}`;
  if (live.elapsed) label += `（${Math.round(live.elapsed)}秒）`;
  stxt.textContent = label;
  let pct = 3;
  const order = ["queued", "enhancing", "loading", "preparing", "conditioning", "vae_encode", "sampling", "vae_decode", "encoding"];
  const idx = order.indexOf(live.stage);
  if (idx >= 0) pct = 5 + idx * 4;
  if (live.stage === "sampling" && live.total) pct = 25 + 65 * ((live.completed || 0) / live.total);
  if (live.stage === "vae_decode") pct = 93;
  if (live.stage === "encoding") pct = 98;
  bar.style.width = pct + "%";
  if (live.meta?.width && ph) ph.style.aspectRatio = `${live.meta.width} / ${live.meta.height}`;
  if (live.preview && ph) {
    let img = ph.querySelector("img");
    if (!img) { img = document.createElement("img"); ph.prepend(img); }
    img.src = live.preview;
  }
}

function startPoller(mid, body) {
  if (state.pollers.has(mid)) clearInterval(state.pollers.get(mid));
  const t = setInterval(async () => {
    let live;
    try { live = await api(`/api/live/${mid}`); } catch { return; }
    if (live.status === "gone") {
      clearInterval(t);
      state.pollers.delete(mid);
      await reloadMessage(mid);
      return;
    }
    const cur = document.querySelector(`.msg[data-id="${mid}"] .body`) || body;
    updateLive(cur, live);
  }, 800);
  state.pollers.set(mid, t);
}

function stopAllPollers() {
  for (const t of state.pollers.values()) clearInterval(t);
  state.pollers.clear();
}

async function reloadMessage(mid) {
  if (!state.convId) return;
  const conv = await api(`/api/conversations/${state.convId}`);
  state.conv = conv;
  const m = conv.messages.find((x) => x.id === mid);
  const el = document.querySelector(`.msg[data-id="${mid}"]`);
  if (m && el) {
    const atBottom = isNearBottom();
    el.replaceWith(renderMsg(m));
    if (atBottom) scrollBottom();
  }
  updateSendState();
}

function isNearBottom() {
  const c = $("#chat");
  return c.scrollHeight - c.scrollTop - c.clientHeight < 200;
}

// ------------------------------------------------------------ attachments
function addFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    if (state.attachments.length >= 10) { alert("参照画像は最大10枚までです"); break; }
    state.attachments.push({ file: f, url: URL.createObjectURL(f) });
  }
  renderAttachments();
}

function addReuseAttachment(name) {
  if (!name || state.attachments.some((a) => a.reuse === name)) return;
  state.attachments.push({ reuse: name, url: `/images/${name}` });
  renderAttachments();
}

function renderAttachments() {
  const box = $("#attachments");
  box.innerHTML = "";
  state.attachments.forEach((a, i) => {
    const el = document.createElement("div");
    el.className = "att";
    el.innerHTML = `<img src="${a.url}" alt=""><button title="外す">✕</button>`;
    el.querySelector("button").onclick = () => { state.attachments.splice(i, 1); renderAttachments(); };
    box.appendChild(el);
  });
  updateSendState();
}

// ------------------------------------------------------------ sending
function updateSendState() {
  const has = $("#input").value.trim() || state.attachments.length;
  $("#sendBtn").disabled = !has;
}

async function send() {
  const text = $("#input").value.trim();
  if (!text && !state.attachments.length) return;
  if (!state.convId) {
    const c = await api("/api/conversations", { method: "POST" });
    state.convId = c.id;
    state.conv = c;
    history.replaceState(null, "", `/?c=${c.id}`);
  }
  const fd = new FormData();
  fd.append("text", text);
  const opts = {
    mode: $("#modeSelect").value,
    aspect: $("#aspectSelect").value,
    quality: $("#qualitySelect").value,
    model: $("#modelSelect").value,
    use_llm: $("#llmToggle").checked && !$("#llmToggle").disabled,
    reuse_images: state.attachments.filter((a) => a.reuse).map((a) => a.reuse),
  };
  fd.append("options", JSON.stringify(opts));
  for (const a of state.attachments) if (a.file) fd.append("files", a.file, a.file.name || "image.png");

  $("#input").value = "";
  autoGrow();
  const sentAtt = state.attachments;
  state.attachments = [];
  renderAttachments();
  $("#modeSelect").value = "auto";
  try {
    const res = await api(`/api/conversations/${state.convId}/messages`, { method: "POST", body: fd });
    state.conv.messages.push(res.user, res.assistant);
    if (!state.conv.title) state.conv.title = text.slice(0, 30);
    render(true);
    loadConvList();
  } catch (e) {
    alert("送信できませんでした: " + e.message);
    $("#input").value = text;
    state.attachments = sentAtt;
    renderAttachments();
  }
}

async function regenerate(mid) {
  const res = await api(`/api/conversations/${state.convId}/messages/${mid}/regenerate`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quality: $("#qualitySelect").value, model: $("#modelSelect").value }),
  });
  state.conv.messages.push(res.assistant);
  render(true);
}

// ------------------------------------------------------------ misc UI
function autoGrow() {
  const t = $("#input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 220) + "px";
  updateSendState();
}

function openLightbox(src) {
  $("#lightboxImg").src = src;
  $("#lightbox").hidden = false;
}

function bind() {
  $("#newChat").onclick = newChat;
  $("#toggleSidebar").onclick = () => $("#app").classList.toggle("side-hidden");
  $("#sendBtn").onclick = send;
  $("#input").addEventListener("input", autoGrow);
  $("#input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  });
  $("#attachBtn").onclick = () => $("#fileInput").click();
  $("#fileInput").onchange = (e) => { addFiles(e.target.files); e.target.value = ""; };
  document.addEventListener("paste", (e) => {
    const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith("image/"));
    if (files.length) { e.preventDefault(); addFiles(files); }
  });
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { dragDepth++; $("#dropOverlay").classList.add("show"); } });
  window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("#dropOverlay").classList.remove("show"); } });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault(); dragDepth = 0; $("#dropOverlay").classList.remove("show");
    if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
  });
  document.querySelectorAll(".chip").forEach((c) => (c.onclick = () => { $("#input").value = c.textContent; autoGrow(); send(); }));
  $("#lightbox").onclick = () => ($("#lightbox").hidden = true);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").hidden = true; });
  $("#modelSelect").onchange = (e) => saveSetting({ model: e.target.value });
  $("#qualitySelect").onchange = (e) => saveSetting({ quality: e.target.value });
  $("#llmToggle").onchange = (e) => saveSetting({ use_llm: e.target.checked });
  $("#openSettings").onclick = () => { refreshStatus(); $("#settingsDlg").showModal(); };
  $("#setPreview").onchange = (e) => saveSetting({ preview: e.target.checked });
}

const isNarrow = () => window.matchMedia("(max-width: 760px)").matches;

(async function init() {
  bind();
  // On phones the sidebar is an overlay: start closed, close after picking a chat.
  if (isNarrow()) $("#app").classList.add("side-hidden");
  $("#convList").addEventListener("click", () => { if (isNarrow()) $("#app").classList.add("side-hidden"); });
  $("#newChat").addEventListener("click", () => { if (isNarrow()) $("#app").classList.add("side-hidden"); });
  $("#main").addEventListener("click", (e) => {
    if (isNarrow() && !$("#app").classList.contains("side-hidden") && !e.target.closest("#toggleSidebar")) {
      $("#app").classList.add("side-hidden");
    }
  });
  await refreshStatus();
  applySettingsToUI();
  setInterval(refreshStatus, 15000);
  const c = new URLSearchParams(location.search).get("c");
  if (c) await openConv(c); else { render(); loadConvList(); }
  updateSendState();
})();
