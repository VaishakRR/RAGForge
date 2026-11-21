"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const AVATAR_SVG = `
  <svg viewBox="0 0 32 32" aria-hidden="true">
    <rect x="3" y="6" width="26" height="7" rx="3.5" fill="currentColor" opacity="0.95" />
    <rect x="6" y="15" width="20" height="6" rx="3" fill="currentColor" opacity="0.7" />
    <rect x="9" y="23" width="14" height="5" rx="2.5" fill="currentColor" opacity="0.45" />
  </svg>`;

const SEND_SVG = `
  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
       stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 19V5" /><path d="m5 12 7-7 7 7" />
  </svg>`;

const STOP_SVG = `
  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
       stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
    <rect x="6" y="6" width="12" height="12" rx="2.5" />
  </svg>`;

const DOC_ICON_SVG = `
  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor"
       stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5" /><path d="M9 13h6M9 17h4" />
  </svg>`;

const TRASH_SVG = `
  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor"
       stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 6h18" /><path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
  </svg>`;

const state = {
  models: [],
  documents: [],
  health: null,
  history: [],
  streaming: false,
  abort: null,
};

const els = {
  dropzone: $("#dropzone"),
  fileInput: $("#fileInput"),
  browseBtn: $("#browseBtn"),
  uploadTitle: $("#uploadTitle"),
  docList: $("#docList"),
  docCount: $("#docCount"),
  resetBtn: $("#resetBtn"),
  statusDot: $("#statusDot"),
  statusText: $("#statusText"),
  statusDetail: $("#statusDetail"),
  connection: $("#connection"),
  indexStats: $("#indexStats"),
  embedderStats: $("#embedderStats"),
  indexSummary: $("#indexSummary"),
  modelSelect: $("#modelSelect"),
  topK: $("#topK"),
  topKVal: $("#topKVal"),
  temp: $("#temp"),
  tempVal: $("#tempVal"),
  messages: $("#messages"),
  welcome: $("#welcome"),
  chatScroll: $("#chatScroll"),
  chatForm: $("#chatForm"),
  query: $("#query"),
  sendBtn: $("#sendBtn"),
  suggestions: $("#suggestions"),
  sidebar: $("#sidebar"),
  menuBtn: $("#menuBtn"),
  backdrop: $("#sidebarBackdrop"),
  toasts: $("#toasts"),
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&": return "&amp;";
      case "<": return "&lt;";
      case ">": return "&gt;";
      case '"': return "&quot;";
      default: return "&#39;";
    }
  });
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatSeconds(nanoseconds) {
  if (!nanoseconds) return null;
  return `${(nanoseconds / 1e9).toFixed(1)}s`;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!response.ok) {
    throw new Error((data && data.detail) || `Request failed (${response.status})`);
  }
  return data;
}

function showToast(message, type = "info", timeout = 4500) {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  els.toasts.append(toast);
  while (els.toasts.children.length > 4) {
    els.toasts.firstElementChild.remove();
  }
  window.setTimeout(() => {
    toast.classList.add("leaving");
    window.setTimeout(() => toast.remove(), 240);
  }, timeout);
}

/* ---------- Status, models, documents ---------- */

function updateStats() {
  const docs = state.documents.length;
  const chunks = state.documents.reduce((sum, doc) => sum + (doc.num_chunks || 0), 0);
  const index = state.health?.index || {};
  const totalDocs = index.documents ?? docs;
  const totalChunks = index.chunks ?? chunks;
  const embedder = index.embedder;

  els.docCount.textContent = `${totalDocs} doc${totalDocs === 1 ? "" : "s"}`;

  if (totalDocs) {
    els.indexStats.textContent = `${totalDocs} document${totalDocs === 1 ? "" : "s"} · ${totalChunks} chunks`;
  } else {
    els.indexStats.textContent = "No index";
  }

  if (embedder) {
    const label = embedder.kind === "local" ? "hashing fallback" : embedder.model;
    els.embedderStats.textContent = `Embedder: ${label} · ${embedder.dim}-d`;
  } else {
    els.embedderStats.textContent = "Embedder: —";
  }

  if (totalDocs) {
    const model = embedder ? (embedder.kind === "local" ? "hashing" : embedder.model) : "local";
    els.indexSummary.textContent = `${totalDocs} document${totalDocs === 1 ? "" : "s"} · ${totalChunks} chunks · ${model}`;
  } else {
    els.indexSummary.textContent = "No documents indexed yet";
  }
}

function renderHealth(health) {
  state.health = health;
  const ollama = health?.ollama || {};
  const connected = Boolean(ollama.connected);

  els.statusDot.classList.toggle("connected", connected);
  els.statusDot.classList.toggle("offline", !connected);
  els.statusText.textContent = connected ? "Ollama connected" : "Ollama offline";
  els.statusDetail.textContent = connected
    ? `v${ollama.version || "?"} · ${(ollama.models || []).length} models`
    : "Run setup_ollama.ps1";
  els.connection.title = connected
    ? `Endpoint: ${ollama.base_url}`
    : (ollama.error || "Ollama is not reachable");

  updateStats();
}

async function loadHealth() {
  try {
    const health = await api("/api/health");
    renderHealth(health);
  } catch (error) {
    renderHealth({ ok: false, ollama: { connected: false, error: error.message } });
  }
}

function renderModelOptions() {
  const preferred =
    localStorage.getItem("ragforge.model") ||
    state.health?.config?.chat_model ||
    "";

  const names = state.models.map((model) => model.name).filter(Boolean);
  if (preferred && !names.includes(preferred)) names.unshift(preferred);

  els.modelSelect.innerHTML = "";
  if (!names.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No models found";
    els.modelSelect.append(option);
    return;
  }

  names.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.title = name;
    if (name === preferred) option.selected = true;
    els.modelSelect.append(option);
  });

  if (!els.modelSelect.value) els.modelSelect.value = names[0];
}

async function loadModels() {
  try {
    const data = await api("/api/models");
    state.models = data.models || [];
    renderModelOptions();
  } catch (error) {
    const fallback = state.health?.config?.chat_model;
    state.models = fallback ? [{ name: fallback }] : [];
    renderModelOptions();
    showToast(`Could not load Ollama models: ${error.message}`, "error", 6500);
  }
}

function renderDocuments() {
  els.docList.innerHTML = "";

  if (!state.documents.length) {
    const empty = document.createElement("div");
    empty.className = "doc-empty";
    empty.innerHTML = "No documents yet.<br />Upload a file to begin.";
    els.docList.append(empty);
    updateStats();
    return;
  }

  state.documents.forEach((doc) => {
    const item = document.createElement("div");
    item.className = "doc-item";
    item.dataset.id = doc.id;

    const icon = document.createElement("span");
    icon.className = "doc-icon";
    icon.innerHTML = DOC_ICON_SVG;

    const info = document.createElement("div");
    info.className = "doc-info";

    const name = document.createElement("span");
    name.className = "doc-name";
    name.textContent = doc.name;
    name.title = doc.name;

    const meta = document.createElement("span");
    meta.className = "doc-meta";
    meta.textContent = `${doc.num_chunks} chunks · ${formatBytes(doc.size_bytes)}`;

    info.append(name, meta);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-btn doc-delete";
    remove.title = `Delete ${doc.name}`;
    remove.setAttribute("aria-label", `Delete ${doc.name}`);
    remove.innerHTML = TRASH_SVG;

    item.append(icon, info, remove);
    els.docList.append(item);
  });

  updateStats();
}

async function loadDocuments() {
  try {
    const data = await api("/api/documents");
    state.documents = data.documents || [];
    renderDocuments();
  } catch (error) {
    showToast(`Could not load documents: ${error.message}`, "error", 6500);
  }
}

/* ---------- Uploads ---------- */

function setUploading(uploading) {
  els.dropzone.classList.toggle("uploading", uploading);
  els.uploadTitle.textContent = uploading ? "Indexing…" : "Drop documents here";
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []).filter(Boolean);
  if (!files.length) return;

  const form = new FormData();
  files.forEach((file) => form.append("files", file));

  setUploading(true);
  showToast(`Indexing ${files.length} file${files.length === 1 ? "" : "s"}…`, "info", 20000);

  try {
    const data = await api("/api/documents", { method: "POST", body: form });
    (data.documents || []).forEach((doc) => {
      showToast(`Indexed ${doc.name} (${doc.num_chunks} chunks)`, "success", 6000);
    });
    (data.errors || []).forEach((error) => {
      showToast(`${error.file}: ${error.error}`, "error", 9000);
    });
    await Promise.all([loadDocuments(), loadHealth()]);
  } catch (error) {
    showToast(error.message, "error", 9000);
  } finally {
    setUploading(false);
    els.fileInput.value = "";
  }
}

/* ---------- Markdown ---------- */

function renderMarkdown(text) {
  if (!text) return "";

  let html = escapeHtml(text);
  const codeBlocks = [];
  const inlineCode = [];

  html = html.replace(/```[a-zA-Z0-9_+-]*\n?([\s\S]*?)```/g, (match, code) => {
    const token = `@@CODEBLOCK${codeBlocks.length}@@`;
    codeBlocks.push(`<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
    return token;
  });

  html = html.replace(/`([^`\n]+)`/g, (match, code) => {
    const token = `@@INLINECODE${inlineCode.length}@@`;
    inlineCode.push(`<code>${code}</code>`);
    return token;
  });

  const lines = html.split("\n");
  let out = "";
  let paragraph = [];
  let listType = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      out += `<p>${paragraph.join("<br>")}</p>`;
      paragraph = [];
    }
  };
  const flushList = () => {
    if (listType) {
      out += `</${listType}>`;
      listType = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }

    if (line.includes("@@CODEBLOCK")) {
      flushParagraph();
      flushList();
      out += line;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length + 1, 4);
      out += `<h${level}>${heading[2]}</h${level}>`;
      continue;
    }

    const unordered = line.match(/^\s*[-*+]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (unordered || ordered) {
      flushParagraph();
      const type = unordered ? "ul" : "ol";
      if (listType !== type) {
        flushList();
        out += `<${type}>`;
        listType = type;
      }
      out += `<li>${(unordered || ordered)[1]}</li>`;
      continue;
    }

    const quote = line.match(/^&gt;\s?(.*)$/);
    if (quote) {
      flushParagraph();
      flushList();
      out += `<blockquote>${quote[1]}</blockquote>`;
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();
  html = out;

  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  html = html.replace(
    /\[(\d{1,2})\]/g,
    (match, number) =>
      `<button type="button" class="cite" data-cite="${number}" title="View source ${number}">${number}</button>`
  );

  inlineCode.forEach((code, index) => {
    html = html.replaceAll(`@@INLINECODE${index}@@`, code);
  });
  codeBlocks.forEach((code, index) => {
    html = html.replaceAll(`@@CODEBLOCK${index}@@`, code);
  });

  return html;
}

/* ---------- Messages ---------- */

function scrollToBottom(force = false) {
  const container = els.chatScroll;
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 180;
  if (force || nearBottom) {
    container.scrollTop = container.scrollHeight;
  }
}

function addUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  row.append(bubble);
  els.messages.append(row);
  scrollToBottom(true);
}

function addMeta(bubble, meta, sourceCount) {
  if (!meta) return;
  const parts = [];
  if (meta.model) parts.push(meta.model);
  if (sourceCount) parts.push(`${sourceCount} source${sourceCount === 1 ? "" : "s"}`);
  if (meta.eval_count && meta.eval_duration) {
    parts.push(`${(meta.eval_count / (meta.eval_duration / 1e9)).toFixed(1)} tok/s`);
  }
  const seconds = formatSeconds(meta.total_duration);
  if (seconds) parts.push(seconds);
  if (!parts.length) return;

  const line = document.createElement("div");
  line.className = "msg-meta";
  parts.forEach((part) => {
    const span = document.createElement("span");
    span.textContent = part;
    line.append(span);
  });
  bubble.append(line);
}

function addAssistantMessage() {
  const row = document.createElement("div");
  row.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = AVATAR_SVG;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const content = document.createElement("div");
  content.className = "md streaming-text";

  const typing = document.createElement("span");
  typing.className = "typing";
  typing.innerHTML = "<i></i><i></i><i></i>";
  content.append(typing);

  const sourcesSlot = document.createElement("div");

  bubble.append(content, sourcesSlot);
  row.append(avatar, bubble);
  els.messages.append(row);
  scrollToBottom(true);

  let hasContent = false;
  let errored = false;

  return {
    row,
    bubble,
    sourcesSlot,
    render(text) {
      if (!hasContent) {
        content.textContent = "";
        hasContent = true;
      }
      content.textContent = text;
      scrollToBottom();
    },
    error(message) {
      errored = true;
      bubble.classList.add("error");
      const note = document.createElement("div");
      note.className = "error-note";
      note.textContent = message;
      bubble.append(note);
      scrollToBottom();
    },
    finish(text, meta, sourceCount) {
      content.classList.remove("streaming-text");
      if (text) {
        content.innerHTML = renderMarkdown(text);
      } else if (!hasContent && !errored) {
        content.textContent = "No response.";
      }
      addMeta(bubble, meta, sourceCount);
      scrollToBottom();
    },
  };
}

function renderSources(container, sources) {
  container.innerHTML = "";
  if (!sources || !sources.length) return;

  const details = document.createElement("details");
  details.className = "sources";
  details.open = false;

  const summary = document.createElement("summary");
  summary.textContent = "Sources ";
  const count = document.createElement("span");
  count.className = "count";
  count.textContent = String(sources.length);
  summary.append(count);

  const list = document.createElement("div");
  list.className = "source-list";

  sources.forEach((source) => {
    const card = document.createElement("div");
    card.className = "source-card";
    card.dataset.n = source.n;

    const head = document.createElement("div");
    head.className = "source-head";

    const number = document.createElement("span");
    number.className = "source-n";
    number.textContent = `[${source.n}]`;

    const doc = document.createElement("span");
    doc.className = "source-doc";
    doc.textContent = source.document;
    doc.title = source.document;

    const page = document.createElement("span");
    page.className = "source-page";
    page.textContent = `page ${source.page}`;

    const score = document.createElement("span");
    score.className = "source-score";
    const scores = source.scores || {};
    score.textContent = `dense ${(scores.dense ?? 0).toFixed(3)} · bm25 ${(scores.bm25 ?? 0).toFixed(2)}`;

    head.append(number, doc, page, score);

    const text = document.createElement("div");
    text.className = "source-text";
    text.textContent = source.text;

    card.append(head, text);
    card.addEventListener("click", () => card.classList.toggle("expanded"));
    list.append(card);
  });

  details.append(summary, list);
  container.append(details);
}

async function readSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      raw.split("\n").forEach((line) => {
        if (!line.startsWith("data: ")) return;
        const payload = line.slice(6).trim();
        if (!payload) return;
        try {
          onEvent(JSON.parse(payload));
        } catch {
          /* ignore malformed events */
        }
      });

      boundary = buffer.indexOf("\n\n");
    }
  }
}

function setStreaming(streaming) {
  state.streaming = streaming;
  els.sendBtn.classList.toggle("stop", streaming);
  els.sendBtn.innerHTML = streaming ? STOP_SVG : SEND_SVG;
  els.sendBtn.setAttribute("aria-label", streaming ? "Stop generating" : "Send message");
  els.sendBtn.title = streaming ? "Stop generating" : "Send message";
}

function stopStreaming() {
  if (state.abort) state.abort.abort();
}

async function sendMessage(text) {
  if (state.streaming) return;

  els.welcome.hidden = true;
  addUserMessage(text);

  const historyForRequest = state.history.slice(-6);
  state.history.push({ role: "user", content: text });

  els.query.value = "";
  autosizeQuery();

  const assistant = addAssistantMessage();
  const payload = {
    query: text,
    history: historyForRequest,
    model: els.modelSelect.value || null,
    top_k: Number(els.topK.value),
    temperature: Number(els.temp.value),
  };

  state.abort = new AbortController();
  setStreaming(true);

  let buffer = "";
  let sources = [];
  let meta = null;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: state.abort.signal,
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `Request failed (${response.status})`);
    }

    await readSSE(response, (event) => {
      if (event.type === "sources") {
        sources = event.sources || [];
        renderSources(assistant.sourcesSlot, sources);
      } else if (event.type === "token") {
        buffer += event.content || "";
        assistant.render(buffer);
      } else if (event.type === "error") {
        assistant.error(event.message);
        showToast(event.message, "error", 8000);
      } else if (event.type === "done") {
        meta = event;
      }
    });
  } catch (error) {
    if (error.name === "AbortError") {
      assistant.error("Generation stopped.");
    } else {
      assistant.error(error.message);
      showToast(error.message, "error", 8000);
    }
  } finally {
    setStreaming(false);
    state.abort = null;
    assistant.finish(buffer, meta, sources.length);

    if (buffer.trim()) {
      state.history.push({ role: "assistant", content: buffer });
    }
    if (state.history.length > 12) {
      state.history = state.history.slice(-12);
    }
    scrollToBottom(true);
  }
}

/* ---------- Interactions ---------- */

function autosizeQuery() {
  els.query.style.height = "auto";
  els.query.style.height = `${Math.min(els.query.scrollHeight, 190)}px`;
}

function restorePreferences() {
  const rawTopK = localStorage.getItem("ragforge.topK");
  const storedTopK = Number(rawTopK);
  if (rawTopK !== null && storedTopK >= 1 && storedTopK <= 12) {
    els.topK.value = String(storedTopK);
  }

  const rawTemp = localStorage.getItem("ragforge.temp");
  const storedTemp = Number(rawTemp);
  if (rawTemp !== null && Number.isFinite(storedTemp) && storedTemp >= 0 && storedTemp <= 1) {
    els.temp.value = String(storedTemp);
  }

  els.topKVal.textContent = els.topK.value;
  els.tempVal.textContent = Number(els.temp.value).toFixed(1);
}

function toggleSidebar(open) {
  els.sidebar.classList.toggle("open", open);
  els.backdrop.hidden = !open;
}

function bindEvents() {
  els.dropzone.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    els.fileInput.click();
  });

  els.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      els.fileInput.click();
    }
  });

  els.browseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    els.fileInput.click();
  });

  els.fileInput.addEventListener("change", () => {
    uploadFiles(els.fileInput.files);
  });

  ["dragenter", "dragover"].forEach((name) => {
    els.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      els.dropzone.classList.add("dragging");
    });
  });

  ["dragleave", "dragend"].forEach((name) => {
    els.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      els.dropzone.classList.remove("dragging");
    });
  });

  els.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    els.dropzone.classList.remove("dragging");
    uploadFiles(event.dataTransfer.files);
  });

  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  els.docList.addEventListener("click", async (event) => {
    const button = event.target.closest(".doc-delete");
    if (!button) return;

    const item = button.closest(".doc-item");
    const id = item?.dataset.id;
    const doc = state.documents.find((entry) => entry.id === id);
    if (!id) return;
    if (!window.confirm(`Delete "${doc?.name || "this document"}" from the index?`)) return;

    button.disabled = true;
    try {
      await api(`/api/documents/${id}`, { method: "DELETE" });
      showToast("Document deleted.", "success");
      await Promise.all([loadDocuments(), loadHealth()]);
    } catch (error) {
      showToast(error.message, "error", 8000);
      button.disabled = false;
    }
  });

  els.resetBtn.addEventListener("click", async () => {
    if (!window.confirm("Delete every document and reset the index?")) return;
    els.resetBtn.disabled = true;
    try {
      await api("/api/admin/reset", { method: "POST" });
      state.history = [];
      $$(".message", els.messages).forEach((message) => message.remove());
      els.welcome.hidden = false;
      showToast("Index reset.", "success");
      await Promise.all([loadDocuments(), loadHealth()]);
    } catch (error) {
      showToast(error.message, "error", 8000);
    } finally {
      els.resetBtn.disabled = false;
    }
  });

  els.chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.streaming) {
      stopStreaming();
      return;
    }
    const text = els.query.value.trim();
    if (text) sendMessage(text);
  });

  els.query.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.chatForm.requestSubmit();
    }
  });

  els.query.addEventListener("input", autosizeQuery);

  els.modelSelect.addEventListener("change", () => {
    localStorage.setItem("ragforge.model", els.modelSelect.value);
  });

  els.topK.addEventListener("input", () => {
    els.topKVal.textContent = els.topK.value;
    localStorage.setItem("ragforge.topK", els.topK.value);
  });

  els.temp.addEventListener("input", () => {
    els.tempVal.textContent = Number(els.temp.value).toFixed(1);
    localStorage.setItem("ragforge.temp", els.temp.value);
  });

  els.suggestions.addEventListener("click", (event) => {
    const button = event.target.closest(".suggestion");
    if (!button) return;
    const prompt = button.dataset.prompt;
    if (prompt) sendMessage(prompt);
  });

  els.messages.addEventListener("click", (event) => {
    const cite = event.target.closest(".cite");
    if (!cite) return;
    const message = cite.closest(".message");
    const details = message?.querySelector(".sources");
    if (!details) return;
    details.open = true;
    const card = details.querySelector(`.source-card[data-n="${cite.dataset.cite}"]`);
    if (!card) return;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.remove("flash");
    void card.offsetWidth;
    card.classList.add("flash");
  });

  els.menuBtn.addEventListener("click", () => toggleSidebar(true));
  els.backdrop.addEventListener("click", () => toggleSidebar(false));

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") toggleSidebar(false);
  });

  els.chatScroll.addEventListener("scroll", () => {
    /* keep native scrolling; helper reads position on demand */
  });
}

async function init() {
  bindEvents();
  restorePreferences();
  await loadHealth();
  await Promise.all([loadModels(), loadDocuments()]);
  autosizeQuery();
  window.setInterval(loadHealth, 30000);
}

init();
