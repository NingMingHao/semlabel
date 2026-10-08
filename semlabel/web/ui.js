// Small DOM helpers: element builder, toasts, modals, formatting.
export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "html") el.innerHTML = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function toast(msg, err = false, ms = 3200) {
  const t = h("div", { class: "toast" + (err ? " err" : "") }, msg);
  document.getElementById("toasts").append(t);
  setTimeout(() => t.remove(), err ? ms * 2 : ms);
}

export function modal(title, body, buttons = []) {
  return new Promise((resolve) => {
    const root = document.getElementById("modal-root");
    const close = (v) => { bg.remove(); document.removeEventListener("keydown", key); resolve(v); };
    const foot = h("div", { class: "foot" },
      buttons.map((b) => h("button", { class: "btn " + (b.cls || ""), onclick: async () => {
        if (b.action) { const r = await b.action(); if (r === false) return; close(r === undefined ? b.value : r); }
        else close(b.value);
      } }, b.label)));
    const box = h("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-label": title }, h("h2", {}, title), body, foot);
    const bg = h("div", { class: "modal-bg", onmousedown: (e) => { if (e.target === bg) close(null); } }, box);
    const key = (e) => { if (e.key === "Escape") close(null); };
    document.addEventListener("keydown", key);
    root.append(bg);
    const first = box.querySelector("input,select,textarea,button");
    if (first) first.focus();
  });
}

export async function confirmBox(title, text, okLabel = "OK", danger = false) {
  return (await modal(title, h("p", {}, text), [
    { label: "Cancel", value: false }, { label: okLabel, value: true, cls: danger ? "danger" : "primary" }])) === true;
}

export const fmt = {
  pct: (v) => (v === null || v === undefined ? "–" : (v * 100).toFixed(0) + "%"),
  num: (v) => (v === null || v === undefined ? "–" : Number(v).toLocaleString()),
  bytes: (b) => (b > 1 << 30 ? (b / (1 << 30)).toFixed(1) + " GB" : b > 1 << 20 ? (b / (1 << 20)).toFixed(1) + " MB" : (b / 1024).toFixed(0) + " KB"),
  time: (t) => (t ? new Date(t * 1000).toLocaleString() : "–"),
  ago(t) {
    if (!t) return "–";
    const s = Date.now() / 1000 - t;
    if (s < 60) return "just now";
    if (s < 3600) return Math.round(s / 60) + " min ago";
    if (s < 86400) return Math.round(s / 3600) + " h ago";
    return Math.round(s / 86400) + " d ago";
  },
};

export const STATUS_LABEL = { new: "New", predicted: "Auto-labeled", in_progress: "In progress", reviewed: "Reviewed" };
export const STATUS_COLOR = { new: "#c3c8d1", predicted: "#e8b04b", in_progress: "#6d95e8", reviewed: "#4caf6e" };

export function hexToRgb(hex) {
  const v = hex.replace("#", "");
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}
