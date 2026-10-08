// Mask editor: paint class indices into a Uint8Array with brush / polygon / superpixel / bucket /
// DINOv3 magic wand / SAM tools, review auto-labels and save. Coordinates: image pixels (x right, y down).
import { api, I, P } from "./api.js";
import { h, toast, fmt, hexToRgb, STATUS_LABEL } from "./ui.js";

const ICONS = {
  brush: '<path d="M4 20c2 0 4-1 4-4 0-1.7 1.3-3 3-3l7-7-3-3-7 7c0 1.7-1.3 3-3 3-3 0-1 7-1 7z"/>',
  erase: '<path d="M7 21h12M5 15l8-8 6 6-6 6H9z"/><path d="M13 7l6 6"/>',
  poly: '<path d="M5 18l2-12 11 3 1 9z"/><circle cx="5" cy="18" r="1.5"/><circle cx="7" cy="6" r="1.5"/><circle cx="18" cy="9" r="1.5"/><circle cx="19" cy="18" r="1.5"/>',
  fill: '<path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4z"/><path d="M13 13h7v7h-7z" fill="currentColor"/>',
  bucket: '<path d="M5 11l7-7 7 7-7 7z"/><path d="M19 15s2 2.5 2 4a2 2 0 0 1-4 0c0-1.5 2-4 2-4z"/>',
  wand: '<path d="M4 20L15 9"/><path d="M15 3v3M15 12v3M18 6h3M9 6h3M17 4l2-2M17 8l2 2M13 8l-2-2"/>',
  sam: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2" fill="currentColor"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
  pick: '<path d="M16 3l5 5-9 9H7v-5z"/><path d="M3 21l4-4"/>',
  pan: '<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>',
  undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>',
  redo: '<path d="M15 14l5-5-5-5"/><path d="M20 9H10a6 6 0 0 0 0 12h3"/>',
};
const TOOLS = [
  ["brush", "Brush", "B"], ["erase", "Eraser", "E"], ["poly", "Polygon", "P"], ["fill", "Superpixel fill", "F"],
  ["bucket", "Bucket fill (same-label region)", "G"], ["wand", "Magic wand (DINOv3 similarity)", "W"],
  ["sam", "SAM click (left +, right −)", "S"], ["pick", "Pick class", "I"], ["pan", "Pan", "H"],
];

export async function openEditor(root, pid, iid, query, crumbs) {
  const st = {
    pid, iid, query, tool: "brush", cls: 1, brush: 12, protect: false, opacity: 0.55, layer: "label", outline: false,
    undo: [], redo: [], dirty: false, view: { s: 1, x: 0, y: 0 }, poly: [], sam: { pts: [], mask: null, score: null },
    wandThresh: 0.62, confThresh: 0.5, hidden: new Set(), autoAdvance: true, busy: 0, mouse: null,
  };
  const [project, img] = await Promise.all([api.get(P(pid)), api.get(I(pid, iid) + `?status=${query.status || ""}&sort=${query.sort || "id"}`)]);
  st.project = project; st.img = img;
  st.W = img.width; st.H = img.height;
  st.classes = project.classes.map((c) => ({ ...c, rgb: hexToRgb(c.color) }));
  if (st.classes.length) st.cls = st.classes[0].idx;
  crumbs([[project.name, `#/p/${pid}/images`], [img.name, null]]);

  // ------------------------------------------------------------------ layout
  const canvas = h("canvas", { tabindex: 0, "aria-label": "Image and label canvas" });
  const hud = h("div", { class: "hud" }, "");
  const busy = h("div", { class: "busy", hidden: true }, "Working…");
  const opacity = h("input", { type: "range", min: 0, max: 100, value: st.opacity * 100, "aria-label": "Label opacity",
    oninput: (e) => { st.opacity = e.target.value / 100; draw(); } });
  const layerSel = h("select", { "aria-label": "Layer", onchange: (e) => { st.layer = e.target.value; renderOverlay(); draw(); } },
    h("option", { value: "label" }, "Label"), h("option", { value: "pred" }, "Auto-label"),
    h("option", { value: "conf" }, "Low confidence"), h("option", { value: "none" }, "Image only"));
  const outlineCb = h("input", { type: "checkbox", onchange: (e) => { st.outline = e.target.checked; renderOverlay(); draw(); } });
  const opts = h("div", { class: "opts" }, "Show", layerSel, "Opacity", opacity, h("label", { class: "row", style: { gap: "4px" } }, outlineCb, "Outline"));
  const stage = h("div", { class: "stage" }, canvas, opts, hud, busy);
  const toolBtns = {};
  const tools = h("div", { class: "tools", role: "toolbar", "aria-label": "Tools" },
    TOOLS.map(([k, label, key]) => (toolBtns[k] = h("button", { class: "tool", title: `${label} (${key})`, "aria-label": label,
      onclick: () => setTool(k), html: `<svg viewBox="0 0 24 24">${ICONS[k]}</svg><kbd>${key}</kbd>` }))),
    h("hr"),
    h("button", { class: "tool", title: "Undo (Ctrl+Z)", "aria-label": "Undo", onclick: () => undo(), html: `<svg viewBox="0 0 24 24">${ICONS.undo}</svg>` }),
    h("button", { class: "tool", title: "Redo (Ctrl+Shift+Z)", "aria-label": "Redo", onclick: () => redo(), html: `<svg viewBox="0 0 24 24">${ICONS.redo}</svg>` }));

  const toolOpts = h("div", { class: "stack" });
  const classList = h("div", { class: "classes", role: "listbox", "aria-label": "Classes" });
  const info = h("dl", { class: "kv" });
  const statusBadge = h("span", { class: "badge" });
  const navPos = h("span", { class: "muted num" });
  const saveState = h("span", { class: "muted" }, "");
  const prevBtn = h("button", { class: "btn small", title: "Previous (Alt+←)", onclick: () => go(st.img.nav.prev) }, "‹ Prev");
  const nextBtn = h("button", { class: "btn small", title: "Next (Alt+→)", onclick: () => go(st.img.nav.next) }, "Next ›");
  const memBtn = h("button", { class: "btn", onclick: () => toggleMemory() }, "");
  const top = h("div", { class: "ed-top" },
    h("a", { class: "btn small", href: `#/p/${pid}/images${qs(query)}` }, "← Gallery"),
    h("span", { class: "title", title: img.name }, img.name), statusBadge, navPos, prevBtn, nextBtn,
    h("div", { class: "spacer" }), saveState,
    h("button", { class: "btn", title: "Save progress (Ctrl+S)", onclick: () => save("in_progress") }, "Save"),
    h("button", { class: "btn primary", title: "Mark reviewed and go to the next image (Ctrl+Enter)", onclick: () => submit(false) }, "Submit"),
    h("button", { class: "btn primary", title: "Mark reviewed, add as a reference to the memory bank (Ctrl+Shift+Enter)", onclick: () => submit(true) }, "Submit + add to memory"));
  const side = h("div", { class: "side" },
    h("section", {}, h("h3", {}, "Classes"), classList,
      h("p", { class: "muted", style: { fontSize: "12px", margin: "8px 0 0" } }, "Keys 1–9, 0 select · Alt-click picks · right-click a class to hide it")),
    h("section", {}, h("h3", {}, "Tool"), toolOpts),
    h("section", {}, h("h3", {}, "Auto-label (in-context, DINOv3)"),
      h("div", { class: "actions" },
        h("button", { class: "btn", title: "Run the memory bank on this image (A)", onclick: () => autolabel("replace") }, "Auto-label image"),
        h("button", { class: "btn", title: "Only fill pixels that are still unlabeled", onclick: () => autolabel("fill") }, "Auto-label unlabeled pixels"),
        h("button", { class: "btn", title: "SegFormer ADE20K mapped to project classes (cold start)", onclick: () => prelabel() }, "Pre-label (SegFormer)"),
        memBtn,
        h("button", { class: "btn danger", onclick: () => { push(); st.mask.fill(0); changed(); } }, "Clear label"))),
    h("section", {}, h("h3", {}, "Image"), info));
  const wrap = h("div", { class: "editor" }, top, tools, stage, side);
  root.replaceChildren(wrap);

  // ------------------------------------------------------------------ data
  const ctx = canvas.getContext("2d");
  const ov = document.createElement("canvas");
  ov.width = st.W; ov.height = st.H;
  const octx = ov.getContext("2d");
  const ovData = octx.createImageData(st.W, st.H);
  const ov32 = new Uint32Array(ovData.data.buffer);
  const image = new Image();
  image.src = I(pid, iid) + "/file";
  await image.decode().catch(() => {});
  st.image = image;

  const lab = await api.bytes("GET", I(pid, iid) + "/label");
  st.mask = lab.data;
  st.pred = null; st.conf = null;
  if (img.has_pred) {
    try {
      const [p, c] = await Promise.all([api.bytes("GET", I(pid, iid) + "/pred"), api.bytes("GET", I(pid, iid) + "/conf")]);
      st.pred = p.data; st.conf = c.data;
    } catch (e) { /* no prediction */ }
  }
  if (lab.empty && st.pred) {             // review flow: start from the auto-label
    st.mask = st.pred.slice();
    st.fromPred = true;
    toast(`Loaded the auto-label (confidence ${fmt.pct(img.conf)}). Correct it and Submit.`);
  }

  function lut() {
    const a = Math.round(255);
    const l = new Uint32Array(256);
    for (const c of st.classes) if (!st.hidden.has(c.idx)) l[c.idx] = (a << 24) | (c.rgb[2] << 16) | (c.rgb[1] << 8) | c.rgb[0];
    return l;
  }
  function renderOverlay(rect) {
    const W = st.W, H = st.H;
    let x0 = 0, y0 = 0, x1 = W, y1 = H;
    if (rect && !st.outline) { x0 = Math.max(0, rect[0]); y0 = Math.max(0, rect[1]); x1 = Math.min(W, rect[2]); y1 = Math.min(H, rect[3]); }
    const L = lut();
    const src = st.layer === "pred" ? st.pred : st.mask;
    if (st.layer === "none" || !src) { ov32.fill(0); octx.putImageData(ovData, 0, 0); return; }
    if (st.layer === "conf") {
      const c = st.conf;
      for (let i = 0; i < W * H; i++) {
        const v = c ? c[i] / 255 : 1;
        ov32[i] = v < st.confThresh ? ((Math.round(230 * (1 - v / st.confThresh) + 25) << 24) | (40 << 16) | (40 << 8) | 235) : 0;
      }
      octx.putImageData(ovData, 0, 0);
      return;
    }
    for (let y = y0; y < y1; y++) {
      let i = y * W + x0;
      for (let x = x0; x < x1; x++, i++) {
        const v = src[i];
        if (st.outline) {
          const edge = (x > 0 && src[i - 1] !== v) || (y > 0 && src[i - W] !== v) || (x < W - 1 && src[i + 1] !== v) || (y < H - 1 && src[i + W] !== v);
          ov32[i] = edge ? L[v] || 0xffffffff : 0;
        } else ov32[i] = L[v];
      }
    }
    octx.putImageData(ovData, 0, 0, x0, y0, x1 - x0, y1 - y0);
  }

  // ------------------------------------------------------------------ view
  function resize() {
    const r = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    st.dpr = dpr; st.cw = r.width; st.ch = r.height;
  }
  function fit() {
    resize();
    const s = Math.min(st.cw / st.W, st.ch / st.H) * 0.96;
    st.view = { s, x: (st.cw - st.W * s) / 2, y: (st.ch - st.H * s) / 2 };
    draw();
  }
  function toImg(e) {
    const r = canvas.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    return [(sx - st.view.x) / st.view.s, (sy - st.view.y) / st.view.s, sx, sy];
  }
  function draw() {
    const d = st.dpr || 1, v = st.view;
    ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.clearRect(0, 0, st.cw, st.ch);
    ctx.setTransform(d * v.s, 0, 0, d * v.s, d * v.x, d * v.y);
    ctx.imageSmoothingEnabled = v.s < 2;
    if (st.image.complete) ctx.drawImage(st.image, 0, 0, st.W, st.H);
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = st.layer === "conf" ? 1 : st.outline ? Math.min(1, st.opacity + 0.35) : st.opacity;
    ctx.drawImage(ov, 0, 0);
    ctx.globalAlpha = 1;
    if (st.sam.mask) {
      if (!st.samCanvas) {
        st.samCanvas = document.createElement("canvas"); st.samCanvas.width = st.W; st.samCanvas.height = st.H;
      }
      if (st.samDirty) {
        const sc = st.samCanvas.getContext("2d"), id = sc.createImageData(st.W, st.H), u = new Uint32Array(id.data.buffer);
        const c = clsObj(st.cls).rgb;
        const col = (150 << 24) | (c[2] << 16) | (c[1] << 8) | c[0];
        for (let i = 0; i < u.length; i++) u[i] = st.sam.mask[i] ? col : 0;
        sc.putImageData(id, 0, 0); st.samDirty = false;
      }
      ctx.drawImage(st.samCanvas, 0, 0);
    }
    ctx.lineWidth = 1.5 / v.s;
    for (const p of st.sam.pts) {
      ctx.fillStyle = p[2] ? "#3ddc84" : "#ff5252"; ctx.strokeStyle = "#000";
      ctx.beginPath(); ctx.arc(p[0], p[1], 5 / v.s, 0, 7); ctx.fill(); ctx.stroke();
    }
    if (st.poly.length) {
      const c = clsObj(st.cls).color;
      ctx.strokeStyle = c; ctx.fillStyle = c + "55"; ctx.lineWidth = 2 / v.s;
      ctx.beginPath(); ctx.moveTo(st.poly[0][0], st.poly[0][1]);
      for (const p of st.poly.slice(1)) ctx.lineTo(p[0], p[1]);
      if (st.mouse) ctx.lineTo(st.mouse[0], st.mouse[1]);
      ctx.fill(); ctx.stroke();
      for (const p of st.poly) { ctx.fillStyle = "#fff"; ctx.fillRect(p[0] - 3 / v.s, p[1] - 3 / v.s, 6 / v.s, 6 / v.s); }
    }
    if (st.mouse && (st.tool === "brush" || st.tool === "erase")) {
      ctx.strokeStyle = st.tool === "erase" ? "#fff" : clsObj(st.cls).color; ctx.lineWidth = 1.5 / v.s;
      ctx.beginPath(); ctx.arc(st.mouse[0], st.mouse[1], st.brush, 0, 7); ctx.stroke();
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    updateHud();
  }
  function updateHud() {
    const m = st.mouse;
    let t = `${Math.round(st.view.s * 100)}%`;
    if (m) {
      const x = Math.floor(m[0]), y = Math.floor(m[1]);
      if (x >= 0 && y >= 0 && x < st.W && y < st.H) {
        const i = y * st.W + x, c = clsObj(st.mask[i]);
        t += ` · ${x}, ${y} · ${c ? c.name : "unlabeled"}`;
        if (st.conf) t += ` · conf ${(st.conf[i] / 2.55).toFixed(0)}%`;
      }
    }
    hud.textContent = t;
  }
  const clsObj = (idx) => st.classes.find((c) => c.idx === idx);

  // ------------------------------------------------------------------ editing primitives
  const MAX_UNDO_BYTES = 300e6;
  function push() {
    st.undo.push(st.mask.slice());
    st.redo = [];
    while (st.undo.length > 2 && st.undo.length * st.mask.length > MAX_UNDO_BYTES) st.undo.shift();
  }
  function undo() { if (!st.undo.length) return; st.redo.push(st.mask); st.mask = st.undo.pop(); changed(); }
  function redo() { if (!st.redo.length) return; st.undo.push(st.mask); st.mask = st.redo.pop(); changed(); }
  function changed(rect) {
    st.dirty = true; st.edited = true; saveState.textContent = "unsaved changes";
    if (st.layer !== "label") { st.layer = "label"; layerSel.value = "label"; rect = null; }
    renderOverlay(rect); draw(); scheduleStats();
  }
  function setPix(i, v) {
    if (st.protect && v !== 0 && st.mask[i] !== 0) return;
    st.mask[i] = v;
  }
  function stamp(cx, cy, r, v) {
    const W = st.W, H = st.H, r2 = r * r;
    const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(H - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++) {
      const dy = y + 0.5 - cy, dx = Math.sqrt(Math.max(0, r2 - dy * dy));
      const x0 = Math.max(0, Math.floor(cx - dx)), x1 = Math.min(W - 1, Math.ceil(cx + dx) - 1);
      for (let x = x0; x <= x1; x++) setPix(y * W + x, v);
    }
  }
  function strokeTo(a, b, v) {
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const n = Math.max(1, Math.ceil(d / Math.max(1, st.brush * 0.4)));
    for (let k = 1; k <= n; k++) stamp(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, st.brush, v);
    const r = st.brush + 2;
    changed([Math.floor(Math.min(a[0], b[0]) - r), Math.floor(Math.min(a[1], b[1]) - r), Math.ceil(Math.max(a[0], b[0]) + r), Math.ceil(Math.max(a[1], b[1]) + r)]);
  }
  function fillPoly(pts) {
    if (pts.length < 3) return;
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const x0 = Math.max(0, Math.floor(Math.min(...xs))), y0 = Math.max(0, Math.floor(Math.min(...ys)));
    const x1 = Math.min(st.W, Math.ceil(Math.max(...xs)) + 1), y1 = Math.min(st.H, Math.ceil(Math.max(...ys)) + 1);
    const w = x1 - x0, hh = y1 - y0;
    if (w <= 0 || hh <= 0) return;
    const c = document.createElement("canvas"); c.width = w; c.height = hh;
    const g = c.getContext("2d");
    g.fillStyle = "#fff"; g.beginPath(); g.moveTo(pts[0][0] - x0, pts[0][1] - y0);
    for (const p of pts.slice(1)) g.lineTo(p[0] - x0, p[1] - y0);
    g.closePath(); g.fill();
    const a = g.getImageData(0, 0, w, hh).data;
    push();
    const v = st.tool === "erase" ? 0 : st.cls;
    for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) if (a[(y * w + x) * 4 + 3] > 127) setPix((y + y0) * st.W + x + x0, v);
    changed([x0, y0, x1, y1]);
  }
  function applyMask(m, v) {
    push();
    for (let i = 0; i < m.length; i++) if (m[i]) setPix(i, v);
    changed();
  }
  function bucket(x, y) {
    const W = st.W, H = st.H, i0 = y * W + x, from = st.mask[i0], to = st.cls;
    if (from === to) return;
    push();
    const stack = new Int32Array(W * H);
    const seen = new Uint8Array(W * H);
    let n = 0; stack[n++] = i0; seen[i0] = 1;
    while (n) {
      const i = stack[--n]; st.mask[i] = to;
      const px = i % W, py = (i / W) | 0;
      if (px > 0 && !seen[i - 1] && st.mask[i - 1] === from) { seen[i - 1] = 1; stack[n++] = i - 1; }
      if (px < W - 1 && !seen[i + 1] && st.mask[i + 1] === from) { seen[i + 1] = 1; stack[n++] = i + 1; }
      if (py > 0 && !seen[i - W] && st.mask[i - W] === from) { seen[i - W] = 1; stack[n++] = i - W; }
      if (py < H - 1 && !seen[i + W] && st.mask[i + W] === from) { seen[i + W] = 1; stack[n++] = i + W; }
    }
    changed();
  }
  async function superpixels() {
    if (st.sp) return st.sp;
    const r = await api.bytes("GET", I(pid, iid) + "/superpixels");
    const sp = new Uint16Array(r.data.buffer, r.data.byteOffset, st.W * st.H);
    let n = 0; for (let i = 0; i < sp.length; i++) if (sp[i] > n) n = sp[i];
    const start = new Int32Array(n + 2), idx = new Int32Array(sp.length);
    for (let i = 0; i < sp.length; i++) start[sp[i] + 1]++;
    for (let k = 0; k <= n; k++) start[k + 1] += start[k];
    const fillp = start.slice();
    for (let i = 0; i < sp.length; i++) idx[fillp[sp[i]]++] = i;
    st.sp = { sp, start, idx };
    return st.sp;
  }
  function fillSuper(x, y, v) {
    const { sp, start, idx } = st.sp;
    const s = sp[y * st.W + x];
    let x0 = st.W, y0 = st.H, x1 = 0, y1 = 0;
    for (let k = start[s]; k < start[s + 1]; k++) {
      const i = idx[k]; setPix(i, v);
      const px = i % st.W, py = (i / st.W) | 0;
      if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py;
    }
    changed([x0, y0, x1 + 1, y1 + 1]);
  }

  // ------------------------------------------------------------------ server tools
  async function withBusy(label, fn) {
    st.busy++; busy.textContent = label; busy.hidden = false;
    try { return await fn(); } catch (e) { toast(e.message, true); } finally { if (--st.busy === 0) busy.hidden = true; }
  }
  async function wand(x, y, global) {
    await withBusy("Magic wand…", async () => {
      const r = await api.bytes("POST", I(pid, iid) + "/wand", { x, y, thresh: st.wandThresh, scope: global ? "global" : "connected" });
      if (!r.count) { toast("Nothing similar at this threshold — lower it."); return; }
      applyMask(r.data, st.tool === "erase" ? 0 : st.cls);
    });
  }
  async function samQuery() {
    if (!st.sam.pts.length) { st.sam.mask = null; draw(); return; }
    await withBusy("SAM…", async () => {
      const r = await api.bytes("POST", I(pid, iid) + "/sam", { points: st.sam.pts });
      st.sam.mask = r.data; st.sam.score = r.score; st.samDirty = true; draw();
      renderToolOpts();
    });
  }
  function samApply() {
    if (!st.sam.mask) return;
    applyMask(st.sam.mask, st.cls);
    st.sam = { pts: [], mask: null, score: null }; draw(); renderToolOpts();
  }
  async function autolabel(how) {
    await withBusy("Auto-labeling…", async () => {
      const r = await api.post(I(pid, iid) + "/predict", {});
      const [p, c] = await Promise.all([api.bytes("GET", I(pid, iid) + "/pred"), api.bytes("GET", I(pid, iid) + "/conf")]);
      st.pred = p.data; st.conf = c.data;
      push();
      if (how === "replace") st.mask = st.pred.slice();
      else for (let i = 0; i < st.mask.length; i++) if (!st.mask[i]) st.mask[i] = st.pred[i];
      st.img = { ...st.img, ...r.image };
      changed(); renderInfo();
      toast(`Auto-labeled in ${r.seconds.toFixed(2)} s · mean confidence ${fmt.pct(r.conf)}`);
    });
  }
  async function prelabel() {
    await withBusy("SegFormer pre-label…", async () => {
      await api.post(I(pid, iid) + "/prelabel");
      const p = await api.bytes("GET", I(pid, iid) + "/pred");
      push();
      for (let i = 0; i < st.mask.length; i++) if (!st.mask[i]) st.mask[i] = p.data[i];
      st.pred = p.data; st.conf = null;
      changed();
      toast("Pre-labeled unlabeled pixels with SegFormer (ADE20K → project classes).");
    });
  }

  // ------------------------------------------------------------------ save / navigation
  async function save(status, addToMemory = null) {
    const url = I(pid, iid) + `/label?status=${status}` + (addToMemory === null ? "" : `&add_to_memory=${addToMemory ? 1 : 0}`);
    saveState.textContent = "saving…";
    try {
      const r = await api.put(url, st.mask);
      st.img = { ...st.img, ...r }; st.dirty = false; st.edited = false;
      saveState.textContent = "saved " + new Date().toLocaleTimeString();
      renderInfo();
      return true;
    } catch (e) { saveState.textContent = "save failed"; toast(e.message, true); return false; }
  }
  async function submit(addMem) {
    if (!(await save("reviewed", addMem ? true : null))) return;
    toast(addMem ? "Reviewed and added to the memory bank." : "Reviewed.");
    if (st.autoAdvance && st.img.nav && st.img.nav.next) go(st.img.nav.next, true);
  }
  async function toggleMemory() {
    try {
      if (st.img.in_memory) await api.post(P(pid) + "/memory/remove", { ids: [iid] });
      else {
        if (st.dirty || !st.img.has_label) await save(st.img.status === "reviewed" ? "reviewed" : "in_progress");
        await api.post(P(pid) + "/memory/add", { ids: [iid] });
      }
      st.img = await api.get(I(pid, iid) + `?status=${query.status || ""}&sort=${query.sort || "id"}`);
      renderInfo();
      toast(st.img.in_memory ? "Added to the memory bank." : "Removed from the memory bank.");
    } catch (e) { toast(e.message, true); }
  }
  async function go(target, skipSave = false) {
    if (!target) { toast("No more images in this view."); return; }
    if (st.edited && st.dirty && !skipSave) await save(st.img.status === "reviewed" ? "reviewed" : "in_progress");
    location.hash = `#/p/${pid}/edit/${target}${qs(query)}`;
  }

  // ------------------------------------------------------------------ panels
  function renderClasses(stats) {
    classList.replaceChildren(...st.classes.map((c, k) => h("button", {
      class: "cls" + (c.idx === st.cls ? " on" : "") + (st.hidden.has(c.idx) ? " hidden" : ""), role: "option",
      "aria-selected": c.idx === st.cls ? "true" : "false", title: `${c.name} (right-click: hide/show)`,
      onclick: () => { st.cls = c.idx; st.samDirty = true; renderClasses(st.lastStats); draw(); },
      oncontextmenu: (e) => { e.preventDefault(); st.hidden.has(c.idx) ? st.hidden.delete(c.idx) : st.hidden.add(c.idx); renderOverlay(); draw(); renderClasses(st.lastStats); },
    }, h("span", { class: "sw", style: { background: c.color } }), h("span", { class: "nm" }, c.name),
      h("span", { class: "pct" }, stats && stats[c.idx] ? (stats[c.idx] * 100).toFixed(stats[c.idx] < 0.01 ? 1 : 0) + "%" : ""),
      h("kbd", {}, k < 10 ? String((k + 1) % 10) : ""))));
  }
  let statTimer = null;
  function scheduleStats() {
    clearTimeout(statTimer);
    statTimer = setTimeout(() => {
      const cnt = new Float64Array(256);
      for (let i = 0; i < st.mask.length; i++) cnt[st.mask[i]]++;
      const s = {}; for (let k = 0; k < 256; k++) if (cnt[k]) s[k] = cnt[k] / st.mask.length;
      st.lastStats = s; renderClasses(s);
    }, 150);
  }
  function renderToolOpts() {
    const t = st.tool;
    const kids = [];
    if (t === "brush" || t === "erase") {
      kids.push(h("label", { class: "field" }, h("span", {}, `Brush size: ${st.brush}px  ([ and ])`),
        h("input", { type: "range", min: 1, max: 120, value: st.brush, oninput: (e) => { st.brush = +e.target.value; renderToolOpts(); draw(); } })));
    }
    if (t === "wand") {
      kids.push(h("label", { class: "field" }, h("span", {}, `Similarity threshold: ${st.wandThresh.toFixed(2)}`),
        h("input", { type: "range", min: 30, max: 95, value: st.wandThresh * 100, oninput: (e) => { st.wandThresh = e.target.value / 100; renderToolOpts(); } })),
        h("p", { class: "muted", style: { fontSize: "12px", margin: 0 } }, "Click: fill the connected region that looks like the clicked point (DINOv3 features, snapped to superpixels). Shift-click: every similar region in the image."));
    }
    if (t === "sam") {
      kids.push(h("p", { class: "muted", style: { fontSize: "12px", margin: 0 } }, "Left click adds a positive point, right click a negative one. Enter applies the current class, Esc clears."),
        h("div", { class: "row" }, h("button", { class: "btn small primary", onclick: samApply, disabled: !st.sam.mask }, "Apply (Enter)"),
          h("button", { class: "btn small", onclick: () => { st.sam = { pts: [], mask: null, score: null }; draw(); renderToolOpts(); } }, "Clear"),
          st.sam.score !== null ? h("span", { class: "muted" }, `score ${st.sam.score.toFixed(2)}`) : null));
    }
    if (t === "poly") kids.push(h("p", { class: "muted", style: { fontSize: "12px", margin: 0 } }, "Click to add points; double-click, Enter or click the first point to close; Backspace removes the last point; Esc cancels."));
    if (t === "fill") kids.push(h("p", { class: "muted", style: { fontSize: "12px", margin: 0 } }, "Click or drag over superpixels to paint whole image segments."));
    if (t === "bucket") kids.push(h("p", { class: "muted", style: { fontSize: "12px", margin: 0 } }, "Click a region to relabel every connected pixel that has the same label (fast fix for a wrong auto-label region)."));
    if (["brush", "poly", "fill", "wand", "sam"].includes(t)) {
      kids.push(h("label", { class: "row", title: "Only paint pixels that are still unlabeled (X)" },
        h("input", { type: "checkbox", checked: st.protect, onchange: (e) => { st.protect = e.target.checked; } }), "Protect labeled pixels (X)"));
    }
    if (st.layer === "conf" || st.conf) {
      kids.push(h("label", { class: "field" }, h("span", {}, `Low-confidence threshold: ${Math.round(st.confThresh * 100)}%`),
        h("input", { type: "range", min: 10, max: 95, value: st.confThresh * 100, oninput: (e) => { st.confThresh = e.target.value / 100; if (st.layer === "conf") { renderOverlay(); draw(); } renderToolOpts(); } })));
    }
    kids.push(h("label", { class: "row" }, h("input", { type: "checkbox", checked: st.autoAdvance, onchange: (e) => { st.autoAdvance = e.target.checked; } }), "Go to next image after Submit"));
    toolOpts.replaceChildren(...kids);
  }
  function renderInfo() {
    const i = st.img;
    statusBadge.className = "badge " + i.status; statusBadge.textContent = STATUS_LABEL[i.status] || i.status;
    navPos.textContent = i.nav && i.nav.pos ? `${i.nav.pos} / ${i.nav.total}` : "";
    prevBtn.disabled = !(i.nav && i.nav.prev); nextBtn.disabled = !(i.nav && i.nav.next);
    memBtn.textContent = i.in_memory ? "Remove from memory bank" : "Add to memory bank";
    info.replaceChildren(
      h("dt", {}, "Size"), h("dd", {}, `${st.W} × ${st.H}`),
      h("dt", {}, "Status"), h("dd", {}, STATUS_LABEL[i.status] || i.status),
      h("dt", {}, "Reference"), h("dd", {}, i.in_memory ? "yes (in memory bank)" : "no"),
      h("dt", {}, "Auto-label confidence"), h("dd", {}, fmt.pct(i.conf)),
      h("dt", {}, "Labeled"), h("dd", {}, fmt.pct(i.labeled_frac)),
      h("dt", {}, "Source"), h("dd", { title: i.source, style: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" } }, i.source || "–"),
      h("dt", {}, "Updated"), h("dd", {}, fmt.ago(i.updated)));
  }
  function setTool(t) {
    if (st.poly.length && t !== "poly") st.poly = [];
    st.tool = t;
    for (const [k, b] of Object.entries(toolBtns)) b.classList.toggle("on", k === t);
    canvas.style.cursor = t === "pan" ? "grab" : t === "pick" ? "copy" : "crosshair";
    if (t === "fill") superpixels().catch((e) => toast(e.message, true));
    renderToolOpts(); draw();
  }

  // ------------------------------------------------------------------ input
  let drag = null;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [, , sx, sy] = toImg(e);
    const f = Math.exp(-e.deltaY * 0.0015);
    const s = Math.min(40, Math.max(0.05, st.view.s * f));
    st.view.x = sx - (sx - st.view.x) * s / st.view.s; st.view.y = sy - (sy - st.view.y) * s / st.view.s; st.view.s = s;
    draw();
  }, { passive: false });
  canvas.addEventListener("pointerdown", async (e) => {
    canvas.focus();
    canvas.setPointerCapture(e.pointerId);
    const [x, y, sx, sy] = toImg(e);
    const xi = Math.floor(x), yi = Math.floor(y), inside = xi >= 0 && yi >= 0 && xi < st.W && yi < st.H;
    if (e.button === 1 || st.tool === "pan" || st.space) { drag = { pan: true, sx, sy, vx: st.view.x, vy: st.view.y }; canvas.style.cursor = "grabbing"; return; }
    if (e.altKey || st.tool === "pick") { if (inside && st.mask[yi * st.W + xi]) { st.cls = st.mask[yi * st.W + xi]; renderClasses(st.lastStats); } return; }
    if (!inside && st.tool !== "poly") return;
    if (st.tool === "brush" || st.tool === "erase") {
      if (e.button === 2) return;
      push(); const v = st.tool === "erase" ? 0 : st.cls;
      stamp(x, y, st.brush, v); changed([xi - st.brush - 1, yi - st.brush - 1, xi + st.brush + 2, yi + st.brush + 2]);
      drag = { paint: true, last: [x, y], v };
    } else if (st.tool === "poly") {
      if (st.poly.length >= 3) {
        const p0 = st.poly[0];
        if (Math.hypot(p0[0] - x, p0[1] - y) * st.view.s < 9) { fillPoly(st.poly); st.poly = []; draw(); return; }
      }
      st.poly.push([Math.min(st.W, Math.max(0, x)), Math.min(st.H, Math.max(0, y))]); draw();
    } else if (st.tool === "fill") {
      await superpixels(); push(); fillSuper(xi, yi, e.button === 2 ? 0 : st.cls); drag = { fill: true, v: e.button === 2 ? 0 : st.cls };
    } else if (st.tool === "bucket") {
      bucket(xi, yi);
    } else if (st.tool === "wand") {
      wand(x, y, e.shiftKey);
    } else if (st.tool === "sam") {
      st.sam.pts.push([x, y, e.button === 2 ? 0 : 1]); draw(); samQuery();
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    const [x, y, sx, sy] = toImg(e);
    st.mouse = [x, y];
    if (drag && drag.pan) { st.view.x = drag.vx + sx - drag.sx; st.view.y = drag.vy + sy - drag.sy; draw(); return; }
    if (drag && drag.paint) { strokeTo(drag.last, [x, y], drag.v); drag.last = [x, y]; return; }
    if (drag && drag.fill && st.sp) {
      const xi = Math.floor(x), yi = Math.floor(y);
      if (xi >= 0 && yi >= 0 && xi < st.W && yi < st.H) fillSuper(xi, yi, drag.v);
      return;
    }
    draw();
  });
  const endDrag = () => { if (drag && drag.pan) canvas.style.cursor = st.tool === "pan" ? "grab" : "crosshair"; drag = null; };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", () => { st.mouse = null; draw(); });
  canvas.addEventListener("dblclick", () => { if (st.tool === "poly" && st.poly.length >= 3) { st.poly.pop(); fillPoly(st.poly); st.poly = []; draw(); } });

  const keydown = (e) => {
    if (!document.body.contains(wrap)) { document.removeEventListener("keydown", keydown); document.removeEventListener("keyup", keyup); return; }
    if (e.target.matches("input[type=text],input[type=search],textarea,select")) return;
    const k = e.key, ctrl = e.ctrlKey || e.metaKey;
    if (ctrl && k.toLowerCase() === "s") { e.preventDefault(); save(st.img.status === "reviewed" ? "reviewed" : "in_progress"); return; }
    if (ctrl && k === "Enter") { e.preventDefault(); submit(e.shiftKey); return; }
    if (ctrl && k.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (ctrl && k.toLowerCase() === "y") { e.preventDefault(); redo(); return; }
    if (e.altKey && k === "ArrowRight") { e.preventDefault(); go(st.img.nav.next); return; }
    if (e.altKey && k === "ArrowLeft") { e.preventDefault(); go(st.img.nav.prev); return; }
    if (ctrl || e.altKey) return;
    if (k === " ") { st.space = true; canvas.style.cursor = "grab"; e.preventDefault(); return; }
    if (/^[0-9]$/.test(k)) { const n = (+k + 9) % 10; if (st.classes[n]) { st.cls = st.classes[n].idx; st.samDirty = true; renderClasses(st.lastStats); draw(); } return; }
    const map = { b: "brush", e: "erase", p: "poly", f: "fill", g: "bucket", w: "wand", s: "sam", i: "pick", h: "pan" };
    if (map[k.toLowerCase()]) { setTool(map[k.toLowerCase()]); return; }
    if (k === "[") { st.brush = Math.max(1, Math.round(st.brush / 1.25)); renderToolOpts(); draw(); }
    else if (k === "]") { st.brush = Math.min(200, Math.round(st.brush * 1.25) + 1); renderToolOpts(); draw(); }
    else if (k === "z" || k === "Z") fit();
    else if (k === "x" || k === "X") { st.protect = !st.protect; renderToolOpts(); toast(st.protect ? "Protecting labeled pixels" : "Painting over labels"); }
    else if (k === "v" || k === "V") { st.layer = st.layer === "none" ? "label" : "none"; layerSel.value = st.layer; renderOverlay(); draw(); }
    else if (k === "c" || k === "C") { st.layer = st.layer === "conf" ? "label" : "conf"; layerSel.value = st.layer; renderOverlay(); draw(); renderToolOpts(); }
    else if (k === "o" || k === "O") { st.outline = !st.outline; outlineCb.checked = st.outline; renderOverlay(); draw(); }
    else if (k === "a" || k === "A") autolabel(st.mask.some((v) => v) ? "fill" : "replace");
    else if (k === "Enter") { if (st.tool === "poly" && st.poly.length >= 3) { fillPoly(st.poly); st.poly = []; draw(); } else if (st.tool === "sam") samApply(); }
    else if (k === "Escape") { st.poly = []; st.sam = { pts: [], mask: null, score: null }; draw(); renderToolOpts(); }
    else if (k === "Backspace" && st.poly.length) { st.poly.pop(); draw(); }
  };
  const keyup = (e) => { if (e.key === " ") { st.space = false; canvas.style.cursor = st.tool === "pan" ? "grab" : "crosshair"; } };
  document.addEventListener("keydown", keydown);
  document.addEventListener("keyup", keyup);
  const ro = new ResizeObserver(() => { const keep = st.view.s; resize(); if (!st.fitted) { fit(); st.fitted = true; } else draw(); });
  ro.observe(stage);
  window.addEventListener("beforeunload", (e) => { if (st.edited && st.dirty && document.body.contains(wrap)) { e.preventDefault(); e.returnValue = ""; } });

  renderOverlay(); renderClasses(); scheduleStats(); renderInfo(); setTool("brush");
  st.dirty = !!st.fromPred;
  saveState.textContent = st.fromPred ? "auto-label loaded (not saved)" : "";
  fit();
  // expose for tests
  window.__editor = st;
  return () => {
    // leaving by any route (gallery link, breadcrumb, browser back): keep real edits as "in progress"
    if (st.edited && st.dirty) save(st.img.status === "reviewed" ? "reviewed" : "in_progress");
    ro.disconnect(); document.removeEventListener("keydown", keydown); document.removeEventListener("keyup", keyup);
  };
}

export function qs(q) {
  const e = Object.entries(q || {}).filter(([, v]) => v !== undefined && v !== null && v !== "");
  return e.length ? "?" + e.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&") : "";
}
