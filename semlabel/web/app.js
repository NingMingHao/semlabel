// SemLabel single-page app: router, project views, gallery, memory bank, evaluation, data, settings.
import { api, P, I, waitJob } from "./api.js";
import { h, toast, modal, confirmBox, fmt, STATUS_LABEL, STATUS_COLOR } from "./ui.js";
import { openEditor, qs } from "./editor.js";

const view = document.getElementById("view");
const crumbsEl = document.getElementById("crumbs");
let cleanup = null;
let system = null;

function crumbs(items) {
  crumbsEl.replaceChildren(...items.flatMap(([label, href], k) => [
    k ? h("span", {}, "/") : null, href ? h("a", { href }, label) : h("span", {}, label)]).filter(Boolean));
}

function parseHash() {
  const [path, query] = location.hash.replace(/^#/, "").split("?");
  const q = {};
  for (const kv of (query || "").split("&")) if (kv) { const [k, v] = kv.split("="); q[k] = decodeURIComponent(v || ""); }
  return { parts: (path || "/").split("/").filter(Boolean), q };
}

async function route() {
  if (cleanup) { try { cleanup(); } catch (e) { /* ignore */ } cleanup = null; }
  const { parts, q } = parseHash();
  try {
    if (parts[0] === "p" && parts[2] === "edit") {
      document.body.style.overflow = "hidden";
      cleanup = await openEditor(view, +parts[1], +parts[3], q, crumbs);
      return;
    }
    document.body.style.overflow = "";
    if (parts[0] === "p") return await projectView(+parts[1], parts[2] || "overview", q);
    return await projectsView();
  } catch (e) {
    view.replaceChildren(h("div", { class: "page" }, h("div", { class: "empty" }, "Error: " + e.message)));
  }
}
window.addEventListener("hashchange", route);

// ====================================================================== projects
async function projectsView() {
  crumbs([["Projects", null]]);
  const list = await api.get("/api/projects");
  const grid = h("div", { class: "proj-grid" }, list.map((p) => {
    const t = Math.max(1, p.total);
    return h("a", { class: "card proj", href: `#/p/${p.id}` },
      h("div", { class: "row", style: { justifyContent: "space-between" } }, h("h2", {}, p.name), h("span", { class: "badge" }, p.feature_key)),
      p.description ? h("div", { class: "muted" }, p.description) : null,
      h("div", { class: "statusbar", title: "new / auto-labeled / in progress / reviewed" },
        ["reviewed", "in_progress", "predicted", "new"].map((s) => h("i", { style: { width: (100 * p.counts[s] / t) + "%", background: STATUS_COLOR[s] } }))),
      h("div", { class: "row muted", style: { fontSize: "12.5px" } },
        `${fmt.num(p.total)} images · ${p.counts.reviewed} reviewed · ${p.counts.predicted} auto-labeled · ${p.references} references`));
  }));
  view.replaceChildren(h("div", { class: "page" },
    h("div", { class: "row", style: { justifyContent: "space-between" } },
      h("div", {}, h("h1", {}, "Projects"), h("p", { class: "muted", style: { margin: "4px 0 0" } },
        "In-context semantic labeling: label a few images, add them to the memory bank, and DINOv3 + INSID3 auto-labels the rest.")),
      h("button", { class: "btn primary", onclick: newProject }, "New project")),
    list.length ? grid : h("div", { class: "empty", style: { marginTop: "20px" } }, "No projects yet. Create one to start labeling.")));
}

async function newProject() {
  const name = h("input", { type: "text", placeholder: "e.g. Office scans, floor 2" });
  const desc = h("input", { type: "text", placeholder: "optional" });
  const tmpl = h("select", {}, h("option", { value: "indoor" }, "Indoor (16 common indoor classes)"),
    h("option", { value: "custom" }, "Custom class list"), h("option", { value: "empty" }, "Empty (add classes later)"));
  const custom = h("textarea", { rows: 5, placeholder: "one class per line, e.g.\nfloor\nwall\ndoor", hidden: true });
  tmpl.onchange = () => { custom.hidden = tmpl.value !== "custom"; };
  const model = h("select", {}, (system?.models || []).map((m) => h("option", { value: m.key, selected: m.key === "large", disabled: !m.available }, `${m.key} (${m.hub}, ${m.dim}-d)${m.available ? "" : " — weights missing"}`)));
  const body = h("div", { class: "stack" }, h("label", { class: "field" }, h("span", {}, "Name"), name),
    h("label", { class: "field" }, h("span", {}, "Description"), desc),
    h("label", { class: "field" }, h("span", {}, "Classes"), tmpl, custom),
    h("label", { class: "field" }, h("span", {}, "DINOv3 backbone"), model));
  const r = await modal("New project", body, [{ label: "Cancel", value: null }, { label: "Create", cls: "primary", action: async () => {
    if (!name.value.trim()) { toast("Give the project a name.", true); return false; }
    const classes = tmpl.value === "custom" ? custom.value.split("\n").map((s) => s.trim()).filter(Boolean) : tmpl.value === "empty" ? [] : null;
    try {
      return await api.post("/api/projects", { name: name.value.trim(), description: desc.value, template: tmpl.value === "indoor" ? "indoor" : "empty", classes, config: { model: model.value } });
    } catch (e) { toast(e.message, true); return false; }
  } }]);
  if (r && r.id) location.hash = `#/p/${r.id}`;
}

// ====================================================================== project shell
const TABS = [["overview", "Overview"], ["images", "Images"], ["memory", "Memory bank"], ["evaluate", "Evaluate"],
  ["classes", "Classes"], ["data", "Import / Export"], ["settings", "Settings"]];

async function projectView(pid, tab, q) {
  const p = await api.get(P(pid));
  crumbs([["Projects", "#/"], [p.name, `#/p/${pid}`], [TABS.find((t) => t[0] === tab)?.[1] || tab, null]]);
  const body = h("div");
  view.replaceChildren(h("div", { class: "page" },
    h("div", { class: "row", style: { justifyContent: "space-between" } }, h("h1", {}, p.name),
      h("div", { class: "row" }, h("span", { class: "badge", title: "backbone + input size + debiasing (memory banks are per key)" }, p.feature_key))),
    h("nav", { class: "tabs" }, TABS.map(([k, label]) => h("a", { href: `#/p/${pid}/${k}`, class: k === tab ? "on" : "" }, label))),
    body));
  const fn = { overview: overviewTab, images: imagesTab, memory: memoryTab, evaluate: evaluateTab, classes: classesTab,
    data: dataTab, settings: settingsTab }[tab] || overviewTab;
  await fn(body, p, q);
}

async function runJob(pid, kind, params, label) {
  const { job } = await api.post(P(pid) + "/jobs", { kind, params });
  toast(`${label} started`);
  pollJobs();
  const j = await waitJob(job);
  pollJobs();
  if (j.status === "done") toast(`${label} finished`);
  else toast(`${label} ${j.status}: ${j.message}`, true);
  return j;
}

// ====================================================================== overview
async function overviewTab(body, p) {
  const [ev, events] = await Promise.all([api.get(P(p.id) + "/evals"), api.get(P(p.id) + "/events?limit=12")]);
  const c = p.counts, t = Math.max(1, p.total);
  const next = p.total === 0 ? ["Import images", `#/p/${p.id}/data`]
    : p.references === 0 ? ["Label a first image", `#/p/${p.id}/images?status=new`]
    : c.new > 0 ? ["Auto-label new images", `#/p/${p.id}/images?status=new`]
    : c.predicted > 0 ? ["Review auto-labels, lowest confidence first", `#/p/${p.id}/images?status=predicted&sort=conf`]
    : ["Export the dataset", `#/p/${p.id}/data`];
  body.append(
    h("div", { class: "grid3" },
      h("div", { class: "card stack" }, h("h2", {}, "Progress"),
        h("div", { class: "statusbar" }, ["reviewed", "in_progress", "predicted", "new"].map((s) => h("i", { style: { width: (100 * c[s] / t) + "%", background: STATUS_COLOR[s] } }))),
        h("div", { class: "row", style: { gap: "18px" } }, ["new", "predicted", "in_progress", "reviewed"].map((s) =>
          h("a", { class: "stat", href: `#/p/${p.id}/images?status=${s}`, style: { textDecoration: "none", color: "inherit" } }, h("b", {}, fmt.num(c[s])), h("span", {}, STATUS_LABEL[s]))))),
      h("div", { class: "card stack" }, h("h2", {}, "Memory bank"),
        h("div", { class: "row", style: { gap: "18px" } },
          h("div", { class: "stat" }, h("b", {}, fmt.num(p.references)), h("span", {}, "reference images")),
          h("div", { class: "stat" }, h("b", {}, fmt.num(p.memory.size)), h("span", {}, "patch features")),
          h("div", { class: "stat" }, h("b", {}, fmt.bytes(p.memory.bytes || 0)), h("span", {}, "size"))),
        p.memory.stale ? h("div", { class: "badge predicted" }, "Bank needs a rebuild for the current backbone") : null),
      h("div", { class: "card stack" }, h("h2", {}, "Auto-label quality"),
        ev.length ? h("div", { class: "row", style: { gap: "18px" } },
          h("div", { class: "stat" }, h("b", {}, ev[0].miou.toFixed(1)), h("span", {}, "mIoU on held-out reviewed images")),
          h("div", { class: "stat" }, h("b", {}, ev[0].pixacc.toFixed(1) + "%"), h("span", {}, "pixel accuracy")))
          : h("p", { class: "muted" }, "Run an evaluation once some reviewed images are not references."),
        h("a", { class: "btn small", href: `#/p/${p.id}/evaluate` }, "Evaluate"))),
    h("div", { class: "card", style: { marginTop: "14px" } },
      h("div", { class: "row", style: { justifyContent: "space-between" } }, h("h2", {}, "Workflow"),
        h("a", { class: "btn primary", href: next[1] }, "Next: " + next[0])),
      h("div", { class: "steps", style: { marginTop: "12px" } },
        [["Import", "Upload images, a folder, a video or a ROS 2 bag (color rig topics)."],
          ["Label references", "Paint a few images fully: brush, polygon, superpixel fill, SAM clicks, or start from the SegFormer pre-label."],
          ["Add to memory", "Submit + add to memory stores their DINOv3 patch features with labels."],
          ["Auto-label", "INSID3-style in-context matching labels every new image in about 0.3 s."],
          ["Review", "Fix the lowest-confidence images first; add corrected ones to the memory bank."],
          ["Compress & export", "Coreset-compress the bank when it grows, evaluate, export PNG / COCO / CVAT."]]
          .map(([t, d]) => h("div", { class: "card" }, h("h3", {}, t), h("p", { class: "muted", style: { margin: "4px 0 0", fontSize: "13px" } }, d))))),
    h("div", { class: "card", style: { marginTop: "14px" } }, h("h2", {}, "Recent activity"),
      events.length ? h("table", { class: "t", style: { marginTop: "8px" } }, h("tbody", {}, events.map((e) => h("tr", {},
        h("td", { class: "muted", style: { whiteSpace: "nowrap" } }, fmt.ago(e.ts)), h("td", {}, e.kind.replace(/_/g, " ")),
        h("td", { class: "muted mono" }, e.image_id ? `image ${e.image_id}` : ""), h("td", { class: "muted mono" }, summarize(e.payload)))))) : h("p", { class: "muted" }, "Nothing yet.")));
}

function summarize(payload) {
  try {
    const d = JSON.parse(payload);
    return Object.entries(d).filter(([k, v]) => typeof v !== "object").map(([k, v]) => `${k}=${v}`).join(" ").slice(0, 120);
  } catch (e) { return ""; }
}

// ====================================================================== images / gallery
async function imagesTab(body, p, q) {
  const state = { status: q.status || "", sort: q.sort || "id", q: q.q || "", page: +(q.page || 0), refs: q.refs || "", ids: q.ids || "", sel: new Set() };
  const PAGE = 120;
  const chips = h("div", { class: "chips" });
  const search = h("input", { type: "search", placeholder: "Search name / source", value: state.q });
  const sortSel = h("select", { "aria-label": "Sort" }, [["id", "Oldest first"], ["-id", "Newest first"], ["conf", "Lowest confidence first"],
    ["-conf", "Highest confidence first"], ["updated", "Recently updated"], ["name", "Name"]].map(([v, l]) => h("option", { value: v, selected: v === state.sort }, l)));
  const grid = h("div", { class: "gallery" });
  const pager = h("div", { class: "pager" });
  const selbar = h("div", { class: "selbar", hidden: true });
  const counts = p.counts;
  const top = h("div", { class: "toolbar" }, chips, h("div", { class: "spacer" }), search, sortSel,
    h("button", { class: "btn primary", title: "Auto-label every image with status New", onclick: () => batchPredict({ status: "new" }, counts.new) }, `Auto-label new (${counts.new})`),
    h("button", { class: "btn", title: "Re-run auto-labeling on auto-labeled (not reviewed) images, e.g. after the memory bank changed", onclick: () => batchPredict({ status: "predicted" }, counts.predicted) }, "Re-label auto-labeled"),
    h("button", { class: "btn", title: "Pick the images least like the current references (k-center greedy on DINOv3 image descriptors)", onclick: suggest }, "Suggest images to label"));
  body.append(top, grid, pager, selbar);

  async function suggest() {
    const n = h("input", { type: "number", value: 10, min: 1, max: 200 });
    const pool = h("select", {}, h("option", { value: "new" }, "from New images"), h("option", { value: "new,predicted" }, "from New + Auto-labeled"));
    const ok = await modal("Suggest images to label", h("div", { class: "stack" },
      h("p", { class: "muted", style: { margin: 0 } }, "Chooses the most diverse images, starting with those least covered by the current memory bank. Labeling these first gives the biggest gain per click."),
      h("label", { class: "field" }, h("span", {}, "How many"), n), pool), [{ label: "Cancel", value: false }, { label: "Suggest", value: true, cls: "primary" }]);
    if (!ok) return;
    const j = await runJob(p.id, "suggest", { n: +n.value, status: pool.value }, "Suggestion");
    if (j.status !== "done" || !j.result.ids.length) return;
    state.ids = j.result.ids.join(","); state.status = ""; state.refs = ""; state.page = 0; state.sort = "id";
    j.result.ids.forEach((i) => state.sel.add(i));
    load();
  }
  const setHash = () => {
    const qq = { status: state.status, sort: state.sort, q: state.q, page: state.page || "", refs: state.refs, ids: state.ids };
    history.replaceState(null, "", `#/p/${p.id}/images${qs(qq)}`);
  };
  function renderChips() {
    const items = [["", `All (${p.total})`], ...["new", "predicted", "in_progress", "reviewed"].map((s) => [s, `${STATUS_LABEL[s]} (${counts[s]})`])];
    chips.replaceChildren(...items.map(([s, l]) => h("button", { class: !state.refs && !state.ids && state.status === s ? "on" : "", onclick: () => { state.status = s; state.refs = ""; state.ids = ""; state.page = 0; load(); } }, l)),
      h("button", { class: state.refs ? "on" : "", onclick: () => { state.refs = "1"; state.status = ""; state.ids = ""; state.page = 0; load(); } }, `★ References (${p.references})`),
      ...(state.ids ? [h("button", { class: "on", title: "click to clear", onclick: () => { state.ids = ""; load(); } }, `Suggested (${state.ids.split(",").length}) ×`)] : []));
  }
  async function batchPredict(params, n) {
    if (!n) { toast("Nothing to auto-label in that group."); return; }
    if (!p.memory.size && !p.references) { toast("The memory bank is empty. Label a few images and add them to memory first.", true); return; }
    await runJob(p.id, "predict", params, `Auto-labeling ${n} images`);
    route();
  }
  async function load() {
    renderChips(); setHash();
    const url = P(p.id) + `/images?limit=${PAGE}&offset=${state.page * PAGE}&sort=${state.sort}` + (state.status ? `&status=${state.status}` : "") +
      (state.q ? `&q=${encodeURIComponent(state.q)}` : "") + (state.refs ? "&in_memory=1" : "") + (state.ids ? `&ids=${state.ids}` : "");
    const r = await api.get(url);
    const editQ = qs({ status: state.status, sort: state.sort });
    grid.replaceChildren(...(r.items.length ? r.items.map((im) => {
      const cb = h("input", { type: "checkbox", class: "check", "aria-label": "select", checked: state.sel.has(im.id),
        onclick: (e) => { e.stopPropagation(); toggleSel(im.id, e.shiftKey, r.items); } });
      return h("div", { class: "thumb" + (state.sel.has(im.id) ? " sel" : ""), "data-id": im.id, tabindex: 0,
        onclick: () => { location.hash = `#/p/${p.id}/edit/${im.id}${editQ}`; },
        onkeydown: (e) => { if (e.key === "Enter") location.hash = `#/p/${p.id}/edit/${im.id}${editQ}`; } },
        h("img", { src: I(p.id, im.id) + "/thumb", loading: "lazy", alt: im.name }), cb,
        im.in_memory ? h("span", { class: "ref", title: "reference in the memory bank" }, "★ ref") : null,
        im.conf !== null && im.status === "predicted" ? h("span", { class: "conf", title: "mean auto-label confidence" }, fmt.pct(im.conf)) : null,
        h("div", { class: "meta" }, h("span", { class: "name", title: im.name }, im.name), h("span", { class: "badge " + im.status }, STATUS_LABEL[im.status])));
    }) : [h("div", { class: "empty", style: { gridColumn: "1/-1" } }, p.total ? "No images match this filter." : "No images yet — import some on the Import / Export tab.")]));
    const pages = Math.ceil(r.total / PAGE);
    pager.replaceChildren(pages > 1 ? h("div", { class: "row" },
      h("button", { class: "btn small", disabled: state.page === 0, onclick: () => { state.page--; load(); } }, "‹"),
      h("span", { class: "muted num" }, `page ${state.page + 1} / ${pages} · ${fmt.num(r.total)} images`),
      h("button", { class: "btn small", disabled: state.page >= pages - 1, onclick: () => { state.page++; load(); } }, "›"),
      h("button", { class: "btn small", onclick: () => { r.items.forEach((im) => state.sel.add(im.id)); load(); } }, "Select page")) : h("span", { class: "muted" }, `${fmt.num(r.total)} images`));
    renderSel();
  }
  let lastSel = null;
  function toggleSel(id, shift, items) {
    if (shift && lastSel !== null) {
      const ids = items.map((i) => i.id), a = ids.indexOf(lastSel), b = ids.indexOf(id);
      if (a >= 0 && b >= 0) for (const k of ids.slice(Math.min(a, b), Math.max(a, b) + 1)) state.sel.add(k);
    } else if (state.sel.has(id)) state.sel.delete(id); else state.sel.add(id);
    lastSel = id;
    grid.querySelectorAll(".thumb").forEach((t) => { const on = state.sel.has(+t.dataset.id); t.classList.toggle("sel", on); t.querySelector(".check").checked = on; });
    renderSel();
  }
  function renderSel() {
    const n = state.sel.size, ids = [...state.sel];
    selbar.hidden = !n;
    selbar.replaceChildren(h("b", {}, `${n} selected`),
      h("button", { class: "btn small primary", onclick: async () => { await runJob(p.id, "predict", { ids }, `Auto-labeling ${n} images`); state.sel.clear(); load(); } }, "Auto-label"),
      h("button", { class: "btn small", onclick: async () => { await runJob(p.id, "prelabel", { ids }, `SegFormer pre-label of ${n} images`); state.sel.clear(); load(); } }, "Pre-label (SegFormer)"),
      h("button", { class: "btn small", title: "Uses each image's saved label", onclick: async () => { const r = await api.post(P(p.id) + "/memory/add", { ids }); if (r.job) await waitJob(r.job); toast("Added to the memory bank"); state.sel.clear(); route(); } }, "Add to memory"),
      h("button", { class: "btn small", onclick: async () => { await api.post(P(p.id) + "/memory/remove", { ids }); toast("Removed from the memory bank"); state.sel.clear(); route(); } }, "Remove from memory"),
      h("button", { class: "btn small", title: "Accept the auto-label as the reviewed label", onclick: async () => {
        if (!(await confirmBox("Accept auto-labels", `Mark ${n} images reviewed using their current auto-labels?`, "Accept"))) return;
        let ok = 0; for (const i of ids) { try { await api.post(I(p.id, i) + "/accept", {}); ok++; } catch (e) { /* no prediction */ } }
        toast(`Accepted ${ok} auto-labels`); state.sel.clear(); route(); } }, "Accept auto-labels"),
      h("button", { class: "btn small danger", onclick: async () => {
        if (!(await confirmBox("Delete images", `Delete ${n} images and their labels? This cannot be undone.`, "Delete", true))) return;
        await api.post(P(p.id) + "/images/delete", { ids }); state.sel.clear(); route(); } }, "Delete"),
      h("button", { class: "btn small", onclick: () => { state.sel.clear(); load(); } }, "Clear selection"));
  }
  let st = null;
  search.oninput = () => { clearTimeout(st); st = setTimeout(() => { state.q = search.value; state.page = 0; load(); }, 300); };
  sortSel.onchange = () => { state.sort = sortSel.value; state.page = 0; load(); };
  await load();
}

// ====================================================================== memory bank
async function memoryTab(body, p) {
  const m = await api.get(P(p.id) + "/memory");
  const cls = Object.fromEntries(p.classes.map((c) => [c.idx, c]));
  const maxc = Math.max(1, ...Object.values(m.per_class || {}));
  const ratio = h("input", { type: "range", min: 1, max: 100, value: 20 });
  const ratioLbl = h("span", { class: "num" }, "20%");
  const mpc = h("input", { type: "number", min: 0, placeholder: "no limit", style: { width: "110px" } });
  ratio.oninput = () => { ratioLbl.textContent = ratio.value + "%"; est.textContent = `≈ ${fmt.num(Math.round(m.size * ratio.value / 100))} features kept`; };
  const est = h("span", { class: "muted" }, `≈ ${fmt.num(Math.round(m.size * 0.2))} features kept`);
  body.append(
    h("div", { class: "grid2" },
      h("div", { class: "card stack" },
        h("h2", {}, "Bank"),
        h("div", { class: "row", style: { gap: "22px" } },
          h("div", { class: "stat" }, h("b", {}, fmt.num(m.references)), h("span", {}, "reference images")),
          h("div", { class: "stat" }, h("b", {}, fmt.num(m.size)), h("span", {}, "patch features")),
          h("div", { class: "stat" }, h("b", {}, fmt.bytes(m.bytes)), h("span", {}, `${m.dim || "–"}-d, fp16`))),
        h("p", { class: "muted", style: { margin: 0 } }, `Key ${m.feature_key}. Each reference adds the debiased DINOv3 features of patches covered ≥ ${Math.round(p.config.purity * 100)}% by one class.`),
        m.stale || (m.other_banks || []).length ? h("div", { class: "row" },
          m.stale ? h("span", { class: "badge predicted" }, "References are missing from this bank") : null,
          (m.other_banks || []).length ? h("span", { class: "muted" }, `Other banks on disk: ${m.other_banks.join(", ")}`) : null) : null,
        h("div", { class: "row" },
          h("button", { class: "btn", onclick: async () => { await runJob(p.id, "memory_rebuild", {}, "Memory rebuild"); route(); } }, "Rebuild from references"),
          h("button", { class: "btn danger", onclick: async () => {
            if (!(await confirmBox("Clear memory bank", "Remove every reference from the memory bank? Labels are kept.", "Clear", true))) return;
            await api.post(P(p.id) + "/memory/clear"); route(); } }, "Clear bank"))),
      h("div", { class: "card stack" },
        h("h2", {}, "Coreset compression"),
        h("p", { class: "muted", style: { margin: 0 } }, "PatchCore-style k-center greedy (as in anomalib), per class: keeps the features that cover the most different appearances and drops near-duplicates, e.g. the same wall seen in many frames."),
        h("p", { class: "muted", style: { margin: 0 } }, "Measured on NYUv2 (40 references, 3 random splits): keeping 20% changed mIoU by +0.1 to +1.5 and made matching 3–6× faster; when held-out images are near-copies of references it cost 1.7. Rebuild restores the full bank at any time."),
        h("label", { class: "field" }, h("span", {}, "Keep ratio per class ", ratioLbl), ratio),
        h("label", { class: "field" }, h("span", {}, "Max features per class (optional)"), mpc),
        h("div", { class: "row" }, est,
          h("button", { class: "btn primary", disabled: !m.size, onclick: async () => {
            const j = await runJob(p.id, "memory_compress", { ratio: ratio.value / 100, max_per_class: mpc.value ? +mpc.value : null }, "Compression");
            if (j.status === "done") toast(`Kept ${fmt.num(j.result.after)} of ${fmt.num(j.result.before)} features`);
            route(); } }, "Compress")),
        (m.history || []).length ? h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "When"), h("th", {}, "Before"), h("th", {}, "After"), h("th", {}, "Ratio"))),
          h("tbody", {}, m.history.slice().reverse().map((r) => h("tr", {}, h("td", {}, fmt.ago(r.time)), h("td", { class: "num" }, fmt.num(r.before)),
            h("td", { class: "num" }, fmt.num(r.after)), h("td", {}, r.ratio ? Math.round(r.ratio * 100) + "%" : r.max_per_class ? `≤ ${r.max_per_class}/class` : "–"))))) : null)),
    h("div", { class: "card", style: { marginTop: "14px" } }, h("h2", {}, "Features per class"),
      Object.keys(m.per_class || {}).length ? h("table", { class: "t", style: { marginTop: "8px" } }, h("tbody", {}, Object.entries(m.per_class).sort((a, b) => b[1] - a[1]).map(([k, n]) => h("tr", {},
        h("td", { style: { width: "220px" } }, h("span", { class: "swatch", style: { background: cls[k]?.color || "#999" } }), " ", cls[k]?.name || `class ${k}`),
        h("td", {}, h("div", { class: "bar" }, h("i", { style: { width: (100 * n / maxc) + "%", background: cls[k]?.color } }))),
        h("td", { class: "num", style: { width: "90px", textAlign: "right" } }, fmt.num(n)))))) : h("p", { class: "muted" }, "Empty. Submit an image with “Submit + add to memory”, or select images in the gallery and choose Add to memory."),
      p.classes.filter((c) => !(m.per_class || {})[c.idx]).length ? h("p", { class: "muted", style: { marginTop: "8px" } },
        "No examples yet for: " + p.classes.filter((c) => !(m.per_class || {})[c.idx]).map((c) => c.name).join(", ") + ". These classes cannot be auto-labeled until a reference contains them.") : null),
    h("div", { class: "card", style: { marginTop: "14px" } }, h("h2", {}, "Reference images"),
      (m.items || []).length ? h("div", { class: "gallery", style: { marginTop: "10px" } }, m.items.map((it) => h("div", { class: "thumb" },
        h("img", { src: I(p.id, it.id) + "/thumb", alt: it.name, onclick: () => { location.hash = `#/p/${p.id}/edit/${it.id}?refs=1`; } }),
        h("div", { class: "meta" }, h("span", { class: "name", title: it.name }, it.name), h("span", { class: "muted num", title: "features kept / added" }, `${fmt.num(it.kept ?? 0)}/${fmt.num(it.patches ?? 0)}`)),
        h("div", { class: "meta" }, h("button", { class: "btn small danger", onclick: async () => { await api.post(P(p.id) + "/memory/remove", { ids: [it.id] }); route(); } }, "Remove"))))) : h("p", { class: "muted" }, "No references yet.")));
}

// ====================================================================== evaluate
async function evaluateTab(body, p) {
  const ev = await api.get(P(p.id) + "/evals");
  const reviewedHeld = (await api.get(P(p.id) + "/images?status=reviewed&in_memory=0&limit=1")).total;
  const runBtn = h("button", { class: "btn primary", disabled: !reviewedHeld || !p.references, onclick: async () => {
    const j = await runJob(p.id, "evaluate", {}, `Evaluation on ${reviewedHeld} images`);
    if (j.status === "done") route(); } }, `Evaluate on ${reviewedHeld} held-out reviewed images`);
  const last = ev[0];
  const chart = ev.length > 1 ? lineChart(ev.slice().reverse().map((e) => [e.memory_size, e.miou])) : null;
  body.append(h("div", { class: "card stack" }, h("h2", {}, "How good is auto-labeling right now?"),
    h("p", { class: "muted", style: { margin: 0 } }, "Auto-labels every reviewed image that is not a reference and compares it with your label. Add more varied references to raise the score; mark a few reviewed images as held out by not adding them to memory."),
    h("div", {}, runBtn)),
    last ? h("div", { class: "grid2", style: { marginTop: "14px" } },
      h("div", { class: "card stack" }, h("h2", {}, `Latest: ${last.miou.toFixed(1)} mIoU · ${last.pixacc.toFixed(1)}% pixels`),
        h("p", { class: "muted", style: { margin: 0 } }, `${last.n_images} images · memory ${fmt.num(last.memory_size)} features · ${last.config.model} · mode ${last.config.mode} · ${fmt.ago(last.ts)}`),
        h("table", { class: "t" }, h("tbody", {}, Object.entries(last.per_class).sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0)).map(([n, v]) => {
          const c = p.classes.find((x) => x.name === n);
          return h("tr", {}, h("td", { style: { width: "200px" } }, h("span", { class: "swatch", style: { background: c?.color || "#999" } }), " ", n),
            h("td", {}, h("div", { class: "bar" }, h("i", { style: { width: (v ?? 0) + "%", background: c?.color } }))),
            h("td", { class: "num", style: { width: "60px", textAlign: "right" } }, v === null ? "–" : v.toFixed(1)));
        })))),
      h("div", { class: "card stack" }, h("h2", {}, "History"), chart || h("p", { class: "muted" }, "Run more evaluations to see the trend."),
        h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "When"), h("th", {}, "mIoU"), h("th", {}, "Pixel acc."), h("th", {}, "Images"), h("th", {}, "Memory"))),
          h("tbody", {}, ev.map((e) => h("tr", {}, h("td", {}, fmt.ago(e.ts)), h("td", { class: "num" }, e.miou.toFixed(1)), h("td", { class: "num" }, e.pixacc.toFixed(1)),
            h("td", { class: "num" }, e.n_images), h("td", { class: "num" }, fmt.num(e.memory_size)))))))) : null);
}

function lineChart(pts) {
  const W = 460, H = 180, L = 36, B = 26;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1, y0 = Math.floor(Math.min(...ys) / 10) * 10, y1 = Math.ceil(Math.max(...ys) / 10) * 10 || 10;
  const X = (v) => L + (W - L - 10) * (x1 === x0 ? 0.5 : (v - x0) / (x1 - x0));
  const Y = (v) => H - B - (H - B - 10) * (y1 === y0 ? 0.5 : (v - y0) / (y1 - y0));
  const svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="mIoU against memory size">
    <line x1="${L}" y1="${H - B}" x2="${W - 10}" y2="${H - B}" stroke="currentColor" stroke-opacity=".2"/>
    <text x="${L - 6}" y="${Y(y1) + 4}" text-anchor="end">${y1}</text><text x="${L - 6}" y="${Y(y0) + 4}" text-anchor="end">${y0}</text>
    <text x="${L}" y="${H - 6}">${x0.toLocaleString()}</text><text x="${W - 10}" y="${H - 6}" text-anchor="end">${x1.toLocaleString()} features</text>
    <polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${pts.map((p) => `${X(p[0])},${Y(p[1])}`).join(" ")}"/>
    ${pts.map((p) => `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3.5" fill="var(--accent)"/>`).join("")}</svg>`;
  return h("div", { html: svg });
}

// ====================================================================== classes
async function classesTab(body, p) {
  const name = h("input", { type: "text", placeholder: "new class name" });
  const color = h("input", { type: "color", value: "#888888", "aria-label": "colour" });
  const rows = p.classes.map((c) => {
    const nm = h("input", { type: "text", value: c.name, onchange: async () => { await api.patch(P(p.id) + `/classes/${c.idx}`, { name: nm.value }); toast("Renamed"); } });
    const col = h("input", { type: "color", value: c.color, onchange: async () => { await api.patch(P(p.id) + `/classes/${c.idx}`, { color: col.value }); toast("Colour updated"); } });
    return h("tr", {}, h("td", { class: "num muted" }, c.idx), h("td", {}, col), h("td", {}, nm),
      h("td", {}, h("button", { class: "btn small danger", onclick: async () => {
        const into = h("select", {}, h("option", { value: 0 }, "unlabeled"), p.classes.filter((x) => x.idx !== c.idx).map((x) => h("option", { value: x.idx }, x.name)));
        const ok = await modal(`Delete "${c.name}"`, h("div", { class: "stack" }, h("p", {}, "Pixels of this class in every label and auto-label become:"), into),
          [{ label: "Cancel", value: false }, { label: "Delete class", value: true, cls: "danger" }]);
        if (!ok) return;
        const r = await api.del(P(p.id) + `/classes/${c.idx}?merge_into=${into.value}`);
        toast(`Deleted; ${r.masks_changed} masks updated`); route(); } }, "Delete")));
  });
  body.append(h("div", { class: "card" },
    h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "Index"), h("th", {}, "Colour"), h("th", {}, "Name"), h("th", {}, ""))), h("tbody", {}, rows)),
    h("div", { class: "row", style: { marginTop: "12px" } }, color, name, h("button", { class: "btn primary", onclick: async () => {
      if (!name.value.trim()) return;
      try { await api.post(P(p.id) + "/classes", { name: name.value.trim(), color: color.value }); route(); } catch (e) { toast(e.message, true); } } }, "Add class")),
    h("p", { class: "muted", style: { marginBottom: 0 } }, "Pixel value in exported index masks = class index (0 = unlabeled). Indices never change when classes are renamed.")));
}

// ====================================================================== import / export
async function dataTab(body, p) {
  const exports = await api.get(P(p.id) + "/exports");
  // --- upload
  const fileIn = h("input", { type: "file", multiple: true, accept: "image/*", hidden: true });
  const drop = h("div", { class: "dropzone", tabindex: 0 }, "Drop images here or ", h("a", { href: "#", onclick: (e) => { e.preventDefault(); fileIn.click(); } }, "choose files"));
  const upload = async (files) => {
    const imgs = [...files].filter((f) => f.type.startsWith("image/"));
    for (let k = 0; k < imgs.length; k += 20) {
      const fd = new FormData(); imgs.slice(k, k + 20).forEach((f) => fd.append("files", f));
      drop.textContent = `Uploading ${Math.min(k + 20, imgs.length)} / ${imgs.length}…`;
      await api.post(P(p.id) + "/images", fd);
    }
    toast(`Uploaded ${imgs.length} images`); route();
  };
  fileIn.onchange = () => upload(fileIn.files);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files); };
  // --- folder / video / bag
  const pathIn = h("input", { type: "text", placeholder: "/path/to/folder, video.mp4 or bag folder / .mcap", style: { flex: 1 } });
  const kind = h("select", {}, h("option", { value: "folder" }, "Image folder"), h("option", { value: "video" }, "Video file"), h("option", { value: "bag" }, "ROS 2 bag (.mcap)"));
  const stride = h("input", { type: "number", value: 1, min: 1, style: { width: "80px" }, title: "folder: every Nth file" });
  const every = h("input", { type: "number", value: 1, min: 0.05, step: 0.05, style: { width: "80px" }, title: "video / bag: seconds between frames" });
  const maxN = h("input", { type: "number", value: 0, min: 0, style: { width: "90px" }, title: "0 = no limit" });
  const topicsBox = h("div", { class: "stack" });
  const browser = h("div", { class: "browser", hidden: true });
  async function browse(path) {
    try {
      const r = await api.get(`/api/browse?path=${encodeURIComponent(path)}`);
      browser.hidden = false;
      browser.replaceChildren(h("div", { onclick: () => browse(r.parent) }, "⬆ ..", h("span", { class: "muted" }, r.path)),
        ...r.entries.map((e) => h("div", { onclick: () => { pathIn.value = e.path; if (e.dir) browse(e.path); else browser.hidden = true; } }, e.dir ? "📁" : "📄", e.name)));
      if (r.images) toast(`${r.images} images in ${r.path}`);
    } catch (e) { toast(e.message, true); }
  }
  async function loadTopics() {
    try {
      const t = await api.get(`/api/bag_topics?path=${encodeURIComponent(pathIn.value)}`);
      topicsBox.replaceChildren(...t.map((x) => h("label", { class: "row" }, h("input", { type: "checkbox", value: x.topic, checked: /left|color|rgb/.test(x.topic) }), h("span", { class: "mono" }, x.topic), h("span", { class: "muted" }, `${x.type.split("/").pop()} · ${fmt.num(x.count)} msgs`))));
      if (!t.length) toast("No image topics found in this bag.", true);
    } catch (e) { toast(e.message, true); }
  }
  const importBtn = h("button", { class: "btn primary", onclick: async () => {
    if (!pathIn.value) return;
    const k = kind.value;
    const params = { path: pathIn.value, max: +maxN.value };
    if (k === "folder") params.stride = +stride.value;
    else params.every_s = +every.value;
    if (k === "bag") {
      params.topics = [...topicsBox.querySelectorAll("input:checked")].map((i) => i.value);
      if (!params.topics.length) { toast("List the bag topics and pick at least one.", true); return; }
    }
    const j = await runJob(p.id, "import_" + k, params, "Import");
    if (j.status === "done") toast(`Imported ${j.result.added} images` + (j.result.duplicates ? ` (${j.result.duplicates} duplicates skipped)` : ""));
    route(); } }, "Import");
  // --- masks
  const maskPath = h("input", { type: "text", placeholder: "/path/to/masks (index or colour PNG, matched by file name) or CVAT .zip", style: { flex: 1 } });
  const maskKind = h("select", {}, h("option", { value: "index" }, "Index PNG (pixel = class id)"), h("option", { value: "color" }, "Colour PNG (labelmap.txt or project colours)"));
  const maskStatus = h("select", {}, h("option", { value: "reviewed" }, "as Reviewed"), h("option", { value: "in_progress" }, "as In progress"));
  const maskMem = h("input", { type: "checkbox" });
  // --- export
  const fmts = ["index_png", "color_png", "coco", "cvat_mask"].map((f) => h("label", { class: "row" }, h("input", { type: "checkbox", value: f, checked: f === "index_png" || f === "coco" }),
    { index_png: "Index PNG masks", color_png: "Colour PNG masks", coco: "COCO JSON (RLE per class)", cvat_mask: "CVAT Segmentation mask 1.1" }[f]));
  const include = h("select", {}, h("option", { value: "reviewed" }, "Reviewed images only"), h("option", { value: "labeled" }, "Every image with a saved label"),
    h("option", { value: "all" }, "All: saved labels, else auto-labels"));
  const withImgs = h("input", { type: "checkbox", checked: true });

  body.append(h("div", { class: "grid2" },
    h("div", { class: "card stack" }, h("h2", {}, "Import images"), drop, fileIn,
      h("div", { class: "row" }, kind, pathIn, h("button", { class: "btn", onclick: () => browse(pathIn.value || "~") }, "Browse")), browser,
      h("div", { class: "row" }, h("label", { class: "field" }, h("span", {}, "Every Nth file"), stride), h("label", { class: "field" }, h("span", {}, "Seconds between frames"), every),
        h("label", { class: "field" }, h("span", {}, "Max images"), maxN), h("button", { class: "btn", onclick: loadTopics }, "List bag topics")),
      topicsBox, h("div", {}, importBtn),
      h("p", { class: "muted", style: { margin: 0, fontSize: "12.5px" } }, "Bags: depth topics are hidden; pick the camera image topics to sample. Exact duplicates are skipped.")),
    h("div", { class: "card stack" }, h("h2", {}, "Import existing labels"),
      h("div", { class: "row" }, maskPath), h("div", { class: "row" }, maskKind, maskStatus, h("label", { class: "row" }, maskMem, "add to memory")),
      h("div", {}, h("button", { class: "btn", onclick: async () => {
        if (!maskPath.value) return;
        const j = await runJob(p.id, "import_masks", { path: maskPath.value, kind: maskKind.value, status: maskStatus.value, add_to_memory: maskMem.checked }, "Label import");
        if (j.status === "done") toast(`Imported ${j.result.imported} labels, ${j.result.n_unmatched} unmatched files`);
        route(); } }, "Import labels")),
      h("h2", { style: { marginTop: "10px" } }, "Export"), h("div", { class: "stack" }, fmts), include, h("label", { class: "row" }, withImgs, "include images"),
      h("div", {}, h("button", { class: "btn primary", onclick: async () => {
        const formats = fmts.map((l) => l.querySelector("input")).filter((i) => i.checked).map((i) => i.value);
        const j = await runJob(p.id, "export", { formats, include: include.value, with_images: withImgs.checked }, "Export");
        if (j.status === "done") { toast(`Exported ${j.result.n_images} images`); route(); } } }, "Export zip")),
      exports.length ? h("table", { class: "t" }, h("tbody", {}, exports.map((e) => h("tr", {}, h("td", {}, h("a", { href: P(p.id) + `/exports/${encodeURIComponent(e.name)}` }, e.name)),
        h("td", { class: "num muted" }, fmt.bytes(e.size)), h("td", { class: "muted" }, fmt.ago(e.time)))))) : null)));
}

// ====================================================================== settings
async function settingsTab(body, p) {
  const c = p.config;
  const f = {};
  const num = (k, step, min, max) => (f[k] = h("input", { type: "number", value: c[k], step, min, max }));
  const sel = (k, opts) => (f[k] = h("select", {}, opts.map(([v, l, dis]) => h("option", { value: v, selected: String(c[k]) === String(v), disabled: !!dis }, l))));
  const models = (system?.models || []).map((m) => [m.key, `${m.key} — ${m.hub} (${m.dim}-d)${m.available ? "" : " — weights missing"}`, !m.available]);
  const name = h("input", { type: "text", value: p.name });
  const desc = h("input", { type: "text", value: p.description || "" });
  const row = (label, el, help) => h("label", { class: "field" }, h("span", {}, label), el, help ? h("small", {}, help) : null);
  body.append(h("div", { class: "grid2" },
    h("div", { class: "card stack" }, h("h2", {}, "Backbone"),
      row("DINOv3 model", sel("model", models), "Large (ViT-L/16) is the default; see the benchmark in docs/BENCHMARK.md."),
      row("Input long side (px)", num("image_size", 16, 256, 2048), "Multiple of 16. Larger = finer patches, slower."),
      row("Positional debiasing (fraction of feature dims)", num("svd_frac", 0.05, 0, 0.9), "INSID3 removes the subspace that encodes absolute position."),
      h("p", { class: "muted", style: { margin: 0 } }, "Changing these needs a memory-bank rebuild (Memory bank → Rebuild).")),
    h("div", { class: "card stack" }, h("h2", {}, "Auto-labeling"),
      row("Mode", sel("mode", [["hybrid", "hybrid — kNN votes + INSID3 clusters (default, ~0.2 s)"], ["smooth_insid3", "hybrid + affinity smoothing (best with many references)"],
        ["smooth", "smooth — kNN + affinity smoothing (fast, ~0.02 s)"], ["knn", "knn — per-patch memory vote"], ["cluster", "cluster — votes per DINOv3 cluster"],
        ["insid3", "insid3 — strict, leaves uncertain regions unlabeled"]]), "NYUv2, 10 / 40 references: hybrid 56.5 / 63.8 mIoU, smooth 55.4 / 63.5, knn 55.0 / 62.9."),
      row("Boundary refinement", sel("refine", [["bilinear", "bilinear (default)"], ["superpixel", "superpixel snapping (SLIC) — crisper edges"]])),
      row("Neighbours k", num("knn_k", 1, 1, 100)), row("Neighbour temperature", num("knn_temp", 0.01, 0.005, 1)),
      row("Cluster threshold τ (INSID3)", num("tau", 0.05, 0.2, 0.95)), row("Merge threshold (INSID3)", num("merge_thresh", 0.05, 0, 1)),
      row("Patch purity for the memory bank", num("purity", 0.05, 0.3, 1)), row("Sub-prototypes per class", num("prototypes_per_class", 1, 1, 16)),
      row("Superpixels per image", num("sp_segments", 100, 200, 20000)),
      row("Leave pixels below this confidence unlabeled", num("min_conf", 0.05, 0, 0.99)),
      row("Auto-compress the memory bank to this fraction (0 = off)", num("auto_compress", 0.05, 0, 0.95),
        "After every new reference, keep each class at this fraction of all features ever added (k-center coreset). NYUv2 with 40 references: 0.2 kept accuracy (+0.1 to +1.5 mIoU), 5× smaller, 3–6× faster matching.")),
    h("div", { class: "card stack" }, h("h2", {}, "Project"), row("Name", name), row("Description", desc),
      h("div", { class: "row" },
        h("button", { class: "btn primary", onclick: async () => {
          const cfg = {};
          for (const [k, el] of Object.entries(f)) cfg[k] = el.tagName === "SELECT" ? el.value : +el.value;
          try {
            const r = await api.patch(P(p.id), { name: name.value, description: desc.value, config: cfg });
            toast("Saved"); if (r.feature_key !== p.feature_key) toast("Backbone changed: rebuild the memory bank from the Memory bank tab.");
            route();
          } catch (e) { toast(e.message, true); } } }, "Save settings"),
        h("button", { class: "btn danger", onclick: async () => {
          if (!(await confirmBox("Delete project", `Delete "${p.name}" with all images, labels and memory banks? This cannot be undone.`, "Delete project", true))) return;
          await api.del(P(p.id)); location.hash = "#/"; } }, "Delete project")))));
}

// ====================================================================== jobs drawer / chip
const chip = document.getElementById("jobchip");
const drawer = document.getElementById("jobs-drawer");
chip.onclick = () => { drawer.hidden = !drawer.hidden; if (!drawer.hidden) renderJobs(); };
drawer.querySelector("[data-close]").onclick = () => { drawer.hidden = true; };
let pollTimer = null;
async function renderJobs() {
  const jobs = await api.get("/api/jobs?limit=30");
  document.getElementById("jobs-list").replaceChildren(...(jobs.length ? jobs.map((j) => h("div", { class: "job" },
    h("div", { class: "row" }, h("b", {}, j.kind.replace(/_/g, " ")), h("span", { class: "badge " + (j.status === "done" ? "reviewed" : j.status === "running" ? "in_progress" : j.status === "failed" ? "predicted" : "") }, j.status)),
    j.total ? h("div", { class: "bar", style: { margin: "6px 0" } }, h("i", { style: { width: (100 * j.progress / j.total) + "%" } })) : null,
    h("div", { class: "muted", style: { fontSize: "12px" } }, `project ${j.project_id} · ${j.progress}/${j.total || "?"} · ${fmt.ago(j.created)} ${j.message || ""}`),
    ["queued", "running"].includes(j.status) ? h("button", { class: "btn small", onclick: async () => { await api.post(`/api/jobs/${j.id}/cancel`); renderJobs(); } }, "Cancel") : null)) : [h("p", { class: "muted" }, "No jobs yet.")]));
}
async function pollJobs() {
  clearTimeout(pollTimer);
  try {
    const jobs = await api.get("/api/jobs?limit=10");
    const active = jobs.filter((j) => ["queued", "running"].includes(j.status));
    chip.hidden = !active.length && drawer.hidden;
    const run = active.find((j) => j.status === "running");
    document.getElementById("jobtext").textContent = run ? `${run.kind.replace(/_/g, " ")} ${run.total ? Math.round(100 * run.progress / run.total) + "%" : "…"}` + (active.length > 1 ? ` (+${active.length - 1})` : "") : active.length ? `${active.length} queued` : "idle";
    if (!drawer.hidden) renderJobs();
    pollTimer = setTimeout(pollJobs, active.length ? 1000 : 5000);
  } catch (e) { pollTimer = setTimeout(pollJobs, 5000); }
}

(async () => {
  try {
    system = await api.get("/api/system");
    if (system.gpu) document.getElementById("gpu").textContent = `${system.gpu.name} · ${fmt.num(system.gpu.free_mb)} MB free`;
  } catch (e) { /* offline */ }
  pollJobs();
  route();
})();
