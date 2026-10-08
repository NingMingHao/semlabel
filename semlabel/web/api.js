// Thin API client. Masks are raw bytes: one uint8 class index per pixel, row-major.
async function req(method, url, body, raw = false) {
  const opt = { method, headers: {} };
  if (body instanceof Uint8Array || body instanceof ArrayBuffer) {
    opt.body = body; opt.headers["Content-Type"] = "application/octet-stream";
  } else if (body instanceof FormData) {
    opt.body = body;
  } else if (body !== undefined) {
    opt.body = JSON.stringify(body); opt.headers["Content-Type"] = "application/json";
  }
  const r = await fetch(url, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  if (raw) return r;
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r;
}

export const api = {
  get: (u) => req("GET", u),
  post: (u, b = {}) => req("POST", u, b),
  patch: (u, b) => req("PATCH", u, b),
  put: (u, b) => req("PUT", u, b),
  del: (u) => req("DELETE", u),
  async bytes(method, url, body) {
    const r = await req(method, url, body, true);
    const buf = new Uint8Array(await r.arrayBuffer());
    const h = (k) => r.headers.get(k);
    return { data: buf, width: +h("X-Width"), height: +h("X-Height"), empty: !!h("X-Empty"),
             count: h("X-Count") ? +h("X-Count") : null, score: h("X-Score") ? +h("X-Score") : null };
  },
};

export const P = (pid) => `/api/projects/${pid}`;
export const I = (pid, iid) => `/api/projects/${pid}/images/${iid}`;

export async function waitJob(id, onProgress) {
  for (;;) {
    const j = await api.get(`/api/jobs/${id}`);
    if (onProgress) onProgress(j);
    if (["done", "failed", "cancelled"].includes(j.status)) return j;
    await new Promise((r) => setTimeout(r, 600));
  }
}
