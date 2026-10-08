"""HTTP API + static web UI.

Masks travel as raw bytes (one uint8 per pixel, row-major, image width x height) so the browser editor
can paint directly into a Uint8Array; responses are gzip-compressed.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import DATA_DIR, MODELS, WEIGHTS_DIR, EngineConfig
from ..store import io as sio
from ..store.files import color_mask
from ..store.project import Workspace
from .jobs import JobRunner

WEB = Path(__file__).resolve().parents[1] / "web"


def create_app(data_dir: Path | None = None) -> FastAPI:
    ws = Workspace(Path(data_dir or DATA_DIR))
    jobs = JobRunner(ws)
    app = FastAPI(title="SemLabel", version="1.0")
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.state.ws, app.state.jobs = ws, jobs

    @app.exception_handler(KeyError)
    async def _nf(_, exc):
        return JSONResponse({"detail": str(exc).strip("'")}, status_code=404)

    @app.exception_handler(ValueError)
    async def _bad(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    P = ws.project

    # ------------------------------------------------------------------ system
    @app.get("/api/system")
    def system():
        gpu = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu = dict(name=torch.cuda.get_device_name(0), free_mb=free // 2 ** 20, total_mb=total // 2 ** 20)
        models = [dict(key=k, hub=v[0], dim=v[2], available=(WEIGHTS_DIR / v[1]).exists()) for k, v in MODELS.items()]
        return dict(gpu=gpu, models=models, data_dir=str(ws.data_dir), defaults=EngineConfig().to_dict(),
                    running_job=jobs.current)

    # ------------------------------------------------------------------ projects
    @app.get("/api/projects")
    def projects():
        return ws.projects()

    @app.post("/api/projects")
    def create_project(body: dict = Body(...)):
        p = ws.create_project(body["name"], body.get("template", "indoor"), body.get("classes"), body.get("config"),
                              body.get("description", ""))
        return p.info()

    @app.get("/api/projects/{pid}")
    def project(pid: int):
        return P(pid).info()

    @app.patch("/api/projects/{pid}")
    def update_project(pid: int, body: dict = Body(...)):
        p = P(pid)
        if "name" in body or "description" in body:
            r = p.row()
            ws.db.x("UPDATE projects SET name=?, description=?, updated=? WHERE id=?",
                    (body.get("name", r["name"]), body.get("description", r["description"]), time.time(), pid))
        if "config" in body:
            p.set_config(body["config"])
        return p.info()

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: int):
        ws.delete_project(pid)
        return dict(ok=True)

    @app.get("/api/projects/{pid}/events")
    def events(pid: int, limit: int = 100):
        return ws.db.q("SELECT * FROM events WHERE project_id=? ORDER BY id DESC LIMIT ?", (pid, limit))

    # ------------------------------------------------------------------ classes
    @app.get("/api/projects/{pid}/classes")
    def classes(pid: int):
        return P(pid).classes()

    @app.post("/api/projects/{pid}/classes")
    def add_class(pid: int, body: dict = Body(...)):
        return P(pid).add_class(body["name"], body.get("color"))

    @app.patch("/api/projects/{pid}/classes/{idx}")
    def update_class(pid: int, idx: int, body: dict = Body(...)):
        P(pid).update_class(idx, body.get("name"), body.get("color"))
        return P(pid).classes()

    @app.delete("/api/projects/{pid}/classes/{idx}")
    def delete_class(pid: int, idx: int, merge_into: int = 0):
        n = P(pid).delete_class(idx, merge_into)
        return dict(masks_changed=n, classes=P(pid).classes())

    # ------------------------------------------------------------------ images
    @app.get("/api/projects/{pid}/images")
    def images(pid: int, status: str | None = None, q: str | None = None, sort: str = "id", offset: int = 0,
               limit: int = 200, in_memory: int | None = None, ids: str | None = None):
        id_list = [int(i) for i in ids.split(",") if i] if ids else None
        return P(pid).list_images(status, q, sort, offset, limit, in_memory, id_list)

    @app.post("/api/projects/{pid}/images")
    async def upload(pid: int, files: list[UploadFile] = File(...)):
        p = P(pid)
        added, dup = [], 0
        with tempfile.TemporaryDirectory() as td:
            for f in files:
                tmp = Path(td) / Path(f.filename).name
                with open(tmp, "wb") as out:
                    shutil.copyfileobj(f.file, out)
                iid = p.add_image_file(tmp, name=Path(f.filename).name, source="upload", move=True)
                if iid is None:
                    dup += 1
                else:
                    added.append(iid)
        ws.db.event(pid, "upload", added=len(added), duplicates=dup)
        return dict(added=added, duplicates=dup)

    @app.get("/api/projects/{pid}/images/{iid}")
    def image(pid: int, iid: int, status: str | None = None, sort: str = "id"):
        p = P(pid)
        r = p.image(iid)
        r["has_label"] = (p.dir / "labels" / f"{iid}.png").exists()
        r["has_pred"] = (p.dir / "preds" / f"{iid}.png").exists()
        r["nav"] = p.neighbours(iid, status, sort)
        return r

    @app.patch("/api/projects/{pid}/images/{iid}")
    def update_image(pid: int, iid: int, body: dict = Body(...)):
        P(pid).update_image(iid, **body)
        return P(pid).image(iid)

    @app.delete("/api/projects/{pid}/images/{iid}")
    def delete_image(pid: int, iid: int):
        P(pid).delete_image(iid)
        return dict(ok=True)

    @app.post("/api/projects/{pid}/images/delete")
    def delete_images(pid: int, body: dict = Body(...)):
        for i in body["ids"]:
            P(pid).delete_image(int(i))
        return dict(deleted=len(body["ids"]))

    @app.get("/api/projects/{pid}/images/{iid}/file")
    def image_file(pid: int, iid: int):
        return FileResponse(P(pid).image_path(iid), headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/projects/{pid}/images/{iid}/thumb")
    def image_thumb(pid: int, iid: int):
        f = P(pid).dir / "thumbs" / f"{iid}.jpg"
        if not f.exists():
            raise HTTPException(404)
        return FileResponse(f, headers={"Cache-Control": "max-age=3600"})

    def raw(a: np.ndarray, extra: dict | None = None) -> Response:
        h = {"X-Width": str(a.shape[1]), "X-Height": str(a.shape[0]), "Cache-Control": "no-store"}
        if extra:
            h.update({k: str(v) for k, v in extra.items()})
        return Response(np.ascontiguousarray(a).tobytes(), media_type="application/octet-stream", headers=h)

    # ------------------------------------------------------------------ labels / predictions
    @app.get("/api/projects/{pid}/images/{iid}/label")
    def get_label(pid: int, iid: int):
        p = P(pid)
        m = p.label(iid)
        if m is None:
            r = p.image(iid)
            m = np.zeros((r["height"], r["width"]), np.uint8)
            return raw(m, {"X-Empty": 1})
        return raw(m)

    @app.put("/api/projects/{pid}/images/{iid}/label")
    async def put_label(pid: int, iid: int, request: Request, status: str = "in_progress",
                        add_to_memory: int | None = None):
        p = P(pid)
        r = p.image(iid)
        data = await request.body()
        if len(data) != r["width"] * r["height"]:
            raise HTTPException(400, f"expected {r['width'] * r['height']} bytes, got {len(data)}")
        m = np.frombuffer(data, np.uint8).reshape(r["height"], r["width"])
        p.save_label(iid, m, status, None if add_to_memory is None else bool(add_to_memory))
        return p.image(iid)

    @app.delete("/api/projects/{pid}/images/{iid}/label")
    def delete_label(pid: int, iid: int):
        P(pid).clear_label(iid)
        return P(pid).image(iid)

    @app.get("/api/projects/{pid}/images/{iid}/label.png")
    def label_png(pid: int, iid: int, color: int = 1):
        p = P(pid)
        m = p.label(iid)
        if m is None:
            raise HTTPException(404)
        from io import BytesIO
        from PIL import Image
        buf = BytesIO()
        Image.fromarray(color_mask(m, p.classes()) if color else m).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.get("/api/projects/{pid}/images/{iid}/pred")
    def get_pred(pid: int, iid: int):
        m, _ = P(pid).pred(iid)
        if m is None:
            raise HTTPException(404, "no prediction")
        return raw(m)

    @app.get("/api/projects/{pid}/images/{iid}/conf")
    def get_conf(pid: int, iid: int):
        _, c = P(pid).pred(iid)
        if c is None:
            raise HTTPException(404, "no prediction")
        return raw(c)

    @app.post("/api/projects/{pid}/images/{iid}/predict")
    def predict(pid: int, iid: int, body: dict = Body(default={})):
        p = P(pid)
        t0 = time.time()
        pr = p.predict(iid, save=True, mode=body.get("mode"))
        counts = np.bincount(pr.labels.ravel(), minlength=256)
        return dict(conf=pr.mean_conf, seconds=round(time.time() - t0, 3), timing=pr.timing,
                    classes={int(i): int(c) for i, c in enumerate(counts) if c}, image=p.image(iid))

    @app.post("/api/projects/{pid}/images/{iid}/accept")
    def accept(pid: int, iid: int, body: dict = Body(default={})):
        p = P(pid)
        p.accept_prediction(iid, body.get("status", "reviewed"))
        if body.get("add_to_memory"):
            p.memory_add([iid])
        return p.image(iid)

    @app.post("/api/projects/{pid}/images/{iid}/prelabel")
    def prelabel(pid: int, iid: int):
        p = P(pid)
        lab = p.prelabel_segformer(iid, save=True)
        return dict(labeled=float((lab > 0).mean()), image=p.image(iid))

    # ------------------------------------------------------------------ tools
    @app.post("/api/projects/{pid}/images/{iid}/wand")
    def wand(pid: int, iid: int, body: dict = Body(...)):
        m = P(pid).wand(iid, float(body["x"]), float(body["y"]), float(body.get("thresh", 0.6)),
                        body.get("scope", "connected"), bool(body.get("snap", True)))
        return raw(m, {"X-Count": int(m.sum())})

    @app.get("/api/projects/{pid}/images/{iid}/superpixels")
    def get_superpixels(pid: int, iid: int, n: int | None = None):
        p = P(pid)
        sp = p.superpixels(iid)
        if n and n != p.config.sp_segments:
            from ..engine.segmenter import superpixels
            sp = superpixels(p.rgb(iid), n)
        return raw(sp.astype(np.uint16).view(np.uint8).reshape(sp.shape[0], -1), {"X-Width": sp.shape[1], "X-Bytes": 2})

    @app.post("/api/projects/{pid}/images/{iid}/sam")
    def sam(pid: int, iid: int, body: dict = Body(...)):
        from ..engine import sam as samlib
        if not samlib.available():
            raise HTTPException(501, samlib.unavailable_reason())
        p = P(pid)
        m, score = samlib.segment(iid, p.rgb(iid), body.get("points", []), body.get("box"), p.slug)
        return raw(m, {"X-Count": int(m.sum()), "X-Score": round(score, 3)})

    # ------------------------------------------------------------------ memory bank
    @app.get("/api/projects/{pid}/memory")
    def memory(pid: int):
        return P(pid).memory_stats()

    @app.post("/api/projects/{pid}/memory/add")
    def memory_add(pid: int, body: dict = Body(...)):
        ids = [int(i) for i in body["ids"]]
        if len(ids) > 8:
            return dict(job=jobs.submit(pid, "memory_add", dict(ids=ids)))
        return P(pid).memory_add(ids)

    @app.post("/api/projects/{pid}/memory/remove")
    def memory_remove(pid: int, body: dict = Body(...)):
        return P(pid).memory_remove([int(i) for i in body["ids"]])

    @app.post("/api/projects/{pid}/memory/compress")
    def memory_compress(pid: int, body: dict = Body(...)):
        ratio = body.get("ratio")
        mpc = body.get("max_per_class")
        return dict(job=jobs.submit(pid, "memory_compress", dict(ratio=float(ratio) if ratio else None,
                                                                   max_per_class=int(mpc) if mpc else None)))

    @app.post("/api/projects/{pid}/memory/rebuild")
    def memory_rebuild(pid: int):
        return dict(job=jobs.submit(pid, "memory_rebuild", {}))

    @app.post("/api/projects/{pid}/memory/clear")
    def memory_clear(pid: int):
        P(pid).memory_clear()
        return P(pid).memory_stats()

    # ------------------------------------------------------------------ jobs
    @app.post("/api/projects/{pid}/jobs")
    def submit_job(pid: int, body: dict = Body(...)):
        P(pid)
        return dict(job=jobs.submit(pid, body["kind"], body.get("params", {})))

    @app.get("/api/jobs")
    def list_jobs(project: int | None = None, limit: int = 30):
        if project is None:
            rows = ws.db.q("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = ws.db.q("SELECT * FROM jobs WHERE project_id=? ORDER BY id DESC LIMIT ?", (project, limit))
        for r in rows:
            r["params"] = json.loads(r["params"] or "{}")
            r["result"] = json.loads(r["result"] or "{}")
        return rows

    @app.get("/api/jobs/{jid}")
    def get_job(jid: int):
        r = ws.db.one("SELECT * FROM jobs WHERE id=?", (jid,))
        if r is None:
            raise HTTPException(404)
        r["params"] = json.loads(r["params"] or "{}")
        r["result"] = json.loads(r["result"] or "{}")
        return r

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: int):
        jobs.request_cancel(jid)
        return dict(ok=True)

    # ------------------------------------------------------------------ evaluation / export / bag topics
    @app.get("/api/projects/{pid}/evals")
    def evals(pid: int):
        rows = ws.db.q("SELECT * FROM evals WHERE project_id=? ORDER BY ts DESC LIMIT 50", (pid,))
        for r in rows:
            r["per_class"] = json.loads(r["per_class"] or "{}")
            r["config"] = json.loads(r["config"] or "{}")
        return rows

    @app.get("/api/projects/{pid}/exports")
    def exports(pid: int):
        d = P(pid).dir / "exports"
        return [dict(name=f.name, size=f.stat().st_size, time=f.stat().st_mtime)
                for f in sorted(d.glob("*.zip"), key=lambda f: -f.stat().st_mtime)]

    @app.get("/api/projects/{pid}/exports/{name}")
    def export_file(pid: int, name: str):
        f = P(pid).dir / "exports" / Path(name).name
        if not f.exists():
            raise HTTPException(404)
        return FileResponse(f, filename=f.name, media_type="application/zip")

    @app.get("/api/bag_topics")
    def bag_topics(path: str):
        return sio.bag_image_topics(path)

    @app.get("/api/browse")
    def browse(path: str = "~"):
        d = Path(os.path.expanduser(path))
        if not d.is_dir():
            raise HTTPException(400, "not a folder")
        out = []
        for f in sorted(d.iterdir(), key=lambda f: (not f.is_dir(), f.name.lower()))[:500]:
            if f.name.startswith("."):
                continue
            out.append(dict(name=f.name, dir=f.is_dir(), path=str(f)))
        n_img = sum(1 for f in d.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
        return dict(path=str(d), parent=str(d.parent), entries=out, images=n_img)

    # ------------------------------------------------------------------ web UI
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def main():
    """Kept for `python -m semlabel.server.app`; prefer the `semlabel serve` command."""
    from ..cli import main as cli_main
    cli_main(["serve"] + __import__("sys").argv[1:])


if __name__ == "__main__":
    main()
