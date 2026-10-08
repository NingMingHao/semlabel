"""Background jobs: one GPU worker thread, progress and cancellation stored in SQLite."""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback

from ..store import io as sio


class JobRunner:
    def __init__(self, ws):
        self.ws = ws
        self.q: queue.Queue = queue.Queue()
        self.cancel: set[int] = set()
        self.current: int | None = None
        # jobs left running by a previous server process can never finish
        ws.db.x("UPDATE jobs SET status='failed', message='server restarted' WHERE status IN ('queued','running')")
        self.t = threading.Thread(target=self._loop, daemon=True, name="semlabel-jobs")
        self.t.start()

    def submit(self, project_id: int, kind: str, params: dict) -> int:
        if kind not in HANDLERS:
            raise ValueError(f"unknown job kind {kind}")
        jid = self.ws.db.x("INSERT INTO jobs(project_id, kind, status, params, created) VALUES (?,?,?,?,?)",
                           (project_id, kind, "queued", json.dumps(params), time.time()))
        self.q.put(jid)
        return jid

    def request_cancel(self, jid: int):
        self.cancel.add(jid)
        self.ws.db.x("UPDATE jobs SET status='cancelled', finished=? WHERE id=? AND status='queued'", (time.time(), jid))

    def _loop(self):
        while True:
            jid = self.q.get()
            job = self.ws.db.one("SELECT * FROM jobs WHERE id=?", (jid,))
            if job is None or job["status"] != "queued":
                continue
            self.current = jid
            db = self.ws.db
            db.x("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), jid))
            last = [0.0]

            def progress(done, total, msg=""):
                now = time.time()
                if now - last[0] > 0.3 or done == total:
                    last[0] = now
                    db.x("UPDATE jobs SET progress=?, total=?, message=? WHERE id=?", (int(done), int(total), msg, jid))

            try:
                p = self.ws.project(job["project_id"])
                res = HANDLERS[job["kind"]](p, json.loads(job["params"]), progress, lambda: jid in self.cancel)
                status = "cancelled" if jid in self.cancel else "done"
                db.x("UPDATE jobs SET status=?, result=?, finished=? WHERE id=?",
                     (status, json.dumps(res, default=str), time.time(), jid))
            except Exception as e:  # noqa: BLE001
                db.x("UPDATE jobs SET status='failed', message=?, result=?, finished=? WHERE id=?",
                     (f"{type(e).__name__}: {e}", json.dumps(dict(trace=traceback.format_exc()[-3000:])), time.time(), jid))
            finally:
                self.current = None
                self.cancel.discard(jid)


def _ids(p, params):
    if params.get("ids"):
        return [int(i) for i in params["ids"]]
    status = params.get("status") or "new"
    return [r["id"] for r in p.list_images(status=status, limit=10 ** 7)["items"]]


def job_predict(p, params, progress, cancelled):
    ids = _ids(p, params)
    done, failed, confs = 0, [], []
    t0 = time.time()
    for n, iid in enumerate(ids):
        if cancelled():
            break
        try:
            pr = p.predict(iid, save=True, mode=params.get("mode"))
            confs.append(pr.mean_conf)
            done += 1
        except Exception as e:  # noqa: BLE001
            failed.append(dict(id=iid, error=str(e)))
            if "memory bank is empty" in str(e):
                raise
        progress(n + 1, len(ids), f"{(time.time() - t0) / (n + 1):.2f} s/image")
    return dict(predicted=done, failed=failed, mean_conf=float(sum(confs) / len(confs)) if confs else None,
                seconds=round(time.time() - t0, 1))


def job_prelabel(p, params, progress, cancelled):
    ids = _ids(p, params)
    for n, iid in enumerate(ids):
        if cancelled():
            break
        p.prelabel_segformer(iid, save=True)
        progress(n + 1, len(ids))
    return dict(prelabeled=len(ids))


def job_memory_add(p, params, progress, cancelled):
    return p.memory_add([int(i) for i in params["ids"]], progress)


def job_memory_rebuild(p, params, progress, cancelled):
    return p.memory_rebuild(progress)


def job_memory_compress(p, params, progress, cancelled):
    rec = p.memory_compress(params.get("ratio"), params.get("max_per_class"))
    return {k: v for k, v in rec.items() if k != "per_class"} | {"per_class": {str(k): v for k, v in rec["per_class"].items()}}


def job_suggest(p, params, progress, cancelled):
    return p.suggest(int(params.get("n", 10)), params.get("status", "new"), progress)


def job_evaluate(p, params, progress, cancelled):
    return p.evaluate(params.get("ids"), progress)


def job_import_folder(p, params, progress, cancelled):
    return sio.import_folder(p, params["path"], params.get("recursive", True), int(params.get("stride", 1)),
                             int(params.get("max", 0)), progress, cancelled)


def job_import_video(p, params, progress, cancelled):
    return sio.import_video(p, params["path"], float(params.get("every_s", 1.0)), int(params.get("max", 0)),
                            progress, cancelled)


def job_import_bag(p, params, progress, cancelled):
    return sio.import_bag(p, params["path"], params["topics"], float(params.get("every_s", 1.0)),
                          int(params.get("max", 0)), progress, cancelled)


def job_import_masks(p, params, progress, cancelled):
    if params.get("path", "").endswith(".zip"):
        return sio.import_cvat_zip(p, params["path"], status=params.get("status", "reviewed"),
                                   add_to_memory=params.get("add_to_memory", False), progress=progress, cancelled=cancelled)
    return sio.import_masks(p, params["path"], params.get("kind", "index"), params.get("class_names"),
                            params.get("status", "reviewed"), params.get("add_to_memory", False),
                            progress=progress, cancelled=cancelled)


def job_export(p, params, progress, cancelled):
    return sio.export_project(p, params.get("formats", ["index_png", "coco"]), params.get("include", "reviewed"),
                              params.get("with_images", True), progress, cancelled)


HANDLERS = {
    "predict": job_predict, "prelabel": job_prelabel, "memory_add": job_memory_add,
    "memory_rebuild": job_memory_rebuild, "suggest": job_suggest, "memory_compress": job_memory_compress, "evaluate": job_evaluate,
    "import_folder": job_import_folder, "import_video": job_import_video, "import_bag": job_import_bag,
    "import_masks": job_import_masks, "export": job_export,
}
