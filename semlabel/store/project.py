"""A labeling project: classes, images, labels, predictions, memory bank and engine glue."""
from __future__ import annotations

import collections
import json
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..config import INDOOR_CLASSES, EngineConfig, MODELS
from ..engine.backbone import Backbone, Features, get_backbone
from ..engine.memory import MemoryBank, patch_labels
from ..engine.segmenter import Prediction, Segmenter, superpixels
from .db import DB
from .files import (color_mask, load_mask, load_rgb, make_thumb, rgb_to_hex, save_mask, sha1_file)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
STATUSES = ("new", "predicted", "in_progress", "reviewed")
# Kelly-style maximally distinct colours for classes created without one
DISTINCT = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
            "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
            "#000075", "#808080"]


def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", s.strip().lower()).strip("-")
    return s or "project"


class LRU:
    def __init__(self, n: int):
        self.n, self.d, self.lock = n, collections.OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        return None

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.n:
                self.d.popitem(last=False)

    def drop(self, pred):
        with self.lock:
            for k in [k for k in self.d if pred(k)]:
                del self.d[k]


class Project:
    def __init__(self, ws, row: dict):
        self.ws = ws
        self.db: DB = ws.db
        self.id = row["id"]
        self.slug = row["slug"]
        self.dir: Path = ws.projects_dir / self.slug
        for sub in ("images", "thumbs", "labels", "preds", "memory", "exports"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self._mem: MemoryBank | None = None
        self._mem_key = None
        self._feat_cache = LRU(24)
        self._sp_cache = LRU(48)
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ meta
    def row(self) -> dict:
        return self.db.one("SELECT * FROM projects WHERE id=?", (self.id,))

    @property
    def config(self) -> EngineConfig:
        return EngineConfig.from_dict(json.loads(self.row()["config"] or "{}"))

    def set_config(self, updates: dict) -> EngineConfig:
        cfg = EngineConfig.from_dict({**self.config.to_dict(), **updates})
        if cfg.model not in MODELS:
            raise ValueError(f"unknown model {cfg.model}")
        cfg.image_size = int(round(cfg.image_size / 16)) * 16
        old_key = self.feature_key()
        self.db.x("UPDATE projects SET config=?, updated=? WHERE id=?", (json.dumps(cfg.to_dict()), time.time(), self.id))
        if self.feature_key() != old_key:
            self._feat_cache.drop(lambda k: True)
            self._mem = None
        return cfg

    def feature_key(self, cfg: EngineConfig | None = None) -> str:
        cfg = cfg or self.config
        return f"{cfg.model}_{cfg.image_size}_svd{cfg.svd_components()}"

    def info(self) -> dict:
        r = self.row()
        counts = {s: 0 for s in STATUSES}
        for x in self.db.q("SELECT status, COUNT(*) n FROM images WHERE project_id=? GROUP BY status", (self.id,)):
            counts[x["status"]] = x["n"]
        refs = self.db.one("SELECT COUNT(*) n FROM images WHERE project_id=? AND in_memory=1", (self.id,))["n"]
        last_eval = self.db.one("SELECT * FROM evals WHERE project_id=? ORDER BY ts DESC LIMIT 1", (self.id,))
        return dict(id=self.id, slug=self.slug, name=r["name"], description=r["description"], created=r["created"],
                    config=self.config.to_dict(), feature_key=self.feature_key(), classes=self.classes(),
                    counts=counts, total=sum(counts.values()), references=refs,
                    memory=self.memory_stats(light=True), last_eval=last_eval)

    # ------------------------------------------------------------------ classes
    def classes(self) -> list[dict]:
        return self.db.q("SELECT idx, name, color FROM classes WHERE project_id=? ORDER BY idx", (self.id,))

    def n_classes(self) -> int:
        r = self.db.one("SELECT MAX(idx) m FROM classes WHERE project_id=?", (self.id,))
        return int(r["m"] or 0)

    def add_class(self, name: str, color: str | None = None) -> dict:
        used = {c["idx"] for c in self.classes()}
        if any(c["name"].lower() == name.strip().lower() for c in self.classes()):
            raise ValueError(f"class '{name}' already exists")
        idx = next(i for i in range(1, 256) if i not in used)
        if color is None:
            color = DISTINCT[(idx - 1) % len(DISTINCT)] if idx <= len(DISTINCT) else \
                rgb_to_hex(np.random.RandomState(idx * 7919).randint(40, 230, 3))
        self.db.x("INSERT INTO classes(project_id, idx, name, color, created) VALUES (?,?,?,?,?)",
                  (self.id, idx, name.strip(), color, time.time()))
        if self._mem is not None:
            self._mem.set_classes(self.n_classes())
        self.db.event(self.id, "class_add", name=name, idx=idx)
        return dict(idx=idx, name=name.strip(), color=color)

    def update_class(self, idx: int, name: str | None = None, color: str | None = None):
        if name is not None:
            self.db.x("UPDATE classes SET name=? WHERE project_id=? AND idx=?", (name.strip(), self.id, idx))
        if color is not None:
            self.db.x("UPDATE classes SET color=? WHERE project_id=? AND idx=?", (color, self.id, idx))

    def delete_class(self, idx: int, merge_into: int = 0) -> int:
        """Remove a class: its pixels become `merge_into` (0 = unlabeled) in every label and prediction."""
        n = 0
        classes = [c for c in self.classes() if c["idx"] != idx]
        for sub in ("labels", "preds"):
            for p in (self.dir / sub).glob("*.png"):
                if p.name.endswith("_conf.png"):
                    continue
                m = load_mask(p)
                if m is not None and (m == idx).any():
                    m[m == idx] = merge_into
                    save_mask(p, m, classes)
                    n += 1
        self.db.x("DELETE FROM classes WHERE project_id=? AND idx=?", (self.id, idx))
        mem = self.memory()
        with mem.lock:
            if merge_into:
                mem.labels[mem.labels == idx] = merge_into
                mem._dirty()
            else:
                mem.drop_class(idx)
        self.save_memory()
        self.db.event(self.id, "class_delete", idx=idx, merge_into=merge_into, masks=n)
        return n

    # ------------------------------------------------------------------ images
    def image(self, iid: int) -> dict:
        r = self.db.one("SELECT * FROM images WHERE id=? AND project_id=?", (iid, self.id))
        if r is None:
            raise KeyError(f"image {iid} not found")
        return r

    def image_path(self, iid: int) -> Path:
        return self.dir / "images" / self.image(iid)["file"]

    def add_image_file(self, src: Path, name: str | None = None, source: str = "", move: bool = False,
                       dedupe: bool = True) -> int | None:
        src = Path(src)
        sha = sha1_file(src)
        if dedupe and self.db.one("SELECT id FROM images WHERE project_id=? AND sha1=?", (self.id, sha)):
            return None
        from PIL import Image
        with Image.open(src) as im:
            w, h = im.size
        now = time.time()
        name = name or src.name
        iid = self.db.x("INSERT INTO images(project_id, name, file, width, height, status, source, added, updated, sha1)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?)", (self.id, name, "", w, h, "new", source, now, now, sha))
        ext = src.suffix.lower() if src.suffix.lower() in IMG_EXT else ".png"
        fname = f"{iid}{ext}"
        dst = self.dir / "images" / fname
        (shutil.move if move else shutil.copy2)(src, dst)
        self.db.x("UPDATE images SET file=? WHERE id=?", (fname, iid))
        make_thumb(dst, self.dir / "thumbs" / f"{iid}.jpg")
        return iid

    def add_image_array(self, rgb: np.ndarray, name: str, source: str = "") -> int | None:
        from PIL import Image
        tmp = self.dir / "exports" / f"_import_{threading.get_ident()}.jpg"
        Image.fromarray(rgb).save(tmp, quality=95)
        try:
            return self.add_image_file(tmp, name=name, source=source, move=True)
        finally:
            tmp.unlink(missing_ok=True)

    def list_images(self, status: str | None = None, q: str | None = None, sort: str = "id", offset: int = 0,
                    limit: int = 200, in_memory: int | None = None, ids: list[int] | None = None) -> dict:
        where, args = ["project_id=?"], [self.id]
        if ids:
            where.append("id IN (%s)" % ",".join("?" * len(ids)))
            args += [int(i) for i in ids]
        if status:
            st = [s for s in status.split(",") if s]
            where.append("status IN (%s)" % ",".join("?" * len(st)))
            args += st
        if q:
            where.append("(name LIKE ? OR source LIKE ? OR notes LIKE ?)")
            args += [f"%{q}%"] * 3
        if in_memory is not None:
            where.append("in_memory=?")
            args.append(int(in_memory))
        order = {"id": "id", "-id": "id DESC", "name": "name", "conf": "conf IS NULL, conf", "-conf": "conf DESC",
                 "updated": "updated DESC", "labeled": "labeled_frac"}.get(sort, "id")
        w = " AND ".join(where)
        total = self.db.one(f"SELECT COUNT(*) n FROM images WHERE {w}", args)["n"]
        rows = self.db.q(f"SELECT * FROM images WHERE {w} ORDER BY {order} LIMIT ? OFFSET ?", args + [limit, offset])
        return dict(total=total, items=rows)

    def neighbours(self, iid: int, status: str | None = None, sort: str = "id") -> dict:
        ids = [r["id"] for r in self.list_images(status=status, sort=sort, limit=100000)["items"]]
        if iid not in ids:
            return dict(prev=None, next=ids[0] if ids else None, pos=None, total=len(ids))
        i = ids.index(iid)
        return dict(prev=ids[i - 1] if i > 0 else None, next=ids[i + 1] if i + 1 < len(ids) else None,
                    pos=i + 1, total=len(ids))

    def update_image(self, iid: int, **fields):
        allowed = {k: v for k, v in fields.items() if k in ("status", "split", "notes", "name")}
        if not allowed:
            return
        sets = ", ".join(f"{k}=?" for k in allowed)
        self.db.x(f"UPDATE images SET {sets}, updated=? WHERE id=?", list(allowed.values()) + [time.time(), iid])

    def delete_image(self, iid: int):
        r = self.image(iid)
        if r["in_memory"]:
            self.memory().remove(iid)
            self.save_memory()
        for p in (self.dir / "images" / r["file"], self.dir / "thumbs" / f"{iid}.jpg", self.dir / "labels" / f"{iid}.png",
                  self.dir / "preds" / f"{iid}.png", self.dir / "preds" / f"{iid}_conf.png"):
            p.unlink(missing_ok=True)
        self.db.x("DELETE FROM images WHERE id=?", (iid,))
        self._feat_cache.drop(lambda k: k[0] == iid)
        self._sp_cache.drop(lambda k: k[0] == iid)

    def rgb(self, iid: int) -> np.ndarray:
        return load_rgb(self.image_path(iid))

    # ------------------------------------------------------------------ labels
    def label(self, iid: int) -> np.ndarray | None:
        return load_mask(self.dir / "labels" / f"{iid}.png")

    def pred(self, iid: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        return load_mask(self.dir / "preds" / f"{iid}.png"), load_mask(self.dir / "preds" / f"{iid}_conf.png")

    def save_label(self, iid: int, mask: np.ndarray, status: str = "in_progress", add_to_memory: bool | None = None):
        r = self.image(iid)
        if mask.shape != (r["height"], r["width"]):
            raise ValueError(f"mask shape {mask.shape} != image {(r['height'], r['width'])}")
        valid = {0} | {c["idx"] for c in self.classes()}
        bad = set(np.unique(mask).tolist()) - valid
        if bad:
            mask = mask.copy()
            mask[np.isin(mask, list(bad))] = 0
        save_mask(self.dir / "labels" / f"{iid}.png", mask, self.classes())
        frac = float((mask > 0).mean())
        if status not in STATUSES:
            status = "in_progress"
        self.db.x("UPDATE images SET status=?, labeled_frac=?, updated=? WHERE id=?", (status, frac, time.time(), iid))
        self.db.event(self.id, "label_save", iid, status=status, labeled=round(frac, 4))
        # a reference whose label changed must be refreshed in the memory bank
        if add_to_memory or (add_to_memory is None and r["in_memory"]):
            self.memory_add([iid])

    def clear_label(self, iid: int):
        (self.dir / "labels" / f"{iid}.png").unlink(missing_ok=True)
        r = self.image(iid)
        if r["in_memory"]:
            self.memory_remove([iid])
        has_pred = (self.dir / "preds" / f"{iid}.png").exists()
        self.db.x("UPDATE images SET status=?, labeled_frac=0, updated=? WHERE id=?",
                  ("predicted" if has_pred else "new", time.time(), iid))

    # ------------------------------------------------------------------ engine
    def backbone(self) -> Backbone:
        cfg = self.config
        return get_backbone(cfg.model, cfg.image_size, cfg.svd_components())

    def features(self, iid: int, rgb: np.ndarray | None = None) -> Features:
        key = (iid, self.feature_key())
        ft = self._feat_cache.get(key)
        if ft is None:
            ft = self.backbone().features(self.rgb(iid) if rgb is None else rgb).cpu_half()
            self._feat_cache.put(key, ft)
        return ft

    def superpixels(self, iid: int, rgb: np.ndarray | None = None) -> np.ndarray:
        n = self.config.sp_segments
        sp = self._sp_cache.get((iid, n))
        if sp is None:
            sp = superpixels(self.rgb(iid) if rgb is None else rgb, n).astype(np.int32)
            self._sp_cache.put((iid, n), sp)
        return sp

    def memory_path(self, key: str | None = None) -> Path:
        return self.dir / "memory" / f"{key or self.feature_key()}.pt"

    def memory(self) -> MemoryBank:
        with self.lock:
            key = self.feature_key()
            if self._mem is None or self._mem_key != key:
                p = self.memory_path(key)
                dim = MODELS[self.config.model][2]
                self._mem = MemoryBank.load(p) if p.exists() else MemoryBank(dim, self.n_classes())
                self._mem.set_classes(max(self.n_classes(), self._mem.n_classes))
                self._mem_key = key
            return self._mem

    def save_memory(self):
        if self._mem is not None:
            self._mem.save(self.memory_path(self._mem_key))

    def memory_stats(self, light: bool = False) -> dict:
        refs = self.db.q("SELECT id, name, labeled_frac FROM images WHERE project_id=? AND in_memory=1 ORDER BY id",
                         (self.id,))
        p = self.memory_path()
        if light and self._mem is None and not p.exists():
            st = dict(size=0, images=0, per_class={}, bytes=0, history=[])
        else:
            st = self.memory().stats()
        st["references"] = len(refs)
        st["feature_key"] = self.feature_key()
        st["stale"] = len(refs) > 0 and st["images"] < len(refs)
        other = [q.stem for q in (self.dir / "memory").glob("*.pt") if q.stem != self.feature_key()]
        st["other_banks"] = other
        if not light:
            info = self.memory().images
            st["items"] = [dict(id=r["id"], name=r["name"], **{k: v for k, v in info.get(r["id"], {}).items()
                                                            if k in ("patches", "kept", "classes", "added")})
                           for r in refs]
        return st

    def memory_add(self, iids: list[int], progress=None) -> dict:
        mem = self.memory()
        cfg = self.config
        added, skipped = 0, []
        for n, iid in enumerate(iids):
            lab = self.label(iid)
            if lab is None or not (lab > 0).any():
                skipped.append(iid)
                continue
            ft = self.features(iid)
            pl = patch_labels(lab, ft.grid, mem.n_classes, cfg.purity)
            added += mem.add(iid, ft.fd.float(), pl, name=self.image(iid)["name"])
            self.db.x("UPDATE images SET in_memory=1 WHERE id=?", (iid,))
            if progress:
                progress(n + 1, len(iids))
        rec = None
        if cfg.auto_compress and 0 < cfg.auto_compress < 1 and added:
            rec = mem.compress(ratio=cfg.auto_compress, relative_to_added=True)
        self.save_memory()
        self.db.event(self.id, "memory_add", images=iids, patches=added, size=len(mem),
                      auto_compressed=bool(rec))
        return dict(patches=added, skipped=skipped, size=len(mem), auto_compressed=bool(rec))

    def memory_remove(self, iids: list[int]) -> dict:
        mem = self.memory()
        n = sum(mem.remove(i) for i in iids)
        self.db.xmany("UPDATE images SET in_memory=0 WHERE id=?", [(i,) for i in iids])
        self.save_memory()
        self.db.event(self.id, "memory_remove", images=iids, patches=n)
        return dict(removed=n, size=len(mem))

    def memory_compress(self, ratio: float | None, max_per_class: int | None) -> dict:
        rec = self.memory().compress(ratio=ratio, max_per_class=max_per_class)
        self.save_memory()
        self.db.event(self.id, "memory_compress", **{k: v for k, v in rec.items() if k != "per_class"})
        return rec

    def memory_rebuild(self, progress=None) -> dict:
        """Re-extract every reference with the current backbone settings (after a model change)."""
        refs = [r["id"] for r in self.db.q("SELECT id FROM images WHERE project_id=? AND in_memory=1", (self.id,))]
        self._mem = MemoryBank(MODELS[self.config.model][2], self.n_classes())
        self._mem_key = self.feature_key()
        return self.memory_add(refs, progress)

    def memory_clear(self):
        self._mem = MemoryBank(MODELS[self.config.model][2], self.n_classes())
        self._mem_key = self.feature_key()
        self.db.x("UPDATE images SET in_memory=0 WHERE project_id=?", (self.id,))
        self.save_memory()

    def segmenter(self) -> Segmenter:
        return Segmenter(self.backbone(), self.config)

    def predict(self, iid: int, save: bool = True, mode: str | None = None) -> Prediction:
        rgb = self.rgb(iid)
        seg = self.segmenter()
        need_sp = (seg.cfg.refine == "superpixel")
        p = seg.predict(rgb, self.memory(), self.n_classes(), feats=self.features(iid, rgb),
                        sp=self.superpixels(iid, rgb) if need_sp else None, mode=mode)
        if save:
            save_mask(self.dir / "preds" / f"{iid}.png", p.labels, self.classes())
            from PIL import Image
            Image.fromarray((p.conf * 255).round().clip(0, 255).astype(np.uint8)).save(self.dir / "preds" / f"{iid}_conf.png")
            r = self.image(iid)
            status = r["status"] if r["status"] in ("reviewed", "in_progress") else "predicted"
            self.db.x("UPDATE images SET status=?, conf=?, pred_model=?, updated=? WHERE id=?",
                      (status, p.mean_conf, self.feature_key(), time.time(), iid))
        return p

    def accept_prediction(self, iid: int, status: str = "reviewed"):
        pred, _ = self.pred(iid)
        if pred is None:
            raise ValueError("no prediction for this image")
        self.save_label(iid, pred, status=status)

    def suggest(self, n: int = 10, status: str = "new", progress=None) -> dict:
        """Most diverse images to label next: k-center greedy over whole-image DINOv3 descriptors,
        seeded with the current references so suggestions cover what the memory bank does not."""
        from ..engine.memory import k_center_greedy
        pool = [r["id"] for r in self.list_images(status=status, limit=10 ** 7)["items"] if not r["in_memory"]]
        refs = [r["id"] for r in self.db.q("SELECT id FROM images WHERE project_id=? AND in_memory=1", (self.id,))]
        if not pool:
            return dict(ids=[], pool=0)
        def desc(iid):
            f = self.features(iid).f.float()
            v = torch.cat([f.mean((1, 2)), f.std((1, 2))])
            return torch.nn.functional.normalize(v, dim=0)
        X = []
        for k, iid in enumerate(pool):
            X.append(desc(iid))
            if progress:
                progress(k + 1, len(pool) + len(refs))
        R = [desc(i) for i in refs]
        X = torch.stack(X)
        init = torch.stack(R) if R else None
        idx = k_center_greedy(X, min(n, len(pool)), proj_dim=10 ** 6, init=init).cpu().tolist()
        ids = [pool[i] for i in idx]
        self.db.event(self.id, "suggest", ids=ids, pool=len(pool))
        return dict(ids=ids, pool=len(pool), references=len(refs))

    # ------------------------------------------------------------------ tools
    def wand(self, iid: int, x: float, y: float, thresh: float = 0.6, scope: str = "connected",
             snap: bool = True) -> np.ndarray:
        """Region of patches whose DINOv3 feature is similar to the clicked patch (cosine >= thresh)."""
        import cv2
        import torch.nn.functional as F
        r = self.image(iid)
        H, W = r["height"], r["width"]
        ft = self.features(iid)
        f = ft.f.float()
        C, h, w = f.shape
        i = int(np.clip(y / H * h, 0, h - 1))
        j = int(np.clip(x / W * w, 0, w - 1))
        sim = torch.einsum("chw,c->hw", f, f[:, i, j])
        up = F.interpolate(sim[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
        if snap:
            sp = self.superpixels(iid)
            sums = np.bincount(sp.ravel(), weights=up.ravel())
            cnt = np.bincount(sp.ravel()).clip(min=1)
            up = (sums / cnt)[sp]
        m = (up >= thresh).astype(np.uint8)
        yi, xi = int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))
        if scope == "connected":
            n, cc = cv2.connectedComponents(m, connectivity=4)
            if m[yi, xi]:
                m = (cc == cc[yi, xi]).astype(np.uint8)
            else:
                m[:] = 0
        return m

    # ------------------------------------------------------------------ prelabel (closed-set model)
    def prelabel_segformer(self, iid: int, save: bool = True) -> np.ndarray:
        from .prelabel import segformer_prelabel
        rgb = self.rgb(iid)
        lab = segformer_prelabel(rgb, self.classes())
        if save:
            save_mask(self.dir / "preds" / f"{iid}.png", lab, self.classes())
            from PIL import Image
            Image.fromarray(np.where(lab > 0, 200, 0).astype(np.uint8)).save(self.dir / "preds" / f"{iid}_conf.png")
            r = self.image(iid)
            status = r["status"] if r["status"] in ("reviewed", "in_progress") else "predicted"
            self.db.x("UPDATE images SET status=?, pred_model=?, updated=? WHERE id=?",
                      (status, "segformer-ade20k", time.time(), iid))
        return lab

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, iids: list[int] | None = None, progress=None) -> dict:
        """Auto-label reviewed images that are NOT references and compare with their human labels."""
        if iids is None:
            iids = [r["id"] for r in self.db.q(
                "SELECT id FROM images WHERE project_id=? AND status='reviewed' AND in_memory=0", (self.id,))]
        K = self.n_classes()
        conf = np.zeros((K + 1, K + 1), np.int64)
        per_image = []
        for n, iid in enumerate(iids):
            gt = self.label(iid)
            if gt is None:
                continue
            p = self.predict(iid, save=False)
            v = gt > 0
            cm = np.bincount(gt[v].astype(np.int64) * (K + 1) + p.labels[v], minlength=(K + 1) ** 2).reshape(K + 1, K + 1)
            conf += cm
            per_image.append(dict(id=iid, acc=float(np.diag(cm).sum() / max(cm.sum(), 1)), conf=p.mean_conf))
            if progress:
                progress(n + 1, len(iids))
        inter = np.diag(conf)[1:]
        union = conf.sum(0)[1:] + conf.sum(1)[1:] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        names = {c["idx"]: c["name"] for c in self.classes()}
        per_class = {names.get(i + 1, str(i + 1)): (None if np.isnan(v) else round(float(v) * 100, 1))
                     for i, v in enumerate(iou) if union[i] > 0}
        res = dict(n_images=len(per_image), miou=round(float(np.nanmean(iou)) * 100, 2) if len(per_image) else None,
                   pixacc=round(float(inter.sum() / max(conf[1:].sum(), 1)) * 100, 2), per_class=per_class,
                   per_image=per_image, memory_size=len(self.memory()), config=self.config.to_dict())
        if per_image:
            self.db.x("INSERT INTO evals(project_id, ts, miou, pixacc, n_images, memory_size, config, per_class)"
                      " VALUES (?,?,?,?,?,?,?,?)", (self.id, time.time(), res["miou"], res["pixacc"], len(per_image),
                                                    res["memory_size"], json.dumps(res["config"]), json.dumps(per_class)))
        return res


class Workspace:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.projects_dir = self.data_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.db = DB(self.data_dir / "semlabel.db")
        self._projects: dict[int, Project] = {}
        self.lock = threading.Lock()

    def project(self, pid: int) -> Project:
        with self.lock:
            if pid not in self._projects:
                r = self.db.one("SELECT * FROM projects WHERE id=?", (pid,))
                if r is None:
                    raise KeyError(f"project {pid} not found")
                self._projects[pid] = Project(self, r)
            return self._projects[pid]

    def projects(self) -> list[dict]:
        out = []
        for r in self.db.q("SELECT id FROM projects ORDER BY id"):
            p = self.project(r["id"])
            info = p.info()
            out.append({k: info[k] for k in ("id", "slug", "name", "description", "counts", "total", "references",
                                              "created", "feature_key")})
        return out

    def create_project(self, name: str, template: str = "indoor", classes: list | None = None,
                       config: dict | None = None, description: str = "") -> Project:
        slug = slugify(name)
        base, n = slug, 2
        while self.db.one("SELECT id FROM projects WHERE slug=?", (slug,)):
            slug, n = f"{base}-{n}", n + 1
        now = time.time()
        cfg = EngineConfig.from_dict(config).to_dict()
        pid = self.db.x("INSERT INTO projects(slug, name, description, created, updated, config) VALUES (?,?,?,?,?,?)",
                        (slug, name, description, now, now, json.dumps(cfg)))
        p = self.project(pid)
        if classes is None:
            classes = [(n_, rgb_to_hex(c)) for n_, c in INDOOR_CLASSES] if template == "indoor" else []
        for item in classes:
            if isinstance(item, dict):
                p.add_class(item["name"], item.get("color"))
            elif isinstance(item, (list, tuple)):
                p.add_class(item[0], item[1] if len(item) > 1 else None)
            else:
                p.add_class(str(item))
        self.db.event(pid, "project_create", name=name, template=template)
        return p

    def delete_project(self, pid: int):
        p = self.project(pid)
        shutil.rmtree(p.dir, ignore_errors=True)
        for t in ("images", "classes", "events", "evals", "jobs"):
            self.db.x(f"DELETE FROM {t} WHERE project_id=?", (pid,))
        self.db.x("DELETE FROM projects WHERE id=?", (pid,))
        with self.lock:
            self._projects.pop(pid, None)
