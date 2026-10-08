"""Importers (folders, videos, ROS 2 bags, existing masks) and exporters (PNG, COCO, CVAT)."""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .files import color_mask, hex_to_rgb, load_mask, save_mask
from .project import IMG_EXT, Project


# ====================================================================== image importers
def import_folder(p: Project, folder: str, recursive: bool = True, stride: int = 1, max_images: int = 0,
                  progress=None, cancelled=lambda: False) -> dict:
    folder = Path(os.path.expanduser(folder))
    if not folder.is_dir():
        raise ValueError(f"not a folder: {folder}")
    files = sorted(f for f in (folder.rglob("*") if recursive else folder.iterdir())
                   if f.suffix.lower() in IMG_EXT and f.is_file())
    files = files[::max(1, stride)]
    if max_images:
        files = files[:max_images]
    added, dup = [], 0
    for n, f in enumerate(files):
        if cancelled():
            break
        rel = str(f.relative_to(folder))
        iid = p.add_image_file(f, name=rel.replace(os.sep, "__"), source=f"folder:{folder}")
        if iid is None:
            dup += 1
        else:
            added.append(iid)
        if progress:
            progress(n + 1, len(files))
    p.db.event(p.id, "import_folder", folder=str(folder), added=len(added), duplicates=dup)
    return dict(added=len(added), duplicates=dup, ids=added)


def import_video(p: Project, path: str, every_s: float = 1.0, max_images: int = 0, progress=None,
                 cancelled=lambda: False) -> dict:
    import cv2
    path = os.path.expanduser(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(every_s * fps)))
    added, i = [], 0
    stem = Path(path).stem
    while True:
        if cancelled():
            break
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                iid = p.add_image_array(frame[..., ::-1].copy(), f"{stem}_{i:06d}.jpg", source=f"video:{path}")
                if iid:
                    added.append(iid)
                if max_images and len(added) >= max_images:
                    break
        i += 1
        if progress and total and i % 10 == 0:
            progress(i, total)
    cap.release()
    p.db.event(p.id, "import_video", path=path, added=len(added))
    return dict(added=len(added), ids=added)


def decode_ros_image(msg) -> np.ndarray | None:
    """sensor_msgs/Image or CompressedImage -> RGB uint8."""
    import cv2
    if hasattr(msg, "format"):                       # CompressedImage
        buf = np.frombuffer(bytes(msg.data), np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return None if im is None else im[..., ::-1].copy()
    enc = msg.encoding.lower()
    data = np.frombuffer(bytes(msg.data), np.uint8)
    h, w, step = msg.height, msg.width, msg.step
    rows = data.reshape(h, step)
    if enc in ("rgb8", "bgr8"):
        im = rows[:, :w * 3].reshape(h, w, 3)
        return im.copy() if enc == "rgb8" else im[..., ::-1].copy()
    if enc in ("rgba8", "bgra8"):
        im = rows[:, :w * 4].reshape(h, w, 4)[..., :3]
        return im.copy() if enc == "rgba8" else im[..., ::-1].copy()
    if enc in ("mono8", "8uc1"):
        g = rows[:, :w]
        return np.repeat(g[..., None], 3, axis=2)
    if enc.startswith("bayer_"):
        code = {"bayer_rggb8": cv2.COLOR_BayerBG2RGB, "bayer_bggr8": cv2.COLOR_BayerRG2RGB,
                "bayer_gbrg8": cv2.COLOR_BayerGR2RGB, "bayer_grbg8": cv2.COLOR_BayerGB2RGB}.get(enc)
        return cv2.cvtColor(rows[:, :w].copy(), code) if code is not None else None
    if enc in ("yuv422", "yuyv", "uyvy"):
        conv = cv2.COLOR_YUV2RGB_YUYV if enc in ("yuyv", "yuv422") else cv2.COLOR_YUV2RGB_UYVY
        return cv2.cvtColor(rows[:, :w * 2].reshape(h, w, 2).copy(), conv)
    return None


def bag_image_topics(path: str) -> list[dict]:
    from mcap.reader import make_reader
    path = _bag_file(path)
    with open(path, "rb") as f:
        summ = make_reader(f).get_summary()
    out = []
    counts = summ.statistics.channel_message_counts if summ.statistics else {}
    for cid, ch in summ.channels.items():
        schema = summ.schemas.get(ch.schema_id)
        name = schema.name if schema else ""
        if name in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage") and "depth" not in ch.topic:
            out.append(dict(topic=ch.topic, type=name, count=int(counts.get(cid, 0))))
    return sorted(out, key=lambda d: d["topic"])


def _bag_file(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        mc = sorted(Path(path).glob("*.mcap"))
        if not mc:
            raise ValueError(f"no .mcap in {path}")
        return str(mc[0])
    return path


def import_bag(p: Project, path: str, topics: list[str], every_s: float = 1.0, max_images: int = 0,
               progress=None, cancelled=lambda: False) -> dict:
    from mcap_ros2.reader import read_ros2_messages
    path = _bag_file(path)
    last = {}
    added = []
    n = 0
    stem = Path(path).stem
    for m in read_ros2_messages(path, topics=topics):
        if cancelled():
            break
        n += 1
        t = m.log_time_ns * 1e-9
        topic = m.channel.topic
        if topic in last and t - last[topic] < every_s:
            continue
        rgb = decode_ros_image(m.ros_msg)
        if rgb is None:
            continue
        last[topic] = t
        name = f"{stem}__{topic.strip('/').replace('/', '_')}__{m.log_time_ns}.jpg"
        iid = p.add_image_array(rgb, name, source=f"bag:{path}:{topic}")
        if iid:
            added.append(iid)
        if progress and len(added) % 5 == 0:
            progress(len(added), max_images or 0)
        if max_images and len(added) >= max_images:
            break
    p.db.event(p.id, "import_bag", path=path, topics=topics, added=len(added))
    return dict(added=len(added), ids=added)


# ====================================================================== mask importers
def _labelmap_from_txt(text: str) -> dict:
    """CVAT/VOC labelmap.txt: 'name:r,g,b::' lines -> {(r,g,b): name}."""
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, col = line.split(":")[:2]
        if col:
            out[tuple(int(v) for v in col.split(","))] = name
    return out


def import_masks(p: Project, folder: str, kind: str = "index", class_names: list[str] | None = None,
                 status: str = "reviewed", add_to_memory: bool = False, create_classes: bool = True,
                 progress=None, cancelled=lambda: False) -> dict:
    """Attach existing masks to images with the same file stem.

    kind="index": pixel value v is a class id; class_names[v-1] (or classes.json next to the masks) gives
                  its name, matched to project classes by name (created if missing).
    kind="color": RGB masks; colours are matched to a labelmap.txt next to the masks or to project colours.
    """
    folder = Path(os.path.expanduser(folder))
    files = sorted(f for f in folder.rglob("*.png"))
    by_stem = {}
    for r in p.db.q("SELECT id, name FROM images WHERE project_id=?", (p.id,)):
        by_stem.setdefault(Path(r["name"]).stem, r["id"])
        by_stem.setdefault(Path(r["name"].split("__")[-1]).stem, r["id"])
    if class_names is None and (folder / "classes.json").exists():
        class_names = json.load(open(folder / "classes.json"))["classes"]
    existing = {c["name"].lower(): c["idx"] for c in p.classes()}

    def idx_for(name):
        if name.lower() not in existing:
            if not create_classes:
                return 0
            existing[name.lower()] = p.add_class(name)["idx"]
        return existing[name.lower()]

    lut = np.zeros(256, np.uint8)
    if kind == "index":
        if class_names:
            for v, name in enumerate(class_names, start=1):
                lut[v] = idx_for(name)
        else:
            lut = np.arange(256, dtype=np.uint8)
    color_map = {}
    if kind == "color":
        lm = folder / "labelmap.txt"
        if lm.exists():
            for rgb, name in _labelmap_from_txt(lm.read_text()).items():
                if name.lower() not in ("background", "_background_"):
                    color_map[rgb] = idx_for(name)
        else:
            color_map = {hex_to_rgb(c["color"]): c["idx"] for c in p.classes()}
    done, missing = [], []
    for n, f in enumerate(files):
        if cancelled():
            break
        iid = by_stem.get(f.stem)
        if iid is None:
            missing.append(f.name)
            continue
        im = Image.open(f)
        if kind == "index":
            m = lut[np.array(im if im.mode in ("P", "L") else im.convert("L"), np.uint8)]
        else:
            a = np.array(im.convert("RGB"))
            m = np.zeros(a.shape[:2], np.uint8)
            for rgb, idx in color_map.items():
                m[(a == rgb).all(-1)] = idx
        r = p.image(iid)
        if m.shape != (r["height"], r["width"]):
            m = np.array(Image.fromarray(m).resize((r["width"], r["height"]), Image.NEAREST))
        p.save_label(iid, m, status=status, add_to_memory=False)
        done.append(iid)
        if progress:
            progress(n + 1, len(files))
    if add_to_memory and done:
        p.memory_add(done)
    p.db.event(p.id, "import_masks", folder=str(folder), mask_kind=kind, imported=len(done), unmatched=len(missing))
    return dict(imported=len(done), unmatched=missing[:50], n_unmatched=len(missing), ids=done)


def import_cvat_zip(p: Project, zip_path: str, **kw) -> dict:
    """CVAT 'Segmentation mask 1.1' export (labelmap.txt + SegmentationClass/*.png)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(os.path.expanduser(zip_path)) as z:
            z.extractall(td)
        seg = next(Path(td).rglob("SegmentationClass"), None)
        if seg is None:
            raise ValueError("SegmentationClass folder not found in archive")
        lm = next(Path(td).rglob("labelmap.txt"), None)
        if lm is not None:
            (seg / "labelmap.txt").write_text(lm.read_text())
        return import_masks(p, str(seg), kind="color", **kw)


# ====================================================================== exporters
def export_project(p: Project, formats: list[str], include: str = "reviewed", with_images: bool = True,
                   progress=None, cancelled=lambda: False) -> dict:
    """Write a zip with the chosen formats.

    include: "reviewed" (human-verified only), "labeled" (any saved label), "all" (labels, else predictions).
    formats: index_png, color_png, coco, cvat_mask
    """
    import pycocotools.mask as mask_util
    classes = p.classes()
    rows = p.db.q("SELECT * FROM images WHERE project_id=? ORDER BY id", (p.id,))
    items = []
    for r in rows:
        lab = p.label(r["id"])
        src = "label"
        if lab is None and include == "all":
            lab, _ = p.pred(r["id"])
            src = "prediction"
        if lab is None or (include == "reviewed" and r["status"] != "reviewed"):
            continue
        items.append((r, lab, src))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = p.dir / "exports" / f"{p.slug}_{include}_{ts}.zip"
    stems = {}
    coco = dict(info=dict(description=f"semlabel export of {p.slug}", date_created=ts),
                images=[], annotations=[], licenses=[],
                categories=[dict(id=c["idx"], name=c["name"], supercategory="", color=c["color"]) for c in classes])
    ann_id = 1
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, (r, lab, src) in enumerate(items):
            if cancelled():
                break
            stem = Path(r["name"]).stem
            if stem in stems:
                stem = f"{stem}_{r['id']}"
            stems[stem] = r["id"]
            img_name = stem + Path(r["file"]).suffix
            if with_images:
                z.write(p.dir / "images" / r["file"], f"images/{img_name}")
            if "index_png" in formats:
                buf = io.BytesIO()
                Image.fromarray(lab).save(buf, format="PNG")
                z.writestr(f"masks/{stem}.png", buf.getvalue())
            if "color_png" in formats:
                buf = io.BytesIO()
                Image.fromarray(color_mask(lab, classes)).save(buf, format="PNG")
                z.writestr(f"color_masks/{stem}.png", buf.getvalue())
            if "cvat_mask" in formats:
                buf = io.BytesIO()
                Image.fromarray(color_mask(lab, classes)).save(buf, format="PNG")
                z.writestr(f"cvat_mask/SegmentationClass/{stem}.png", buf.getvalue())
                if with_images:
                    z.write(p.dir / "images" / r["file"], f"cvat_mask/JPEGImages/{img_name}")
            if "coco" in formats:
                coco["images"].append(dict(id=r["id"], file_name=img_name, width=r["width"], height=r["height"],
                                           status=r["status"], label_source=src))
                for c in classes:
                    m = lab == c["idx"]
                    if not m.any():
                        continue
                    rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
                    bbox = mask_util.toBbox(rle).tolist()
                    rle["counts"] = rle["counts"].decode("ascii")
                    coco["annotations"].append(dict(id=ann_id, image_id=r["id"], category_id=c["idx"], segmentation=rle,
                                                    area=float(m.sum()), bbox=bbox, iscrowd=1))
                    ann_id += 1
            if progress:
                progress(n + 1, len(items))
        meta = dict(project=p.slug, classes=classes, include=include, n_images=len(items), exported=ts,
                    note="masks/*.png: pixel value = class idx (0 = unlabeled)")
        z.writestr("classes.json", json.dumps(dict(classes=[c["name"] for c in classes], detail=classes), indent=1))
        z.writestr("meta.json", json.dumps(meta, indent=1))
        if "coco" in formats:
            z.writestr("annotations/coco_semantic.json", json.dumps(coco))
        if "cvat_mask" in formats:
            lm = "# label:color_rgb:parts:actions\nbackground:0,0,0::\n" + "".join(
                f"{c['name']}:{','.join(map(str, hex_to_rgb(c['color'])))}::\n" for c in classes)
            z.writestr("cvat_mask/labelmap.txt", lm)
            z.writestr("cvat_mask/ImageSets/Segmentation/default.txt", "\n".join(stems) + "\n")
    p.db.event(p.id, "export", file=out.name, formats=formats, include=include, n=len(items))
    return dict(file=out.name, n_images=len(items), size=out.stat().st_size)
