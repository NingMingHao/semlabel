"""On-disk layout and image / mask helpers.

<data>/projects/<slug>/
    images/<id>.<ext>      original image (as imported)
    thumbs/<id>.jpg        256 px thumbnail
    labels/<id>.png        human label, palette PNG, pixel value = class index (0 = unlabeled)
    preds/<id>.png         latest auto label, same format
    preds/<id>_conf.png    confidence of the auto label, uint8 (255 = 1.0)
    memory/<feature>.pt    memory bank for one backbone/input size
    exports/               export archives
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

THUMB = 256


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in c[:3])


def palette(classes: list[dict]) -> list[int]:
    pal = [0, 0, 0] * 256
    for c in classes:
        r, g, b = hex_to_rgb(c["color"])
        pal[3 * c["idx"]:3 * c["idx"] + 3] = [r, g, b]
    return pal


def save_mask(path: Path, mask: np.ndarray, classes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(mask.astype(np.uint8), mode="P")
    im.putpalette(palette(classes))
    tmp = path.with_suffix(".tmp.png")
    im.save(tmp, optimize=False, compress_level=3)
    tmp.replace(path)


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    im = Image.open(path)
    if im.mode not in ("P", "L"):
        im = im.convert("L")
    return np.array(im, dtype=np.uint8)


def color_mask(mask: np.ndarray, classes: list[dict]) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for c in classes:
        lut[c["idx"]] = hex_to_rgb(c["color"])
    return lut[mask]


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def make_thumb(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.thumbnail((THUMB, THUMB))
    im.save(dst, quality=85)


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
