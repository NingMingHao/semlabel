"""Closed-set pre-labeling with SegFormer-B4 (ADE20K) mapped onto the project's classes.

Useful to bootstrap the first references: pre-label, correct by hand, add to memory. ADE20K classes are
mapped through data/ade20k_to_indoor.json (16 indoor groups) and then by name onto the project's
classes; anything without a match stays unlabeled.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import torch

_MODEL = {}
_LOCK = threading.Lock()
MAP_FILE = Path(__file__).resolve().parents[1] / "data" / "ade20k_to_indoor.json"
HF_ID = "nvidia/segformer-b4-finetuned-ade-512-512"


def _load():
    with _LOCK:
        if "m" not in _MODEL:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            proc = SegformerImageProcessor.from_pretrained(HF_ID)
            model = SegformerForSemanticSegmentation.from_pretrained(HF_ID).eval()
            if torch.cuda.is_available():
                model = model.cuda()
            _MODEL["m"] = (proc, model)
        return _MODEL["m"]


def ade_to_class_lut(classes: list[dict]) -> np.ndarray:
    """(150,) ADE20K id -> project class idx (0 = unlabeled)."""
    proc, model = _load()
    id2label = model.config.id2label
    mapping = json.load(open(MAP_FILE))["ade_to_group"]
    by_name = {c["name"].lower(): c["idx"] for c in classes}
    lut = np.zeros(len(id2label), np.uint8)
    for i in range(len(id2label)):
        ade = id2label[i]
        group = mapping.get(ade, mapping.get(ade + " ", ade))
        for cand in (group, ade, ade.split(",")[0]):
            if cand and cand.lower() in by_name:
                lut[i] = by_name[cand.lower()]
                break
    return lut


@torch.no_grad()
def segformer_prelabel(rgb: np.ndarray, classes: list[dict]) -> np.ndarray:
    proc, model = _load()
    lut = ade_to_class_lut(classes)
    inp = proc(images=rgb, return_tensors="pt")
    dev = next(model.parameters()).device
    logits = model(pixel_values=inp["pixel_values"].to(dev)).logits
    up = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
    ade = up.argmax(1)[0].cpu().numpy()
    return lut[ade]
