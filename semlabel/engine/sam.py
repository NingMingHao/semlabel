"""Click-to-segment with SAM 2.1 (transformers). Image embeddings are cached per image for fast re-clicks."""
from __future__ import annotations

import os
import threading

import numpy as np
import torch

HF_ID = os.environ.get("SEMLABEL_SAM", "facebook/sam2.1-hiera-large")
_STATE: dict = {}
_LOCK = threading.Lock()


def _cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(HF_ID, "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


def available() -> bool:
    try:
        import transformers  # noqa: F401
        return hasattr(transformers, "Sam2Model") and _cached()
    except Exception:  # noqa: BLE001
        return False


def unavailable_reason() -> str:
    return f"SAM weights {HF_ID} are not in the Hugging Face cache (download them once with huggingface-cli download {HF_ID})"


def _load():
    with _LOCK:
        if "model" not in _STATE:
            from transformers import Sam2Model, Sam2Processor
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            proc = Sam2Processor.from_pretrained(HF_ID)
            model = Sam2Model.from_pretrained(HF_ID).to(dev).eval()
            _STATE.update(model=model, proc=proc, dev=dev, emb={})
        return _STATE


@torch.no_grad()
def segment(iid: int, rgb: np.ndarray, points: list, box: list | None = None, scope: str = "") -> tuple[np.ndarray, float]:
    """points: [[x, y, label(1 fg / 0 bg)], ...] in image pixels; box: [x0, y0, x1, y1] or None."""
    st = _load()
    model, proc, dev = st["model"], st["proc"], st["dev"]
    kw = {}
    if points:
        kw["input_points"] = [[[[float(p[0]), float(p[1])] for p in points]]]
        kw["input_labels"] = [[[int(p[2]) if len(p) > 2 else 1 for p in points]]]
    if box:
        kw["input_boxes"] = [[[float(v) for v in box]]]
    inputs = proc(images=rgb, return_tensors="pt", **kw)
    key = (scope, iid, rgb.shape)
    with _LOCK:
        emb = st["emb"].get(key)
        if emb is None:
            emb = model.get_image_embeddings(inputs["pixel_values"].to(dev, model.dtype))
            if len(st["emb"]) > 6:
                st["emb"].pop(next(iter(st["emb"])))
            st["emb"][key] = emb
        multi = len(points) <= 1 and not box
        call = {k: v.to(dev) for k, v in inputs.items() if k in ("input_points", "input_labels", "input_boxes")}
        out = model(image_embeddings=emb, multimask_output=multi, **call)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]   # (objects, n, H, W)
    scores = out.iou_scores.cpu()[0, 0]
    best = int(scores.argmax())
    m = masks[0, best].numpy().astype(np.uint8)
    return m, float(scores[best])
