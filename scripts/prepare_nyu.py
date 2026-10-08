#!/usr/bin/env python3
"""Export the NYUv2 13-class validation split (tanganke/nyuv2 parquet) to PNG images + index masks.

usage: python scripts/prepare_nyu.py [parquet_dir] [out_dir]   (downloads the split if it is missing;
needs `pip install -e ".[bench]"`)

mask value 0 = void/unlabeled, 1..13 = classes below (NYUv2-13 order).
"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CLASSES = ["bed", "books", "ceiling", "chair", "floor", "furniture", "objects", "picture", "sofa",
           "table", "tv", "wall", "window"]
src = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "datasets/nyuv2_hf/data")
dst = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "datasets/nyuv2_13")
if not glob.glob(f"{src}/val-*.parquet"):                    # fetch the validation split (~1.3 GB) once
    from huggingface_hub import snapshot_download
    root = os.path.dirname(os.path.abspath(src)) if src.rstrip("/").endswith("data") else src
    snapshot_download("tanganke/nyuv2", repo_type="dataset", allow_patterns=["data/val-*"], local_dir=root)
    src = os.path.join(root, "data")
os.makedirs(f"{dst}/images", exist_ok=True)
os.makedirs(f"{dst}/masks", exist_ok=True)
n = 0
for f in sorted(glob.glob(f"{src}/val-*.parquet")):
    for b in pq.ParquetFile(f).iter_batches(batch_size=16, columns=["image", "segmentation"]):
        for img, seg in zip(b.column("image").to_pylist(), b.column("segmentation").to_pylist()):
            im = (np.clip(np.array(img, np.float32).transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
            m = (np.array(seg, np.int64) + 1).clip(0, 13).astype(np.uint8)
            Image.fromarray(im).save(f"{dst}/images/{n:04d}.png")
            Image.fromarray(m).save(f"{dst}/masks/{n:04d}.png")
            n += 1
json.dump({"classes": CLASSES}, open(f"{dst}/classes.json", "w"))
print("exported", n, "to", dst)
