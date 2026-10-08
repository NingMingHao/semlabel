#!/usr/bin/env python3
"""Benchmark the in-context engine on an indexed dataset (images/*.png, masks/*.png, classes.json).

Builds a memory bank from N reference images, auto-labels M held-out images and reports mIoU for every
(mode, refine) combination, plus timing. Example:
  python scripts/bench_engine.py --model large --refs 10 20 --test 100
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semlabel.config import EngineConfig  # noqa: E402
from semlabel.engine.backbone import get_backbone  # noqa: E402
from semlabel.engine.memory import MemoryBank, patch_labels  # noqa: E402
from semlabel.engine.segmenter import Segmenter, superpixels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="datasets/nyuv2_13", help="output of scripts/prepare_nyu.py")
ap.add_argument("--model", default="large")
ap.add_argument("--image-size", type=int, default=1024)
ap.add_argument("--svd-frac", type=float, default=0.49)
ap.add_argument("--refs", type=int, nargs="+", default=[10])
ap.add_argument("--test", type=int, default=100)
ap.add_argument("--modes", nargs="+", default=["knn", "cluster", "insid3", "hybrid"])
ap.add_argument("--refines", nargs="+", default=["bilinear", "superpixel"])
ap.add_argument("--tau", type=float, nargs="+", default=[0.6])
ap.add_argument("--merge", type=float, nargs="+", default=[0.2])
ap.add_argument("--k", type=int, nargs="+", default=[10])
ap.add_argument("--purity", type=float, default=0.6)
ap.add_argument("--protos", type=int, default=4)
ap.add_argument("--compress", type=float, nargs="+", default=[1.0])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()

classes = json.load(open(f"{a.data}/classes.json"))["classes"]
K = len(classes)
names = sorted(os.listdir(f"{a.data}/images"))
rng = np.random.RandomState(a.seed)
order = rng.permutation(len(names))
max_refs = max(a.refs)
# references from the front of the permutation, held-out images from the end: the held-out set no longer
# depends on how many references a run uses (results in bench/ before this change used order[max_refs:...])
ref_ids, test_ids = order[:max_refs], order[len(order) - a.test:]
assert max_refs + a.test <= len(order), "references and held-out images would overlap"
load = lambda i: (np.array(Image.open(f"{a.data}/images/{names[i]}").convert("RGB")),
                  np.array(Image.open(f"{a.data}/masks/{names[i]}")))

cfg = EngineConfig(model=a.model, image_size=a.image_size, svd_frac=a.svd_frac, purity=a.purity,
                   prototypes_per_class=a.protos)
t0 = time.time()
bb = get_backbone(cfg.model, cfg.image_size, cfg.svd_components())
print(f"loaded {a.model} in {time.time() - t0:.1f}s, dim {bb.dim}, svd {bb.svd_components}", flush=True)

ref_feats = []
for i in ref_ids:
    im, m = load(i)
    ft = bb.features(im)
    ref_feats.append((i, ft.cpu_half(), m))
test = []
tf = []
for i in test_ids:
    im, m = load(i)
    t1 = time.time()
    ft = bb.features(im)
    torch.cuda.synchronize()
    tf.append(time.time() - t1)
    test.append((im, m, ft.cpu_half(), superpixels(im, cfg.sp_segments)))
print(f"features: {np.median(tf) * 1000:.0f} ms/image (median), grid {test[0][2].grid}", flush=True)


def miou(conf):
    inter = np.diag(conf)[1:]
    union = conf.sum(0)[1:] + conf.sum(1)[1:] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return np.nanmean(iou), iou


results = []
for n_refs, comp, k, tau, merge in itertools.product(a.refs, a.compress, a.k, a.tau, a.merge):
    mem = MemoryBank(bb.dim, K)
    for i, ft, m in ref_feats[:n_refs]:
        mem.add(int(i), ft.fd.float(), patch_labels(m, ft.grid, K, cfg.purity))
    full = len(mem)
    if comp < 1.0:
        mem.compress(ratio=comp)
    cfg2 = EngineConfig(**{**cfg.to_dict(), "knn_k": k, "tau": tau, "merge_thresh": merge})
    seg = Segmenter(bb, cfg2)
    for mode, refine in itertools.product(a.modes, a.refines):
        conf = np.zeros((K + 1, K + 1), np.int64)
        times = []
        cover = []
        for im, m, ft, sp in test:
            t1 = time.time()
            p = seg.predict(im, mem, K, feats=ft, sp=sp, mode=mode, refine=refine)
            times.append(time.time() - t1)
            valid = m > 0
            cover.append(float((p.labels[valid] > 0).mean()) if valid.any() else 1.0)
            conf += np.bincount((m[valid].astype(np.int64) * (K + 1) + p.labels[valid]),
                                minlength=(K + 1) ** 2).reshape(K + 1, K + 1)
        mi, iou = miou(conf)
        pix = np.diag(conf)[1:].sum() / conf[1:].sum()
        r = dict(model=a.model, size=a.image_size, refs=n_refs, mem=len(mem), mem_full=full, compress=comp,
                 k=k, tau=tau, merge=merge, mode=mode, refine=refine, miou=round(float(mi) * 100, 2),
                 pixacc=round(float(pix) * 100, 2), coverage=round(float(np.mean(cover)) * 100, 1),
                 ms=round(float(np.median(times)) * 1000), iou={c: (None if np.isnan(v) else round(float(v) * 100, 1))
                                                               for c, v in zip(classes, iou)})
        results.append(r)
        print(f"refs {n_refs:3d} mem {len(mem):6d} comp {comp:.2f} k {k:2d} tau {tau:.2f} merge {merge:.2f} "
              f"{mode:8s} {refine:10s} mIoU {r['miou']:5.1f} pix {r['pixacc']:5.1f} cover {r['coverage']:5.1f} "
              f"{r['ms']:5d} ms", flush=True)
if a.out:
    json.dump(results, open(a.out, "w"), indent=1)
