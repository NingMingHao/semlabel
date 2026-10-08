"""Memory bank of labelled DINOv3 patch features (PatchCore-style) with coreset compression.

Every reference image contributes its debiased patch features whose patch is covered by one class for
at least `purity` of its area. Queries are k-nearest-neighbour label votes (the multi-class form of
INSID3's backward matching) and per-class sub-prototypes (INSID3's forward matching, with k-means
sub-prototypes so that varied classes such as "wall" or "storage" keep several modes).

Compression follows anomalib's PatchCore `KCenterGreedy`: features are randomly projected to a low
dimension and the coreset is grown by repeatedly adding the feature farthest from everything already
selected. That keeps coverage of every appearance while dropping near-duplicates (the same wall seen
in 30 frames). It runs per class so rare classes are never squeezed out by frequent ones.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def patch_labels(mask: np.ndarray, grid: tuple[int, int], n_classes: int, purity: float) -> torch.Tensor:
    """mask: (H, W) uint8, 0 = unlabeled, 1..n_classes = class ids.
    Returns (h, w) int64 class id per patch, -1 where no class covers >= purity of the patch."""
    import cv2
    h, w = grid
    frac = torch.zeros(n_classes + 1, h, w)
    for c in np.unique(mask):
        if 0 < c <= n_classes:                                     # unlabeled (0) never wins
            frac[int(c)] = torch.from_numpy(
                cv2.resize((mask == c).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA))
    best, cls = frac.max(0)
    cls[best < purity] = -1
    return cls


def k_center_greedy(x: torch.Tensor, n_select: int, proj_dim: int = 128, seed: int = 0,
                    init: torch.Tensor | None = None) -> torch.Tensor:
    """Indices of a k-center coreset of x (N, D) (anomalib KCenterGreedy, on the GPU).

    init: optional (M, D) points that are already covered (e.g. existing references); the selection then
    starts from the point farthest from all of them."""
    N = x.shape[0]
    if n_select >= N and init is None:
        return torch.arange(N, device=x.device)
    n_select = min(n_select, N)
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = x.float()
    zi = init.float().to(z.device) if init is not None and len(init) else None
    if z.shape[1] > proj_dim:                                       # Johnson-Lindenstrauss projection
        R = torch.randn(z.shape[1], proj_dim, generator=g).to(z.device) / proj_dim ** 0.5
        z = z @ R
        zi = zi @ R if zi is not None else None
    sel = torch.empty(n_select, dtype=torch.long, device=x.device)
    if zi is not None:
        min_d = torch.cdist(z, zi).pow(2).min(1).values
        start = int(torch.argmax(min_d))
    else:
        start = int(torch.randint(N, (1,), generator=g))
    sel[0] = start
    first = ((z - z[start]) ** 2).sum(1)
    min_d = torch.minimum(min_d, first) if zi is not None else first
    for i in range(1, n_select):
        idx = int(torch.argmax(min_d))
        sel[i] = idx
        min_d = torch.minimum(min_d, ((z - z[idx]) ** 2).sum(1))
    return sel


def spherical_kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0) -> torch.Tensor:
    """(k, D) unit-norm centroids of unit-norm rows x (N, D)."""
    N = x.shape[0]
    k = max(1, min(k, N))
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = x[torch.randperm(N, generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        newc = torch.zeros_like(c).index_add_(0, a, x)
        empty = torch.bincount(a, minlength=k) == 0
        newc[empty] = c[empty]
        c = F.normalize(newc, dim=1)
    return c


class MemoryBank:
    def __init__(self, dim: int, n_classes: int, device: str | None = None):
        self.dim = dim
        self.n_classes = n_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feats = torch.empty(0, dim, dtype=torch.float16)       # CPU master copy (debiased, unit norm)
        self.labels = torch.empty(0, dtype=torch.int16)
        self.img = torch.empty(0, dtype=torch.int32)
        self.images: dict[int, dict] = {}                           # img id -> info
        self.history: list[dict] = []                               # compressions etc.
        self._dev = None                                            # (feats, labels) on device
        self._protos = None
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ editing
    def _dirty(self):
        self._dev = None
        self._protos = None

    def add(self, img_id: int, fd: torch.Tensor, labels: torch.Tensor, name: str = "") -> int:
        """fd: (C, h, w) debiased features; labels: (h, w) class id or -1. Replaces earlier entries of img_id."""
        with self.lock:
            self.remove(img_id, record=False)
            keep = labels.reshape(-1) >= 0
            f = fd.reshape(fd.shape[0], -1).T[keep.to(fd.device)]
            y = labels.reshape(-1)[keep]
            if len(y) == 0:
                return 0
            self.feats = torch.cat([self.feats, f.half().cpu()])
            self.labels = torch.cat([self.labels, y.to(torch.int16).cpu()])
            self.img = torch.cat([self.img, torch.full((len(y),), img_id, dtype=torch.int32)])
            counts = torch.bincount(y.long(), minlength=self.n_classes + 1).tolist()
            self.images[img_id] = dict(name=name, added=time.time(), patches=int(len(y)),
                                       kept=int(len(y)), classes={i: c for i, c in enumerate(counts) if c})
            self._dirty()
            return int(len(y))

    def remove(self, img_id: int, record: bool = True) -> int:
        with self.lock:
            sel = self.img != img_id
            n = int((~sel).sum())
            if n:
                self.feats, self.labels, self.img = self.feats[sel], self.labels[sel], self.img[sel]
                self._dirty()
            self.images.pop(img_id, None)
            return n

    def set_classes(self, n_classes: int):
        self.n_classes = n_classes
        self._dirty()

    def drop_class(self, cls: int):
        with self.lock:
            sel = self.labels != cls
            self.feats, self.labels, self.img = self.feats[sel], self.labels[sel], self.img[sel]
            self._dirty()

    def added_per_class(self) -> dict[int, int]:
        """Patches per class as originally added (before any compression)."""
        tot: dict[int, int] = {}
        for info in self.images.values():
            for c, n in info.get("classes", {}).items():
                tot[int(c)] = tot.get(int(c), 0) + int(n)
        return tot

    def compress(self, ratio: float | None = None, max_per_class: int | None = None, seed: int = 0,
                 relative_to_added: bool = False) -> dict:
        """k-center-greedy coreset per class. Keeps max(1, ratio*n) (and at most max_per_class) per class.

        relative_to_added: n is the number of patches originally added for the class rather than the current
        count, so calling it after every new reference keeps the bank at a stable fraction instead of
        shrinking old features again and again."""
        with self.lock:
            before = len(self.labels)
            keep_idx = []
            per_class = {}
            added = self.added_per_class() if relative_to_added else {}
            for c in sorted(set(self.labels.tolist())):
                idx = torch.nonzero(self.labels == c).squeeze(1)
                n = len(idx)
                target = n
                if ratio is not None:
                    base = added.get(int(c), n) if relative_to_added else n
                    target = min(n, max(1, int(round(base * ratio))))
                if max_per_class is not None:
                    target = min(target, max_per_class)
                if target < n:
                    x = self.feats[idx].to(self.device)
                    sub = k_center_greedy(x, target, seed=seed).cpu()
                    idx = idx[sub]
                keep_idx.append(idx)
                per_class[int(c)] = (n, len(idx))
            keep = torch.sort(torch.cat(keep_idx)).values if keep_idx else torch.empty(0, dtype=torch.long)
            self.feats, self.labels, self.img = self.feats[keep], self.labels[keep], self.img[keep]
            kept = torch.bincount(self.img.long(), minlength=1) if len(self.img) else torch.empty(0)
            for i, info in self.images.items():
                info["kept"] = int(kept[i]) if i < len(kept) else 0
            rec = dict(time=time.time(), before=before, after=len(self.labels), ratio=ratio,
                       max_per_class=max_per_class, per_class=per_class)
            self.history.append(rec)
            self._dirty()
            return rec

    # ------------------------------------------------------------------ queries
    def _device_tensors(self):
        if self._dev is None:
            self._dev = (self.feats.to(self.device), self.labels.long().to(self.device))
        return self._dev

    def knn(self, q: torch.Tensor, k: int, chunk: int = 32768) -> tuple[torch.Tensor, torch.Tensor]:
        """q: (P, C) unit rows -> (P, k) similarities and (P, k) labels of the nearest memory features."""
        with self.lock:
            M, L = self._device_tensors()
            qh = q.to(M.device, M.dtype)
            k = min(k, M.shape[0])
            best_s = best_i = None
            for s in range(0, M.shape[0], chunk):
                sim = qh @ M[s:s + chunk].T
                kk = min(k, sim.shape[1])
                v, i = sim.topk(kk, dim=1)
                i = i + s
                if best_s is None:
                    best_s, best_i = v, i
                else:
                    cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, i], 1)
                    best_s, o = cs.topk(k, dim=1)
                    best_i = ci.gather(1, o)
            return best_s.float(), L[best_i]

    def prototypes(self, per_class: int) -> dict[int, torch.Tensor]:
        """class id -> (m, C) unit sub-prototypes on device (spherical k-means, cached)."""
        with self.lock:
            if self._protos is None or self._protos[0] != per_class:
                M, L = self._device_tensors()
                protos = {}
                for c in torch.unique(L).tolist():
                    x = M[L == c].float()
                    m = max(1, min(per_class, len(x) // 20)) if per_class > 1 else 1
                    protos[int(c)] = spherical_kmeans(x, m) if m > 1 else F.normalize(x.mean(0, keepdim=True), dim=1)
                self._protos = (per_class, protos)
            return self._protos[1]

    # ------------------------------------------------------------------ io / stats
    def __len__(self):
        return len(self.labels)

    def stats(self) -> dict:
        with self.lock:
            counts = torch.bincount(self.labels.long(), minlength=self.n_classes + 1).tolist() if len(self.labels) else []
            return dict(size=len(self.labels), dim=self.dim, images=len(self.images),
                        per_class={i: c for i, c in enumerate(counts) if c},
                        bytes=int(self.feats.numel() * 2), history=self.history[-10:])

    def save(self, path: Path):
        with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save(dict(dim=self.dim, n_classes=self.n_classes, feats=self.feats, labels=self.labels,
                            img=self.img, images=self.images, history=self.history), tmp)
            tmp.replace(path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "MemoryBank":
        d = torch.load(path, map_location="cpu", weights_only=False)
        mb = cls(d["dim"], d["n_classes"], device)
        mb.feats, mb.labels, mb.img = d["feats"], d["labels"], d["img"]
        mb.images = {int(k): v for k, v in d["images"].items()}
        mb.history = d.get("history", [])
        return mb
