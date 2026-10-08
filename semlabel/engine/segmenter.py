"""Multi-class in-context segmentation: INSID3 generalised to a labelled memory bank.

For a target image (features f, debiased fd on an h x w patch grid):

1. backward matching (INSID3: target patch -> nearest reference patch must lie inside the reference mask)
   becomes a k-NN vote over the whole memory bank: V[c, t] = softmax-weighted share of the k nearest
   memory features that carry class c.
2. forward matching (INSID3: similarity to the reference prototype > 0) uses per-class sub-prototypes:
   S[c, t] = max_m <fd_t, p_cm>.
3. the target is partitioned by agglomerative clustering of the original (not debiased) features with
   INSID3's threshold tau.
4. INSID3 seed + aggregate, per class: candidates = (S_c > 0) & (argmax V == c); the seed is the
   candidate cluster most similar to the class prototype; every cluster scores
   cross_c(k) * intra(k, seed_c) * area_c(k) and is kept above merge_thresh.

Modes: "knn" (patch votes), "cluster" (votes averaged per cluster), "insid3" (strict per-class INSID3,
unclaimed clusters stay unlabeled), "hybrid" (cluster votes boosted by the INSID3 score).
Patch-level class probabilities are upsampled and optionally snapped to SLIC superpixels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

from ..config import EngineConfig
from .backbone import Backbone, Features
from .memory import MemoryBank


@dataclass
class Prediction:
    labels: np.ndarray                  # (H, W) uint8 class ids, 0 = unlabeled
    conf: np.ndarray                    # (H, W) float32 in [0, 1]
    patch_probs: torch.Tensor           # (K+1, h, w) on cpu
    timing: dict = field(default_factory=dict)

    @property
    def mean_conf(self) -> float:
        return float(self.conf.mean())


def agglomerative(f: torch.Tensor, tau: float) -> torch.Tensor:
    """f: (C, P) unit columns -> (P,) cluster ids (cosine distance, average linkage; INSID3)."""
    X = f.T
    D = (1.0 - (X @ X.T).clamp(-1, 1)).float().cpu().numpy()
    ac = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                 distance_threshold=float(1.0 - tau))
    return torch.from_numpy(ac.fit_predict(D)).long().to(f.device)


def cluster_means(x: torch.Tensor, lab: torch.Tensor, K: int, normalize: bool = True) -> torch.Tensor:
    """x: (D, P), lab: (P,) -> (K, D) per-cluster means."""
    s = torch.zeros(K, x.shape[0], device=x.device).index_add_(0, lab, x.T)
    n = torch.bincount(lab, minlength=K).clamp(min=1).unsqueeze(1).float()
    m = s / n
    return F.normalize(m, dim=1) if normalize else m


def superpixels(image: np.ndarray, n_segments: int, max_side: int = 1280) -> np.ndarray:
    """SLIC superpixels. Large images are segmented at <= max_side px and the labels upscaled (4K: 5 s -> <1 s)."""
    import cv2
    from skimage.segmentation import slic
    H, W = image.shape[:2]
    s = max_side / max(H, W)
    small = image if s >= 1 else cv2.resize(image, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    sp = slic(small, n_segments=n_segments, compactness=12, start_label=0, enforce_connectivity=True)
    if sp.max() + 1 < max(2, n_segments // 10):                    # SLIC collapsed (e.g. pure noise): use a grid
        h, w = small.shape[:2]
        step = max(2, int(round((h * w / n_segments) ** 0.5)))
        gy, gx = np.mgrid[0:h, 0:w]
        sp = (gy // step) * ((w + step - 1) // step) + gx // step
    if s < 1:
        sp = cv2.resize(sp.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    return sp


class Segmenter:
    def __init__(self, backbone: Backbone, cfg: EngineConfig):
        self.bb = backbone
        self.cfg = cfg

    # ------------------------------------------------------------------ patch level
    @torch.no_grad()
    def patch_scores(self, feats: Features, mem: MemoryBank, n_classes: int, mode: str | None = None):
        cfg = self.cfg
        mode = mode or cfg.mode
        dev = self.bb.device
        f = feats.f.to(dev).float()
        fd = feats.fd.to(dev).float()
        C, h, w = f.shape
        P = h * w
        K1 = n_classes + 1
        t0 = time.time()
        # 1. backward: k-NN vote over the memory bank
        sims, labs = mem.knn(fd.reshape(C, P).T, cfg.knn_k)
        wts = torch.softmax(sims / cfg.knn_temp, dim=1)
        V = torch.zeros(K1, P, device=dev).scatter_add_(0, labs.T.clamp(0, K1 - 1), wts.T)
        t1 = time.time()
        if mode == "knn":
            return V.reshape(K1, h, w), dict(knn=t1 - t0)
        if mode in ("smooth", "smooth_insid3"):
            Vs = self._affinity_smooth(V, f.reshape(C, P))
            if mode == "smooth":
                return Vs.reshape(K1, h, w), dict(knn=t1 - t0, smooth=time.time() - t1)
            V = Vs
        # 2. forward: sub-prototype similarity
        protos = mem.prototypes(cfg.prototypes_per_class)
        S = torch.full((K1, P), -1.0, device=dev)
        fdP = fd.reshape(C, P)
        for c, pc in protos.items():
            if 0 < c < K1:
                S[c] = (pc.to(dev).float() @ fdP).max(0).values
        # 3. clustering of the target (original features)
        cl = agglomerative(f.reshape(C, P), cfg.tau)
        Kc = int(cl.max()) + 1
        t2 = time.time()
        knn_c = cluster_means(V, cl, Kc, normalize=False)            # (Kc, K1) mean vote per cluster
        if mode == "cluster":
            return knn_c[cl].T.reshape(K1, h, w), dict(knn=t1 - t0, cluster=t2 - t1, clusters=Kc)
        # 4. INSID3 seed + aggregate per class
        comb = self._insid3_scores(V, S, cl, Kc, f.reshape(C, P), fd.reshape(C, P), protos, K1)
        t3 = time.time()
        if mode == "insid3":
            score = torch.where(comb > cfg.merge_thresh, comb, torch.zeros_like(comb))
            best, arg = score.max(1)
            probs = torch.zeros(Kc, K1, device=dev)
            probs[torch.arange(Kc), arg] = torch.where(best > 0, best.clamp(max=1), torch.zeros_like(best))
            probs[:, 0] = torch.where(best > 0, 1 - best.clamp(max=1), torch.ones_like(best))
            return probs[cl].T.reshape(K1, h, w), dict(knn=t1 - t0, cluster=t2 - t1, insid3=t3 - t2, clusters=Kc)
        # hybrid (and smooth_insid3): per-patch votes smoothed by the cluster vote, plus the INSID3 evidence
        boost = torch.where(comb > cfg.merge_thresh, comb, torch.zeros_like(comb))     # (Kc, K1)
        mix = 0.5 * V + 0.5 * knn_c[cl].T + boost[cl].T
        mix = mix / mix.sum(0, keepdim=True).clamp(min=1e-6)
        return mix.reshape(K1, h, w), dict(knn=t1 - t0, cluster=t2 - t1, insid3=t3 - t2, clusters=Kc)

    @staticmethod
    def _affinity_smooth(V, fP, k: int = 16, temp: float = 0.05, alpha: float = 0.5, iters: int = 2):
        """Soft label propagation inside the target image: each patch mixes its memory vote with the votes
        of its k most similar patches (original DINOv3 features). A soft version of INSID3's clustering."""
        X = fP.T
        sim = X @ X.T
        v, i = sim.topk(k + 1, dim=1)
        wgt = torch.softmax((v[:, 1:] - 1) / temp, dim=1)
        nb = i[:, 1:]
        out = V
        for _ in range(iters):
            out = alpha * V + (1 - alpha) * (out[:, nb] * wgt[None]).sum(-1)
        return out / out.sum(0, keepdim=True).clamp(min=1e-6)

    def _insid3_scores(self, V, S, cl, Kc, fP, fdP, protos, K1):
        """(Kc, K1) INSID3 combined score of every cluster for every class."""
        dev = V.device
        back = V.argmax(0)                                          # backward winner per patch
        area_all = torch.bincount(cl, minlength=Kc).float().clamp(min=1)
        proto_o = cluster_means(fP, cl, Kc)                         # (Kc, C) original space
        cross = cluster_means(S.clamp(min=-1), cl, Kc, normalize=False)   # (Kc, K1) mean forward sim
        comb = torch.zeros(Kc, K1, device=dev)
        for c in protos:
            if not 0 < c < K1:
                continue
            cand = (S[c] > 0) & (back == c)
            if not bool(cand.any()):
                continue
            in_c = torch.bincount(cl[cand], minlength=Kc).float()
            area = in_c / area_all
            matched = in_c > 0
            ids = torch.nonzero(matched).squeeze(1)
            seed = ids[cross[ids, c].argmax()]
            intra = proto_o @ proto_o[seed]
            area[seed] = 1.0
            comb[:, c] = cross[:, c].clamp(min=0) * intra.clamp(min=0) * area
        return comb

    # ------------------------------------------------------------------ pixel level
    @torch.no_grad()
    def predict(self, image: np.ndarray, mem: MemoryBank, n_classes: int, feats: Features | None = None,
                sp: np.ndarray | None = None, mode: str | None = None, refine: str | None = None) -> Prediction:
        if len(mem) == 0:
            raise ValueError("memory bank is empty: label a few images and add them as references first")
        cfg = self.cfg
        t0 = time.time()
        if feats is None:
            feats = self.bb.features(image)
        t1 = time.time()
        probs, timing = self.patch_scores(feats, mem, n_classes, mode)
        H, W = image.shape[:2]
        up = F.interpolate(probs[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        refine = refine or cfg.refine
        if refine == "superpixel":
            if sp is None:
                sp = superpixels(image, cfg.sp_segments)
            spt = torch.from_numpy(sp.astype(np.int64)).to(up.device).reshape(-1)
            ns = int(spt.max()) + 1
            sums = torch.zeros(up.shape[0], ns, device=up.device).index_add_(1, spt, up.reshape(up.shape[0], -1))
            cnt = torch.bincount(spt, minlength=ns).clamp(min=1).float()
            up = (sums / cnt)[:, spt].reshape(up.shape)
        conf, lab = up.max(0)
        if cfg.min_conf > 0:
            lab[conf < cfg.min_conf] = 0
        t2 = time.time()
        timing.update(features=t1 - t0, total=t2 - t0)
        return Prediction(lab.to(torch.uint8).cpu().numpy(), conf.float().cpu().numpy(), probs.cpu(), timing)
