"""Frozen DINOv3 patch features with INSID3 positional debiasing.

Images keep their aspect ratio: the long side is resized to `image_size` and both sides are rounded to
the 16 px patch size (INSID3 squashes to a square; indoor robot frames are 16:10 and squashing distorts
thin structures). Features are the last-layer patch tokens, L2-normalised. The positional subspace is
estimated per grid shape from a blank image (INSID3: SVD of features of an image with no content) and
projected out to give the "debiased" features used for cross-image matching.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..config import DINOV3_REPO, MODELS, WEIGHTS_DIR

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PATCH = 16


@dataclass
class Features:
    """Per-image features. f: (C, h, w) normalised; fd: (C, h, w) debiased + normalised."""
    f: torch.Tensor
    fd: torch.Tensor
    image_hw: tuple[int, int]       # original image size
    input_hw: tuple[int, int]       # size fed to the network

    @property
    def grid(self) -> tuple[int, int]:
        return tuple(self.f.shape[-2:])

    def cpu_half(self) -> "Features":
        return Features(self.f.half().cpu(), self.fd.half().cpu(), self.image_hw, self.input_hw)

    def to(self, device) -> "Features":
        return Features(self.f.to(device).float(), self.fd.to(device).float(), self.image_hw, self.input_hw)


def input_size(h: int, w: int, image_size: int) -> tuple[int, int]:
    s = image_size / max(h, w)
    return max(PATCH, int(round(h * s / PATCH)) * PATCH), max(PATCH, int(round(w * s / PATCH)) * PATCH)


class Backbone:
    """Loads one DINOv3 model and serves (debiased) patch features. Thread-safe."""

    def __init__(self, model: str = "large", image_size: int = 1024, svd_components: int | None = None,
                 device: str | None = None):
        if model not in MODELS:
            raise ValueError(f"unknown model {model}; choose from {list(MODELS)}")
        self.model_name = model
        self.image_size = image_size
        hub_name, weight_file, self.dim = MODELS[model]
        self.svd_components = svd_components if svd_components is not None else int(round(0.49 * self.dim))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = WEIGHTS_DIR / weight_file
        if not weights.exists():
            raise FileNotFoundError(weights)
        # build the architecture without weights, then load the local file directly (torch.hub would copy
        # the checkpoint into ~/.cache first)
        self.net = torch.hub.load(str(DINOV3_REPO), hub_name, source="local", pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if [m for m in missing if not m.endswith("bias_mask")] or unexpected:
            raise RuntimeError(f"weight mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.net.eval().to(self.device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self._bases: dict[tuple[int, int], torch.Tensor] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ core
    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) normalised -> (B, C, h, w) L2-normalised patch features (float32)."""
        if x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                f = self.net.get_intermediate_layers(x, n=1, reshape=True)[0]
        else:
            f = self.net.get_intermediate_layers(x, n=1, reshape=True)[0]
        return F.normalize(f.float(), dim=1)

    @torch.no_grad()
    def positional_basis(self, hw: tuple[int, int]) -> torch.Tensor:
        """(C, k) orthonormal basis of the positional subspace for an input of size hw (cached)."""
        if hw not in self._bases:
            blank = ((torch.zeros(1, 3, *hw) - MEAN) / STD).to(self.device)
            f = self._forward(blank)[0]                          # (C, h, w)
            E = f.reshape(f.shape[0], -1)
            E = E - E.mean(dim=1, keepdim=True)
            U, _, _ = torch.linalg.svd(E, full_matrices=False)
            k = min(self.svd_components, U.shape[1] - 1)
            self._bases[hw] = U[:, :k].contiguous()
        return self._bases[hw]

    def debias(self, f: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """f: (C, ...) normalised -> projection onto the orthogonal complement, renormalised."""
        if self.svd_components <= 0:
            return f
        B = self.positional_basis(hw)
        shp = f.shape
        X = f.reshape(shp[0], -1)
        X = X - B @ (B.T @ X)
        return F.normalize(X, dim=0).reshape(shp)

    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """image: (H, W, 3) uint8 RGB -> normalised (1, 3, h, w) tensor on device."""
        H, W = image.shape[:2]
        hw = input_size(H, W, self.image_size)
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float().div(255)
        x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False, antialias=True)
        return ((x - MEAN) / STD).to(self.device), hw

    @torch.no_grad()
    def features(self, image: np.ndarray) -> Features:
        with self.lock:
            x, hw = self.preprocess(image)
            f = self._forward(x)[0]
            fd = self.debias(f, hw)
        return Features(f, fd, tuple(image.shape[:2]), hw)


_CACHE: dict[tuple, Backbone] = {}
_CACHE_LOCK = threading.Lock()


def get_backbone(model: str, image_size: int, svd_components: int) -> Backbone:
    """One live backbone per (model, image_size, svd); switching models frees the previous one."""
    key = (model, image_size, svd_components)
    with _CACHE_LOCK:
        if key not in _CACHE:
            for k in list(_CACHE):
                if k[0] != model:
                    del _CACHE[k]
            torch.cuda.empty_cache()
            same = next((b for k, b in _CACHE.items() if k[0] == model), None)
            if same is not None:          # reuse the loaded network, only the settings differ
                bb = object.__new__(Backbone)
                bb.__dict__.update(same.__dict__)
                bb.image_size, bb.svd_components, bb._bases = image_size, svd_components, {}
                bb.lock = same.lock
                _CACHE[key] = bb
            else:
                _CACHE[key] = Backbone(model, image_size, svd_components)
        return _CACHE[key]
