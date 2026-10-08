"""Paths, model registry and engine defaults.

Locations come from environment variables, which may also be set in a `semlabel.env` file
(KEY=value lines) in the current directory or in $SEMLABEL_HOME (default ~/.semlabel):

    SEMLABEL_HOME         ~/.semlabel
    SEMLABEL_DATA         $SEMLABEL_HOME/data       projects, images, labels, memory banks
    SEMLABEL_WEIGHTS      $SEMLABEL_HOME/weights    DINOv3 .pth checkpoints
    SEMLABEL_DINOV3_REPO  $SEMLABEL_HOME/dinov3     clone of facebookresearch/dinov3 (model code)
    SEMLABEL_SAM          facebook/sam2.1-hiera-large (Hugging Face id, optional click tool)
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env_files() -> list[Path]:
    """Read KEY=value lines from semlabel.env files without overriding real environment variables."""
    home = Path(os.path.expanduser(os.environ.get("SEMLABEL_HOME", "~/.semlabel")))
    loaded = []
    for f in (Path.cwd() / "semlabel.env", home / "semlabel.env"):
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            os.environ.setdefault(k, os.path.expandvars(v.strip().strip("'\"")))
        loaded.append(f)
    return loaded


ENV_FILES = _load_env_files()
HOME = Path(os.path.expanduser(os.environ.get("SEMLABEL_HOME", "~/.semlabel")))


def _path(var: str, default: Path) -> Path:
    return Path(os.path.expanduser(os.environ[var])) if os.environ.get(var) else default


DATA_DIR = _path("SEMLABEL_DATA", HOME / "data")
WEIGHTS_DIR = _path("SEMLABEL_WEIGHTS", HOME / "weights")
DINOV3_REPO = _path("SEMLABEL_DINOV3_REPO", HOME / "dinov3")
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"   # facebookresearch/dinov3 revision tested

# DINOv3 backbones that fit a 12 GB GPU. key -> (hub entry point, weight file, feature dim)
MODELS = {
    "small": ("dinov3_vits16", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth", 384),
    "small_plus": ("dinov3_vits16plus", "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth", 384),
    "base": ("dinov3_vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", 768),
    "large": ("dinov3_vitl16", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", 1024),
    "huge_plus": ("dinov3_vith16plus", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", 1280),
}


@dataclass
class EngineConfig:
    """Everything that changes features or predictions. Stored per project."""
    model: str = "large"
    image_size: int = 1024          # long side fed to DINOv3 (multiple of 16)
    svd_frac: float = 0.49          # positional subspace size as a fraction of the feature dim (INSID3: 500/1024)
    tau: float = 0.6                # agglomerative clustering similarity threshold (INSID3)
    merge_thresh: float = 0.2       # INSID3 cluster aggregation threshold
    knn_k: int = 10                 # neighbours for the memory-bank vote
    knn_temp: float = 0.05          # softmax temperature on neighbour similarities
    purity: float = 0.6             # min fraction of a patch covered by one class to enter the memory bank
    prototypes_per_class: int = 4   # k-means sub-prototypes per class (forward matching)
    mode: str = "hybrid"            # knn | smooth | cluster | insid3 | hybrid | smooth_insid3 (see docs/BENCHMARK.md)
    refine: str = "bilinear"        # bilinear | superpixel (superpixel snapping did not raise mIoU on NYUv2)
    sp_segments: int = 1500         # SLIC superpixels per image for boundary snapping
    min_conf: float = 0.0           # pixels below this confidence are left unlabeled (0 = label everything)
    auto_compress: float = 0.0      # >0: after adding references keep each class at this fraction of the
                                    # features ever added (k-center coreset). NYUv2: 0.2 = +0.1..+1.5 mIoU, 5x smaller

    def feature_key(self) -> str:
        """Memory banks are only valid for the backbone + input size they were built with."""
        return f"{self.model}_{self.image_size}"

    def svd_components(self) -> int:
        return int(round(self.svd_frac * MODELS[self.model][2]))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "EngineConfig":
        d = d or {}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# Default indoor class set (16 groups; ADE20K classes map onto them in data/ade20k_to_indoor.json).
INDOOR_CLASSES = [
    ("floor", (205, 190, 160)), ("wall", (175, 185, 200)), ("ceiling", (225, 225, 230)),
    ("window", (90, 175, 230)), ("door", (120, 90, 200)), ("table / desk", (230, 140, 40)),
    ("chair", (220, 60, 60)), ("sofa / bed", (60, 160, 90)), ("storage", (150, 95, 55)),
    ("box / bin", (240, 200, 40)), ("appliance", (40, 170, 170)), ("plant", (100, 200, 80)),
    ("curtain", (175, 150, 215)), ("person", (235, 0, 130)), ("lamp / decor", (240, 230, 120)),
    ("other", (150, 150, 150)),
]
