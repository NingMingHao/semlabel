"""Engine unit tests: patch labels, coreset, memory bank, segmenter (small DINOv3 on synthetic scenes)."""
import numpy as np
import pytest
import torch

from semlabel.config import EngineConfig
from semlabel.engine.memory import MemoryBank, k_center_greedy, patch_labels, spherical_kmeans

from conftest import needs_weights

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_patch_labels_purity():
    m = np.zeros((32, 32), np.uint8)
    m[:, :16] = 1
    m[:, 16:] = 2
    m[:8, :8] = 0                      # unlabeled corner
    lab = patch_labels(m, (4, 4), 2, purity=0.6)
    assert lab.shape == (4, 4)
    assert lab[0, 0] == -1             # unlabeled never wins
    assert (lab[:, 2:] == 2).all() and (lab[1:, :2] == 1).all()
    mixed = np.zeros((16, 16), np.uint8)
    mixed[:, :8], mixed[:, 8:] = 1, 2
    assert (patch_labels(mixed, (1, 1), 2, purity=0.6) == -1).all()   # 50/50 patch is impure


def test_k_center_greedy_covers_clusters():
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(5, 64, generator=g) * 10
    x = torch.cat([c + 0.01 * torch.randn(200, 64, generator=g) for c in centers])   # 5 tight blobs
    idx = k_center_greedy(x.to(DEV), 5).cpu()
    assert len(set(idx.tolist())) == 5
    assert len({int(i) // 200 for i in idx}) == 5        # one pick per blob: duplicates are dropped first


def test_spherical_kmeans_unit_norm():
    x = torch.nn.functional.normalize(torch.randn(300, 16), dim=1)
    c = spherical_kmeans(x, 4)
    assert c.shape == (4, 16)
    assert torch.allclose(c.norm(dim=1), torch.ones(4), atol=1e-5)


def _bank(n_per_class=300, dim=32):
    torch.manual_seed(0)
    mb = MemoryBank(dim, 3, device=DEV)
    for img, cls in ((1, 1), (2, 2), (3, 3)):
        base = torch.nn.functional.normalize(torch.randn(dim), dim=0)
        f = torch.nn.functional.normalize(base[:, None] + 0.05 * torch.randn(dim, n_per_class), dim=0)
        mb.add(img, f.reshape(dim, 10, n_per_class // 10), torch.full((10, n_per_class // 10), cls))
    return mb


def test_memory_add_remove_knn_and_io(tmp_path):
    mb = _bank()
    assert len(mb) == 900 and mb.stats()["per_class"] == {1: 300, 2: 300, 3: 300}
    q = mb.feats[:5].float()
    s, lab = mb.knn(q, 3)
    assert (lab[:, 0] == 1).all() and s.shape == (5, 3)
    assert mb.remove(2) == 300 and len(mb) == 600 and 2 not in mb.images
    mb.save(tmp_path / "m.pt")
    mb2 = MemoryBank.load(tmp_path / "m.pt", device=DEV)
    assert len(mb2) == 600 and set(mb2.images) == {1, 3}
    protos = mb2.prototypes(2)
    assert set(protos) == {1, 3}


def test_memory_compress_per_class():
    mb = _bank()
    rec = mb.compress(ratio=0.1)
    assert rec["before"] == 900 and rec["after"] == 90
    assert mb.stats()["per_class"] == {1: 30, 2: 30, 3: 30}
    rec = mb.compress(max_per_class=10)
    assert len(mb) == 30
    assert sum(info["kept"] for info in mb.images.values()) == 30


@pytest.fixture(scope="module")
def small_backbone():
    from semlabel.engine.backbone import get_backbone
    cfg = EngineConfig(model="small", image_size=256)
    return get_backbone(cfg.model, cfg.image_size, cfg.svd_components()), cfg


def _scene(seed, swap=False):
    """Synthetic 2-class scene: wood-like stripes vs. noisy blue carpet; halves swap between images."""
    rng = np.random.RandomState(seed)
    H, W = 192, 256
    yy, xx = np.mgrid[0:H, 0:W]
    stripes = np.stack([150 + 60 * np.sin(xx / 4.0), 90 + 40 * np.sin(xx / 4.0), 40 + 0 * xx], -1)
    carpet = np.stack([30 + 0 * xx, 60 + 0 * xx, 170 + 0 * xx], -1) + rng.randint(-40, 40, (H, W, 3))
    left = (xx < W // 2) if not swap else (xx >= W // 2)
    img = np.where(left[..., None], stripes, carpet).clip(0, 255).astype(np.uint8)
    mask = np.where(left, 1, 2).astype(np.uint8)
    return img, mask


@needs_weights
@pytest.mark.parametrize("mode", ["knn", "cluster", "hybrid", "smooth", "insid3"])
def test_segmenter_transfers_labels(small_backbone, mode):
    from semlabel.engine.segmenter import Segmenter
    bb, cfg = small_backbone
    ref, ref_mask = _scene(0)
    tgt, tgt_mask = _scene(1, swap=True)
    mem = MemoryBank(bb.dim, 2, device=bb.device)
    ft = bb.features(ref)
    mem.add(1, ft.fd, patch_labels(ref_mask, ft.grid, 2, cfg.purity))
    # the synthetic textures are 0.73-similar across halves, so cluster-based modes need a finer tau than
    # INSID3's 0.6 (which merges anything above 0.6 average similarity into one region)
    tau = 0.85 if mode in ("cluster", "insid3") else cfg.tau
    seg = Segmenter(bb, EngineConfig(**{**cfg.to_dict(), "mode": mode, "tau": tau}))
    p = seg.predict(tgt, mem, 2, refine="bilinear")
    assert p.labels.shape == tgt_mask.shape and p.conf.shape == tgt_mask.shape
    labeled = p.labels > 0
    acc = (p.labels[labeled] == tgt_mask[labeled]).mean()
    assert acc > 0.9, f"{mode}: accuracy {acc:.2f}"
    if mode != "insid3":
        assert labeled.mean() > 0.95


@needs_weights
def test_segmenter_empty_memory_raises(small_backbone):
    from semlabel.engine.segmenter import Segmenter
    bb, cfg = small_backbone
    with pytest.raises(ValueError):
        Segmenter(bb, cfg).predict(_scene(0)[0], MemoryBank(bb.dim, 2), 2)


def test_compress_relative_to_added_is_stable():
    mb = _bank()
    mb.compress(ratio=0.2, relative_to_added=True)
    assert len(mb) == 180
    mb.compress(ratio=0.2, relative_to_added=True)       # does not shrink again
    assert len(mb) == 180
    assert mb.added_per_class() == {1: 300, 2: 300, 3: 300}


def test_superpixels_large_and_degenerate():
    from semlabel.engine.segmenter import superpixels
    rng = np.random.RandomState(0)
    noise = (rng.rand(300, 400, 3) * 255).astype(np.uint8)
    sp = superpixels(noise, 500)
    assert sp.shape == (300, 400) and sp.max() + 1 >= 50          # never one segment for the fill tool
    big = np.zeros((2160, 3840, 3), np.uint8)
    big[:, 1920:] = 200
    sp = superpixels(big, 800)
    assert sp.shape == (2160, 3840)
    assert len(np.unique(sp[:, :1900])) and not (set(np.unique(sp[:, :1900])) & set(np.unique(sp[:, 1940:])))
