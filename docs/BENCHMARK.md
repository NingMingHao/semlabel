# Engine benchmark (NYUv2-13)

How the in-context engine was chosen and tuned. Everything here is reproducible with
`scripts/bench_engine.py` and `bench/run_sweeps.sh`; raw results are in `bench/*.json`.

## Setup

- **Data**: NYUv2 validation split, 13-class labels (bed, books, ceiling, chair, floor, furniture,
  objects, picture, sofa, table, tv, wall, window), 288 × 384 px, from `tanganke/nyuv2`
  (`scripts/prepare_nyu.py`). Void pixels are ignored.
- **Protocol**: a random permutation (seed 0 unless noted) gives N reference images, whose ground-truth
  masks go into the memory bank, and the next 100 images, which are auto-labeled and scored. Metric:
  mean IoU over the 13 classes, accumulated over the 100 images.
- **Hardware**: RTX 3080 Ti, bf16 inference. Times are per image, median.
- The target use case, low-viewpoint indoor robot frames (640 × 400, camera ~8 cm above the floor), has no ground truth,
  so NYUv2 is the stand-in. Treat the absolute numbers as indicative; the comparisons are what matter.

## Backbone: ViT-L/16 is the best fit

mIoU by number of reference images (input long side 1024 px, debiasing 49 % of dims, k = 10):

| DINOv3 backbone | features | 5 refs | 10 refs | 20 refs | 40 refs |
|---|---|---|---|---|---|
| ViT-S/16 (21 M) | 21 ms | 37.3 | 45.1 | 47.9 | 53.3 |
| ViT-B/16 (86 M) | 35 ms | 45.8 | 52.0 | 54.9 | 61.4 |
| **ViT-L/16 (300 M)** | 86 ms | **49.6** | **56.7** | **59.1** | **63.8** |
| ViT-H+/16 (840 M) | 175 ms | 46.2 | 52.4 | 56.7 | 61.7 |

(best of the knn / cluster / hybrid modes for each cell; per-mode numbers are in `bench/model_*.json`.)
ViT-L wins at every budget and is twice as fast as ViT-H+, which matches INSID3's own default. The 7B
model does not fit 12 GB. ViT-S / ViT-B are selectable per project when speed matters more.

## Positional debiasing (INSID3) matters: +5 mIoU

ViT-L, 10 references, held-out images `order[10:110]`; the fraction of feature dimensions projected out
as the positional subspace:

| debiasing | kNN | hybrid |
|---|---|---|
| none (0 %) | 48.6 | 49.5 |
| 25 % | 53.3 | 54.0 |
| **49 % (INSID3: 500 of 1024)** | **53.6** | **54.5** |
| 75 % | 53.1 | 54.2 |

Removing the positional subspace is worth about +5 mIoU; the exact size matters little between 25 % and
75 %. INSID3's 500 components are kept as a fraction (0.49 × feature dim) so they transfer to the other
backbone widths; a fixed 500 would remove every dimension of ViT-S (384-d).

## Labeling mode

ViT-L, bilinear upsampling:

| mode | 10 refs | 40 refs | time / image |
|---|---|---|---|
| knn — per-patch memory vote (INSID3 backward matching, multi-class) | 55.0 | 62.9 | 8–19 ms |
| smooth — knn + soft affinity propagation inside the image | 55.4 | 63.5 | 9–20 ms |
| cluster — votes averaged per INSID3 agglomerative cluster | 56.7 | 62.5 | ~150 ms |
| **hybrid** — votes + cluster votes + INSID3 seed/aggregate score (default) | **56.5** | 63.8 | ~150–200 ms |
| smooth_insid3 — hybrid on smoothed votes | 56.4 | **64.2** | ~200 ms |
| insid3 — strict per-class INSID3, unclaimed regions stay unlabeled | 29.2* (labels 33 % of pixels, ~88 % correct) | – | ~150 ms |

\* first smoke run, 40 held-out images.

The strict INSID3 rule (keep a cluster only if its seed-and-aggregate score exceeds 0.2) is built for
one object per prompt; for dense multi-class labeling it leaves two thirds of the image empty. Its score
is therefore used as extra evidence in `hybrid` rather than as a gate. `insid3` remains available as a
high-precision mode.

Other parameters (ViT-L, 10 references, held-out `order[10:110]`, same images for every row):

- k ∈ {5, 10, 20, 40} and τ ∈ {0.5, 0.6, 0.7}: hybrid stays within 54.1–54.6, so neither needs tuning;
  `cluster` mode alone drops at τ = 0.5 (52.1–53.3) and is best at INSID3's τ = 0.6.
- Input long side: 768 px 53.8 / 54.8 (kNN / hybrid), 1024 px 53.6 / 54.5, 1280 px 54.0 / 54.9, with
  features taking 46 / 86 / 152 ms. NYUv2 frames are only 384 px wide, so upscaling adds nothing here;
  1024 px (INSID3's default) is kept for 640 × 400 robot frames, and 768 px is a safe 2× speed-up.
- Superpixel boundary snapping: within ±0.2 mIoU on NYUv2's low-resolution labels, so the default is
  bilinear; it remains an option for crisper edges.

Protocol note: in these runs the held-out set was the 100 images after the largest reference count of a run, so the
backbone and mode tables (runs up to 40 references) and this section (runs with 10 only) use different
held-out images. Compare numbers within a table, not across tables. `bench_engine.py` now takes the held-out images from the end of the permutation, so new runs share one held-out set.

## Coreset compression of the memory bank (anomalib PatchCore k-center greedy)

ViT-L, 40 references, three random splits. The bank is compressed per class:

| kept | features | seed 0 knn / hybrid | seed 1 knn / hybrid | seed 2 knn / hybrid | kNN time |
|---|---|---|---|---|---|
| 100 % | ~105 k | 62.9 / 63.8 | 65.8 / 65.6 | 63.0 / 63.5 | 17–25 ms |
| 50 % | ~52 k | 63.5 / 64.7 | 65.5 / 65.2 | 63.0 / 62.7 | 10–16 ms |
| **20 %** | ~21 k | 64.6 / 65.3 | 66.1 / 65.7 | 63.3 / 63.9 | 6–10 ms |
| 10 % | ~10.5 k | 64.9 / 64.9 | 65.7 / 65.9 | 63.5 / 63.7 | 4 ms |

Keeping 20 % is accuracy-neutral to slightly positive on random splits (+0.1 to +1.7 mIoU, mean +0.7),
5× smaller and 3–6× faster to search: near-duplicate patches (the same floor seen in 40 frames) stop
crowding the k nearest neighbours. On a split where held-out frames are near-copies of references
(NYUv2 file order, consecutive frames of the same room) it cost 1.7 mIoU (75.6 → 73.9), because exact
look-alikes are what got removed. Compression is therefore manual or opt-in (`auto_compress`), and
"Rebuild" restores the full bank from the reference labels at any time.

## Throughput in the platform (ViT-L, hybrid)

- 237 frames (two camera topics, one per second) imported from a 12 GB ROS 2 bag in 14 s.
- Batch auto-labeling: 0.20–0.30 s per image including disk I/O (225 robot frames of 640 × 400 in 49 s;
  3840 × 2160 frames ≈ 0.8 s).
- Single image from the editor: ~0.6 s for the first call on an image, then cached features.
- "Suggest images to label" on 237 images: 37 s the first time (extracts and caches every image's features).
