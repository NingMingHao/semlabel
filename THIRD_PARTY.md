# Third-party components and licences

SemLabel's own code is under the Apache License 2.0 (see `LICENSE`). It builds on the following work.
Nothing below is vendored in this repository except where noted; models and model code are fetched at
install time or on first use, under their own terms.

| component | how SemLabel uses it | licence |
|---|---|---|
| [INSID3](https://github.com/visinf/INSID3) (Cuttano et al., CVPR 2026) | The in-context segmentation recipe (positional debiasing by SVD of blank-image features, forward / backward matching, agglomerative clustering with seed-and-aggregate scoring) is re-implemented and generalised to multi-class labeling in `semlabel/engine/`. | Apache-2.0 |
| [DINOv3](https://github.com/facebookresearch/dinov3) (Siméoni et al., 2025) | Model code is cloned by `scripts/install.sh` (not vendored) and loaded through `torch.hub`; weights must be requested from Meta and placed in the weights folder. | DINOv3 License (code and weights); read it before use, it has its own terms for use and redistribution |
| [anomalib](https://github.com/open-edge-platform/anomalib) PatchCore | The k-center-greedy coreset with random projection is re-implemented in `semlabel/engine/memory.py` for memory-bank compression. | Apache-2.0 |
| [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large) via [transformers](https://github.com/huggingface/transformers) | Optional click-to-segment tool; weights downloaded on request (`semlabel download-extras`). | Apache-2.0 |
| [SegFormer-B4 ADE20K](https://huggingface.co/nvidia/segformer-b4-finetuned-ade-512-512) | Optional pre-labeling (ADE20K classes mapped to project classes); weights downloaded on request. | NVIDIA's SegFormer licence (listed as "other" on the model card): it restricts commercial use, so check it before using pre-labeling commercially |
| [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) via [tanganke/nyuv2](https://huggingface.co/datasets/tanganke/nyuv2) | Benchmark only: `scripts/prepare_nyu.py` downloads it; the screenshots in `docs/images/` show NYUv2 images. Not redistributed otherwise. | NYU Depth V2 dataset terms |
| `semlabel/data/ade20k_to_indoor.json` | ADE20K-150 to 16-group indoor mapping and palette, written for this project. | Apache-2.0 (this repository) |

Python dependencies (PyTorch, FastAPI, scikit-learn, scikit-image, OpenCV, pycocotools, mcap, …) are
installed from PyPI under their own licences.
