# How SemLabel works

## Auto-labeling engine (`semlabel/engine/`)

For a target image (ViT-L/16, long side 1024 px: a 640 × 400 frame becomes a 64 × 40 patch grid):

1. **Features** (`backbone.py`): last-layer DINOv3 patch tokens, L2-normalised. Images keep their aspect
   ratio; INSID3 squashes to a square, which distorts thin structures in wide frames.
2. **Positional debiasing** (INSID3): DINOv3 features partly encode *where* a patch is. The features of
   a blank image span that positional subspace. Its top singular vectors (49 % of the dimensions,
   INSID3's 500 / 1024) are projected out, giving the *debiased* features used for matching across images.
3. **Backward matching → k-NN vote** (`segmenter.py`, `memory.py`): INSID3 keeps a target patch when its
   nearest reference patch lies inside the reference mask. With a multi-class bank this becomes a vote:
   the k = 10 nearest bank features (cosine, softmax-weighted with temperature 0.05) give each patch a
   probability per class.
4. **Forward matching**: similarity to per-class sub-prototypes (spherical k-means of the class's bank
   features, up to 4 per class), INSID3's "similar to the reference prototype" test.
5. **Clustering**: average-linkage agglomerative clustering of the target's own (not debiased) features at
   τ = 0.6. For each class, INSID3 picks a seed cluster (candidate patches that pass both matching tests,
   most similar to the class prototype). It then scores every cluster as cross-similarity × similarity to
   the seed × fraction of candidates.
6. **Modes**:
   - `knn`: the patch votes;
   - `cluster`: votes averaged per cluster;
   - `insid3`: the strict INSID3 rule, keeping a cluster above the 0.2 merge threshold and leaving others
     unlabeled;
   - `hybrid` (default): ½ patch vote + ½ cluster vote + the INSID3 score;
   - `smooth`: patch votes propagated to each patch's 16 most similar patches in the same image;
   - `smooth_insid3`: hybrid on smoothed votes.
7. **Pixels**: class probabilities are upsampled bilinearly (optionally averaged inside SLIC superpixels).
   Label = most probable class; confidence = its probability.

The **memory bank** stores, for every reference patch covered by one class for at least 60 % of its area,
the debiased feature (fp16), the class and the source image. **Compression** runs per class: random
projection to 128 dimensions, then k-center greedy (anomalib PatchCore) on the GPU. **Suggest** uses the
same greedy selection on whole-image descriptors (mean and standard deviation of the patch features),
seeded with the current references.

## Storage (`semlabel/store/`)

```
$SEMLABEL_DATA/
  semlabel.db                      SQLite: projects, classes, images, jobs, events, evals
  projects/<slug>/
    images/<id>.<ext>              imported originals
    thumbs/<id>.jpg                256 px thumbnails
    labels/<id>.png                human labels: palette PNG, pixel value = class index, 0 = unlabeled
    preds/<id>.png, <id>_conf.png  latest auto-label and its confidence (uint8, 255 = 1.0)
    memory/<model>_<size>_svd<k>.pt  memory banks (torch.save)
    exports/*.zip
```

## Server (`semlabel/server/`)

FastAPI serves the API and the static single-page app (`semlabel/web/`, plain ES modules, no build
step). Long operations run as **jobs** on one background worker thread (GPU work is serialised); progress
and cancellation are stored in SQLite. Masks travel as raw bytes (one `uint8` class index per pixel,
row-major), gzip-compressed.

Main endpoints (interactive docs at `/docs` on a running server):

| | |
|---|---|
| projects | `GET/POST /api/projects`, `GET/PATCH/DELETE /api/projects/{pid}`, `GET …/events` |
| classes | `GET/POST …/classes`, `PATCH/DELETE …/classes/{idx}?merge_into=` |
| images | `GET …/images?status=&sort=&q=&ids=&in_memory=`, `POST …/images` (multipart), `GET/PATCH/DELETE …/images/{iid}`, `GET …/file`, `GET …/thumb` |
| labels | `GET/PUT/DELETE …/images/{iid}/label` (raw bytes; `PUT ?status=&add_to_memory=`), `GET …/label.png` |
| auto-labels | `POST …/images/{iid}/predict`, `GET …/pred`, `GET …/conf`, `POST …/accept`, `POST …/prelabel` |
| tools | `POST …/images/{iid}/wand`, `POST …/sam`, `GET …/superpixels` |
| memory | `GET …/memory`, `POST …/memory/add`, `…/remove`, `…/compress`, `…/rebuild`, `…/clear` |
| jobs | `POST …/jobs {kind, params}` with kind = `predict`, `prelabel`, `suggest`, `memory_add`, `memory_rebuild`, `memory_compress`, `evaluate`, `import_folder`, `import_video`, `import_bag`, `import_masks`, `export`; `GET /api/jobs[/{id}]`, `POST /api/jobs/{id}/cancel` |
| other | `GET …/evals`, `GET …/exports[/{name}]`, `GET /api/bag_topics?path=`, `GET /api/browse?path=`, `GET /api/system` |

(`…` = `/api/projects/{pid}`.)

## Code map

```
semlabel/
  cli.py              semlabel serve | doctor | download-extras
  config.py           paths (env / semlabel.env), model registry, EngineConfig
  engine/             backbone.py, memory.py, segmenter.py, sam.py
  store/              db.py, project.py, io.py (import/export), prelabel.py, files.py
  server/             app.py (FastAPI), jobs.py (worker)
  web/                index.html, app.js (views), editor.js (canvas editor), api.js, ui.js, style.css
  data/               ade20k_to_indoor.json (pre-label class mapping + indoor palette)
scripts/              install.sh, prepare_nyu.py, bench_engine.py
bench/                run_sweeps.sh and the published results (*.json)
tests/                pytest suites, Playwright UI scripts
```
