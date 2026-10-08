# SemLabel user guide

## Concepts

- **Project**: a set of images, a class list and its own engine settings.
- **Class**: a name, a colour and a fixed index (the pixel value in exported index masks; 0 = unlabeled).
  Renaming never changes the index.
- **Image status**:

  | status | meaning |
  |---|---|
  | New | imported, no label |
  | Auto-labeled | has an auto-label, not checked by a person |
  | In progress | a person saved a label but has not submitted it |
  | Reviewed | a person submitted the label |

- **Reference**: an image whose label is in the **memory bank**, marked ★ in the gallery. Auto-labeling
  only knows what the references show.
- **Memory bank**: the DINOv3 patch features of every reference, with their classes. There is one bank per
  backbone and input size, shown as a key such as `large_1024_svd502`.

## Workflow

1. **Import** (*Import / Export* tab):
   - drag and drop files;
   - a folder on the server (*Browse* helps), with *every Nth file*;
   - a video, sampled every N seconds;
   - a ROS 2 bag (`.mcap` file or bag folder). Use *List bag topics* and tick the camera topics; depth
     topics are hidden. Frames are sampled every N seconds per topic.

   Identical files are skipped.
2. **Choose what to label first**: in *Images*, *Suggest images to label* runs a k-center greedy
   selection over whole-image DINOv3 descriptors, seeded with the current references. The suggestions are
   the images least like anything already in the bank. They are selected and filtered so you can open
   them in order.
3. **Label references** in the editor (below). Label the whole image; unlabeled pixels are simply ignored.
   With the *Indoor* class set, *Pre-label (SegFormer)* fills unlabeled pixels with a closed-set model's
   guess as a starting point.
4. **Submit + add to memory** (Ctrl+Shift+Enter) marks the image reviewed and adds it to the bank.
   Editing a reference later updates its bank entry automatically.
5. **Auto-label**: *Auto-label new* in the gallery (background job), or `A` in the editor.
6. **Review**: filter *Auto-labeled*, sort *Lowest confidence first*, open the first image. The
   auto-label loads as an editable label, so fix it and press Ctrl+Enter (Submit), and the next image
   opens. Use Ctrl+Shift+Enter instead when the image shows something the bank was missing.
   *Re-label auto-labeled* refreshes older auto-labels after the bank has grown.
7. **Evaluate** (*Evaluate* tab): every reviewed image that is **not** a reference is auto-labeled and
   compared with its label (mIoU per class, pixel accuracy, history against bank size). Keep some reviewed
   images out of the bank so this has data.
8. **Compress** the bank when it grows (below) and **Export**.

## Editor

| tool | key | use |
|---|---|---|
| Brush / eraser | `B` / `E` | paint the current class / clear; `[` `]` change the size |
| Polygon | `P` | click points; double-click, Enter or click the first point to close; Backspace undoes a point; Esc cancels |
| Superpixel fill | `F` | click or drag to paint whole SLIC segments (snaps to image edges) |
| Bucket fill | `G` | relabel the connected region that has the clicked pixel's label: the fastest fix for a wrong auto-label region |
| Magic wand | `W` | selects the connected region whose DINOv3 features resemble the clicked point (threshold slider); Shift-click selects every similar region in the image |
| SAM | `S` | left click adds a positive point, right click a negative one; the preview updates; Enter applies the current class, Esc clears |
| Pick class | `I` or Alt-click | take the class under the cursor |
| Pan | `H`, Space-drag, middle-drag | wheel zooms, `Z` fits the image |

Other keys: `1`–`9`, `0` select classes 1–10 · `X` protect labeled pixels (paint only unlabeled ones) ·
`A` auto-label (fills only unlabeled pixels if some are labeled) · `C` low-confidence overlay · `V`
hide/show labels · `O` outline mode · Ctrl+Z / Ctrl+Shift+Z undo / redo · Ctrl+S save (in progress) ·
Ctrl+Enter submit · Ctrl+Shift+Enter submit + add to memory · Alt+← / Alt+→ previous / next image in the
gallery's current filter and order. Right-click a class in the list to hide it.

The *Show* menu switches between the editable label, the last auto-label and the low-confidence overlay
(red where confidence is below the threshold slider). Your edits are saved as *In progress* when you
leave an image any way (next / previous, gallery, browser back); an auto-label you only looked at stays an
auto-label.

## Memory bank

The *Memory bank* tab shows the number of references and features, features per class (and classes with
no example yet), every reference with the features it still contributes, and the compression history.

- **Add / remove references** from the editor, or for many images at once from the gallery selection
  bar.
- **Coreset compression** keeps, per class, the features chosen by k-center greedy (the PatchCore method
  used in anomalib). Each pick is the feature farthest from everything already kept, so many copies of
  the same wall collapse to a few while rare appearances stay.

  On NYUv2 with 40 references, keeping 20 % changed mIoU by +0.1 to +1.5 on three random splits and made
  matching 3–6× faster. It cost 1.7 when held-out images were near-copies of references.
- **Auto-compress** (project *Settings*): after every new reference, keep each class at a fixed
  fraction of all features ever added. Repeated compression does not shrink old features further.
- **Rebuild from references** re-extracts every reference with the current backbone. Use it after
  changing the backbone or input size, or to undo compression.

## Classes

Add, rename and recolour classes in the *Classes* tab. Deleting a class asks which class its pixels
should become (or unlabeled) and rewrites every label, auto-label and memory entry.

## Import and export formats

**Export** (zip, *Import / Export* tab). Choose reviewed images only, every saved label, or saved labels
plus auto-labels for the rest; with or without the images.

| format | content |
|---|---|
| Index PNG | `masks/<name>.png`, pixel value = class index, plus `classes.json` |
| Colour PNG | `color_masks/<name>.png` in class colours |
| COCO | `annotations/coco_semantic.json`: one RLE annotation per class and image (`iscrowd: 1`), categories with colours |
| CVAT | `cvat_mask/`: CVAT *Segmentation mask 1.1* (`labelmap.txt`, `SegmentationClass/`, `ImageSets/`), importable into CVAT |

**Import existing labels**. Masks are matched to images by file name.

- Index PNG folder: with a `classes.json` listing names in index order, classes are matched by name
  and created if missing.
- Colour PNG folder: uses a `labelmap.txt`, or else the project colours.
- A CVAT *Segmentation mask 1.1* zip.

## Settings

| setting | default | notes |
|---|---|---|
| DINOv3 model | large | `small` / `base` for speed or small GPUs; `huge_plus` was not better in the benchmark |
| Input long side | 1024 px | multiple of 16; 768 is ~2× faster with similar accuracy on small images |
| Positional debiasing | 0.49 | fraction of feature dimensions removed (INSID3) |
| Mode | hybrid | `smooth` is ~10× faster for large batches; `insid3` leaves uncertain regions unlabeled |
| Boundary refinement | bilinear | `superpixel` snaps label edges to image edges |
| Neighbours k, temperature | 10, 0.05 | accuracy is insensitive to k between 5 and 40 |
| Cluster threshold τ, merge threshold | 0.6, 0.2 | INSID3 defaults |
| Patch purity | 0.6 | a reference patch enters the bank only if one class covers this fraction of it |
| Leave pixels below this confidence unlabeled | 0 | raise it to keep only confident auto-labels |
| Auto-compress | 0 (off) | e.g. 0.2 |

Changing the model, input size or debiasing creates a new bank key: rebuild the bank from the *Memory
bank* tab.

## Keep it running

Background: `setsid nohup ./semlabel.sh > ~/.semlabel/server.log 2>&1 &`, stop with `pkill -f semlabel`.

As a systemd user service, create `~/.config/systemd/user/semlabel.service` (adjust the path), then
run `systemctl --user enable --now semlabel`:

```ini
[Unit]
Description=SemLabel labeling server

[Service]
ExecStart=%h/semlabel/semlabel.sh --port 8765
Restart=on-failure

[Install]
WantedBy=default.target
```

To label from another computer, start with `--host 0.0.0.0` and open `http://<server>:8765`. There is
no login, so only do this on a network you trust.

## Troubleshooting

- **Anything odd**: run `./semlabel.sh doctor --load`. It checks the GPU, folders, DINOv3 code and
  weights, Python modules and optional models, and says how to fix each problem.
- **"memory bank is empty"**: auto-labeling needs at least one reference. Label an image and use
  *Submit + add to memory*.
- **A class is never predicted**: no reference contains it. The *Memory bank* tab lists classes without
  examples.
- **Bank needs a rebuild**: the backbone settings changed. Use *Memory bank → Rebuild from references*.
- **CUDA out of memory**: switch the project to `base` or lower the input size; close other GPU programs.
- **Errors mentioning ROS packages or pytest plugins**: start through `semlabel.sh` (or `env -u
  PYTHONPATH …`); ROS 2 puts its packages on `PYTHONPATH`.
- **SAM or pre-label buttons report missing weights**: `./semlabel.sh download-extras`.
