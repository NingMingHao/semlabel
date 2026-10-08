"""Command line: `semlabel serve | doctor | download-extras`."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import DATA_DIR, DINOV3_COMMIT, DINOV3_REPO, ENV_FILES, HOME, MODELS, WEIGHTS_DIR

OK, WARN, BAD = "\033[32m✓\033[0m", "\033[33m!\033[0m", "\033[31m✗\033[0m"
if not sys.stdout.isatty():
    OK, WARN, BAD = "[ok]", "[warn]", "[missing]"


def serve(a):
    import uvicorn
    from .server.app import create_app
    data = Path(os.path.expanduser(a.data)) if a.data else DATA_DIR
    print(f"SemLabel {__version__}: http://{a.host}:{a.port}   data: {data}", flush=True)
    uvicorn.run(create_app(data), host=a.host, port=a.port, log_level=a.log_level)


def doctor(a) -> int:
    """Check everything SemLabel needs and say how to fix what is missing."""
    problems = 0

    def line(mark, what, detail=""):
        print(f" {mark} {what}" + (f"  — {detail}" if detail else ""))

    print(f"SemLabel {__version__}  (python {sys.version.split()[0]})")
    print(f"  config files: {', '.join(map(str, ENV_FILES)) or 'none (defaults under ' + str(HOME) + ')'}")
    try:
        import torch
        line(OK, f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            line(OK, f"CUDA GPU: {torch.cuda.get_device_name(0)}", f"{total / 2 ** 30:.1f} GB, {free / 2 ** 30:.1f} GB free")
            if total < 7.5 * 2 ** 30:
                line(WARN, "less than 8 GB GPU memory", "use the 'base' or 'small' backbone (project settings)")
        else:
            line(WARN, "no CUDA GPU", "everything works on CPU, but auto-labeling is ~50x slower")
    except Exception as e:  # noqa: BLE001
        line(BAD, "PyTorch not importable", str(e))
        problems += 1
    # data dir
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        probe = DATA_DIR / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        line(OK, f"data directory {DATA_DIR}")
    except Exception as e:  # noqa: BLE001
        line(BAD, f"data directory {DATA_DIR} is not writable", str(e))
        problems += 1
    # DINOv3 code
    hub = DINOV3_REPO / "hubconf.py"
    if hub.exists():
        rev = ""
        if (DINOV3_REPO / ".git").exists() and shutil.which("git"):
            rev = subprocess.run(["git", "-C", str(DINOV3_REPO), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()
        if rev and rev != DINOV3_COMMIT:
            line(WARN, f"DINOv3 code at {DINOV3_REPO}", f"revision {rev[:10]} differs from the tested {DINOV3_COMMIT[:10]}")
        else:
            line(OK, f"DINOv3 code at {DINOV3_REPO}")
    else:
        line(BAD, f"DINOv3 code not found at {DINOV3_REPO}",
             f"git clone https://github.com/facebookresearch/dinov3 {DINOV3_REPO} "
             f"&& git -C {DINOV3_REPO} checkout {DINOV3_COMMIT[:10]}")
        problems += 1
    # weights
    have = [k for k, v in MODELS.items() if (WEIGHTS_DIR / v[1]).exists()]
    for k, (hub_name, fname, dim) in MODELS.items():
        mark = OK if k in have else (WARN if k != "large" else BAD)
        line(mark, f"weights '{k}' ({hub_name}, {dim}-d)", "found" if k in have else f"{WEIGHTS_DIR / fname} missing")
    if "large" not in have:
        print("     The default backbone is 'large'. Request the DINOv3 weights from Meta "
              "(https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and put the .pth files in "
              f"{WEIGHTS_DIR}" + ("; or switch projects to: " + ", ".join(have) if have else ""))
        problems += 0 if have else 1
    # python packages
    for mod, why in (("fastapi", "web server"), ("sklearn", "clustering"), ("skimage", "superpixels"),
                     ("cv2", "image ops"), ("pycocotools", "COCO export"), ("transformers", "SAM / SegFormer"),
                     ("mcap_ros2", "ROS 2 bag import")):
        try:
            __import__(mod)
            line(OK, f"python module {mod}", why)
        except Exception:  # noqa: BLE001
            mark = WARN if mod in ("mcap_ros2", "transformers") else BAD
            line(mark, f"python module {mod} missing", f"{why}; pip install -e .")
            problems += mark == BAD
    # optional models
    from .engine import sam
    line(OK if sam.available() else WARN, f"SAM click tool ({sam.HF_ID})",
         "ready" if sam.available() else "optional: semlabel download-extras")
    try:
        from huggingface_hub import try_to_load_from_cache
        from .store.prelabel import HF_ID as SEG_ID
        ok = isinstance(try_to_load_from_cache(SEG_ID, "config.json"), str)
        line(OK if ok else WARN, f"SegFormer pre-label ({SEG_ID})", "ready" if ok else "optional: semlabel download-extras")
    except Exception:  # noqa: BLE001
        pass
    if a.load and have:
        import numpy as np
        from .engine.backbone import Backbone
        k = "large" if "large" in have else have[0]
        try:
            bb = Backbone(k, 512)
            f = bb.features(np.zeros((240, 320, 3), np.uint8))
            line(OK, f"loaded '{k}' and extracted features", f"grid {tuple(f.f.shape[1:])} on {bb.device}")
        except Exception as e:  # noqa: BLE001
            line(BAD, f"loading '{k}' failed", str(e))
            problems += 1
    print("\nAll set — run: semlabel serve" if not problems else f"\n{problems} problem(s) to fix (see above).")
    return 1 if problems else 0


def download_extras(a):
    from huggingface_hub import snapshot_download
    from .engine.sam import HF_ID as SAM_ID
    from .store.prelabel import HF_ID as SEG_ID
    for rid in ([SAM_ID] if not a.no_sam else []) + ([SEG_ID] if not a.no_segformer else []):
        print("downloading", rid, flush=True)
        print("  ->", snapshot_download(rid))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="semlabel", description="In-context semantic labeling with DINOv3")
    ap.add_argument("--version", action="version", version=f"semlabel {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="start the web app (default)")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other machines")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--data", default=None, help=f"data directory (default {DATA_DIR})")
    s.add_argument("--log-level", default="info")
    d = sub.add_parser("doctor", help="check the installation")
    d.add_argument("--load", action="store_true", help="also load a backbone and extract features once")
    e = sub.add_parser("download-extras", help="download the optional SAM 2.1 and SegFormer weights")
    e.add_argument("--no-sam", action="store_true")
    e.add_argument("--no-segformer", action="store_true")
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or (args[0].startswith("-") and args[0] not in ("-h", "--help", "--version")):
        args = ["serve"] + args                      # `semlabel --port 9000` means serve
    a = ap.parse_args(args)
    if a.cmd == "serve":
        return serve(a)
    if a.cmd == "doctor":
        sys.exit(doctor(a))
    if a.cmd == "download-extras":
        return download_extras(a)


if __name__ == "__main__":
    main()
