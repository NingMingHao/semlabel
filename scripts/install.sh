#!/usr/bin/env bash
# SemLabel installer for Linux.
#
#   scripts/install.sh [--venv PATH] [--torch cu126|cu128|cpu] [--extras] [--test] [--python 3.10]
#
#   --venv     virtual environment to create or reuse          (default: .venv in the repo)
#   --torch    PyTorch wheel flavour from download.pytorch.org (default: cu126; cpu for no GPU)
#   --extras   also download the optional SAM 2.1 and SegFormer weights (~1.2 GB, Hugging Face)
#   --test     also install the test tools (pytest, Playwright)
#
# What it does: creates the venv (uv if available, else python -m venv), installs PyTorch and
# SemLabel, clones the DINOv3 model code at the tested revision into $SEMLABEL_HOME/dinov3, creates
# $SEMLABEL_HOME/{data,weights} and a semlabel.env, then runs `semlabel doctor`.
# The DINOv3 weights are not downloadable without accepting Meta's licence; see the README.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
VENV="$REPO/.venv"
TORCH="cu126"
EXTRAS=0
TEST=0
PY="3.10"
while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$(realpath -m "$2")"; shift 2 ;;
    --torch) TORCH="$2"; shift 2 ;;
    --extras) EXTRAS=1; shift ;;
    --test) TEST=1; shift ;;
    --python) PY="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
unset PYTHONPATH   # ROS 2 environments put packages here that would shadow the venv's

SEMLABEL_HOME="${SEMLABEL_HOME:-$HOME/.semlabel}"
DINOV3_REPO="${SEMLABEL_DINOV3_REPO:-$SEMLABEL_HOME/dinov3}"
WEIGHTS="${SEMLABEL_WEIGHTS:-$SEMLABEL_HOME/weights}"
DATA="${SEMLABEL_DATA:-$SEMLABEL_HOME/data}"
COMMIT="$(sed -n 's/^DINOV3_COMMIT = "\([0-9a-f]*\)".*/\1/p' semlabel/config.py)"
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "virtual environment: $VENV"
if command -v uv >/dev/null 2>&1; then
  [ -x "$VENV/bin/python" ] || uv venv "$VENV" --python "$PY"
  PIP=(uv pip install --python "$VENV/bin/python")
else
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
fi

say "PyTorch ($TORCH)"
"${PIP[@]}" torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH"

say "SemLabel"
if [ "$TEST" = 1 ]; then "${PIP[@]}" -e ".[test]"; else "${PIP[@]}" -e .; fi

say "DINOv3 model code -> $DINOV3_REPO"
if [ ! -f "$DINOV3_REPO/hubconf.py" ]; then
  git clone --quiet https://github.com/facebookresearch/dinov3 "$DINOV3_REPO"
fi
if [ -d "$DINOV3_REPO/.git" ] && [ -n "$COMMIT" ]; then
  git -C "$DINOV3_REPO" fetch --quiet origin "$COMMIT" 2>/dev/null || true
  git -C "$DINOV3_REPO" checkout --quiet "$COMMIT" && echo "checked out $COMMIT"
fi

say "folders and settings in $SEMLABEL_HOME"
mkdir -p "$DATA" "$WEIGHTS"
if [ ! -f "$SEMLABEL_HOME/semlabel.env" ]; then
  cat > "$SEMLABEL_HOME/semlabel.env" <<ENV
# SemLabel settings (read by semlabel and semlabel.sh). Real environment variables take precedence.
SEMLABEL_VENV=$VENV
SEMLABEL_DATA=$DATA
SEMLABEL_WEIGHTS=$WEIGHTS
SEMLABEL_DINOV3_REPO=$DINOV3_REPO
# SEMLABEL_SAM=facebook/sam2.1-hiera-large
ENV
  echo "wrote $SEMLABEL_HOME/semlabel.env"
fi

if [ "$EXTRAS" = 1 ]; then
  say "optional SAM 2.1 + SegFormer weights"
  "$VENV/bin/python" -m semlabel download-extras
fi
if [ "$TEST" = 1 ]; then
  say "Playwright browser for the UI tests"
  "$VENV/bin/python" -m playwright install chromium || echo "(skipped: run '$VENV/bin/python -m playwright install chromium' later)"
fi

say "check"
"$VENV/bin/python" -m semlabel doctor || true
cat <<MSG

Next:
  1. Put DINOv3 weights in $WEIGHTS (at least dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth).
     Request access: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
  2. Start:  ./semlabel.sh        then open http://127.0.0.1:8765
MSG
