"""Shared test helpers: tests that need DINOv3 weights skip when the small backbone is not installed."""
import pytest

from semlabel.config import MODELS, WEIGHTS_DIR

HAVE_SMALL = (WEIGHTS_DIR / MODELS["small"][1]).exists()
needs_weights = pytest.mark.skipif(not HAVE_SMALL, reason=f"DINOv3 ViT-S weights not in {WEIGHTS_DIR}")
