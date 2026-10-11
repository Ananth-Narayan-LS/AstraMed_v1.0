"""Overlay rendering checks (no model needed): cold regions must not be tinted."""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.services.explanation_service import render_explanation


def _decode(encoded):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))


IMAGE = np.random.default_rng(0).integers(0, 4000, (300, 500)).astype(np.float32)


def test_zero_activation_leaves_original_untouched():
    result = render_explanation(np.zeros((224, 224), np.float32), IMAGE, True)
    assert np.array_equal(
        _decode(result["original_image_base64"]), _decode(result["overlay_base64"])
    )


def test_hot_region_changes_only_where_heat_is():
    heat = np.zeros((224, 224), np.float32)
    heat[100:140, 100:140] = 1.0
    result = render_explanation(heat, IMAGE, True)
    diff = np.abs(
        _decode(result["overlay_base64"]).astype(int)
        - _decode(result["original_image_base64"]).astype(int)
    ).max(axis=-1)
    assert diff[:100, :100].max() == 0          # far from the hot region (top-left of the stretched map)
    assert diff.max() > 0


def test_sizes_match_original_and_contract_fields():
    result = render_explanation(np.eye(224, dtype=np.float32), IMAGE, True)
    assert (result["width"], result["height"]) == (500, 300)
    for key in ("original_image_base64", "heatmap_base64", "overlay_base64"):
        assert _decode(result[key]).shape == (300, 500, 3)
    assert render_explanation(np.eye(224, dtype=np.float32), IMAGE, False)["overlay_base64"] is None


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS:", test.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
