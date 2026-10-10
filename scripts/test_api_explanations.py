import numpy as np
from PIL import Image
from io import BytesIO
import base64

from src.api.services.explanation_compatibility import (
    validate_explanation_compatibility,
)
from src.api.services.explanation_service import (
    ExplanationServiceError,
    render_explanation,
)


def test_resnet_gradcam_is_supported():
    validate_explanation_compatibility("resnet50", "gradcam")


def test_vit_attention_is_supported():
    validate_explanation_compatibility("vit_b16", "attention")


def test_resnet_attention_is_rejected():
    try:
        validate_explanation_compatibility("resnet50", "attention")
    except ExplanationServiceError as exc:
        assert exc.code == "INCOMPATIBLE_EXPLANATION_METHOD"
        assert exc.status_code == 422
    else:
        raise AssertionError("ResNet-50 attention should be rejected.")


def test_vit_gradcam_is_rejected():
    try:
        validate_explanation_compatibility("vit_b16", "gradcam")
    except ExplanationServiceError as exc:
        assert exc.code == "INCOMPATIBLE_EXPLANATION_METHOD"
        assert exc.status_code == 422
    else:
        raise AssertionError("ViT Grad-CAM should be rejected.")


def test_render_heatmap_and_overlay():
    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[70:150, 70:150] = 1.0

    original = np.full((512, 512), 100, dtype=np.float32)

    images = render_explanation(
        heatmap=heatmap,
        original_grayscale=original,
        include_overlay=True,
    )

    assert images["heatmap_base64"]
    assert images["overlay_base64"]

    decoded = base64.b64decode(images["heatmap_base64"])
    rendered = Image.open(BytesIO(decoded))
    assert rendered.format == "PNG"
    assert rendered.size == (512, 512)


def test_overlay_can_be_disabled():
    images = render_explanation(
        heatmap=np.ones((224, 224), dtype=np.float32),
        original_grayscale=np.ones((512, 512), dtype=np.float32),
        include_overlay=False,
    )

    assert images["heatmap_base64"]
    assert images["overlay_base64"] is None


if __name__ == "__main__":
    tests = [
        test_resnet_gradcam_is_supported,
        test_vit_attention_is_supported,
        test_resnet_attention_is_rejected,
        test_vit_gradcam_is_rejected,
        test_render_heatmap_and_overlay,
        test_overlay_can_be_disabled,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("M4 service tests passed.")