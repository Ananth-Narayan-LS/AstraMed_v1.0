
"""Smoke test for the AstraMed model factory."""

import gc

import torch

from src.models.model_factory import build_model


ADVANCED_MODELS = [
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_b16",
]


def main():
    # Test on CPU to avoid requiring a GPU.
    device = torch.device("cpu")

    # Simulate one adapted X-ray: [B, 3, 224, 224].
    images = torch.randn(1, 3, 224, 224)

    for model_name in ADVANCED_MODELS:
        print(f"\nTesting {model_name}...")

        # False avoids downloading pretrained weights during this test.
        model = build_model(
            model_name=model_name,
            pretrained=False,
        ).to(device)

        model.eval()

        with torch.no_grad():
            outputs = model(images.to(device))

        print("Output shape:", tuple(outputs.shape))

        assert outputs.shape == (1, 1), (
            f"{model_name} returned {tuple(outputs.shape)}, "
            "expected (1, 1)"
        )

        print(f"{model_name}: PASSED")

        del model, outputs
        gc.collect()

    print("\nAll advanced model factory tests PASSED.")


if __name__ == "__main__":
    main()
