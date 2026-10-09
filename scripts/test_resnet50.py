
"""Smoke test for pretrained AstraMed ResNet-50."""

import torch

from src.data.advanced_input import AdvancedInputAdapter
from src.models.resnet import build_resnet50


def main():
    print("=" * 60)
    print("ASTRAMED RESNET-50 SMOKE TEST")
    print("=" * 60)

    # Check whether CUDA is actually available to this PyTorch install.
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "CUDA is unavailable. Running on CPU; "
            "this does not verify GPU execution."
        )

    # 1. Build ResNet-50 with ImageNet-pretrained weights.
    print("\n[1/5] Loading pretrained ResNet-50...")

    model = build_resnet50(pretrained=True)
    model = model.to(device)
    model.eval()

    print("Pretrained ResNet-50 loaded successfully.")

    # 2. Count parameters.
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\n[2/5] Parameter count")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 3. Simulate one grayscale RSNA image.
    # The existing Dataset returns [B, 1, H, W].
    print("\n[3/5] Testing grayscale input adapter...")

    grayscale_images = torch.rand(
        1, 1, 512, 512,
        dtype=torch.float32,
    )

    adapter = AdvancedInputAdapter(
        image_size=224,
        use_imagenet_normalization=True,
    ).to(device)

    with torch.no_grad():
        model_images = adapter(
            grayscale_images.to(device)
        )

    print("Adapted input shape:", tuple(model_images.shape))

    assert model_images.shape == (1, 3, 224, 224)
    assert model_images.dtype == torch.float32

    # 4. Run one inference forward pass.
    print("\n[4/5] Running forward pass...")

    with torch.no_grad():
        logits = model(model_images)

    print("Output shape:", tuple(logits.shape))
    print("Raw logit:", logits.item())

    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all().item()

    print("Forward pass successful.")

    # 5. Report GPU execution and memory if CUDA was used.
    print("\n[5/5] Device compatibility")

    if cuda_available:
        print("Model device:", next(model.parameters()).device)
        print(
            "Peak allocated GPU memory: "
            f"{torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MiB"
        )
        print("CUDA forward pass successful.")
    else:
        print("CPU forward pass successful.")
        print("GPU compatibility remains unverified.")

    print("\nRESNET-50 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
