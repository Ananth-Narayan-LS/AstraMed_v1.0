
"""Smoke test for pretrained AstraMed DenseNet-121."""

import torch

from src.data.advanced_input import AdvancedInputAdapter
from src.models.densenet import build_densenet121


def main():
    print("=" * 60)
    print("ASTRAMED DENSENET-121 SMOKE TEST")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load pretrained DenseNet-121.
    print("\n[1/5] Loading pretrained DenseNet-121...")

    model = build_densenet121(pretrained=True).to(device)
    model.eval()

    print("Pretrained DenseNet-121 loaded successfully.")

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

    # 3. Adapt a simulated grayscale X-ray.
    print("\n[3/5] Testing grayscale input adapter...")

    grayscale_images = torch.rand(
        1, 1, 512, 512,
        dtype=torch.float32,
    ).to(device)

    adapter = AdvancedInputAdapter(
        image_size=224,
        use_imagenet_normalization=True,
    ).to(device)

    with torch.no_grad():
        model_images = adapter(grayscale_images)

    print("Original input shape: (1, 1, 512, 512)")
    print("Adapted input shape:", tuple(model_images.shape))

    assert model_images.shape == (1, 3, 224, 224)
    assert model_images.dtype == torch.float32

    # Reset peak statistics immediately before model inference.
    if cuda_available:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # 4. Forward pass.
    print("\n[4/5] Running forward pass...")

    with torch.no_grad():
        logits = model(model_images)

    if cuda_available:
        torch.cuda.synchronize()

    print("Output shape:", tuple(logits.shape))
    print("Raw logit:", logits.item())

    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all().item()

    print("Forward pass successful.")

    # 5. Report memory and device verification.
    print("\n[5/5] Device and memory verification")

    if cuda_available:
        allocated_mb = (
            torch.cuda.memory_allocated(device) / (1024 ** 2)
        )
        peak_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

        print("Model device:", next(model.parameters()).device)
        print(f"Current allocated GPU memory: {allocated_mb:.1f} MiB")
        print(f"Peak GPU memory during forward: {peak_mb:.1f} MiB")
        print("CUDA forward pass successful.")
    else:
        print("CPU forward pass successful.")
        print("GPU compatibility remains unverified.")

    print("\nDENSENET-121 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
