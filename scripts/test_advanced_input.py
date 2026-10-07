import torch

from src.data.advanced_input import AdvancedInputAdapter


def main():
    print("Testing Advanced Input Pipeline")
    print("=" * 50)

    # Simulate output from RSNADataset.
    images = torch.randn(
        8,
        1,
        512,
        512,
    )

    print("Original shape:")
    print(images.shape)

    adapter = AdvancedInputAdapter(
        image_size=224,
        use_imagenet_normalization=True,
    )

    adapted_images = adapter(images)

    print("\nAdapted shape:")
    print(adapted_images.shape)

    print("\nExpected:")
    print("(8, 3, 224, 224)")

    assert adapted_images.shape == (
        8,
        3,
        224,
        224,
    )

    assert adapted_images.dtype == torch.float32

    print("\nInput pipeline test PASSED.")


if __name__ == "__main__":
    main()