"""
Advanced model input pipeline for AstraMed.

This module adapts the existing grayscale RSNA dataset output
for ImageNet-pretrained models without modifying the baseline
dataset implementation.

[B,1,H,W]
   ↓
resize
   ↓
channel replication
   ↓
normalization
   ↓
[B,3,224,224]


"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class AdvancedInputAdapter(nn.Module):
    """
    Adapt grayscale X-ray tensors for ImageNet-pretrained models.

    Input:
        [B, 1, H, W]

    Output:
        [B, 3, image_size, image_size]
    """

    IMAGENET_MEAN = [
        0.485,
        0.456,
        0.406,
    ]

    IMAGENET_STD = [
        0.229,
        0.224,
        0.225,
    ]

    def __init__(
        self,
        image_size: int = 224,
        use_imagenet_normalization: bool = True,
    ):
        super().__init__()

        if image_size <= 0:
            raise ValueError(
                "image_size must be greater than 0."
            )

        self.image_size = image_size
        self.use_imagenet_normalization = (
            use_imagenet_normalization
        )

        self.resize = transforms.Resize(
            (image_size, image_size)
        )

        self.normalize = transforms.Normalize(
            mean=self.IMAGENET_MEAN,
            std=self.IMAGENET_STD,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert grayscale images to 3-channel input.

        Parameters
        ----------
        images : torch.Tensor
            Tensor with shape [B, 1, H, W].

        Returns
        -------
        torch.Tensor
            Tensor with shape
            [B, 3, image_size, image_size].
        """

        if images.ndim != 4:
            raise ValueError(
                "Expected input with shape [B, C, H, W], "
                f"got {tuple(images.shape)}."
            )

        if images.shape[1] != 1:
            raise ValueError(
                "Expected grayscale input with 1 channel, "
                f"got {images.shape[1]} channels."
            )

        # Step 1: Resize.
        images = self.resize(images)

        # Step 2: Convert grayscale → 3 channels.
        images = images.repeat(1, 3, 1, 1)

        # Step 3: Apply ImageNet normalization.
        if self.use_imagenet_normalization:
            images = self.normalize(images)

        return images