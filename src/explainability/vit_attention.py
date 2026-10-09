
from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTAttentionCapture:
    """
    Capture multi-head attention weights from a ViT model.

    Compatible with torchvision-style ViT models whose transformer
    blocks contain nn.MultiheadAttention modules.

    Captured attention shape:
        [batch, heads, tokens, tokens]
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = []
        self._original_forwards = []
        self._closed = False

        attention_modules = [
            module
            for module in model.modules()
            if isinstance(module, nn.MultiheadAttention)
        ]

        if not attention_modules:
            raise RuntimeError(
                "No nn.MultiheadAttention modules found. "
                "Inspect the ViT implementation before proceeding."
            )

        for module in attention_modules:
            original_forward = module.forward
            self._original_forwards.append((module, original_forward))
            module._captured_attention = None

            def forward_with_attention(
                this,
                query,
                key,
                value,
                *args,
                _original=original_forward,
                **kwargs,
            ):
                # Override the caller's need_weights=False.
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False

                output, weights = _original(
                    query, key, value, *args, **kwargs
                )

                this._captured_attention = (
                    weights.detach() if weights is not None else None
                )

                # Preserve the tuple interface expected by torchvision ViT.
                return output, weights

            module.forward = MethodType(forward_with_attention, module)
            self.layers.append(module)

    def get_attention_maps(self):
        """Return captured attention matrices in transformer order."""
        maps = []

        for index, layer in enumerate(self.layers):
            attention = layer._captured_attention

            if attention is None:
                raise RuntimeError(
                    f"No attention weights captured for layer {index}."
                )

            maps.append(attention)

        return maps

    def close(self):
        """Restore the original forward methods."""
        if self._closed:
            return

        for module, original_forward in self._original_forwards:
            module.forward = original_forward
            module._captured_attention = None

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def cls_attention_map(
    attention: torch.Tensor,
    image_size: int = 224,
) -> torch.Tensor:
    """
    Convert the final layer's attention into a spatial patch map.

    Args:
        attention: [1, heads, tokens, tokens]
        image_size: desired square output resolution.

    Returns:
        [image_size, image_size] normalized map in [0, 1].

    Assumes token 0 is CLS and remaining tokens form a square
    image-patch grid, as in standard torchvision ViT-B/16.
    """
    if attention.ndim != 4 or attention.shape[0] != 1:
        raise ValueError(
            "Expected attention with shape [1, heads, tokens, tokens]."
        )

    if attention.shape[-1] != attention.shape[-2]:
        raise ValueError("Attention matrix must be square.")

    # Average across attention heads.
    mean_attention = attention.mean(dim=1)[0]

    # CLS token (row 0) attending to all image patches.
    patch_attention = mean_attention[0, 1:]

    patch_count = patch_attention.numel()
    grid_size = int(patch_count ** 0.5)

    if grid_size * grid_size != patch_count:
        raise ValueError(
            f"{patch_count} patch tokens do not form a square grid. "
            "This model may use a different token layout."
        )

    spatial_map = patch_attention.reshape(1, 1, grid_size, grid_size)

    spatial_map = F.interpolate(
        spatial_map,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    # Normalize for visualization.
    minimum = spatial_map.min()
    maximum = spatial_map.max()

    spatial_map = (spatial_map - minimum) / (
        maximum - minimum + 1e-8
    )

    return spatial_map.detach().cpu()
