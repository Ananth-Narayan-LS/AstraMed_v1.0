
"""
Dedicated Vision Transformer (ViT-B/16) interface for AstraMed.

Model construction is delegated to the central model factory.
"""

from torch import nn

from src.models.model_factory import build_model


def build_vit_b16(pretrained: bool = True) -> nn.Module:
    """
    Build ViT-B/16 for binary pneumonia classification.

    Parameters
    ----------
    pretrained : bool
        Whether to load ImageNet-pretrained weights.

    Returns
    -------
    nn.Module
        ViT-B/16 with one binary-classification logit.
    """

    return build_model(
        model_name="vit_b16",
        pretrained=pretrained,
    )
