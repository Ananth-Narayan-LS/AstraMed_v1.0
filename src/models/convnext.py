
"""
Dedicated ConvNeXt-Tiny interface for AstraMed.

Model construction is delegated to the central model factory.
"""

from torch import nn

from src.models.model_factory import build_model


def build_convnext_tiny(
    pretrained: bool = True,
) -> nn.Module:
    """
    Build ConvNeXt-Tiny for binary pneumonia classification.

    Parameters
    ----------
    pretrained : bool
        Whether to load ImageNet-pretrained weights.

    Returns
    -------
    nn.Module
        ConvNeXt-Tiny with one binary-classification logit.
    """

    return build_model(
        model_name="convnext_tiny",
        pretrained=pretrained,
    )
