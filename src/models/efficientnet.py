
"""
Dedicated EfficientNet-B0 interface for AstraMed.

Model construction is delegated to the central model factory.
"""

from torch import nn

from src.models.model_factory import build_model


def build_efficientnet_b0(
    pretrained: bool = True,
) -> nn.Module:
    """
    Build EfficientNet-B0 for binary pneumonia classification.

    Parameters
    ----------
    pretrained : bool
        Whether to load ImageNet-pretrained weights.

    Returns
    -------
    nn.Module
        EfficientNet-B0 with one binary-classification logit.
    """

    return build_model(
        model_name="efficientnet_b0",
        pretrained=pretrained,
    )
