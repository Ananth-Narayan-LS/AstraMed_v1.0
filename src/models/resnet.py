
"""
Dedicated ResNet-50 interface for AstraMed.

Model construction is delegated to the central model factory
to avoid duplicating architecture definitions.
"""

from torch import nn

from src.models.model_factory import build_model


def build_resnet50(pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-50 for binary pneumonia classification.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-pretrained weights when True.

    Returns
    -------
    nn.Module
        ResNet-50 with a single binary logit output.
    """

    return build_model(
        model_name="resnet50",
        pretrained=pretrained,
    )
