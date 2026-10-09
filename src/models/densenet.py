
"""
Dedicated DenseNet-121 interface for AstraMed.

Model construction is delegated to the central model factory.
"""

from torch import nn

from src.models.model_factory import build_model


def build_densenet121(pretrained: bool = True) -> nn.Module:
    """
    Build DenseNet-121 for binary pneumonia classification.

    Parameters
    ----------
    pretrained : bool
        Whether to load ImageNet-pretrained weights.

    Returns
    -------
    nn.Module
        DenseNet-121 with one binary-classification logit.
    """

    return build_model(
        model_name="densenet121",
        pretrained=pretrained,
    )
