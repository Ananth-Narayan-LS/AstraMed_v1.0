
"""
Model factory for AstraMed.

Creates binary pneumonia-classification models using
TorchVision architectures.

Advanced models return raw logits with shape [B, 1].
Use BCEWithLogitsLoss during training.
"""

from torch import nn
from torchvision import models

from src.models.baseline_cnn import BaselineCNN


def build_model(
    model_name: str,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a model for binary pneumonia classification.

    Supported advanced architectures:
        - resnet50
        - densenet121
        - efficientnet_b0
        - convnext_tiny
        - vit_b16

    Parameters
    ----------
    model_name : str
        Architecture name.
    pretrained : bool
        Whether to load ImageNet-pretrained weights.

    Returns
    -------
    nn.Module
        Model with a single binary-classification output.
        Advanced models return raw logits, not probabilities.
    """

    model_name = model_name.lower().strip()

    # Keep the Sprint 4 baseline available without changing it.
    if model_name == "baseline_cnn":
        if pretrained:
            raise ValueError(
                "BaselineCNN does not support ImageNet pretrained weights."
            )
        return BaselineCNN()

    if model_name == "resnet50":
        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained else None
        )
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif model_name == "densenet121":
        weights = (
            models.DenseNet121_Weights.DEFAULT
            if pretrained else None
        )
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(
            model.classifier.in_features, 1
        )

    elif model_name == "efficientnet_b0":
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 1
        )

    elif model_name == "convnext_tiny":
        weights = (
            models.ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained else None
        )
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features, 1
        )

    elif model_name == "vit_b16":
        weights = (
            models.ViT_B_16_Weights.DEFAULT
            if pretrained else None
        )
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, 1
        )

    else:
        supported = [
            "baseline_cnn",
            "resnet50",
            "densenet121",
            "efficientnet_b0",
            "convnext_tiny",
            "vit_b16",
        ]
        raise ValueError(
            f"Unsupported model: {model_name!r}. "
            f"Supported models: {', '.join(supported)}"
        )

    return model
