
from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for binary classification models that return one logit.

    Expected model output:
        [batch_size, 1]

    target_layer:
        The convolutional layer whose activations are used to
        construct the Grad-CAM map.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.activation_gradients = None

        self._forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

        # Retain gradients for this intermediate feature map.
        if output.requires_grad:
            output.retain_grad()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target: str = "positive",
    ) -> tuple[torch.Tensor, float]:
        """
        Generate a normalized Grad-CAM map.

        Args:
            input_tensor: Model input, shape [1, 3, H, W].
            target: "positive" or "negative".

        Returns:
            cam: Tensor of shape [H, W], values in [0, 1].
            probability: Model's predicted pneumonia probability.
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                "Grad-CAM currently expects one image: [1, 3, H, W]."
            )

        if target not in {"positive", "negative"}:
            raise ValueError("target must be 'positive' or 'negative'.")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None

        # Do NOT use torch.no_grad(): Grad-CAM needs gradients.
        with torch.enable_grad():
            model_input = input_tensor.detach().clone().requires_grad_(True)
            output = self.model(model_input)

            if not isinstance(output, torch.Tensor):
                raise TypeError("Model must return a Tensor of logits.")

            if output.numel() != 1:
                raise ValueError(
                    "Expected one logit for binary classification; "
                    f"received output shape {tuple(output.shape)}."
                )

            logit = output.reshape(-1)[0]

            probability = torch.sigmoid(logit.detach()).item()

            # Positive logit explains the pneumonia class.
            # Negative logit explains the non-pneumonia class.
            score = logit if target == "positive" else -logit
            score.backward()

            if self.activations is None:
                raise RuntimeError("Target layer did not capture activations.")

            gradients = self.activations.grad
            if gradients is None:
                raise RuntimeError(
                    "No gradients captured. Check the target layer."
                )

            # Average gradients spatially to obtain channel weights.
            weights = gradients.mean(dim=(2, 3), keepdim=True)

            # Weighted sum across channels, then keep positive evidence.
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            # Resize CAM to model-input image dimensions.
            cam = F.interpolate(
                cam,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            cam = cam[0, 0]
            cam_min = cam.min()
            cam_max = cam.max()

            # Normalize safely, including the all-zero case.
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

            return cam.detach().cpu(), probability

    def close(self):
        """Remove the hook when Grad-CAM is no longer needed."""
        self._forward_handle.remove()
