
from __future__ import annotations

import torch


class IntegratedGradients:
    """
    Integrated Gradients for a binary classifier that returns one logit.

    Args:
        model: Trained model returning one logit per input.
        baseline: Baseline tensor in the same space and shape as the input.
                   If None, a zero tensor is used.
        steps: Number of integration intervals. Larger values generally
               improve the numerical approximation but take longer.
    """

    def __init__(self, model, steps: int = 64):
        if steps < 1:
            raise ValueError("steps must be >= 1")

        self.model = model
        self.steps = steps

    def generate(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor | None = None,
        target: str = "positive",
    ) -> dict:
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                "Expected input shape [1, channels, height, width]."
            )

        if target not in {"positive", "negative"}:
            raise ValueError("target must be 'positive' or 'negative'")

        if not input_tensor.is_floating_point():
            raise TypeError("input_tensor must be floating point")

        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        else:
            baseline = baseline.to(
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        if baseline.shape != input_tensor.shape:
            raise ValueError(
                f"Baseline shape {tuple(baseline.shape)} must match "
                f"input shape {tuple(input_tensor.shape)}"
            )

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        x = input_tensor.detach()
        b = baseline.detach()
        difference = x - b

        # Trapezoidal approximation of the path integral.
        gradient_sum = torch.zeros_like(x)

        with torch.enable_grad():
            for index in range(self.steps + 1):
                alpha = index / self.steps
                interpolated = (
                    b + alpha * difference
                ).detach().requires_grad_(True)

                output = self.model(interpolated)

                if not isinstance(output, torch.Tensor) or output.numel() != 1:
                    raise ValueError(
                        "Model must return exactly one logit for this input."
                    )

                logit = output.reshape(-1)[0]
                score = logit if target == "positive" else -logit

                gradient = torch.autograd.grad(
                    outputs=score,
                    inputs=interpolated,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                weight = 0.5 if index in {0, self.steps} else 1.0
                gradient_sum += weight * gradient

        average_gradient = gradient_sum / self.steps
        attributions = difference * average_gradient

        # Completeness check is calculated in logit space.
        with torch.no_grad():
            input_logit = self.model(x).reshape(-1)[0]
            baseline_logit = self.model(b).reshape(-1)[0]

            input_score = (
                input_logit if target == "positive" else -input_logit
            )
            baseline_score = (
                baseline_logit if target == "positive" else -baseline_logit
            )

            score_difference = (input_score - baseline_score).item()
            attribution_sum = attributions.sum().item()
            completeness_delta = attribution_sum - score_difference

            pneumonia_probability = torch.sigmoid(input_logit).item()

        # Summarize channel contributions into a spatial map.
        # Preserve signed attribution values: positive and negative
        # contributions should remain distinguishable.
        attribution_map = attributions[0].sum(dim=0)

        return {
            "attributions": attributions.detach(),
            "attribution_map": attribution_map.detach().cpu(),
            "probability": pneumonia_probability,
            "input_score": input_score.item(),
            "baseline_score": baseline_score.item(),
            "score_difference": score_difference,
            "attribution_sum": attribution_sum,
            "completeness_delta": completeness_delta,
            "steps": self.steps,
            "target": target,
        }
