
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F


def resize_heatmap(heatmap, output_shape):
    """Resize a 2D heatmap to (height, width) and normalize to [0, 1]."""
    heatmap = torch.as_tensor(heatmap, dtype=torch.float32)

    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be a 2D array or tensor.")

    if not torch.isfinite(heatmap).all():
        raise ValueError("Heatmap contains NaN or infinity.")

    resized = F.interpolate(
        heatmap[None, None],
        size=tuple(output_shape),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    low, high = resized.min(), resized.max()

    if high > low:
        resized = (resized - low) / (high - low)
    else:
        resized = np.zeros_like(resized)

    return resized


def create_box_mask(boxes, image_shape):
    """
    Create a binary mask from RSNA boxes.

    Each box uses original-image pixel coordinates:
    x, y, width, height.
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=bool)

    for box in boxes:
        x1 = int(np.floor(box["x"]))
        y1 = int(np.floor(box["y"]))
        x2 = int(np.ceil(box["x"] + box["width"]))
        y2 = int(np.ceil(box["y"] + box["height"]))

        # Clip boxes to image boundaries.
        x1 = np.clip(x1, 0, width)
        x2 = np.clip(x2, 0, width)
        y1 = np.clip(y1, 0, height)
        y2 = np.clip(y2, 0, height)

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True

    return mask


def threshold_heatmap(heatmap, percentile=80):
    """
    Select the strongest activations.

    At percentile=80, pixels at or above the 80th percentile
    are marked as activated.
    """
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100.")

    threshold = float(np.percentile(heatmap, percentile))
    binary_map = heatmap >= threshold

    return binary_map, threshold


def localization_metrics(heatmap, boxes, image_shape, percentile=80):
    """
    Compare a thresholded explanation map with radiologist boxes.

    Reports pixel overlap metrics, pointing-game hit, and
    fraction of heatmap activation energy inside the boxes.

    These metrics measure spatial overlap, not clinical correctness.
    """
    aligned = resize_heatmap(heatmap, image_shape)
    box_mask = create_box_mask(boxes, image_shape)
    predicted_mask, threshold = threshold_heatmap(
        aligned, percentile=percentile
    )

    intersection = np.logical_and(predicted_mask, box_mask).sum()
    union = np.logical_or(predicted_mask, box_mask).sum()

    tp = int(intersection)
    fp = int(np.logical_and(predicted_mask, ~box_mask).sum())
    fn = int(np.logical_and(~predicted_mask, box_mask).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    iou = tp / union if union else 0.0

    # Pointing game: does the single hottest pixel fall inside a box?
    max_y, max_x = np.unravel_index(np.argmax(aligned), aligned.shape)
    pointing_hit = bool(box_mask[max_y, max_x]) if box_mask.any() else None

    # Fraction of nonnegative heatmap mass falling inside boxes.
    total_energy = float(aligned.sum())
    energy_inside = (
        float(aligned[box_mask].sum() / total_energy)
        if total_energy > 0 and box_mask.any()
        else None
    )

    return {
        "threshold_percentile": percentile,
        "threshold_value": threshold,
        "box_area_pixels": int(box_mask.sum()),
        "predicted_area_pixels": int(predicted_mask.sum()),
        "intersection_pixels": int(intersection),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pointing_game_hit": pointing_hit,
        "energy_inside_boxes": energy_inside,
        "has_boxes": bool(box_mask.any()),
        "aligned_heatmap": aligned,
        "box_mask": box_mask,
        "thresholded_heatmap": predicted_mask,
    }


def save_alignment_visualization(
    original_image,
    heatmap,
    boxes,
    output_path,
    title="Explanation vs. radiologist boxes",
    percentile=80,
):
    """Save four panels for visual inspection and debugging."""
    original_image = np.asarray(original_image, dtype=np.float32)
    aligned = resize_heatmap(heatmap, original_image.shape[:2])

    low, high = np.percentile(original_image, [1, 99])
    if high > low:
        display_image = np.clip(
            (original_image - low) / (high - low), 0, 1
        )
    else:
        display_image = np.zeros_like(original_image)

    metrics = localization_metrics(
        aligned, boxes, original_image.shape[:2], percentile
    )

    thresholded = metrics["thresholded_heatmap"]
    box_mask = metrics["box_mask"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    axes[0, 0].imshow(display_image, cmap="gray")
    axes[0, 0].set_title("Original X-ray + radiologist boxes")

    for box in boxes:
        axes[0, 0].add_patch(
            Rectangle(
                (box["x"], box["y"]),
                box["width"],
                box["height"],
                fill=False,
                edgecolor="lime",
                linewidth=2,
            )
        )

    axes[0, 1].imshow(display_image, cmap="gray")
    axes[0, 1].imshow(aligned, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    axes[0, 1].set_title("Explanation overlay")

    axes[1, 0].imshow(box_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Radiologist box mask")

    axes[1, 1].imshow(display_image, cmap="gray")
    axes[1, 1].imshow(thresholded, cmap="autumn", alpha=0.45)
    axes[1, 1].contour(
        box_mask.astype(float),
        levels=[0.5],
        colors=["cyan"],
        linewidths=1,
    )
    axes[1, 1].set_title(
        f"Thresholded explanation (top {100-percentile}%) + boxes"
    )

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Do not serialize large arrays in the metrics report.
    clean_metrics = {
        key: value
        for key, value in metrics.items()
        if key not in {
            "aligned_heatmap",
            "box_mask",
            "thresholded_heatmap",
        }
    }

    return clean_metrics
