
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.labels import load_labels
from src.data.dicom_loader import load_dicom
from src.data.preprocessing import preprocess_image
from src.data.advanced_input import AdvancedInputAdapter
from src.models.model_factory import build_model
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS
from src.explainability.vit_attention import (
    ViTAttentionCapture,
    cls_attention_map,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PATIENT_ID = "00436515-870c-4b36-a041-de91049b9ab4"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainability" / "vit_attention"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_checkpoint():
    candidates = list(
        EXPERIMENTS_DIR.glob("vit_b16*/checkpoints/best_model.pth")
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No ViT-B/16 checkpoint found. Expected a file matching:\n"
            "outputs/experiments/vit_b16*/checkpoints/best_model.pth\n"
            "Check reports/experiments/vit_b16.json and your experiment "
            "output directory for the actual checkpoint location."
        )

    if len(candidates) > 1:
        paths = "\n".join(str(path) for path in candidates)
        raise RuntimeError(
            "Multiple ViT checkpoints found. Select one explicitly:\n"
            + paths
        )

    return candidates[0]


def load_model(checkpoint_path):
    model = build_model("vit_b16", pretrained=False)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    return model


def prepare_patient(patient_id):
    labels, _ = load_labels(TRAIN_LABELS)

    if patient_id not in labels:
        raise KeyError(f"Patient not found in labels: {patient_id}")

    dicom_path = Path(TRAIN_IMAGES) / f"{patient_id}.dcm"

    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM not found: {dicom_path}")

    raw_image, metadata = load_dicom(
        dicom_path,
        return_metadata=True,
    )

    processed_image = preprocess_image(
        raw_image,
        target_size=(512, 512),
    )

    grayscale_tensor = torch.from_numpy(
        np.asarray(processed_image, dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0)

    adapter = AdvancedInputAdapter(
        image_size=224,
        use_imagenet_normalization=True,
    )

    model_input = adapter(grayscale_tensor).to(DEVICE)

    return raw_image, model_input, labels[patient_id]


def display_normalize(image):
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1, 99])

    if high <= low:
        return np.zeros_like(image)

    return np.clip((image - low) / (high - low), 0, 1)


def save_attention_visualizations(raw_image, attention_map):
    attention_map = attention_map.numpy()
    original = display_normalize(raw_image)

    # Map the 224x224 attention visualization to original image size.
    # The map is interpolated, not a native-resolution explanation.
    attention_original = torch.nn.functional.interpolate(
        torch.from_numpy(attention_map)[None, None],
        size=raw_image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    attention_path = OUTPUT_DIR / "attention_map.png"

    plt.figure(figsize=(7, 7))
    plt.imshow(attention_original, cmap="jet", vmin=0, vmax=1)
    plt.title("ViT-B/16: CLS-to-patch attention")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        attention_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    overlay_path = OUTPUT_DIR / "attention_overlay.png"

    plt.figure(figsize=(7, 7))
    plt.imshow(original, cmap="gray", vmin=0, vmax=1)
    plt.imshow(
        attention_original,
        cmap="jet",
        alpha=0.40,
        vmin=0,
        vmax=1,
    )
    plt.title("ViT-B/16 Attention Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Attention map saved: {attention_path.resolve()}")
    print(f"Overlay saved: {overlay_path.resolve()}")

    return attention_original


def main():
    print("[1/4] Finding checkpoint...", flush=True)
    checkpoint_path = find_checkpoint()

    print("[2/4] Loading ViT-B/16...", flush=True)
    model = load_model(checkpoint_path)

    print("[3/4] Preparing patient image...", flush=True)
    raw_image, model_input, label_record = prepare_patient(PATIENT_ID)

    print("[4/4] Capturing attention...", flush=True)

    # Attention extraction temporarily enables returned attention weights.
    with ViTAttentionCapture(model) as capture:
        with torch.no_grad():
            output = model(model_input)

        attention_layers = capture.get_attention_maps()

        if output.numel() != 1:
            raise ValueError(
                f"Expected one binary-classification logit; got "
                f"{tuple(output.shape)}"
            )

        logit = output.reshape(-1)[0].item()
        probability = float(torch.sigmoid(torch.tensor(logit)).item())

        last_attention = attention_layers[-1]

        print(f"Attention layers captured: {len(attention_layers)}")
        print(f"Final attention shape: {tuple(last_attention.shape)}")
        print(f"Ground-truth target: {label_record['target']}")
        print(f"Predicted pneumonia probability: {probability:.4f}")

        attention_map = cls_attention_map(
            last_attention,
            image_size=224,
        )

    # Basic sanity checks.
    assert attention_map.shape == (224, 224)
    assert torch.isfinite(attention_map).all()
    assert attention_map.min().item() >= 0
    assert attention_map.max().item() <= 1.0 + 1e-6
    assert 0 <= probability <= 1

    attention_original = save_attention_visualizations(
        raw_image,
        attention_map,
    )

    assert attention_original.shape == raw_image.shape[:2]
    assert np.isfinite(attention_original).all()

    print("\n--- ViT attention sanity checks passed ---")


if __name__ == "__main__":
    main()
