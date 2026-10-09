
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.labels import load_labels
from src.data.dicom_loader import load_dicom
from src.data.preprocessing import preprocess_image
from src.data.advanced_input import AdvancedInputAdapter
from src.models.model_factory import build_model
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS
from src.explainability.integrated_gradients import IntegratedGradients


# --------------------------------------------------
# Configuration
# --------------------------------------------------
PATIENT_ID = "00436515-870c-4b36-a041-de91049b9ab4"

CHECKPOINT_PATH = (
    Path("outputs/experiments")
    / "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = Path("outputs/explainability/integrated_gradients")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTEGRATION_STEPS = 128
TARGET = "positive"

# Baseline options:
# "zero" = zeros in normalized model-input space
# "mean" = constant 0.5 image before ImageNet normalization
BASELINE_MODE = "zero"


def load_trained_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print(f"Loading checkpoint: {CHECKPOINT_PATH}", flush=True)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = build_model("resnet50", pretrained=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    print(f"Device: {DEVICE}", flush=True)
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}", flush=True)
    return model


def prepare_input():
    print("Loading labels...", flush=True)
    labels, _ = load_labels(TRAIN_LABELS)

    if PATIENT_ID not in labels:
        raise ValueError(f"Patient not found in labels: {PATIENT_ID}")

    dicom_path = Path(TRAIN_IMAGES) / f"{PATIENT_ID}.dcm"
    print(f"Loading DICOM: {dicom_path}", flush=True)

    raw_image, _ = load_dicom(dicom_path)
    processed_image = preprocess_image(
        raw_image, target_size=(512, 512)
    )

    grayscale_tensor = torch.from_numpy(
        np.asarray(processed_image, dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0)

    adapter = AdvancedInputAdapter(
        image_size=224,
        use_imagenet_normalization=True,
    )

    model_input = adapter(grayscale_tensor).to(DEVICE)

    print(f"Raw image: {raw_image.shape}", flush=True)
    print(f"Processed image: {processed_image.shape}", flush=True)
    print(f"Model input: {tuple(model_input.shape)}", flush=True)

    return raw_image, processed_image, model_input, labels[PATIENT_ID]


def make_baseline(model_input):
    if BASELINE_MODE == "zero":
        return torch.zeros_like(model_input)

    if BASELINE_MODE == "mean":
        # Create a constant 0.5 grayscale image in [0, 1],
        # then apply the same adapter and ImageNet normalization.
        gray = torch.full(
            (1, 1, 512, 512),
            0.5,
            dtype=torch.float32,
        )
        adapter = AdvancedInputAdapter(
            image_size=224,
            use_imagenet_normalization=True,
        )
        return adapter(gray).to(DEVICE)

    raise ValueError(
        f"Unsupported BASELINE_MODE: {BASELINE_MODE}"
    )


def save_visualizations(processed_image, result):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    attribution_map = result["attribution_map"].numpy()
    attribution_tensor = torch.from_numpy(
        attribution_map
    ).float()[None, None]

    attribution_512 = F.interpolate(
        attribution_tensor,
        size=processed_image.shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    # Signed map: positive and negative contributions are different.
    limit = float(np.max(np.abs(attribution_512)))
    if limit < 1e-12:
        limit = 1.0

    signed_path = OUTPUT_DIR / "signed_attributions.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(processed_image, cmap="gray")
    plt.imshow(
        attribution_512,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        alpha=0.65,
    )
    plt.title(
        f"Integrated Gradients | target={TARGET} | "
        f"P(pneumonia)={result['probability']:.3f}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(signed_path, dpi=160, bbox_inches="tight")
    plt.close()

    # Absolute attribution: highlights magnitude regardless of sign.
    absolute_map = np.abs(attribution_512)
    maximum = float(absolute_map.max())
    if maximum > 0:
        absolute_map = absolute_map / maximum

    overlay_path = OUTPUT_DIR / "absolute_attribution.png"
    plt.figure(figsize=(8, 8))
    plt.imshow(processed_image, cmap="gray")
    plt.imshow(absolute_map, cmap="jet", vmin=0, vmax=1, alpha=0.45)
    plt.title("Integrated Gradients | Absolute Attribution")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Signed map saved: {signed_path.resolve()}", flush=True)
    print(f"Absolute attribution saved: {overlay_path.resolve()}", flush=True)


def main():
    print("\n=== M3: Integrated Gradients ===", flush=True)

    model = load_trained_model()
    _, processed_image, model_input, label_record = prepare_input()
    baseline = make_baseline(model_input)

    print(
        f"Generating attributions: steps={INTEGRATION_STEPS}, "
        f"baseline={BASELINE_MODE}, target={TARGET}",
        flush=True,
    )

    explainer = IntegratedGradients(
        model,
        steps=INTEGRATION_STEPS,
    )
    result = explainer.generate(
        model_input,
        baseline=baseline,
        target=TARGET,
    )

    attribution_map = result["attribution_map"]
    assert attribution_map.shape == (224, 224)
    assert torch.isfinite(attribution_map).all()
    assert np.isfinite(result["completeness_delta"])
    assert 0.0 <= result["probability"] <= 1.0

    save_visualizations(processed_image, result)

    report_path = OUTPUT_DIR / "completeness_report.txt"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("Integrated Gradients Completeness Report\n")
        report.write(f"Patient ID: {PATIENT_ID}\n")
        report.write(f"Ground-truth label: {label_record['target']}\n")
        report.write(f"Predicted pneumonia probability: {result['probability']:.6f}\n")
        report.write(f"Target: {TARGET}\n")
        report.write(f"Baseline: {BASELINE_MODE}\n")
        report.write(f"Integration steps: {INTEGRATION_STEPS}\n")
        report.write(f"Input score: {result['input_score']:.8f}\n")
        report.write(f"Baseline score: {result['baseline_score']:.8f}\n")
        report.write(f"Score difference: {result['score_difference']:.8f}\n")
        report.write(f"Attribution sum: {result['attribution_sum']:.8f}\n")
        report.write(f"Completeness delta: {result['completeness_delta']:.8f}\n")
        report.write(
            f"Absolute completeness delta: "
            f"{abs(result['completeness_delta']):.8f}\n"
        )

    print("\n--- Integrated Gradients sanity tests passed ---")
    print(f"Ground-truth label: {label_record['target']}")
    print(f"Predicted pneumonia probability: {result['probability']:.4f}")
    print(f"Attribution map shape: {tuple(attribution_map.shape)}")
    print(f"Attribution range: [{attribution_map.min():.6f}, {attribution_map.max():.6f}]")
    print(f"Score difference: {result['score_difference']:.6f}")
    print(f"Attribution sum: {result['attribution_sum']:.6f}")
    print(f"Completeness delta: {result['completeness_delta']:.6f}")
    print(f"Report saved: {report_path.resolve()}")


if __name__ == "__main__":
    main()
