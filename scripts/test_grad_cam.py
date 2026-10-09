
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
from src.explainability.grad_cam import GradCAM


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PATIENT_ID = "00436515-870c-4b36-a041-de91049b9ab4"

CHECKPOINT_PATH = (
    Path("outputs/experiments/")
    / "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_DIR = Path("outputs/explainability/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Load the checkpoint and model
# --------------------------------------------------

def load_trained_resnet(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}"
        )

    model = build_model("resnet50", pretrained=False)

    # Checkpoints from this project contain model_state_dict.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility fallback for older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Device: {DEVICE}")

    return model


# --------------------------------------------------
# 3. Load and preprocess the patient's image
# --------------------------------------------------

# def prepare_image(patient_id: str):
#     labels, _ = load_labels(TRAIN_LABELS)
    

#     if patient_id not in labels:
#         raise KeyError(f"Patient not found in labels: {patient_id}")

#     dicom_path = Path(TRAIN_IMAGES) / f"{patient_id}.dcm"

#     if not dicom_path.exists():
#         raise FileNotFoundError(f"DICOM not found: {dicom_path}")

#     raw_image, metadata = load_dicom(dicom_path)
#     processed_image = preprocess_image(
#         raw_image,
#         target_size=(512, 512),
#     )

#     # Dataset-format tensor: [1, 1, 512, 512].
#     grayscale_tensor = torch.from_numpy(
#         np.asarray(processed_image, dtype=np.float32)
#     ).unsqueeze(0).unsqueeze(0)

#     print("[DEBUG] Creating AdvancedInputAdapter...", flush=True)

#     adapter = AdvancedInputAdapter(
#         image_size=224,
#         use_imagenet_normalization=True,
#     )

#     print("[DEBUG] Applying adapter...", flush=True)

#     model_input = adapter(grayscale_tensor).to(DEVICE)

#     print(
#         "[DEBUG] Adapter complete:",
#         tuple(model_input.shape),
#         flush=True,
#     )

#     print(f"Patient ID: {patient_id}")
#     print(f"Ground-truth target: {labels[patient_id]['target']}")
#     print(f"Original image shape: {raw_image.shape}")
#     print(f"Processed image shape: {processed_image.shape}")
#     print(f"Model input shape: {tuple(model_input.shape)}")

#     print("[DEBUG] prepare_image() completed.", flush=True)

#     return raw_image, processed_image, model_input, labels[patient_id]




def prepare_image(patient_id):
    print("[DEBUG 1] Entered prepare_image()", flush=True)

    
    print("[DEBUG 2] Loading labels...", flush=True)
    labels, stats = load_labels(TRAIN_LABELS)

    print("[DEBUG 2A] Labels loaded successfully", flush=True)
    print("[DEBUG 2B] Number of patients:", len(labels), flush=True)
    print("[DEBUG 2C] Checking patient ID...", flush=True)

    if patient_id not in labels:
        raise ValueError(f"Patient ID not found: {patient_id}")

    print("[DEBUG 3] Patient ID verified", flush=True)


    if patient_id not in labels:
        raise ValueError(f"Patient ID not found: {patient_id}")

    dicom_path = Path(TRAIN_IMAGES) / f"{patient_id}.dcm"
    print(f"[DEBUG 4] DICOM path: {dicom_path}", flush=True)

    print("[DEBUG 5] Loading DICOM...", flush=True)
    raw_image, metadata = load_dicom(dicom_path)
    print(f"[DEBUG 6] Raw image shape: {raw_image.shape}", flush=True)

    print("[DEBUG 7] Preprocessing image...", flush=True)
    processed_image = preprocess_image(
        raw_image, target_size=(512, 512)
    )
    print(
        f"[DEBUG 8] Processed shape: {processed_image.shape}",
        flush=True
    )

    print("[DEBUG 9] Creating tensor...", flush=True)
    grayscale_tensor = torch.from_numpy(
        np.asarray(processed_image, dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0)
    print(
        f"[DEBUG 10] Tensor shape: {tuple(grayscale_tensor.shape)}",
        flush=True
    )

    print("[DEBUG 11] Creating AdvancedInputAdapter...", flush=True)
    adapter = AdvancedInputAdapter(
        image_size=224,
        use_imagenet_normalization=True
    )
    print("[DEBUG 12] Applying adapter...", flush=True)

    model_input = adapter(grayscale_tensor).to(DEVICE)
    print(
        f"[DEBUG 13] Adapter output: {tuple(model_input.shape)}",
        flush=True
    )

    print("[DEBUG 14] prepare_image() completed", flush=True)

    return raw_image, processed_image, model_input, labels[patient_id]





# --------------------------------------------------
# 4. Generate the heatmap and overlay
# --------------------------------------------------

def save_visualizations(raw_image, cam, probability, target):
    # Resize the CAM from 224 x 224 to original DICOM dimensions.
    cam_original = F.interpolate(
        cam[None, None],
        size=raw_image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    # Normalize original pixels for display only.
    original = np.asarray(raw_image, dtype=np.float32)
    low, high = np.percentile(original, [1, 99])

    if high > low:
        original = np.clip((original - low) / (high - low), 0, 1)
    else:
        original = np.zeros_like(original)

    # Save a standalone color heatmap.
    heatmap_path = OUTPUT_DIR / "heatmap.png"
    plt.figure(figsize=(7, 7))
    plt.imshow(cam_original, cmap="jet", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Save the original image with a transparent heatmap overlay.
    overlay_path = OUTPUT_DIR / "overlay.png"
    plt.figure(figsize=(7, 7))
    plt.imshow(original, cmap="gray", vmin=0, vmax=1)
    plt.imshow(cam_original, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    plt.title(
        f"Grad-CAM | target={target} | "
        f"P(pneumonia)={probability:.3f}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved: {heatmap_path.resolve()}")
    print(f"Overlay saved: {overlay_path.resolve()}")

    return cam_original


# --------------------------------------------------
# 5. Sanity tests and execution
# --------------------------------------------------

def main():
    print("[1/5] Loading trained ResNet-50...", flush=True)
    model = load_trained_resnet(CHECKPOINT_PATH)

    print("[2/5] Loading and preprocessing DICOM...", flush=True)
    raw_image, processed_image, model_input, label_record = prepare_image(
        PATIENT_ID
    )

    print("[3/5] Creating Grad-CAM...", flush=True)
    grad_cam = GradCAM(model, model.layer4[-1])

    try:
        print("[4/5] Generating Grad-CAM heatmap...", flush=True)
        cam, probability = grad_cam.generate(
            model_input,
            target="positive",
        )

        print("[5/5] Saving heatmap and overlay...", flush=True)

        assert cam.ndim == 2
        assert cam.shape == (224, 224)
        assert torch.isfinite(cam).all()
        assert 0.0 <= probability <= 1.0

        cam_original = save_visualizations(
            raw_image,
            cam,
            probability,
            target="positive",
        )

        assert cam_original.shape == raw_image.shape[:2]
        assert np.isfinite(cam_original).all()

        print("\n--- Grad-CAM sanity tests passed ---")
        print(f"Ground-truth label: {label_record['target']}")
        print(f"Predicted pneumonia probability: {probability:.4f}")
        print(f"CAM shape: {tuple(cam.shape)}")
        print(f"Original-resolution CAM: {cam_original.shape}")

    finally:
        grad_cam.close()

    # Standard torchvision ResNet-50's final convolutional stage.
    grad_cam = GradCAM(model, model.layer4[-1])

    try:
        cam, probability = grad_cam.generate(
            model_input,
            target="positive",
        )

        assert cam.ndim == 2, "CAM must be a 2D map."
        assert cam.shape == (224, 224), f"Unexpected CAM shape: {cam.shape}"
        assert torch.isfinite(cam).all(), "CAM contains NaN or infinity."
        assert cam.min().item() >= 0.0, "CAM has negative values."
        assert cam.max().item() <= 1.0 + 1e-6, "CAM exceeds 1."
        assert 0.0 <= probability <= 1.0, "Invalid probability."

        cam_original = save_visualizations(
            raw_image,
            cam,
            probability,
            target="positive",
        )

        assert cam_original.shape == raw_image.shape[:2]
        assert np.isfinite(cam_original).all()

        print("\n--- Grad-CAM sanity tests passed ---")
        print(f"Ground-truth label: {label_record['target']}")
        print(f"Predicted pneumonia probability: {probability:.4f}")
        print(f"CAM shape: {tuple(cam.shape)}")
        print(f"Original-resolution CAM: {cam_original.shape}")
        print(f"CAM range: [{cam.min().item():.4f}, {cam.max().item():.4f}]")

    finally:
        grad_cam.close()


if __name__ == "__main__":
    main()
