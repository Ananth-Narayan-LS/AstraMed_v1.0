
"""Audit AstraMed data, annotations, preprocessing and model checkpoint."""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import load_labels
from src.data.dicom_loader import load_dicom
from src.data.preprocessing import preprocess_image
from src.data.advanced_input import AdvancedInputAdapter
from src.models.model_factory import build_model
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
    / "checkpoints"
    / "best_model.pth"
)


def audit_annotations():
    print("\n" + "=" * 60)
    print("1. ANNOTATION AUDIT")
    print("=" * 60)

    labels, stats = load_labels(TRAIN_LABELS)

    print("Label file:", TRAIN_LABELS)
    print("Statistics:", stats)
    print("Number of patient IDs:", len(labels))

    # Prefer a positive case with an existing DICOM image.
    selected_id = None

    for patient_id, record in labels.items():
        image_path = TRAIN_IMAGES / f"{patient_id}.dcm"

        if not image_path.exists():
            continue

        if isinstance(record, dict) and int(record.get("target", 0)) == 1:
            selected_id = patient_id
            break

    if selected_id is None:
        for patient_id in labels:
            if (TRAIN_IMAGES / f"{patient_id}.dcm").exists():
                selected_id = patient_id
                break

    if selected_id is None:
        raise FileNotFoundError(
            f"No matching DICOM images found under {TRAIN_IMAGES}"
        )

    record = labels[selected_id]

    print("Selected patient ID:", selected_id)
    print("Annotation record type:", type(record).__name__)
    print("Annotation record:", repr(record)[:2000])

    if isinstance(record, dict):
        print("Annotation keys:", list(record.keys()))

        for key, value in record.items():
            if "box" in key.lower():
                print(f"Bounding-box field {key!r}:")
                print("  Type:", type(value).__name__)
                print("  Example:", repr(value)[:1500])

    return selected_id, record


def audit_preprocessing(patient_id):
    print("\n" + "=" * 60)
    print("2. PREPROCESSING AUDIT")
    print("=" * 60)

    image_path = TRAIN_IMAGES / f"{patient_id}.dcm"

    image, metadata = load_dicom(
        image_path,
        return_metadata=True,
    )

    processed = preprocess_image(
        image,
        target_size=(512, 512),
    )

    processed = np.asarray(processed, dtype=np.float32)

    print("DICOM path:", image_path)
    print("Raw image shape:", np.asarray(image).shape)
    print("Preprocessed shape:", processed.shape)
    print("Preprocessed dtype:", processed.dtype)
    print("Pixel minimum:", float(processed.min()))
    print("Pixel maximum:", float(processed.max()))
    print("Metadata type:", type(metadata).__name__)

    # This tests the existing adapter without modifying it.
    image_tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)

    adapter = AdvancedInputAdapter(
        image_size=224,
        use_imagenet_normalization=True,
    )

    model_input = adapter(image_tensor)

    print("Dataset-style tensor shape:", tuple(image_tensor.shape))
    print("Advanced model input shape:", tuple(model_input.shape))
    print("Advanced model input dtype:", model_input.dtype)

    return model_input


def audit_checkpoint(model_input):
    print("\n" + "=" * 60)
    print("3. CHECKPOINT AND PREDICTION AUDIT")
    print("=" * 60)

    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint exists:", CHECKPOINT.exists())

    if not CHECKPOINT.exists():
        print("STOP: checkpoint not found. Do not substitute random weights.")
        return

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        print("Checkpoint top-level keys:", list(checkpoint.keys()))
    else:
        raise TypeError("Unexpected checkpoint format.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Supports a checkpoint that is itself a state dictionary.
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Model state dictionary was not found.")

    print("Number of state-dict entries:", len(state_dict))

    model = build_model(
        model_name="resnet50",
        pretrained=False,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        logits = model(model_input)
        probability = torch.sigmoid(logits)

    print("Model output shape:", tuple(logits.shape))
    print("Pneumonia probability:", probability.reshape(-1).tolist())
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint validation loss:", checkpoint.get("loss"))
    print("Checkpoint loaded successfully.")


def main():
    print("ASTRAMED — SPRINT 6 / MILESTONE 1")
    print("Project root:", PROJECT_ROOT)

    patient_id, _record = audit_annotations()
    model_input = audit_preprocessing(patient_id)
    audit_checkpoint(model_input)

    print("\nAudit finished. Review all three sections before continuing.")


if __name__ == "__main__":
    main()
