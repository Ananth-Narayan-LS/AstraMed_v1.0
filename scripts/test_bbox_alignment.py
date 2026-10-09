
from pathlib import Path
import json

import numpy as np
import torch

from src.data.labels import load_labels
from src.data.dicom_loader import load_dicom
from src.data.preprocessing import preprocess_image
from src.data.advanced_input import AdvancedInputAdapter
from src.models.model_factory import build_model
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS

from src.explainability.grad_cam import GradCAM
from src.explainability.vit_attention import (
    ViTAttentionCapture,
    cls_attention_map,
)
from src.explainability.bbox_alignment import (
    resize_heatmap,
    localization_metrics,
    save_alignment_visualization,
)


PATIENT_ID = "00436515-870c-4b36-a041-de91049b9ab4"

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"
OUTPUT_DIR = ROOT / "outputs" / "explainability" / "bbox_alignment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(model_name, checkpoint_path):
    model = build_model(model_name, pretrained=False)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    return model


def find_one(pattern):
    candidates = list(EXPERIMENTS.glob(pattern))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint for pattern {pattern!r}; "
            f"found {len(candidates)}: {candidates}"
        )

    return candidates[0]


def prepare_patient():
    labels, _ = load_labels(TRAIN_LABELS)

    if PATIENT_ID not in labels:
        raise KeyError(f"Patient not found: {PATIENT_ID}")

    dicom_path = Path(TRAIN_IMAGES) / f"{PATIENT_ID}.dcm"
    if not dicom_path.exists():
        raise FileNotFoundError(dicom_path)

    raw_image, _ = load_dicom(dicom_path, return_metadata=True)
    processed = preprocess_image(raw_image, target_size=(512, 512))

    grayscale = torch.from_numpy(
        np.asarray(processed, dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0)

    adapter = AdvancedInputAdapter(
        image_size=224,
        use_imagenet_normalization=True,
    )
    model_input = adapter(grayscale).to(DEVICE)

    return raw_image, model_input, labels[PATIENT_ID]


def main():
    print("[1/5] Loading annotations and image...", flush=True)
    original_image, model_input, record = prepare_patient()
    boxes = record["bounding_boxes"]

    print(f"Patient: {PATIENT_ID}")
    print(f"Ground-truth target: {record['target']}")
    print(f"Radiologist boxes: {len(boxes)}")
    print(f"Original image shape: {original_image.shape}")

    if not boxes:
        raise ValueError(
            "This patient has no bounding boxes. Choose a positive "
            "patient with annotations for localization evaluation."
        )

    resnet_checkpoint = find_one(
        "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/"
        "checkpoints/best_model.pth"
    )
    vit_checkpoint = find_one(
        "vit_b16_pretrained_img224_bs4_lr0.0001_ep2_seed42/"
        "checkpoints/best_model.pth"
    )

    results = {}

    # --------------------------------------------------
    # ResNet-50 Grad-CAM
    # --------------------------------------------------
    print("[2/5] Generating ResNet-50 Grad-CAM...", flush=True)
    resnet = load_checkpoint_model("resnet50", resnet_checkpoint)
    cam_extractor = GradCAM(resnet, resnet.layer4[-1])

    try:
        gradcam, resnet_probability = cam_extractor.generate(
            model_input, target="positive"
        )
    finally:
        cam_extractor.close()

    resnet_metrics = save_alignment_visualization(
        original_image,
        gradcam,
        boxes,
        OUTPUT_DIR / "resnet50_gradcam_alignment.png",
        title="ResNet-50 Grad-CAM vs. RSNA boxes",
        percentile=80,
    )
    resnet_metrics["pneumonia_probability"] = resnet_probability
    results["resnet50_gradcam"] = resnet_metrics

    # --------------------------------------------------
    # ViT-B/16 CLS-to-patch attention
    # --------------------------------------------------
    print("[3/5] Generating ViT-B/16 attention...", flush=True)
    vit = load_checkpoint_model("vit_b16", vit_checkpoint)

    with ViTAttentionCapture(vit) as capture:
        with torch.no_grad():
            vit_output = vit(model_input)

        attention_layers = capture.get_attention_maps()
        vit_probability = torch.sigmoid(
            vit_output.reshape(-1)[0]
        ).item()

        vit_map = cls_attention_map(
            attention_layers[-1],
            image_size=224,
        )

    vit_metrics = save_alignment_visualization(
        original_image,
        vit_map,
        boxes,
        OUTPUT_DIR / "vit_attention_alignment.png",
        title="ViT-B/16 attention vs. RSNA boxes",
        percentile=80,
    )
    vit_metrics["pneumonia_probability"] = float(vit_probability)
    results["vit_b16_attention"] = vit_metrics

    # --------------------------------------------------
    # JSON report
    # --------------------------------------------------
    print("[4/5] Saving localization metrics...", flush=True)
    report = {
        "patient_id": PATIENT_ID,
        "ground_truth_target": int(record["target"]),
        "original_image_shape": list(original_image.shape),
        "bounding_boxes": boxes,
        "threshold_percentile": 80,
        "coordinate_space": "original_dicom_pixels",
        "results": results,
        "limitations": [
            "Single-patient result; not a dataset-level evaluation.",
            "Threshold percentile affects pixel overlap metrics.",
            "Attention and Grad-CAM are not causal explanations.",
            "Bounding boxes annotate regions, not every diseased pixel.",
            "Metrics do not establish clinical validity.",
        ],
    }

    report_path = OUTPUT_DIR / "localization_metrics.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("[5/5] Finished.", flush=True)
    print(f"ResNet visualization: {OUTPUT_DIR / 'resnet50_gradcam_alignment.png'}")
    print(f"ViT visualization: {OUTPUT_DIR / 'vit_attention_alignment.png'}")
    print(f"Metrics report: {report_path}")

    for name, metrics in results.items():
        print(f"\n{name}")
        for key in (
            "iou",
            "precision",
            "recall",
            "f1",
            "pointing_game_hit",
            "energy_inside_boxes",
        ):
            print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
