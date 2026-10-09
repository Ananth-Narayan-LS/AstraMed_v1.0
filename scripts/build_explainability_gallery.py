
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
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
    save_alignment_visualization,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"
EVAL_DIR = ROOT / "outputs" / "explainability" / "quantitative_evaluation"
GALLERY_DIR = ROOT / "outputs" / "explainability" / "gallery"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name, experiment):
    checkpoint_path = (
        EXPERIMENTS / experiment / "checkpoints" / "best_model.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = build_model(name, pretrained=False)

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint),
        strict=True,
    )
    return model.to(DEVICE).eval()


def prepare_patient(patient_id, labels):
    path = Path(TRAIN_IMAGES) / f"{patient_id}.dcm"
    if not path.exists():
        raise FileNotFoundError(path)

    raw_image, _ = load_dicom(path, return_metadata=True)
    processed = preprocess_image(raw_image, target_size=(512, 512))

    grayscale = torch.from_numpy(
        np.asarray(processed, dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0)

    adapter = AdvancedInputAdapter(
        image_size=224,
        use_imagenet_normalization=True,
    )
    model_input = adapter(grayscale).to(DEVICE)

    return raw_image, model_input, labels[patient_id]


def generate_maps(resnet, vit, model_input):
    extractor = GradCAM(resnet, resnet.layer4[-1])
    try:
        gradcam, resnet_probability = extractor.generate(
            model_input, target="positive"
        )
    finally:
        extractor.close()

    with ViTAttentionCapture(vit) as capture:
        with torch.no_grad():
            vit_output = vit(model_input)
        attentions = capture.get_attention_maps()
        vit_probability = torch.sigmoid(
            vit_output.reshape(-1)[0]
        ).item()
        vit_map = cls_attention_map(attentions[-1], image_size=224)

    return {
        "resnet50_gradcam": (gradcam.numpy(), float(resnet_probability)),
        "vit_b16_attention": (vit_map.numpy(), float(vit_probability)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=12)
    args = parser.parse_args()

    if args.cases < 1:
        raise ValueError("--cases must be at least 1.")

    csv_path = EVAL_DIR / "per_patient_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run M6 evaluation first."
        )

    evaluation = pd.read_csv(csv_path)
    if "patient_id" not in evaluation.columns:
        raise ValueError("M6 CSV must contain patient_id.")

    # Preserve the same evaluated cohort, with deterministic selection.
    patient_ids = sorted(evaluation["patient_id"].dropna().unique())[:args.cases]

    labels, _ = load_labels(TRAIN_LABELS)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ResNet-50...", flush=True)
    resnet = load_model(
        "resnet50",
        "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42",
    )

    print("Loading ViT-B/16...", flush=True)
    vit = load_model(
        "vit_b16",
        "vit_b16_pretrained_img224_bs4_lr0.0001_ep2_seed42",
    )

    failures = []

    for index, patient_id in enumerate(patient_ids, start=1):
        print(f"[{index}/{len(patient_ids)}] {patient_id}", flush=True)

        try:
            raw_image, model_input, record = prepare_patient(
                patient_id, labels
            )
            maps = generate_maps(resnet, vit, model_input)

            for method, (heatmap, probability) in maps.items():
                output_path = (
                    GALLERY_DIR / f"{patient_id}_{method}.png"
                )

                metrics = save_alignment_visualization(
                    original_image=raw_image,
                    heatmap=heatmap,
                    boxes=record["bounding_boxes"],
                    output_path=output_path,
                    title=(
                        f"{method} | P(pneumonia)={probability:.3f}"
                    ),
                    percentile=80,
                )

                print(
                    f"  Saved {output_path.name}; "
                    f"IoU={metrics['iou']:.3f}",
                    flush=True,
                )

        except Exception as exc:
            failures.append({"patient_id": patient_id, "error": repr(exc)})
            print(f"  FAILED: {exc}", flush=True)

    failure_path = GALLERY_DIR / "gallery_failures.txt"
    failure_path.write_text(
        "\n".join(f"{item['patient_id']}: {item['error']}" for item in failures),
        encoding="utf-8",
    )

    print(f"\nGallery directory: {GALLERY_DIR.resolve()}")
    print(f"Patients selected: {len(patient_ids)}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
