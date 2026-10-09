
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch

from src.data.labels import load_labels
from src.data.split_data import split_patients
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
    localization_metrics,
)
from src.explainability.quantitative_eval import (
    aggregate_metrics,
    write_csv,
    write_comparison_report,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"
OUTPUT_DIR = ROOT / "outputs" / "explainability" / "quantitative_evaluation"

THRESHOLDS = (50, 70, 80, 90)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate explanation localization on positive test patients."
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=100,
        help="Maximum positive test patients; use 0 for all available.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_checkpoint(experiment_name):
    path = (
        EXPERIMENTS / experiment_name
        / "checkpoints" / "best_model.pth"
    )

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return path


def load_model(model_name, checkpoint_path):
    model = build_model(model_name, pretrained=False)

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

    return model


def prepare_patient(patient_id, labels):
    dicom_path = Path(TRAIN_IMAGES) / f"{patient_id}.dcm"

    if not dicom_path.exists():
        raise FileNotFoundError(f"Missing DICOM: {dicom_path}")

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

    return raw_image, model_input, labels[patient_id]


def compute_explanations(resnet, vit, model_input):
    gradcam_extractor = GradCAM(resnet, resnet.layer4[-1])

    try:
        gradcam, resnet_probability = gradcam_extractor.generate(
            model_input, target="positive"
        )
    finally:
        gradcam_extractor.close()

    with ViTAttentionCapture(vit) as capture:
        with torch.no_grad():
            vit_output = vit(model_input)

        attention_layers = capture.get_attention_maps()
        vit_probability = torch.sigmoid(
            vit_output.reshape(-1)[0]
        ).item()

        attention_map = cls_attention_map(
            attention_layers[-1],
            image_size=224,
        )

    return {
        "resnet50_gradcam": (
            gradcam.numpy(),
            float(resnet_probability),
        ),
        "vit_b16_attention": (
            attention_map.numpy(),
            float(vit_probability),
        ),
    }


def main():
    args = parse_args()

    if args.max_cases < 0:
        raise ValueError("--max-cases must be 0 or a positive integer.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("[1/6] Loading labels and patient split...", flush=True)
    labels, _ = load_labels(TRAIN_LABELS)

    # Use the established split implementation, not a new random split.
    train_ids, val_ids, test_ids = split_patients(labels)

    train_ids, val_ids, test_ids = (
        set(train_ids), set(val_ids), set(test_ids)
    )

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Patient split overlap detected.")

    positive_test_ids = sorted(
        patient_id
        for patient_id in test_ids
        if patient_id in labels
        and labels[patient_id]["target"] == 1
        and labels[patient_id]["bounding_boxes"]
    )

    if not positive_test_ids:
        raise RuntimeError(
            "No positive patients with bounding boxes in the test split."
        )

    if args.max_cases:
        rng = random.Random(args.seed)
        rng.shuffle(positive_test_ids)
        patient_ids = sorted(positive_test_ids[:args.max_cases])
    else:
        patient_ids = positive_test_ids

    print(f"Positive test patients available: {len(positive_test_ids)}")
    print(f"Patients selected: {len(patient_ids)}")

    print("[2/6] Loading trained models...", flush=True)
    resnet = load_model(
        "resnet50",
        find_checkpoint(
            "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
        ),
    )
    vit = load_model(
        "vit_b16",
        find_checkpoint(
            "vit_b16_pretrained_img224_bs4_lr0.0001_ep2_seed42"
        ),
    )

    per_patient_rows = []
    threshold_rows = []
    failures = []

    print("[3/6] Evaluating patients...", flush=True)

    for index, patient_id in enumerate(patient_ids, start=1):
        print(
            f"[{index}/{len(patient_ids)}] {patient_id}",
            flush=True,
        )

        try:
            raw_image, model_input, record = prepare_patient(
                patient_id, labels
            )
            boxes = record["bounding_boxes"]

            explanations = compute_explanations(
                resnet, vit, model_input
            )

            for method, (heatmap, probability) in explanations.items():
                # The M5 metric function resizes the map to the original
                # image shape, preserving the original box coordinates.
                for percentile in THRESHOLDS:
                    metrics = localization_metrics(
                        heatmap,
                        boxes,
                        raw_image.shape[:2],
                        percentile=percentile,
                    )

                    row = {
                        "patient_id": patient_id,
                        "method": method,
                        "ground_truth_target": int(record["target"]),
                        "pneumonia_probability": probability,
                        "threshold_percentile": percentile,
                        "box_count": len(boxes),
                    }

                    for key in (
                        "iou",
                        "precision",
                        "recall",
                        "f1",
                        "pointing_game_hit",
                        "energy_inside_boxes",
                    ):
                        row[key] = metrics[key]

                    threshold_rows.append(row)

                    if percentile == 80:
                        per_patient_rows.append(row.copy())

        except Exception as exc:
            failures.append({
                "patient_id": patient_id,
                "error": repr(exc),
            })
            print(f"  FAILED: {exc}", flush=True)

    if not per_patient_rows:
        raise RuntimeError(
            "No patients were successfully evaluated. "
            "Check the error log and data paths."
        )

    print("[4/6] Aggregating metrics...", flush=True)
    aggregate_rows = aggregate_metrics(
        threshold_rows,
        group_keys=("method", "threshold_percentile"),
    )

    print("[5/6] Writing reports...", flush=True)
    write_csv(
        OUTPUT_DIR / "per_patient_metrics.csv",
        per_patient_rows,
    )
    write_csv(
        OUTPUT_DIR / "threshold_sensitivity.csv",
        threshold_rows,
    )
    write_csv(
        OUTPUT_DIR / "aggregate_metrics.csv",
        aggregate_rows,
    )

    write_comparison_report(
        OUTPUT_DIR / "method_comparison.md",
        aggregate_rows,
        primary_threshold=80,
    )

    metadata = {
        "seed": args.seed,
        "requested_max_cases": args.max_cases,
        "successful_patients": len({
            row["patient_id"] for row in per_patient_rows
        }),
        "failed_patients": len(failures),
        "thresholds": list(THRESHOLDS),
        "evaluated_split": "test",
        "positive_cases_only": True,
        "failures": failures,
        "limitations": [
            "Screening-trained checkpoints, not clinical validation.",
            "Positive patients with bounding boxes only.",
            "Metrics depend on heatmap threshold.",
            "Bounding boxes are imperfect proxies for disease extent.",
            "Per-patient observations may have varying annotation areas.",
        ],
    }

    (OUTPUT_DIR / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("[6/6] Evaluation complete.", flush=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Successfully evaluated: {metadata['successful_patients']}")
    print(f"Failed patients: {metadata['failed_patients']}")

    for row in aggregate_rows:
        if row["threshold_percentile"] != 80:
            continue
        print(
            f"{row['method']}: "
            f"IoU={row['iou_mean']:.4f} ± {row['iou_std']:.4f}, "
            f"F1={row['f1_mean']:.4f}, "
            f"pointing hit={row['pointing_game_hit_mean']:.4f}, "
            f"n={row['patient_count']}"
        )


if __name__ == "__main__":
    main()
