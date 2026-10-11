"""
Evaluate a saved AstraMed checkpoint on the patient-level validation and test splits.

Read-only with respect to checkpoints and experiment folders: results are written to
outputs/evaluation/<experiment_name>/. The test split is scored once at a FIXED
threshold (default 0.5). Threshold exploration is done on the VALIDATION split only.

Usage:
    python scripts/evaluate_model.py --experiment resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.advanced_input import AdvancedInputAdapter
from src.data.dataloader import build_dataloader
from src.data.dataset import RSNADataset
from src.data.labels import load_labels
from src.data.split_data import split_patients
from src.models.model_factory import build_model
from src.utils.config import OUTPUT_DIR, TRAIN_IMAGES, TRAIN_LABELS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_at(targets, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()

    def div(a, b):
        return float(a / b) if b else 0.0

    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": div(tn + tp, len(targets)),
        "precision": precision,
        "sensitivity": recall,
        "specificity": div(tn, tn + fp),
        "f1": div(2 * precision * recall, precision + recall),
    }


def score(model, adapter, ids, labels, device, batch_size):
    loader = build_dataloader(
        RSNADataset(ids, labels, TRAIN_IMAGES),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    targets, probs = [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(adapter(batch["image"].to(device)))
            probs += torch.sigmoid(logits).reshape(-1).cpu().tolist()
            targets += batch["target"].tolist()
    return np.asarray(targets), np.asarray(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fixed threshold used for the test split.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    checkpoint_path = (
        OUTPUT_DIR / "experiments" / args.experiment / "checkpoints" / "best_model.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = build_model(args.model, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    adapter = AdvancedInputAdapter(image_size=224, use_imagenet_normalization=True)
    adapter.to(device).eval()

    labels, label_stats = load_labels(TRAIN_LABELS)
    train_ids, val_ids, test_ids = split_patients(labels)
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_set & val_set or train_set & test_set or val_set & test_set)

    val_t, val_p = score(model, adapter, val_ids, labels, device, args.batch_size)
    test_t, test_p = score(model, adapter, test_ids, labels, device, args.batch_size)

    sweep = [metrics_at(val_t, val_p, th) for th in np.round(np.arange(0.05, 0.96, 0.05), 2)]
    result = {
        "experiment": args.experiment,
        "model": args.model,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": checkpoint.get("loss"),
        "label_stats": label_stats,
        "split": {"seed": 42, "train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "validation": {
            "n": int(len(val_t)), "positives": int(val_t.sum()),
            "roc_auc": float(roc_auc_score(val_t, val_p)),
            "pr_auc": float(average_precision_score(val_t, val_p)),
            "at_0.5": metrics_at(val_t, val_p, 0.5),
            "threshold_sweep": sweep,
        },
        "test": {
            "n": int(len(test_t)), "positives": int(test_t.sum()),
            "note": "Scored once at a fixed threshold; not used for threshold selection.",
            "roc_auc": float(roc_auc_score(test_t, test_p)),
            "pr_auc": float(average_precision_score(test_t, test_p)),
            "at_fixed_threshold": metrics_at(test_t, test_p, args.threshold),
        },
    }

    out_dir = OUTPUT_DIR / "evaluation" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "split_ids.json").write_text(
        json.dumps({"seed": 42, "train": train_ids, "val": val_ids, "test": test_ids}),
        encoding="utf-8",
    )
    print("checkpoint sha256:", result["checkpoint_sha256"], "epoch:", result["checkpoint_epoch"])
    print("TEST:", json.dumps(result["test"]["at_fixed_threshold"]),
          "ROC-AUC %.4f PR-AUC %.4f" % (result["test"]["roc_auc"], result["test"]["pr_auc"]))
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
