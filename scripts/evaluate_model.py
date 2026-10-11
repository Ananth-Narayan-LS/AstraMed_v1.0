"""
Evaluate a saved AstraMed checkpoint on the patient-level splits.

The checkpoint and experiment folder are only read. Outputs go to
outputs/evaluation/<experiment_name>/ and existing output files are never overwritten.

Two modes, deliberately separated:

  validation (default)  Scores the VALIDATION split, writes per-patient predictions
                        (validation_predictions.csv) and a validation-only threshold
                        sweep (validation_threshold_sweep.json). All threshold
                        exploration belongs here.

  test                  Scores the TEST split ONCE at the fixed deployed threshold
                        (0.5). There is no threshold option. It refuses to run if
                        evaluation.json (which already holds test results) exists.

Usage:
    python scripts/evaluate_model.py --experiment <name>
    python scripts/evaluate_model.py --experiment <name> --split test
"""
import argparse
import csv
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

FIXED_TEST_THRESHOLD = 0.5  # the deployed threshold; not configurable here


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
    """Return patient IDs, targets and probabilities in the same order."""
    loader = build_dataloader(
        RSNADataset(ids, labels, TRAIN_IMAGES),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    patient_ids, targets, probs = [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(adapter(batch["image"].to(device)))
            probs += torch.sigmoid(logits).reshape(-1).cpu().tolist()
            targets += batch["target"].tolist()
            patient_ids += list(batch["patient_id"])
    if patient_ids != list(ids):
        raise RuntimeError("Scored patient order does not match the requested split.")
    for pid, target in zip(patient_ids, targets):
        if int(labels[pid]["target"]) != int(target):
            raise RuntimeError(f"Label mismatch for patient {pid}.")
    return patient_ids, np.asarray(targets), np.asarray(probs)


def refuse_overwrite(path: Path):
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite it.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / "evaluation" / args.experiment
    if args.split == "test":
        refuse_overwrite(out_dir / "evaluation.json")
    else:
        refuse_overwrite(out_dir / "validation_predictions.csv")
        refuse_overwrite(out_dir / "validation_threshold_sweep.json")

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
    if (
        len(train_set) != len(train_ids)
        or len(val_set) != len(val_ids)
        or len(test_set) != len(test_ids)
        or train_set & val_set
        or train_set & test_set
        or val_set & test_set
    ):
        raise RuntimeError("Patient splits overlap or contain duplicate patients.")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_file = out_dir / "split_ids.json"
    current_split = {"seed": 42, "train": train_ids, "val": val_ids, "test": test_ids}
    if split_file.exists():
        if json.loads(split_file.read_text(encoding="utf-8")) != current_split:
            raise RuntimeError("Regenerated split differs from the saved split_ids.json.")
    else:
        split_file.write_text(json.dumps(current_split), encoding="utf-8")

    header = {
        "experiment": args.experiment,
        "model": args.model,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_loss": checkpoint.get("loss"),
        "label_stats": label_stats,
        "split": {"seed": 42, "train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
    }

    if args.split == "validation":
        pids, targets, probs = score(model, adapter, val_ids, labels, device, args.batch_size)
        with (out_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient_id", "target", "probability"])
            writer.writerows(zip(pids, targets.tolist(), (f"{p:.8f}" for p in probs)))
        result = {
            **header,
            "split_evaluated": "validation",
            "n": int(len(targets)), "positives": int(targets.sum()),
            "roc_auc": float(roc_auc_score(targets, probs)),
            "pr_auc": float(average_precision_score(targets, probs)),
            "threshold_sweep": [
                metrics_at(targets, probs, th) for th in np.round(np.arange(0.05, 0.96, 0.05), 2)
            ],
        }
        (out_dir / "validation_threshold_sweep.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print("Validation:", result["n"], "patients; ROC-AUC %.4f PR-AUC %.4f"
              % (result["roc_auc"], result["pr_auc"]))
    else:
        pids, targets, probs = score(model, adapter, test_ids, labels, device, args.batch_size)
        result = {
            **header,
            "split_evaluated": "test",
            "note": "Scored once at the fixed deployed threshold; not used for threshold selection.",
            "n": int(len(targets)), "positives": int(targets.sum()),
            "roc_auc": float(roc_auc_score(targets, probs)),
            "pr_auc": float(average_precision_score(targets, probs)),
            "at_fixed_threshold": metrics_at(targets, probs, FIXED_TEST_THRESHOLD),
        }
        (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("TEST:", json.dumps(result["at_fixed_threshold"]))
    print("Wrote results to", out_dir)


if __name__ == "__main__":
    main()
