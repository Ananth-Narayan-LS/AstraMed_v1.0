"""
Validation-only comparison of the frozen image-only baseline and Sprint 8 arms.

Reads per-patient VALIDATION predictions only (no model inference except an optional
parity check, no test split). All arms are scored on the same 4,002 validation patients.
Differences use a paired bootstrap over patients.

    python scripts/compare_multimodal.py --arms image_only_resnet50_seed42 structured_resnet50_seed42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import OUTPUT_DIR

BASELINE_CSV = (OUTPUT_DIR / "evaluation" / "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
                / "validation_predictions.csv")
MULTIMODAL_DIR = OUTPUT_DIR / "multimodal"
THRESHOLD = 0.5  # the deployed baseline threshold; reported for comparison only


def point_metrics(y, p, threshold=THRESHOLD):
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else 0.0
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "sensitivity": recall, "specificity": div(tn, tn + fp), "precision": precision,
        "f1": div(2 * precision * recall, precision + recall),
    }


def paired_bootstrap(y, p_a, p_b, n=2000, seed=0):
    """95% interval of metric(b) - metric(a) over resampled validation patients."""
    rng = np.random.default_rng(seed)
    d_auc, d_ap = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d_auc.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
        d_ap.append(average_precision_score(y[i], p_b[i]) - average_precision_score(y[i], p_a[i]))
    interval = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {
        "delta_roc_auc": float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
        "delta_roc_auc_95ci": interval(d_auc),
        "delta_pr_auc": float(average_precision_score(y, p_b) - average_precision_score(y, p_a)),
        "delta_pr_auc_95ci": interval(d_ap),
    }


def cache_parity(baseline, count):
    """Baseline model on cached inputs vs the saved DICOM-pipeline predictions (read-only)."""
    import torch
    from src.data.advanced_input import AdvancedInputAdapter
    from src.models.model_factory import build_model
    from src.multimodal.image_cache import CachedImageStore
    from src.multimodal.training import BASELINE_CHECKPOINT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location=device, weights_only=True)
    model = build_model("resnet50", pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    adapter = AdvancedInputAdapter(224, True).to(device).eval()
    store = CachedImageStore(MULTIMODAL_DIR / "cache" / "images_val_224_fp16")
    rows = baseline.iloc[:count]
    with torch.inference_mode():
        x = torch.stack([store.get(pid) for pid in rows.patient_id]).to(device)
        cached_p = torch.sigmoid(model(adapter(x))).reshape(-1).cpu().numpy()
    diff = np.abs(cached_p - rows.probability.to_numpy())
    return {"patients_checked": int(len(rows)), "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--parity-check", type=int, default=0)
    args = parser.parse_args()

    baseline = pd.read_csv(BASELINE_CSV)
    y = baseline.target.to_numpy()
    results = {
        "protocol": "validation split only (4,002 patients); no test data used; threshold 0.5 for point metrics",
        "n": int(len(y)), "positives": int(y.sum()),
        "frozen_image_only_baseline": point_metrics(y, baseline.probability.to_numpy()),
        "arms": {},
    }
    if args.parity_check:
        results["cache_parity_check"] = cache_parity(baseline, args.parity_check)

    predictions = {}
    for arm in args.arms:
        frame = pd.read_csv(MULTIMODAL_DIR / arm / "val_predictions.csv")
        if list(frame.patient_id) != list(baseline.patient_id) or not np.array_equal(frame.target, baseline.target):
            raise RuntimeError(f"{arm}: validation patients/labels differ from the baseline's.")
        predictions[arm] = frame.probability.to_numpy()
        meta = json.loads((MULTIMODAL_DIR / arm / "metadata.json").read_text(encoding="utf-8"))
        results["arms"][arm] = {
            "modalities_used": meta["modalities_used"], "best_epoch": meta["best_epoch"],
            "training_seconds": meta["training_seconds"], "peak_gpu_memory_mb": meta["peak_gpu_memory_mb"],
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "metrics": point_metrics(y, predictions[arm]),
            "vs_frozen_baseline": paired_bootstrap(y, baseline.probability.to_numpy(), predictions[arm]),
        }
    for arm in args.arms:
        control = arm.replace("structured_view", "image_only").replace("structured", "image_only")
        if control != arm and control in predictions:
            results["arms"][arm]["vs_image_only_control"] = paired_bootstrap(
                y, predictions[control], predictions[arm])

    out = MULTIMODAL_DIR / "validation_comparison.json"
    if out.exists():
        out = MULTIMODAL_DIR / f"validation_comparison_{'_'.join(args.arms)[:60]}.json"
    if out.exists():
        raise FileExistsError(f"{out} already exists; refusing to overwrite it.")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
