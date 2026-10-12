"""
Train one Sprint 8 multimodal research arm (validation only; the test split is never loaded).

Arms (same recipe, same seed):
    --arm image_only    image encoder + fusion head, no clinical input (control)
    --arm structured    + age and sex
    --arm structured_view   + age, sex and view position (shortcut-risk ablation)

Prerequisites: python scripts/build_image_cache.py
    python scripts/train_multimodal.py --arm structured --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import load_labels
from src.multimodal.clinical_data import extract_header_table
from src.multimodal.training import MULTIMODAL_DIR, MultimodalConfig, train_multimodal
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS

ARMS = {
    "image_only": dict(use_structured=False, use_view_position=False),
    "structured": dict(use_structured=True, use_view_position=False),
    "structured_view": dict(use_structured=True, use_view_position=True),
}


def load_clinical_table() -> pd.DataFrame:
    path = MULTIMODAL_DIR / "clinical_metadata.csv"
    if path.exists():
        return pd.read_csv(path, keep_default_na=True)
    labels, _ = load_labels(TRAIN_LABELS)
    table = extract_header_table(sorted(labels), TRAIN_IMAGES, workers=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    cfg = MultimodalConfig(
        name=f"{args.arm}_resnet50_seed{args.seed}", seed=args.seed, epochs=args.epochs, **ARMS[args.arm]
    )
    result = train_multimodal(cfg, load_clinical_table())
    print(json.dumps({k: result[k] for k in ("name", "status", "best_epoch", "best_val_pr_auc",
                                              "final_val_metrics", "training_seconds")}, indent=2))


if __name__ == "__main__":
    main()
