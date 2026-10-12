"""
Training for the Sprint 8 multimodal research models.

Pre-specified recipe (the same for every arm, set before any validation result was seen):
  * Existing patient-level split; train on TRAIN only; model selection on VALIDATION only.
  * The TEST split is never loaded (forbidden_ids guard).
  * Image encoder initialised from the frozen image-only baseline checkpoint (read-only).
  * Epoch 1: image encoder frozen (BatchNorm in eval mode), new layers trained.
  * Epochs 2+: image encoder unfrozen at a small learning rate.
  * Plain BCE-with-logits (no class weighting), same as the baseline, so arms stay comparable.
  * Best epoch = highest validation PR-AUC (average precision); early stopping patience 2.
Outputs go to outputs/multimodal/<name>/ and are never overwritten.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.labels import load_labels
from src.data.split_data import split_patients
from src.multimodal.clinical_data import (
    ClinicalFeatureEncoder, missingness_report, validate_join,
)
from src.multimodal.dataset import MultimodalDataset, collate_batch, move_to_device
from src.multimodal.image_cache import CachedImageStore
from src.multimodal.model import MultimodalPneumoniaNet, save_multimodal_checkpoint
from src.utils.config import OUTPUT_DIR, TRAIN_LABELS
from src.utils.reproducibility import set_seed

MULTIMODAL_DIR = OUTPUT_DIR / "multimodal"
BASELINE_CHECKPOINT = (
    OUTPUT_DIR / "experiments" / "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
    / "checkpoints" / "best_model.pth"
)
BASELINE_SHA256 = "77A82AAC41C7EE5D4D7B5A55964008E7DA7DFB999AAB602BC14DE045FA53ED26"


@dataclass
class MultimodalConfig:
    name: str
    use_structured: bool = False      # False = image-only control with the same fusion head
    use_view_position: bool = False   # shortcut-risk ablation only
    seed: int = 42
    epochs: int = 4
    warmup_epochs: int = 1
    batch_size: int = 32
    head_lr: float = 3e-4
    backbone_lr: float = 2e-5
    patience: int = 2
    selection_metric: str = "val_pr_auc"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OUTPUT_DIR.parent, text=True
        ).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    ids, targets, probs = [], [], []
    for batch in loader:
        batch_on_device = move_to_device(batch, device)
        probs += model.predict_proba(batch_on_device).cpu().tolist()
        targets += batch["target"].tolist()
        ids += batch["patient_id"]
    return ids, np.asarray(targets), np.asarray(probs)


def validation_metrics(targets, probs) -> dict:
    eps = 1e-7
    clipped = np.clip(probs, eps, 1 - eps)
    loss = float(-(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)).mean())
    return {
        "val_loss": loss,
        "val_roc_auc": float(roc_auc_score(targets, probs)),
        "val_pr_auc": float(average_precision_score(targets, probs)),
    }


def prepare_data(cfg: MultimodalConfig, clinical_table: pd.DataFrame):
    labels, label_stats = load_labels(TRAIN_LABELS)
    train_ids, val_ids, test_ids = split_patients(labels)
    validate_join(clinical_table, labels.keys())

    table_by_id = clinical_table.set_index("patient_id")
    encoder = ClinicalFeatureEncoder(use_view_position=cfg.use_view_position)
    encoder.fit(table_by_id.loc[train_ids].reset_index())   # TRAIN ONLY

    cache = MULTIMODAL_DIR / "cache"
    train_set = MultimodalDataset(
        train_ids, labels, clinical_table, encoder,
        CachedImageStore(cache / "images_train_224_fp16"), forbidden_ids=test_ids,
    )
    val_set = MultimodalDataset(
        val_ids, labels, clinical_table, encoder,
        CachedImageStore(cache / "images_val_224_fp16"), forbidden_ids=test_ids,
    )
    return labels, label_stats, train_ids, val_ids, encoder, train_set, val_set


def train_multimodal(cfg: MultimodalConfig, clinical_table: pd.DataFrame) -> dict:
    experiment_dir = MULTIMODAL_DIR / cfg.name
    checkpoint_path = experiment_dir / "best_model.pth"
    if experiment_dir.exists():
        raise FileExistsError(f"{experiment_dir} already exists; choose a new name.")
    if sha256_file(BASELINE_CHECKPOINT) != BASELINE_SHA256:
        raise RuntimeError("Baseline checkpoint hash differs from the documented value; stopping.")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, label_stats, train_ids, val_ids, encoder, train_set, val_set = prepare_data(
        cfg, clinical_table
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, generator=generator,
        collate_fn=collate_batch, num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=64, shuffle=False, collate_fn=collate_batch, num_workers=0,
    )

    model = MultimodalPneumoniaNet(
        struct_dim=encoder.dim, use_structured=cfg.use_structured
    )
    init_report = model.load_image_encoder_from_baseline(BASELINE_CHECKPOINT)
    model.to(device)

    backbone_params = list(model.image_encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.Adam([
        {"params": head_params, "lr": cfg.head_lr},
        {"params": backbone_params, "lr": cfg.backbone_lr},
    ])
    criterion = torch.nn.BCEWithLogitsLoss()

    experiment_dir.mkdir(parents=True)
    metadata = {
        "name": cfg.name, "status": "running", "config": asdict(cfg),
        "modalities_used": ["image"] + (["structured:age,sex" + (",view_position" if cfg.use_view_position else "")]
                                         if cfg.use_structured else []),
        "modalities_unavailable": ["symptoms_text", "clinical_history_text"],
        "git_commit_at_start": git_commit(),
        "baseline_checkpoint_sha256": BASELINE_SHA256,
        "image_encoder_initialisation": init_report,
        "feature_schema": encoder.to_dict(),
        "split": {"train": len(train_ids), "val": len(val_ids), "test_used": False},
        "label_stats": label_stats,
        "missingness_train": missingness_report(
            clinical_table.set_index("patient_id").loc[train_ids].reset_index()),
        "missingness_val": missingness_report(
            clinical_table.set_index("patient_id").loc[val_ids].reset_index()),
        "environment": {
            "torch": torch.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "preprocessing": "DICOM -> min-max -> 512 -> 224 (cached fp16) -> 3ch -> ImageNet norm; no augmentation",
    }
    (experiment_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    history, best, best_epoch, stale = [], -1.0, 0, 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        for epoch in range(1, cfg.epochs + 1):
            frozen = epoch <= cfg.warmup_epochs
            model.set_image_encoder_trainable(not frozen)
            model.train()
            if frozen:
                model.image_encoder.eval()  # keep BatchNorm statistics fixed while frozen

            running, count = 0.0, 0
            for batch in train_loader:
                batch = move_to_device(batch, device)
                logits = model(batch).reshape(-1)
                loss = criterion(logits, batch["target"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item() * len(logits)
                count += len(logits)

            _, val_targets, val_probs = predict_loader(model, val_loader, device)
            record = {"epoch": epoch, "image_encoder_frozen": frozen,
                      "train_loss": running / count, **validation_metrics(val_targets, val_probs)}
            history.append(record)
            (experiment_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()}),
                  flush=True)

            score = record[cfg.selection_metric]
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                if checkpoint_path.exists():
                    checkpoint_path.unlink()  # only our own earlier best checkpoint in this new run dir
                save_multimodal_checkpoint(
                    model, optimizer, epoch, score, checkpoint_path,
                    extra={"feature_schema": encoder.to_dict(), "config": asdict(cfg)},
                )
            else:
                stale += 1
                if stale >= cfg.patience:
                    print("Early stopping.", flush=True)
                    break
    except Exception as error:
        metadata.update(status="failed", error=repr(error))
        (experiment_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise

    # Final validation predictions from the BEST checkpoint (local file; patient-level).
    from src.multimodal.model import load_multimodal_checkpoint

    best_model, _ = load_multimodal_checkpoint(checkpoint_path, device)
    ids, targets, probs = predict_loader(best_model, val_loader, device)
    pd.DataFrame({"patient_id": ids, "target": targets.astype(int), "probability": probs}).to_csv(
        experiment_dir / "val_predictions.csv", index=False, float_format="%.8f")

    metadata.update(
        status="completed", best_epoch=best_epoch, best_val_pr_auc=best,
        final_val_metrics=validation_metrics(targets, probs),
        training_seconds=time.perf_counter() - start,
        peak_gpu_memory_mb=(torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None),
        checkpoint=str(checkpoint_path), checkpoint_sha256=sha256_file(checkpoint_path),
    )
    (experiment_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
