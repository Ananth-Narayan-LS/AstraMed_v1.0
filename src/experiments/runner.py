
"""
Unified experiment runner for AstraMed Sprint 5.

Supports the Sprint 4 baseline and five advanced architectures.
The existing dataset and baseline implementation remain unchanged.
"""
import random

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.data.advanced_input import AdvancedInputAdapter
from src.data.dataloader import build_dataloader
from src.data.dataset import RSNADataset
from src.data.labels import load_labels
from src.data.split_data import split_patients
from src.evaluations.metrics import calculate_metrics
from src.experiments.config import ExperimentConfig, get_device
from src.experiments.tracker import ExperimentTracker
from src.models.model_factory import build_model
from src.training.losses import build_loss
from src.training.train import train_one_epoch
from src.training.validate import validate
from src.utils.checkpoint import save_checkpoint
from src.utils.config import TRAIN_IMAGES, TRAIN_LABELS
from src.utils.reproducibility import set_seed


ADVANCED_MODELS = {
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_b16",
}


def _balanced_subset(
    patient_ids,
    labels,
    limit: int | None,
    seed: int,
):
    """
    Select a reproducible subset containing both classes
    when both classes are available in the supplied split.
    """

    if limit is None:
        return list(patient_ids)

    if limit < 2:
        raise ValueError(
            "A balanced smoke-test subset needs at least 2 samples."
        )

    by_class = {0: [], 1: []}

    for patient_id in patient_ids:
        target = int(labels[patient_id]["target"])
        by_class[target].append(patient_id)

    rng = random.Random(seed)

    for patient_ids_for_class in by_class.values():
        rng.shuffle(patient_ids_for_class)

    selected = []

    # Alternate classes to avoid taking a one-class prefix.
    while len(selected) < limit:
        added_sample = False

        for target in (0, 1):
            if by_class[target] and len(selected) < limit:
                selected.append(by_class[target].pop())
                added_sample = True

        if not added_sample:
            break

    if len(selected) < limit:
        print(
            f"Requested {limit} samples, but only "
            f"{len(selected)} were available."
        )

    return selected



def _collect_validation_metrics(
    model,
    dataloader,
    device,
    advanced: bool,
) -> dict:
    """Collect probabilities and compute validation metrics."""

    model.eval()
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["target"].reshape(-1)

            outputs = model(images)

            # Advanced models return logits; baseline returns probabilities.
            probabilities = (
                torch.sigmoid(outputs)
                if advanced
                else outputs
            )

            all_targets.extend(targets.cpu().numpy().tolist())
            all_probabilities.extend(
                probabilities.reshape(-1).cpu().numpy().tolist()
            )

    targets = np.asarray(all_targets, dtype=np.int64)
    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float64,
    )

    metrics = calculate_metrics(
        targets,
        probabilities,
        threshold=0.5,
    )

    # PR-AUC is average precision, calculated from probabilities.
    if len(np.unique(targets)) == 2:
        pr_auc = average_precision_score(
            targets,
            probabilities,
        )
    else:
        pr_auc = float("nan")

    # Convert NumPy values to JSON-friendly Python values.
    result = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            result[key] = np.asarray(value).tolist()
        elif isinstance(value, (int, float, np.integer, np.floating)):
            result[key] = float(value)
        else:
            result[key] = value

    result["pr_auc"] = float(pr_auc)

    return result


def run_experiment(
    config: ExperimentConfig,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> dict:
    """
    Run one reproducible model experiment.

    Optional sample limits are intended for short smoke tests.
    They select prefixes of the existing patient-level splits.
    """

    model_name = config.model_name.lower().strip()

    supported_models = ADVANCED_MODELS | {"baseline_cnn"}

    if model_name not in supported_models:
        raise ValueError(
            f"Unsupported model {model_name!r}. "
            f"Choose one of: {sorted(supported_models)}"
        )

    advanced = model_name in ADVANCED_MODELS

    if advanced and not config.pretrained:
        print("Advanced model will use random initialization.")

    # The experiment configuration is the source of truth for the seed.
    set_seed(config.seed)

    device = get_device()
    config.create_directories()

    tracker = ExperimentTracker(config.experiment_dir)

    start_time = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        print(f"Experiment: {config.experiment_name}")
        print(f"Model: {model_name}")
        print(f"Device: {device}")

        # 1. Load labels and create the patient-level split.
        labels, label_stats = load_labels(TRAIN_LABELS)

        train_ids, val_ids, _test_ids = split_patients(
            labels,
            random_seed=config.seed,
        )

        # Smoke-test limits only shorten the train/validation prefixes.
        # They do not change the original patient-level split.
        train_ids = _balanced_subset(
            train_ids,
            labels,
            limit=max_train_samples,
            seed=config.seed,
        )

        val_ids = _balanced_subset(
            val_ids,
            labels,
            limit=max_val_samples,
            seed=config.seed + 1,
        )

        if not train_ids or not val_ids:
            raise ValueError(
                "Training and validation splits must both be non-empty."
            )

        train_dataset = RSNADataset(
            train_ids,
            labels,
            TRAIN_IMAGES,
        )

        val_dataset = RSNADataset(
            val_ids,
            labels,
            TRAIN_IMAGES,
        )

        train_loader = build_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = build_dataloader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # 2. Build the selected model.
        model = build_model(
            model_name=model_name,
            pretrained=config.pretrained if advanced else False,
        ).to(device)

        # Advanced models receive the adapter. The baseline does not.
        adapter = None
        if advanced:
            adapter = AdvancedInputAdapter(
                image_size=config.model_image_size,
                use_imagenet_normalization=(
                    config.use_imagenet_normalization
                ),
            ).to(device)

        # A wrapper lets the existing training utilities receive the
        # correct model input without modifying the Dataset.
        if adapter is not None:
            model_for_training = _InputPipeline(model, adapter)
        else:
            model_for_training = model

        # 3. Select the correct loss for the model's output convention.
        criterion = build_loss(
            "bce_logits" if advanced else "bce"
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        tracker.log_metadata({
            "model_name": model_name,
            "pretrained": config.pretrained if advanced else False,
            "device": str(device),
            "seed": config.seed,
            "dataset_image_size": config.dataset_image_size,
            "model_image_size": (
                config.model_image_size if advanced
                else config.dataset_image_size
            ),
            "use_imagenet_normalization": advanced and (
                config.use_imagenet_normalization
            ),
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "label_stats": label_stats,
        })

        best_val_loss = float("inf")
        best_checkpoint = config.checkpoint_dir / "best_model.pth"

        # 4. Train and validate.
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.perf_counter()

            train_loss = train_one_epoch(
                model_for_training,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_loss = validate(
                model_for_training,
                val_loader,
                criterion,
                device,
            )

            metrics = _collect_validation_metrics(
                model_for_training,
                val_loader,
                device,
                advanced,
            )

            epoch_seconds = time.perf_counter() - epoch_start

            print(
                f"Epoch {epoch}/{config.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | "
                f"PR-AUC={metrics['pr_auc']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"time={epoch_seconds:.1f}s"
            )

            tracker.log_epoch(
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                metrics=metrics,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    path=best_checkpoint,
                )

        elapsed_seconds = time.perf_counter() - start_time

        peak_gpu_memory_mb = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2)
            )

        tracker.log_metadata({
            "status": "completed",
            "training_time_seconds": elapsed_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "best_validation_loss": float(best_val_loss),
            "best_checkpoint": str(best_checkpoint),
        })

        tracker.mark_completed()
        tracker.save()

        result = {
            "experiment_name": config.experiment_name,
            "model_name": model_name,
            "training_time_seconds": elapsed_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "best_validation_loss": float(best_val_loss),
            "best_checkpoint": str(best_checkpoint),
            "history": tracker.history,
        }

        print("\nExperiment completed.")
        print(f"Checkpoint: {best_checkpoint}")
        print(f"Results: {config.experiment_dir}")

        return result

    except Exception as error:
        tracker.mark_failed(str(error))
        tracker.log_metadata({
            "training_time_seconds": time.perf_counter() - start_time,
        })
        tracker.save()
        raise


class _InputPipeline(torch.nn.Module):
    """Apply the advanced input adapter before the selected model."""

    def __init__(self, model, adapter):
        super().__init__()
        self.model = model
        self.adapter = adapter

    def forward(self, images):
        return self.model(self.adapter(images))
