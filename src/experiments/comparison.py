
"""Generate AstraMed model comparison reports from saved experiments."""

import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports" / "experiments"

MODELS = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_b16": "ViT-B/16",
}

# Baseline numbers supplied in the milestone specification.
# Verify their source and evaluation protocol before final model selection.
BASELINE = {
    "Model": "Baseline CNN",
    "ROC-AUC": 0.727,
    "PR-AUC": None,
    "Recall": 0.003,
    "Precision": 0.333,
    "F1": 0.007,
    "Params": None,
    "Time (s)": None,
    "GPU Mem (MB)": None,
    "Status": "provided_baseline_values",
}


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def number(value):
    """Return a finite float, or None when missing/invalid."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def find_value(data, aliases):
    """Find a dictionary value regardless of common key naming styles."""
    if not isinstance(data, dict):
        return None

    normalized = {
        str(key).lower().replace("-", "_").replace(" ", "_"): value
        for key, value in data.items()
    }

    for alias in aliases:
        key = alias.lower().replace("-", "_").replace(" ", "_")
        if key in normalized:
            return normalized[key]

    return None


def load_history(path):
    data = read_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("history", "epochs", "epoch_history"):
            if isinstance(data.get(key), list):
                return data[key]

    return []


def find_latest_experiment(model_name):
    """Find the newest saved experiment matching this model."""
    if not EXPERIMENTS_DIR.exists():
        return None

    candidates = []

    for folder in EXPERIMENTS_DIR.iterdir():
        if not folder.is_dir():
            continue

        metadata_path = folder / "metadata.json"
        history_path = folder / "history.json"

        if not metadata_path.exists() or not history_path.exists():
            continue

        try:
            metadata = read_json(metadata_path)
            history = load_history(history_path)
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping {folder.name}: {error}")
            continue

        saved_model = str(
            find_value(metadata, ["model_name", "model"]) or ""
        ).lower().strip()

        folder_name = folder.name.lower()

        if not (
            saved_model == model_name
            or folder_name == model_name
            or folder_name.startswith(model_name + "_")
        ):
            continue

        status = str(
            find_value(metadata, ["status", "experiment_status"]) or ""
        ).lower()

        if status in {"failed", "error"}:
            continue

        candidates.append({
            "folder": folder,
            "metadata": metadata,
            "history": history,
        })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["folder"].stat().st_mtime,
    )


def get_epoch_metrics(epoch):
    """Read metrics from the actual tracker format or common alternatives."""
    nested = find_value(
        epoch, ["metrics", "validation_metrics", "val_metrics"]
    )
    data = nested if isinstance(nested, dict) else epoch

    aliases = {
        "roc_auc": ["roc_auc", "rocauc"],
        "pr_auc": [
            "pr_auc", "average_precision",
            "average_precision_score",
        ],
        "recall": ["recall", "sensitivity"],
        "precision": ["precision"],
        "f1": ["f1", "f1_score"],
    }

    return {
        name: number(find_value(data, keys))
        for name, keys in aliases.items()
    }


def build_report(model_name):
    experiment = find_latest_experiment(model_name)

    if experiment is None:
        return {
            "model_name": model_name,
            "display_name": MODELS[model_name],
            "status": "not_run_or_no_saved_results",
            "experiment_name": None,
            "metrics": {
                "roc_auc": None,
                "pr_auc": None,
                "recall": None,
                "precision": None,
                "f1": None,
            },
            "parameters": None,
            "training_time_seconds": None,
            "peak_gpu_memory_mb": None,
            "best_validation_loss": None,
            "configured_epochs": None,
            "recorded_epochs": 0,
            "checkpoint": None,
        }

    folder = experiment["folder"]
    metadata = experiment["metadata"]
    history = [
        row for row in experiment["history"]
        if isinstance(row, dict)
    ]

    configured_epochs = number(
        find_value(metadata, ["epochs", "num_epochs"])
    )
    configured_epochs = (
        int(configured_epochs) if configured_epochs is not None else None
    )

    # Select the recorded epoch with the lowest validation loss.
    rows_with_loss = [
        row for row in history
        if number(find_value(row, ["val_loss", "validation_loss"])) is not None
    ]

    if rows_with_loss:
        best_epoch = min(
            rows_with_loss,
            key=lambda row: number(
                find_value(row, ["val_loss", "validation_loss"])
            ),
        )
    else:
        best_epoch = history[-1] if history else None

    metrics = get_epoch_metrics(best_epoch or {})

    params = number(find_value(metadata, [
        "total_parameters", "parameter_count", "params"
    ]))
    runtime = number(find_value(metadata, [
        "training_time_seconds", "total_runtime_seconds"
    ]))
    gpu_memory = number(find_value(metadata, [
        "peak_gpu_memory_mb", "gpu_memory_mb"
    ]))
    train_samples = number(find_value(metadata, [
        "train_samples", "num_train_samples"
    ]))
    val_samples = number(find_value(metadata, [
        "val_samples", "validation_samples", "num_val_samples"
    ]))
    val_loss = number(find_value(best_epoch or {}, [
        "val_loss", "validation_loss"
    ]))

    original_status = str(
        find_value(metadata, ["status", "experiment_status"]) or "unknown"
    ).lower()

    # A tiny smoke test is not a valid benchmark run.
    is_smoke_test = (
        "smoke" in folder.name.lower()
        or (train_samples is not None and train_samples <= 8)
        or (val_samples is not None and val_samples <= 4)
    )

    incomplete_history = (
        configured_epochs is not None
        and len(history) < configured_epochs
    )

    if is_smoke_test:
        status = "smoke_test_not_for_selection"
    elif incomplete_history:
        status = "incomplete_history"
    else:
        status = original_status

    checkpoint = find_value(metadata, [
        "best_checkpoint", "checkpoint_path", "checkpoint"
    ])

    return {
        "model_name": model_name,
        "display_name": MODELS[model_name],
        "status": status,
        "original_metadata_status": original_status,
        "experiment_name": folder.name,
        "experiment_directory": str(folder),
        "pretrained": find_value(metadata, ["pretrained"]),
        "batch_size": find_value(metadata, ["batch_size"]),
        "learning_rate": find_value(metadata, ["learning_rate", "lr"]),
        "seed": find_value(metadata, ["seed"]),
        "configured_epochs": configured_epochs,
        "recorded_epochs": len(history),
        "train_samples": (
            int(train_samples) if train_samples is not None else None
        ),
        "val_samples": (
            int(val_samples) if val_samples is not None else None
        ),
        "metrics": metrics,
        "parameters": int(params) if params is not None else None,
        "training_time_seconds": runtime,
        "peak_gpu_memory_mb": gpu_memory,
        "best_validation_loss": val_loss,
        "selected_epoch": find_value(best_epoch or {}, ["epoch"]),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "note": (
            "Do not use smoke tests or incomplete histories to select "
            "the final model. Missing values are not zero."
        ),
    }


def comparison_row(report):
    metrics = report["metrics"]

    return {
        "Model": report["display_name"],
        "ROC-AUC": metrics.get("roc_auc"),
        "PR-AUC": metrics.get("pr_auc"),
        "Recall": metrics.get("recall"),
        "Precision": metrics.get("precision"),
        "F1": metrics.get("f1"),
        "Params": report.get("parameters"),
        "Time (s)": report.get("training_time_seconds"),
        "GPU Mem (MB)": report.get("peak_gpu_memory_mb"),
        "Status": report.get("status"),
    }


def format_value(value, digits=4):
    value = number(value)
    return f"{value:.{digits}f}" if value is not None else "N/A"


def write_csv(path, rows):
    columns = [
        "Model", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1",
        "Params", "Time (s)", "GPU Mem (MB)", "Status",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    """
        ## Preliminary Model Selection

        ### Selected candidate: ResNet-50

        ResNet-50 is selected as the provisional candidate for the next stage of AstraMed experiments. In the current screening benchmark, it achieved a ROC-AUC of 0.8356, PR-AUC of 0.8179, and F1-score of 0.7923.

        ### Efficiency alternative: EfficientNet-B0

        EfficientNet-B0 is retained as the efficiency-focused alternative. It achieved a ROC-AUC of 0.8207 while using approximately 427 MB of peak GPU memory and taking approximately 201 seconds for the screening run.

        ### Experimental limitations

        The comparison used pretrained models, 224 × 224 model inputs, a batch size of 4, a learning rate of 0.0001, two epochs, 1,000 training samples, and 300 validation samples. Results are preliminary and may change with longer training and larger datasets.

        The baseline CNN values were supplied separately and should be verified against their original experiment configuration before direct comparison.

        ### Next-stage validation

        Before finalizing the model, train the strongest candidates for longer, compare them using the same patient-level data split, examine sensitivity and specificity, and evaluate the selected model on a held-out test set that was not used for model selection.

        **Decision:** ResNet-50 is the provisional candidate; final selection remains subject to further validation.
    
    """
    columns = [
        "Model", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1",
        "Params", "Time (s)", "GPU Mem (MB)", "Status",
    ]

    lines = [
        "# AstraMed Model Comparison",
        "",
        "Generated from saved experiment artifacts.",
        "",
        "N/A means a value was not recorded. It does not mean zero.",
        "Smoke tests and incomplete histories must not be used for final model selection.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in rows:
        values = []

        for column in columns:
            value = row.get(column)

            if column in {
                "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"
            }:
                values.append(format_value(value))
            elif column == "Time (s)":
                values.append(format_value(value, 2))
            elif column == "GPU Mem (MB)":
                values.append(format_value(value, 2))
            elif column == "Params" and value is not None:
                values.append(f"{int(value):,}")
            else:
                values.append(str(value) if value is not None else "N/A")

        lines.append("| " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- ROC-AUC and PR-AUC measure ranking performance.",
        "- Recall measures how many positive cases are detected.",
        "- Precision measures how many predicted positives are correct.",
        "- F1 balances precision and recall.",
        "- Parameter count, runtime, and GPU memory describe model cost.",
        "- Compare only runs using consistent patient splits and evaluation settings.",
        "- The baseline metrics were supplied in the milestone specification; verify their provenance.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def generate_reports():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    reports = []

    for model_name in MODELS:
        report = build_report(model_name)
        reports.append(report)

        output_path = REPORTS_DIR / f"{model_name}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)

        print(f"{model_name}: {report['status']}")
        print(f"  Saved: {output_path}")

    rows = [BASELINE]
    rows.extend(comparison_row(report) for report in reports)

    csv_path = REPORTS_DIR / "model_comparison.csv"
    md_path = REPORTS_DIR / "model_comparison.md"

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"\nCSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print("\nReport generation complete.")


if __name__ == "__main__":
    generate_reports()
