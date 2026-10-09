
from __future__ import annotations

import csv
import statistics
from pathlib import Path


METRICS = (
    "iou",
    "precision",
    "recall",
    "f1",
    "pointing_game_hit",
    "energy_inside_boxes",
)


def aggregate_metrics(rows, group_keys):
    """
    Aggregate per-patient metric rows by the supplied group keys.

    Example group_keys:
        ("method", "threshold_percentile")

    Boolean pointing-game outcomes are converted to 0/1.
    Missing values are excluded from each metric's summary.
    """
    groups = {}

    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    summaries = []

    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = dict(zip(group_keys, key))
        summary["patient_count"] = len(group)

        for metric in METRICS:
            values = []

            for row in group:
                value = row.get(metric)

                if value is None:
                    continue

                if metric == "pointing_game_hit":
                    value = int(bool(value))

                values.append(float(value))

            summary[f"{metric}_n"] = len(values)
            summary[f"{metric}_mean"] = (
                statistics.mean(values) if values else None
            )
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
            summary[f"{metric}_median"] = (
                statistics.median(values) if values else None
            )

        summaries.append(summary)

    return summaries


def write_csv(path, rows):
    """Write dictionaries to CSV using the union of their keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to write to {path}")

    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_report(path, summaries, primary_threshold=80):
    """Generate a readable Markdown method-comparison report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    selected = [
        row for row in summaries
        if row.get("threshold_percentile") == primary_threshold
    ]

    lines = [
        "# AstraMed Quantitative Explanation Evaluation",
        "",
        f"Primary threshold: {primary_threshold}th percentile.",
        "",
        "Metrics are aggregated across evaluated positive patients. "
        "They measure spatial alignment with RSNA boxes, not clinical "
        "correctness.",
        "",
        "| Method | Patients | IoU mean ± SD | Precision mean | "
        "Recall mean | F1 mean | Pointing hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in selected:
        def fmt(key):
            value = row.get(key)
            return "N/A" if value is None else f"{value:.3f}"

        lines.append(
            f"| {row['method']} "
            f"| {row['patient_count']} "
            f"| {fmt('iou_mean')} ± {fmt('iou_std')} "
            f"| {fmt('precision_mean')} "
            f"| {fmt('recall_mean')} "
            f"| {fmt('f1_mean')} "
            f"| {fmt('pointing_game_hit_mean')} |"
        )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Compare methods on the same patients and thresholds.",
        "- A higher overlap score does not establish clinical validity.",
        "- Percentile threshold choice can change localization metrics.",
        "- Report sample size and variability alongside means.",
        "- These metrics depend on radiologist box annotations.",
        "- Do not treat pixels outside boxes as confirmed healthy tissue.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
