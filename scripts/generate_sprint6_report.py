
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPLAIN_DIR = ROOT / "outputs" / "explainability"
EVAL_DIR = EXPLAIN_DIR / "quantitative_evaluation"
REPORT_PATH = EXPLAIN_DIR / "sprint6_report.md"


def main():
    aggregate_path = EVAL_DIR / "aggregate_metrics.csv"
    metadata_path = EVAL_DIR / "evaluation_metadata.json"
    per_patient_path = EVAL_DIR / "per_patient_metrics.csv"

    for path in (aggregate_path, metadata_path, per_patient_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required M6 artifact missing: {path}"
            )

    aggregate = pd.read_csv(aggregate_path)
    per_patient = pd.read_csv(per_patient_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    primary = aggregate[
        aggregate["threshold_percentile"] == 80
    ]

    lines = [
        "# AstraMed — Sprint 6 Explainability Report",
        "",
        "## 1. Objective",
        "",
        "Implement and evaluate visual explanations for pneumonia "
        "classification using ResNet-50 Grad-CAM and ViT-B/16 attention.",
        "",
        "## 2. Implemented milestones",
        "",
        "- M1: Explainability foundation and checkpoint audit.",
        "- M2: Grad-CAM for ResNet-50.",
        "- M3: Integrated Gradients status must be verified separately.",
        "- M4: ViT-B/16 attention extraction and visualization.",
        "- M5: Explanation alignment with RSNA bounding boxes.",
        "- M6: Quantitative localization evaluation.",
        "- M7: Dashboard, sample gallery, and report generation.",
        "",
        "## 3. Quantitative evaluation",
        "",
        f"- Successfully evaluated patients: "
        f"{metadata.get('successful_patients', 'unknown')}",
        f"- Failed patients: {metadata.get('failed_patients', 'unknown')}",
        f"- Evaluated split: {metadata.get('evaluated_split', 'unknown')}",
        f"- Thresholds: {metadata.get('thresholds', [])}",
        "",
        "### Results at the 80th-percentile threshold",
        "",
        "| Method | Patients | Mean IoU | IoU SD | Mean F1 | Pointing hit rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in primary.iterrows():
        lines.append(
            f"| {row['method']} "
            f"| {int(row['patient_count'])} "
            f"| {row['iou_mean']:.4f} "
            f"| {row['iou_std']:.4f} "
            f"| {row['f1_mean']:.4f} "
            f"| {row['pointing_game_hit_mean']:.4f} |"
        )

    lines += [
        "",
        "## 4. Generated artifacts",
        "",
        "- `quantitative_evaluation/per_patient_metrics.csv`",
        "- `quantitative_evaluation/aggregate_metrics.csv`",
        "- `quantitative_evaluation/threshold_sensitivity.csv`",
        "- `quantitative_evaluation/method_comparison.md`",
        "- `gallery/` patient-level visualizations, when generated",
        "",
        "## 5. Limitations",
        "",
        "- The current checkpoint results are exploratory screening results.",
        "- Positive cases with boxes do not represent all possible cases.",
        "- Attention and Grad-CAM are not necessarily causal explanations.",
        "- Bounding boxes are imperfect proxies for disease extent.",
        "- Threshold selection changes pixel-level localization metrics.",
        "- Localization scores do not establish clinical validity.",
        "- Integrated Gradients status must be verified before claiming "
        "that all explanation methods have been implemented.",
        "",
        "## 6. Reproducibility",
        "",
        f"- Seed: {metadata.get('seed', 'unknown')}",
        "- Model checkpoints and preprocessing must be recorded with results.",
        "- The cohort and threshold configuration should be preserved.",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Report generated: {REPORT_PATH.resolve()}")
    print(f"Patient-level rows: {len(per_patient)}")
    print(f"Aggregate rows: {len(aggregate)}")


if __name__ == "__main__":
    main()
