# AstraMed — Sprint 6 Explainability Report

## 1. Objective

Implement and evaluate visual explanations for pneumonia classification using ResNet-50 Grad-CAM and ViT-B/16 attention.

## 2. Implemented milestones

- M1: Explainability foundation and checkpoint audit.
- M2: Grad-CAM for ResNet-50.
- M3: Integrated Gradients status must be verified separately.
- M4: ViT-B/16 attention extraction and visualization.
- M5: Explanation alignment with RSNA bounding boxes.
- M6: Quantitative localization evaluation.
- M7: Dashboard, sample gallery, and report generation.

## 3. Quantitative evaluation

- Successfully evaluated patients: 100
- Failed patients: 0
- Evaluated split: test
- Thresholds: [50, 70, 80, 90]

### Results at the 80th-percentile threshold

| Method | Patients | Mean IoU | IoU SD | Mean F1 | Pointing hit rate |
|---|---:|---:|---:|---:|---:|
| resnet50_gradcam | 100 | 0.2004 | 0.1425 | 0.3106 | 0.5000 |
| vit_b16_attention | 100 | 0.0388 | 0.0541 | 0.0700 | 0.0300 |

## 4. Generated artifacts

- `quantitative_evaluation/per_patient_metrics.csv`
- `quantitative_evaluation/aggregate_metrics.csv`
- `quantitative_evaluation/threshold_sensitivity.csv`
- `quantitative_evaluation/method_comparison.md`
- `gallery/` patient-level visualizations, when generated

## 5. Limitations

- The current checkpoint results are exploratory screening results.
- Positive cases with boxes do not represent all possible cases.
- Attention and Grad-CAM are not necessarily causal explanations.
- Bounding boxes are imperfect proxies for disease extent.
- Threshold selection changes pixel-level localization metrics.
- Localization scores do not establish clinical validity.
- Integrated Gradients status must be verified before claiming that all explanation methods have been implemented.

## 6. Reproducibility

- Seed: 42
- Model checkpoints and preprocessing must be recorded with results.
- The cohort and threshold configuration should be preserved.
