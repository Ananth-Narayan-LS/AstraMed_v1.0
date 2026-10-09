# AstraMed Quantitative Explanation Evaluation

Primary threshold: 80th percentile.

Metrics are aggregated across evaluated positive patients. They measure spatial alignment with RSNA boxes, not clinical correctness.

| Method | Patients | IoU mean ± SD | Precision mean | Recall mean | F1 mean | Pointing hit rate |
|---|---:|---:|---:|---:|---:|---:|
| resnet50_gradcam | 100 | 0.200 ± 0.143 | 0.264 | 0.550 | 0.311 | 0.500 |
| vit_b16_attention | 100 | 0.039 ± 0.054 | 0.067 | 0.109 | 0.070 | 0.030 |

## Interpretation notes

- Compare methods on the same patients and thresholds.
- A higher overlap score does not establish clinical validity.
- Percentile threshold choice can change localization metrics.
- Report sample size and variability alongside means.
- These metrics depend on radiologist box annotations.
- Do not treat pixels outside boxes as confirmed healthy tissue.
