
# M6 — Quantitative Explanation Evaluation

## Objective
Evaluate localization of ResNet-50 Grad-CAM and ViT-B/16 attention
against RSNA radiologist-provided pneumonia bounding boxes across
positive patients in the established test split.

## Evaluation protocol
- Reuse the existing patient-level train/validation/test split.
- Evaluate positive test patients with at least one bounding box.
- Use the same patient cohort for both methods.
- Resize explanation maps to original image coordinates.
- Evaluate thresholds at the 50th, 70th, 80th, and 90th percentiles.
- Report patient-level scores and aggregate mean, standard deviation,
  and median.

## Metrics
- Intersection over Union (IoU)
- Precision, recall, and F1
- Pointing-game hit rate
- Fraction of normalized heatmap mass inside annotated boxes

## Outputs
- `per_patient_metrics.csv`: patient-level scores at the 80th percentile.
- `threshold_sensitivity.csv`: patient-level scores at all thresholds.
- `aggregate_metrics.csv`: aggregate statistics by method and threshold.
- `method_comparison.md`: readable comparison at the primary threshold.
- `evaluation_metadata.json`: cohort, seed, failures, and limitations.

## Interpretation
These metrics measure spatial overlap with the annotated boxes.
They do not establish causal explanations, diagnostic accuracy, or
clinical validity.

Only positive cases with boxes are included. This does not measure
false-positive localization on negative cases.

The current pretrained screening checkpoints were trained using
limited screening runs. Results must be labeled exploratory.

## Run
Start with a small test:

`python -u scripts/run_explanation_evaluation.py --max-cases 10`

Then run a larger evaluation:

`python -u scripts/run_explanation_evaluation.py --max-cases 100`

Use `--max-cases 0` to evaluate all eligible positive test patients.
