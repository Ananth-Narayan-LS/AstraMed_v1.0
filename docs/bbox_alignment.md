
# M5 — Bounding-Box Alignment

## Objective
Compare explanation heatmaps against radiologist-provided RSNA
pneumonia bounding boxes using a common image coordinate system.

## Coordinate handling
- Original DICOM: typically 1024 x 1024 pixels.
- Preprocessed dataset image: 512 x 512 pixels.
- Model input: 224 x 224 pixels.
- RSNA box format: x, y, width, height in original-image coordinates.
- Explanation maps are resized to the original DICOM image dimensions.
- Boxes remain in original coordinates and are clipped to image bounds.

## Outputs
- `resnet50_gradcam_alignment.png`
- `vit_attention_alignment.png`
- `localization_metrics.json`

## Metrics
- Precision: fraction of thresholded explanation pixels inside boxes.
- Recall: fraction of annotated box pixels covered by the thresholded map.
- F1: harmonic mean of precision and recall.
- IoU: intersection-over-union of the thresholded map and box mask.
- Pointing-game hit: whether the hottest pixel falls inside any box.
- Energy inside boxes: fraction of normalized heatmap mass within boxes.

The heatmap threshold uses the 80th percentile by default. This is a
relative threshold and should be tested at multiple percentiles.

## Limitations
- One patient is a pipeline smoke test, not a reliable performance estimate.
- Bounding boxes are imperfect proxies for the full disease region.
- Attention and Grad-CAM are not proof of causal model reasoning.
- Pixel-level overlap metrics depend on threshold choice and annotation size.
- Evaluation should ultimately cover a held-out set of positive patients.
- Localization metrics do not establish clinical safety or validity.
