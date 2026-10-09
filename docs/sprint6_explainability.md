
# AstraMed — Sprint 6: Explainability

## Objective
Provide reusable visual explanations and quantitative localization
evaluation for the pneumonia classification models.

## Components
- Grad-CAM for ResNet-50.
- CLS-to-patch attention visualization for ViT-B/16.
- Alignment with RSNA bounding-box annotations.
- Quantitative localization metrics and threshold sensitivity.
- Streamlit dashboard for patient and cohort exploration.
- Sample gallery and generated Sprint 6 report.

## Run the workflow
1. Run the M6 quantitative evaluation.
2. Generate the sample gallery.
3. Generate the Sprint 6 report.
4. Launch the dashboard.

## Limitations
The current evaluation uses screening-trained checkpoints and annotated
positive cases. Results are exploratory and do not establish clinical
validity. Attention and Grad-CAM should not be interpreted as definitive
evidence of causal model reasoning.

Integrated Gradients must be documented as pending unless its
implementation and tests have been completed.
