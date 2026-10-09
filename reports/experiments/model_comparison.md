# AstraMed Model Comparison

Generated from saved experiment artifacts.

N/A means a value was not recorded. It does not mean zero.
Smoke tests and incomplete histories must not be used for final model selection.

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Params | Time (s) | GPU Mem (MB) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline CNN | 0.7270 | N/A | 0.0030 | 0.3330 | 0.0070 | N/A | N/A | N/A | provided_baseline_values |
| ResNet-50 | 0.8356 | 0.8179 | 0.8267 | 0.7607 | 0.7923 | 23,510,081 | 227.28 | 735.46 | completed |
| DenseNet-121 | 0.8017 | 0.7752 | 0.7733 | 0.7160 | 0.7436 | 6,954,881 | 250.45 | 637.06 | completed |
| EfficientNet-B0 | 0.8207 | 0.7615 | 0.8333 | 0.7102 | 0.7669 | 4,008,829 | 201.35 | 427.26 | completed |
| ConvNeXt-Tiny | 0.8206 | 0.7831 | 0.8333 | 0.7184 | 0.7716 | 27,820,897 | 213.81 | 887.73 | completed |
| ViT-B/16 | 0.7567 | 0.7418 | 0.9000 | 0.5794 | 0.7050 | 85,799,425 | 243.55 | 1826.77 | completed |

## Interpretation

- ROC-AUC and PR-AUC measure ranking performance.
- Recall measures how many positive cases are detected.
- Precision measures how many predicted positives are correct.
- F1 balances precision and recall.
- Parameter count, runtime, and GPU memory describe model cost.
- Compare only runs using consistent patient splits and evaluation settings.
- The baseline metrics were supplied in the milestone specification; verify their provenance.
