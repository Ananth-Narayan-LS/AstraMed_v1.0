
# AstraMed Baseline CNN Report

## 1. Overview

This report documents the first baseline CNN training run
for the AstraMed pneumonia detection system.

The model was trained using the RSNA Pneumonia Detection
Challenge dataset.

The objective of this baseline is to establish a reference
model and identify areas for future improvement.

---

## 2. Dataset

Dataset:
RSNA Pneumonia Detection Challenge

Total patients:
26,684

Training patients:
18,678

Validation patients:
4,002

Test patients:
4,004

The dataset was split at the patient level to prevent
data leakage between training, validation, and test sets.

---

## 3. Model

Model:

BaselineCNN

Input:
1-channel grayscale chest X-ray image

Input resolution:
512 × 512

Architecture:

Conv2D
→ ReLU
→ MaxPool

Conv2D
→ ReLU
→ MaxPool

Adaptive Average Pooling
→ Flatten
→ Linear(64 → 128)
→ ReLU
→ Linear(128 → 1)
→ Sigmoid

The model produces a probability representing the predicted
probability of pneumonia.

---

## 4. Training Configuration

Optimizer:
Adam

Learning rate:
0.001

Loss function:
Binary Cross Entropy (BCELoss)

Batch size:
8

Number of epochs:
5

Device:
NVIDIA GPU using CUDA

---

## 5. Training Results

| Epoch | Training Loss | Validation Loss |
|------:|--------------:|----------------:|
| 1 | 0.5329 | 0.5186 |
| 2 | 0.5189 | 0.4971 |
| 3 | 0.4980 | 0.4790 |
| 4 | 0.4867 | 0.4775 |
| 5 | 0.4812 | 0.4828 |

The training loss decreased consistently across the
five epochs.

Validation loss also decreased from the first epoch
through epoch 4, followed by a small increase in epoch 5.

---

## 6. Classification Metrics

The following metrics were calculated on the full validation
set containing 4,002 patients.

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.7806 |
| Precision  | 0.3333 |
| Recall     | 0.0034 |
| F1 Score   | 0.0068 |
| ROC-AUC    | 0.7269 |

Confusion Matrix:

[[3121    6]
 [ 872    3]]

---


---


## 7. Failure Cases

The baseline model shows a strong tendency to classify validation
images as negative.

### False Negatives

The model produced 872 false negatives out of 875 actual positive
cases.

This resulted in a recall of only 0.0034.

This means the baseline model detected very few of the pneumonia
cases at the default probability threshold of 0.5.

### False Positives

The model produced only 6 false positives.

Although this results in relatively high precision among the few
positive predictions, the model is missing a very large number of
positive cases.

### Overall Observation

The accuracy of 0.7806 is largely influenced by the class
distribution because negative cases are more common than positive
cases.

Therefore, accuracy alone does not adequately represent the
performance of this pneumonia detection baseline.

The ROC-AUC of 0.7269 indicates that the model's probability
scores contain useful discrimination information, despite the
very poor classification performance at the default 0.5 threshold.

## 8. Observed Limitations

The baseline model is intentionally simple.

Limitations include:

- Small CNN architecture
- No data augmentation
- No class weighting
- No advanced image preprocessing
- No transfer learning
- No bounding-box-aware training
- Training pipeline is limited by DICOM loading and CPU preprocessing

The dataset is also class-imbalanced, with substantially more
negative cases than positive cases.

Therefore, accuracy alone is not sufficient to evaluate the
model.

---

## 9. Next Improvements

The baseline results identify several areas for improvement:

1. Investigate the classification threshold instead of relying
   only on the default 0.5 threshold.

2. Address class imbalance so that the model does not strongly
   favor the negative class.

3. Experiment with weighted loss functions.

4. Replace Sigmoid + BCELoss with raw logits +
   BCEWithLogitsLoss.

5. Add data augmentation.

6. Optimize DICOM loading and CPU preprocessing.

7. Use multiple DataLoader workers and pinned memory.

8. Cache preprocessed images to reduce repeated DICOM
   preprocessing.

9. Experiment with deeper CNN architectures.

10. Investigate transfer learning using pretrained models.

11. Investigate bounding-box-aware training using the available
    pneumonia localization annotations.

12. Perform detailed error analysis on false-negative cases.

13. Evaluate Precision-Recall curves in addition to ROC curves.

--

## 10. Baseline Checkpoint

The trained baseline model was saved as:

`models/baseline_cnn_checkpoint.pth`

The checkpoint contains:

- Model state
- Optimizer state
- Training epoch
- Loss value

---

## 11. Conclusion

The baseline CNN successfully completed a five-epoch training
run on the RSNA pneumonia dataset.

Training loss decreased from 0.5329 to 0.4812, while validation
loss decreased from 0.5186 to 0.4828.

This baseline provides a reference point for evaluating future
AstraMed model improvements.