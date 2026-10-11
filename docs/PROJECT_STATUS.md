# AstraMed — Verified Project Status

Research prototype. **Not clinically validated and not for clinical use.** Last verified: 2026-10-11.
Statements below are marked **Verified** (checked by running code or reading a saved artifact), **Reported** (owner's statement, not provable from the repo) or **Unknown**.

## 1. Objective and scope
Detect pneumonia (lung opacity) in chest X-rays with a deep-learning classifier and show a visual explanation. Current scope: a FastAPI backend (`/api/v1/predict`, `/api/v1/health`) serving a ResNet50 binary classifier with Grad-CAM heatmaps. Out of scope so far: multimodal inputs (Sprint 8), Docker/deployment (Sprint 9), any clinical validation.

## 2. Dataset and labels (Verified by script)
- Source: RSNA Pneumonia Detection Challenge, stage-2 train files in `data/raw/rsna-pneumonia-detection-challenge/` (git-ignored). The repo does not record the download URL or licence.
- `stage_2_train_labels.csv`: 30,227 rows → 26,684 unique patients. One row per bounding box; patients with several boxes repeat. No duplicate rows; no patient has conflicting labels.
- Label definition: `Target = 1` means lung opacity (6,012 patients, 9,555 boxes); `Target = 0` (20,672 patients) has no boxes. Target 1 matches `stage_2_detailed_class_info.csv` class "Lung Opacity" exactly. Target 0 mixes "Normal" (8,851) and "No Lung Opacity / Not Normal" (11,821); the model treats both as "normal".
- All 26,684 DICOMs are present, MONOCHROME2, 1024×1024, 8-bit, no rescale slope/intercept; `PatientID` equals the filename.
- Header metadata: age, sex and view position (AP/PA) exist (checked on one header and a 10% sample). Symptoms and clinical history do not.

## 3. Split and leakage prevention (Verified)
- `src/data/split_data.py`: shuffle unique patient IDs with seed 42, cut 70/15/15. Train 18,678 / validation 4,002 / test 4,004 (test: 950 positive, 3,054 negative). Each patient has one image, so a patient-level split is also an image-level split.
- Disjoint, complete and deterministic: `tests/test_data_integrity.py`. IDs saved in `outputs/evaluation/<experiment>/split_ids.json`.
- The split is regenerated from the seed on each run, **and changes if the seed changes**. Not stratified (prevalence: train 22.4%, val 21.9%, test 23.7%).
- Model selection (best epoch by validation loss) uses validation only (`src/experiments/runner.py`); the runner never touches the test split.

## 4. Preprocessing (Verified, same code for training and API)
DICOM `pixel_array` (float32) → per-image min-max to [0,1] → resize 512×512 (bilinear) → resize 224×224 → replicate to 3 channels → ImageNet mean/std normalization (`src/data/dicom_loader.py`, `preprocessing.py`, `advanced_input.py`). API and batch probabilities agreed within ~1e-4 on 3 images. No augmentation. MONOCHROME1 inversion and rescale slopes are not handled (not present in this dataset). PNG/JPEG uploads are converted to 8-bit grayscale.

## 5. Model, checkpoint and experiment configuration
- Architecture (Verified): torchvision ResNet50, ImageNet-pretrained init, final layer replaced by a single logit; `BCEWithLogitsLoss`, Adam, no scheduler (`src/models/model_factory.py`, `src/experiments/runner.py`). Probability = sigmoid(logit); "pneumonia" if probability ≥ 0.5.
- Active checkpoint (Verified): `outputs/experiments/resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/checkpoints/best_model.pth`, SHA-256 `77a82aac41c7ee5d4d7b5a55964008e7da7dfb999aab602bc14de045fa53ed26`, epoch 1, stored loss 0.3677, loads with `strict=True`. It is not the baseline CNN checkpoint (`models/baseline_cnn_checkpoint.pth`, different architecture, unused by the API).
- Configuration (**Reported**): 18,678 training patients, 2 epochs, batch size 4, learning rate 1e-4, seed 42, best checkpoint at epoch 1. Only the epoch and loss are recorded in the checkpoint; the training-set size, epoch count and hyperparameters are not.
- **Provenance problem (Verified):** the `metadata.json` / `history.json` in this experiment folder, `reports/experiments/resnet50.json` and `reports/experiments/model_comparison.md` describe an earlier **1,000-train / 300-validation smoke run** (best epoch 2, val loss 0.5095, ROC-AUC 0.8356) that used the same folder name. They do not describe the active checkpoint. The full-data run's own metadata, history and log were not preserved. Indirect evidence that the checkpoint is the full-data run: file dated after the smoke-run metadata; stored epoch 1 and loss 0.3677 differ from the smoke run's. Do not cite the 1,000-sample numbers for the deployed model.
- Other architectures (DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16) were smoke runs only (1,000/300 samples, 2 epochs, one seed). The baseline CNN row's numbers come from `reports/baseline_model.md` and were not recomputed.
- The runner now refuses to overwrite an existing checkpoint unless `overwrite=True` (`tests/test_runner_guard.py`).

## 6. Evaluation results (Verified, `scripts/evaluate_model.py`, 2026-10-11)
Source: `outputs/evaluation/resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/evaluation.json` (contains the checkpoint hash and epoch). PR-AUC = scikit-learn average precision.

| Split (patients) | Positives | ROC-AUC | PR-AUC | Threshold |
|---|---|---|---|---|
| Validation (4,002) | 875 | 0.862 | 0.681 | 0.5 for point metrics |
| Test (4,004) | 950 | 0.8775 | 0.7160 | fixed 0.5 |

| Split @0.5 | TN | FP | FN | TP | Accuracy | Precision | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Validation | 2910 | 217 | 415 | 460 | 0.842 | 0.680 | 0.526 | 0.931 | 0.593 |
| Test | 2842 | 212 | 416 | 534 | 0.843 | 0.716 | 0.562 | 0.931 | 0.630 |

These match the owner's reported test numbers. Accuracy is not informative here: always predicting "normal" scores ~76%. ROC-AUC rates ranking ability; PR-AUC should be read against a ~0.24 chance level. Single seed, no confidence intervals, no calibration analysis.

**Threshold selection: no threshold has been chosen; the deployed threshold is 0.5.** Validation-only sweep (saved in `evaluation.json`, steps of 0.05): sensitivity/specificity = 0.744/0.813 at 0.35, 0.795/0.758 at 0.30, 0.84/0.72 at 0.25 (0.30 values read from the saved confusion counts: TP 696/875, TN 2369/3127). Choosing among these is a research decision awaiting the owner; the intended method is to choose on validation, then score the test split once at that fixed value. Caveat: validation data also selected the checkpoint, so validation numbers are slightly optimistic. Caveat: test results at validation-chosen thresholds were viewed during an audit session (scratch script, nothing saved or deployed), so a threshold chosen now is not strictly blind to test results.

### Test-set independence (read this)
**Historical independence of the test set cannot be proven from the repository and is not guaranteed.** The training runner does not use the test split, and the evaluator only scores it at a fixed threshold. But: no historical test-evaluation script or log exists, the full-data run's log was not preserved, the notebooks were not audited, and `scripts/run_explanation_evaluation.py` uses test-split patients for explanation metrics. The test numbers above were produced by a test-set pass during this stabilization; they confirm the owner's reported values, not that the test split was untouched earlier.

## 7. Explanation functionality (Verified 2026-10-11 on 25 validation positives; scratch check, not saved as an artifact)
- Grad-CAM on `model.layer4[-1]` with the positive (pneumonia) logit as score, upsampled to the original image size and drawn with a blue→red colour map. Heatmap, overlay and original share the same size; orientation matches the DICOM (no flip); repeated identical input gave an identical map.
- Fix applied: the overlay used a flat 0.4 blend that tinted the whole image blue (mean pixel change 45/255). It now blends proportionally to heat (max 0.6; mean change 2.6/255). API fields are unchanged.
- Weight-randomization check: trained-model maps put 0.39 of their heat inside annotated boxes (box area 0.17 of the image; peak inside a box in 56% of cases); randomized-weights copies gave 0.10–0.20 (peak in box 0–20%), and rank correlation with the trained map was near zero. So the maps depend on learned weights. 25 images, one seed.
- Limits: Grad-CAM shows where the model's score responded, not clinical reasoning, causality or disease boundaries; the map is coarse (7×7 stretched), min-max scaled per image (a "normal" image still shows a hot spot), and was observed to fire on non-lung structures in the example checked. The saved Sprint 6 gallery and `outputs/explainability/quantitative_evaluation/` (ResNet Grad-CAM pointing-game 0.50, IoU ≈ 0.16–0.20; ViT attention pointing-game 0.03) were generated earlier with older checkpoints and the old overlay; they were not regenerated.

## 8. How to run
```
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000   # docs at /docs
python scripts/test_api_foundation.py            # 10 checks
python scripts/test_api_explanations.py          # 6 checks
python scripts/test_api_integration.py --image <dcm>
python scripts/test_api_reliability.py --image <dcm>
python tests/test_data_integrity.py              # 7 checks
python tests/test_runner_guard.py                # 1 check
python scripts/evaluate_model.py --experiment <experiment_name>
```
`pytest` is not installed; every test file runs as a plain script. Results on 2026-10-11 with the project `.venv`: all passed; reliability test GPU memory growth 0.00 MiB. Passing API tests show the service behaves consistently; they do not show model quality, clinical validity or heatmap faithfulness. Not run: notebooks, the Streamlit dashboard, a clean-environment install, concurrent requests.

## 9. Milestones
| Sprint | Status | Evidence |
|---|---|---|
| 1–2 DICOM, preprocessing, labels | Verified | `tests/test_data_integrity.py` 7/7; header scan |
| 3 / 3.5 EDA, split, Dataset, dashboard | Split/Dataset verified; notebooks and dashboard not re-executed | same |
| 4 Baseline CNN | Implemented; metrics not recomputed | `reports/baseline_model.md` |
| 5 Model factory / experiments | Implemented; comparison is small-sample, single-seed | `reports/experiments/` |
| 6 Explainability | Implemented; overlay fixed; randomization check passed on a small sample; saved outputs stale | section 7 |
| 7 API | Verified | section 8 |
| Pending before Sprint 8/9 | Threshold decision; checkpoint backup + provenance record; re-run explainability evaluation on the active checkpoint; pin dependencies and test a clean install | — |
| 8 Multimodal | Not started; RSNA offers only age/sex/view position (view position may correlate with label — check for shortcut) | — |
| 9 Deployment | Not started; no Dockerfile or CI | — |

## 10. Known bugs, limitations, unverified assumptions
- Active model: single seed, 1 effective epoch, sensitivity 0.56 at the default threshold.
- `predict` is `async def` but runs blocking GPU work; one request blocks others (including `/health`). Do not add threads without making Grad-CAM hook use thread-safe.
- `RequestIDMiddleware` is registered twice in `src/api/main.py` (harmless; left in place pending approval). `scripts/run_benchmark.py` now reports FAILED for models whose smoke-run checkpoint exists (intended overwrite guard).
- Unused/duplicate code: `src/api/services/explanation_services.py`; empty `src/explainability/input_pipeline.py` and `model_loader.py`. Several docs are empty placeholders (`ROADMAP.md`, `experiment_log.md`, `change_log.md`, ...).
- ViT-B/16 is selectable in the API, but its only checkpoint is a collapsed smoke-test run.
- `requirements.txt` is unpinned and was not tested in a clean environment.
- Tested environment: Windows, Python 3.14, torch 2.13 (CUDA 12.6). Other environments are untested.

## 11. Obtaining / regenerating checkpoints
Checkpoints are git-ignored (`*.pth`, `outputs/experiments/**/checkpoints/`). The active file exists only on the original machine; back it up (GitHub Release, Git LFS or cloud storage) and verify with the SHA-256 above. To regenerate, run `run_experiment(ExperimentConfig(model_name="resnet50", pretrained=True, batch_size=4, learning_rate=1e-4, epochs=2, seed=42))` with no sample limits and a **new** `experiment_name`; results will be similar but not bit-identical (cuDNN determinism is only partly enforced).
