# AstraMed — Verified Project Status

Research prototype. **Not clinically validated and not for clinical use.** Last verified: 2026-10-11.
Statements below are marked **Verified** (checked by running code or reading a saved artifact), **Reported** (owner's statement, not provable from the repo) or **Unknown**.

## 1. Objective and scope
Detect pneumonia (lung opacity) in chest X-rays with a deep-learning classifier and show a visual explanation. Current scope: a FastAPI backend (`/api/v1/predict`, `/api/v1/health`) serving a ResNet50 binary classifier with Grad-CAM heatmaps. This image-only ResNet50 pipeline (data → model → evaluation → API → explanation) is the **completed baseline** that any later multimodal work must be compared against. Out of scope so far: multimodal inputs (Sprint 8), Docker/deployment (Sprint 9), any clinical validation. Sprint 8 (multimodal research) is partially complete (section 12); further multimodal work is subject to a defensible data-source decision (RSNA has no symptoms or clinical history).

## 2. Dataset and labels (Verified by script)
- Source: RSNA Pneumonia Detection Challenge, stage-2 train files in `data/raw/rsna-pneumonia-detection-challenge/` (git-ignored). The repo does not record the download URL or licence.
- `stage_2_train_labels.csv`: 30,227 rows → 26,684 unique patients. One row per bounding box; patients with several boxes repeat. No duplicate rows; no patient has conflicting labels.
- Label definition: `Target = 1` means lung opacity (6,012 patients, 9,555 boxes); `Target = 0` (20,672 patients) has no boxes. Target 1 matches `stage_2_detailed_class_info.csv` class "Lung Opacity" exactly. Target 0 mixes "Normal" (8,851) and "No Lung Opacity / Not Normal" (11,821); the model treats both as "normal".
- All 26,684 DICOMs are present, MONOCHROME2, 1024×1024, 8-bit, no rescale slope/intercept; `PatientID` equals the filename.
- Header metadata: age, sex and view position (AP/PA) exist (checked on one header and a 10% sample). Symptoms and clinical history do not.

## 3. Split and leakage prevention (Verified)
- `src/data/split_data.py`: shuffle unique patient IDs with seed 42, cut 70/15/15. Train 18,678 / validation 4,002 / test 4,004 (test: 950 positive, 3,054 negative). Each patient has one image, so a patient-level split is also an image-level split.
- Disjoint, complete and deterministic: `tests/test_data_integrity.py`. `outputs/evaluation/<experiment>/split_ids.json` is a local, regenerable copy of the IDs (the evaluator refuses to continue if the regenerated split differs from it).
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
Sources (in `outputs/evaluation/resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/`): `evaluation.json` (test + validation results, checkpoint hash and epoch; created once on 2026-10-11 and not regenerated), `validation_predictions.csv` (4,002 rows: patient_id, target, probability) and `validation_threshold_sweep.json` (a second, independent validation-only run on 2026-10-11 whose sweep matched the saved one exactly). PR-AUC = scikit-learn average precision.
**No per-patient test predictions are saved**: producing them needs a new test-set inference pass, which was deliberately not done. Test metrics are therefore backed by `evaluation.json` only.

Evaluator design (`scripts/evaluate_model.py`): validation mode (default) writes per-patient predictions and the threshold sweep; test mode (`--split test`) has no threshold option (fixed 0.5), and refuses to run if `evaluation.json` already exists. Existing outputs are never overwritten. Split-overlap checks raise `RuntimeError` (active under `python -O`); patient order and labels are verified against the split.

| Split (patients) | Positives | ROC-AUC | PR-AUC | Threshold |
|---|---|---|---|---|
| Validation (4,002) | 875 | 0.862 | 0.681 | 0.5 for point metrics |
| Test (4,004) | 950 | 0.8775 | 0.7160 | fixed 0.5 |

| Split @0.5 | TN | FP | FN | TP | Accuracy | Precision | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Validation | 2910 | 217 | 415 | 460 | 0.842 | 0.680 | 0.526 | 0.931 | 0.593 |
| Test | 2842 | 212 | 416 | 534 | 0.843 | 0.716 | 0.562 | 0.931 | 0.630 |

These match the owner's reported test numbers. Accuracy is not informative here: always predicting "normal" scores ~76%. ROC-AUC rates ranking ability; PR-AUC should be read against a ~0.24 chance level. Single seed, no confidence intervals, no calibration analysis.

**Threshold selection: no threshold has been chosen; the deployed threshold is 0.5.** Validation-only sweep (saved in `evaluation.json`, steps of 0.05): at 0.30 sensitivity = 696/875 ≈ 0.7954 and specificity = 2369/3127 ≈ 0.7576 (re-computed from freshly generated validation predictions, which matched the saved sweep); other points in `validation_threshold_sweep.json`. Note 0.7954 is below 0.80. Choosing among these is a research decision awaiting the owner (decision support below); the intended method is to choose on validation, then score the test split once at that fixed value. Caveat: validation data also selected the checkpoint, so validation numbers are slightly optimistic. Caveat: test results at validation-chosen thresholds were viewed during an audit session (scratch script, nothing saved or deployed), so a threshold chosen now is not strictly blind to test results.

### Threshold decision support (validation data only; nothing deployed)
Computed from `validation_predictions.csv` / `validation_threshold_sweep.json` (4,002 patients, 875 positive); no model re-run, no test data. 95% intervals are bootstrap estimates (1,000 resamples) on validation.

| Threshold | Sensitivity | Specificity | Precision | F1 | False alarms | Missed positives | Flagged |
|---|---|---|---|---|---|---|---|
| 0.20 | 0.867 | 0.671 | 0.425 | 0.570 | 1029 | 116 | 44.7% |
| 0.25 | 0.839 | 0.716 | 0.452 | 0.588 | 889 | 141 | 40.6% |
| **0.30** | 0.7954 (95% CI 0.77–0.82) | 0.7576 (0.74–0.77) | 0.479 | 0.598 | 758 | 179 | 36.3% |
| 0.35 | 0.744 | 0.813 | 0.526 | 0.617 | 586 | 224 | 30.9% |
| 0.40 | 0.689 | 0.858 | 0.577 | 0.628 | 443 | 272 | 26.1% |
| 0.45 | 0.611 | 0.898 | 0.627 | 0.619 | 318 | 340 | 21.3% |
| **0.50 (deployed)** | 0.5257 (0.49–0.56) | 0.9306 (0.92–0.94) | 0.680 | 0.593 | 217 | 415 | 16.9% |

Trade-off in plain terms: lowering the threshold makes the model say "pneumonia" more often. That catches more true cases (fewer misses) but raises false alarms. At 0.50 the model misses about 47% of positives in validation (104 per 1,000 patients at ~22% prevalence) and flags 17%; at 0.30 it misses about 20% (45 per 1,000) but flags 36%, and roughly half of those flags are false (precision 0.48). Neither is "correct"; it depends on what an error costs.
- Objective-based option (a **research-prototype** choice, not a clinical one): if the goal is a screening-style demo that should catch about 80% of positives, the largest threshold with validation sensitivity ≥ 0.80 is **0.29** (sensitivity 0.806, specificity 0.749, precision 0.473; 0.01-step search). 0.30 is the nearest round value but gives 0.7954. If instead the goal is best F1, the validation optimum is ≈ 0.42 (F1 0.629).
- Limits: validation is also the data that selected the checkpoint (slightly optimistic); intervals reflect sampling noise only; single seed and one checkpoint; precision depends on prevalence (~22% here), so it will differ elsewhere; probabilities are not calibrated. **This is not a clinical recommendation and the deployed threshold remains 0.5 pending the owner's decision.** Test-set independence cannot be established historically, and test results at alternative thresholds were already viewed in an earlier session, so any threshold chosen now is not strictly blind; the test split should be used only once, at the chosen fixed value, and described with that caveat.

### Test-set independence (read this)
**Historical independence of the test set cannot be proven from the repository and is not guaranteed.** The training runner does not use the test split, and the evaluator only scores it at a fixed threshold. But: no historical test-evaluation script or log exists, the full-data run's log was not preserved, the notebooks were not audited, and `scripts/run_explanation_evaluation.py` uses test-split patients for explanation metrics. The test numbers above were produced by a test-set pass during this stabilization; they confirm the owner's reported values, not that the test split was untouched earlier.

## 7. Explanation functionality (Verified 2026-10-11 on 25 validation positives; scratch check, not saved as an artifact)
- Grad-CAM on `model.layer4[-1]` with the positive (pneumonia) logit as score, upsampled to the original image size and drawn with a blue→red colour map. Heatmap, overlay and original share the same size; orientation matches the DICOM (no flip); repeated identical input gave an identical map.
- Fix applied (`src/api/services/explanation_service.py`, covered by `tests/test_overlay.py`): the overlay used a flat 0.4 blend that tinted the whole image blue (mean pixel change 45/255). It now blends proportionally to heat (max 0.6; mean change 2.6/255). API fields are unchanged. Edge cases checked: zero or constant heatmap → overlay equals the original exactly; non-square image → all three images keep the original size; NaN heatmap → controlled `INVALID_HEATMAP` error. The randomization check below is a small, one-seed sanity test, not a faithfulness proof.
- Weight-randomization check: trained-model maps put 0.39 of their heat inside annotated boxes (box area 0.17 of the image; peak inside a box in 56% of cases); randomized-weights copies gave 0.10–0.20 (peak in box 0–20%), and rank correlation with the trained map was near zero. So the maps depend on learned weights. 25 images, one seed.
- Limits: Grad-CAM shows where the model's score responded, not clinical reasoning, causality or disease boundaries; the map is coarse (7×7 stretched), min-max scaled per image (a "normal" image still shows a hot spot), and was observed to fire on non-lung structures in the example checked. The saved Sprint 6 gallery and `outputs/explainability/quantitative_evaluation/` (ResNet Grad-CAM pointing-game 0.50, IoU ≈ 0.16–0.20; ViT attention pointing-game 0.03) were generated earlier with older checkpoints and still contain the **old whole-image blue overlay**; they were not regenerated.

## 8. How to run
```
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000   # docs at /docs
python scripts/test_api_foundation.py            # 10 checks
python scripts/test_api_explanations.py          # 6 checks
python scripts/test_api_integration.py --image <dcm>
python scripts/test_api_reliability.py --image <dcm>
python tests/test_data_integrity.py              # 7 checks
python tests/test_runner_guard.py                # 1 check
python tests/test_request_id.py                  # 4 checks
python tests/test_overlay.py                     # 3 checks
python scripts/evaluate_model.py --experiment <experiment_name>              # validation only
# python scripts/evaluate_model.py --experiment <name> --split test           # one-shot, refuses if evaluation.json exists
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
| Pending before Sprint 8/9 | Owner's threshold decision (deployed value stays 0.5 until then); optional re-run of explainability evaluation on the active checkpoint; dependency pinning. Done: owner backup of the checkpoint (hash verified 2026-10-11); a clean install on one machine (CPU, Python 3.14). The checkpoint's exact training provenance cannot be independently established and is documented as such. | — |
| 8 Multimodal | **Partially complete (2026-10-11).** Architecture, leakage-safe pipeline, 21 unit tests and validation experiments done; only image + age + sex trainable; no improvement demonstrated; symptoms/history unavailable; no test inference. See section 12. | `src/multimodal/`, `tests/test_multimodal.py`, `outputs/multimodal/validation_comparison.json` |
| 9 Deployment | **Verified for CPU Docker deployment on one machine (2026-10-12).** Image builds, container healthy under `docker run` and Docker Compose, smoke test passed, security settings confirmed; GPU-in-Docker, performance, concurrency, long-running stability, other platforms and the runner-guard test in the owner's environment not verified; no CI. See section 13. | `Dockerfile`, `docker-compose.yml`, `docs/DEPLOYMENT.md`, `tests/test_deployment.py` |

## 10. Known bugs, limitations, unverified assumptions
- Active model: single seed, 1 effective epoch, sensitivity 0.56 at the default threshold.
- `predict` is `async def` but runs blocking GPU work; one request blocks others (including `/health`). Do not add threads without making Grad-CAM hook use thread-safe.
- Fixed 2026-10-11: `RequestIDMiddleware` was registered twice in `src/api/main.py`; with no `X-Request-ID` header the response header and body `request_id` differed (reproduced, then fixed; `tests/test_request_id.py`). `scripts/run_benchmark.py` now reports FAILED for models whose smoke-run checkpoint exists (intended overwrite guard).
- Unused/duplicate code: `src/api/services/explanation_services.py`; empty `src/explainability/input_pipeline.py` and `model_loader.py`. Several docs are empty placeholders (`ROADMAP.md`, `experiment_log.md`, `change_log.md`, ...).
- ViT-B/16 is selectable in the API, but its only checkpoint is a collapsed smoke-test run.
- Dependencies (VERIFIED 2026-10-11): static scan of imports found no unlisted third-party package. A disposable venv outside the repository (Python 3.14.6, `pip install -r requirements.txt`) installed successfully (torch 2.14.1+cpu, torchvision 0.29.1+cpu; the working `.venv` uses torch 2.13.0+cu126, torchvision 0.28.0+cu126, scipy 1.18.0). In that venv: module imports OK; the checkpoint loads with `strict=True` on CPU; data-integrity 7/7, runner-guard, request-ID 4/4, overlay 3/3, API foundation 10/10, explanation and integration tests passed on CPU. The reliability test passed but its GPU-memory check was SKIPPED (no CUDA). Two findings: (1) the scripts need `pip install -e .` (the dev `.venv` has an editable install; the README now says so); (2) on this machine Windows Application Control blocked the freshly installed scipy 1.18.1 binary (`_rigid_transform_cy`), so `sklearn` failed to import until `scipy==1.18.0` (the version already in the working `.venv`) was installed. That block is a machine policy, not shown to be a project bug, so `requirements.txt` was not pinned. Not tested: Python 3.12, Linux/macOS, CUDA builds, fresh-install performance. Versions remain unpinned.
- Checkpoint integrity (VERIFIED 2026-10-11): active file and the owner's backup in `%USERPROFILE%\AstraMed-backupest_model.pth` both hash to the SHA-256 above.
- Artifact tracking (decision 2026-10-11): `evaluation.json` and `validation_threshold_sweep.json` hold only aggregate numbers, the checkpoint hash and counts (no absolute paths or secrets found) and are suitable to commit. `validation_predictions.csv` (204 KB) and `split_ids.json` (1 MB) list individual RSNA patient IDs with ground-truth labels (and model probabilities); the repository does not record the dataset's redistribution terms, so keep them **local** and regenerate with `python scripts/evaluate_model.py --experiment <name>` (validation only; the split is deterministic). Text format alone does not make a file safe to publish.
- Tested environment: Windows, Python 3.14, torch 2.13 (CUDA 12.6). Other environments are untested.

## 11. Obtaining / regenerating checkpoints
Checkpoints are git-ignored (`*.pth`, `outputs/experiments/**/checkpoints/`). The active file exists only on the original machine; back it up (GitHub Release, Git LFS or cloud storage) and verify with the SHA-256 above. To regenerate, run `run_experiment(ExperimentConfig(model_name="resnet50", pretrained=True, batch_size=4, learning_rate=1e-4, epochs=2, seed=42))` with no sample limits and a **new** `experiment_name`; results will be similar but not bit-identical (cuDNN determinism is only partly enforced).

## 12. Sprint 8 — multimodal research model (implemented 2026-10-11; research only)

**Verdict: partial multimodal implementation. The reusable architecture, data pipeline and tests are done; only image + age + sex could be trained because RSNA has no symptom or clinical-history data, and that added nothing measurable.**

### Feature availability (Verified: all 26,684 DICOM headers read with `src/multimodal/clinical_data.py`)
| Feature | Source | Coverage | Missing | Usable now? |
|---|---|---|---|---|
| Age | DICOM `PatientAge` | 26,679 / 26,684 | 5 (0.02%) | Yes (range 1-92, mean 47) |
| Sex | DICOM `PatientSex` (M 15,166 / F 11,518) | 26,684 | 0 | Yes |
| View position | DICOM `ViewPosition` (PA 14,511 / AP 12,173) | 26,684 | 0 | Available, **excluded from the primary model** (shortcut risk) |
| Symptoms | none in RSNA files or repo | 0 | 100% | **No** |
| Clinical history | none in RSNA files or repo | 0 | 100% | **No** |

- The join of header fields to labels is one-to-one (26,684 unique IDs; DICOM `PatientID` equals the filename). Duplicates, orphans and malformed tables are rejected (`validate_join`, tested).
- View position is a shortcut: pneumonia prevalence is 0.383 for AP vs 0.093 for PA (the label is tied to how the X-ray was taken, and the view is also visible in the image). A metadata-only logistic regression (scratch check: fitted on training patients, scored on validation) gives ROC-AUC 0.53 for age + sex but 0.71 with view position. Age and sex alone are almost uninformative. View position was therefore run only as a labelled ablation.
- No symptom or history text was fabricated. The text encoder (`src/multimodal/text.py`) is implemented, masked, and unit-tested with clearly labelled synthetic strings only. It is inactive in every trained model (its modality masks are always 0). Genuine text would need a different dataset with patient-level linkage and licence checks; none was downloaded.

### Architecture (`src/multimodal/`)
ResNet50 image encoder (same layer names as the baseline; baseline weights loaded read-only, everything except the old 1-logit head; the new projection and fusion head are untrained) + structured encoder (age z-score, age-missing flag, sex one-hot with an unknown slot; small MLP) + optional text encoder, concatenated with modality-mask bits, then an MLP giving one logit. A masked modality contributes exactly zero, so absence is explicit (tested: masked features have no effect on the output). Plain `BCEWithLogitsLoss` with no class weighting (as in the baseline, to keep arms comparable). The age mean/std are fitted on the training patients only. The existing API is **unchanged**; no multimodal endpoint was added (see below).

### Experiment protocol (fixed before any result was seen)
Existing 70/15/15 patient-level split; train on the training patients only; select the epoch by validation PR-AUC (patience 2, at most 4 epochs). The test split is never loaded (`forbidden_ids` guard) and **no test inference was run in Sprint 8**. Epoch 1 trains the new layers with the image encoder frozen; epochs 2+ unfreeze it (encoder lr 2e-5, new layers 3e-4, Adam, batch 32). Seed 42 plus seeds 1 and 2 give a sense of run-to-run noise. To save compute, train and validation images are cached locally as preprocessed 224x224 float16 (`scripts/build_image_cache.py`; test images are not cached). Parity check: the frozen baseline on cached inputs differs from its saved DICOM-pipeline validation probabilities by at most 0.0031 (mean 0.0003) on 500 patients.

An **image-only control** with the identical fusion head and recipe is included, because the multimodal models also receive extra image fine-tuning that the frozen baseline did not. Without the control, any gain could be wrongly credited to the clinical data.

### Validation results (4,002 validation patients, 875 positive; threshold 0.5 for point metrics; `outputs/multimodal/validation_comparison.json`)
| Model (seeds 42 / 1 / 2) | ROC-AUC | PR-AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Frozen image-only baseline (deployed) | 0.8621 | 0.6806 | 0.526 | 0.931 |
| Image-only control | 0.8776 / 0.8765 / 0.8769 | 0.7046 / 0.7047 / 0.7020 | 0.477 / 0.565 / 0.566 | 0.946 / 0.924 / 0.922 |
| Image + age + sex | 0.8768 / 0.8769 / 0.8769 | 0.7027 / 0.7040 / 0.7022 | 0.465 / 0.579 / 0.558 | 0.948 / 0.918 / 0.923 |
| Image + age + sex + view position (ablation, seed 42) | 0.8762 | 0.7013 | 0.469 | 0.948 |

- Image-only control vs the frozen baseline: ROC-AUC about +0.015 in all three seeds (paired-bootstrap 95% intervals roughly +0.007 to +0.023). Extra image training helps on validation (the baseline was effectively one epoch).
- Image + age + sex vs the same-seed image-only control: change in ROC-AUC -0.0008, +0.0004, +0.0000 and in PR-AUC -0.0018, -0.0007, +0.0002; every interval includes 0. **No multimodal improvement was demonstrated.** View position was slightly worse than the control (seed 42: change in ROC-AUC -0.0014, interval excludes 0; PR-AUC -0.0033).
- Point metrics at 0.5 vary by seed (sensitivity 0.47-0.58) because where the probabilities sit moves even when ranking quality (AUC) is stable. Do not read sensitivity differences between arms as effects. All arms overfit after epoch 2 (validation loss rises); epoch 2 was selected every time.
- Cost: about 570 s per 4-epoch run, peak GPU memory about 3.0 GB (RTX 4050 Laptop 6 GB; torch 2.13.0+cu126; Windows, Python 3.14.6).
- Validation data selected the epoch for every arm (same rule), so absolute validation numbers are slightly optimistic. Differences between arms are paired on the same patients; intervals reflect sampling of validation patients only, and three seeds are shown.
- The Sprint 8 models are research artifacts: **not deployed, and the deployed threshold remains 0.5**. They have no held-out result, and the caveats about historical test-set independence still apply.

### Reproduce
```
python scripts/build_image_cache.py                       # local cache, train+val only (~2.3 GB, git-ignored)
python scripts/train_multimodal.py --arm image_only --seed 42
python scripts/train_multimodal.py --arm structured --seed 42
python scripts/train_multimodal.py --arm structured_view --seed 42   # shortcut ablation
python scripts/compare_multimodal.py --parity-check 500 --arms image_only_resnet50_seed42 structured_resnet50_seed42
python tests/test_multimodal.py                           # 21 unit tests
```
Outputs go to `outputs/multimodal/<arm>_resnet50_seed<N>/` (checkpoint, `metadata.json` with config, feature schema, missingness, baseline hash and provenance, `history.json`, and a local `val_predictions.csv`). Training refuses to reuse an existing run folder or overwrite any checkpoint. Local-only (git-ignored): the cache, `clinical_metadata.csv`, checkpoints and per-patient prediction CSVs.

### API integration: deliberately not done
The image-only `/api/v1/predict` is untouched. A multimodal endpoint would need a versioned route (for example `/api/v2/predict`) with optional `age` (0-120) and `sex` (M/F/unknown) and a response listing `modalities_used`. The evidence gives no reason to ship it: age and sex added no measurable value, and symptom/history inputs have no training data.

### Remaining blockers and next actions
- Full fusion (symptoms + history) is blocked by data: it needs a dataset with genuine, patient-linked text and compatible licensing; none is in the repository.
- If multimodal research continues, the data-source decision comes first. The model already accepts text inputs without architecture changes.
- The +0.015 ROC-AUC from further image fine-tuning (early-stopped on validation) is a legitimate follow-up for the image model, but it needs a held-out evaluation decision from the owner and does not change the deployed model.

## 13. Sprint 9 — Docker deployment (2026-10-12)

**Verdict: the image-only API runs correctly in Docker, started both with `docker run` and with Docker Compose. A few items remain unverified (listed below), so Sprint 9 is verified for CPU deployment on this machine only.** Commands and the full status table: [docs/DEPLOYMENT.md](DEPLOYMENT.md).

- **Added:** `Dockerfile` (python:3.14-slim, CPU-only PyTorch, non-root user, health check), `.dockerignore`, `docker-compose.yml` (checkpoint mounted read-only, localhost-only port, read-only filesystem), `requirements-api.txt` (pinned, minimal) and `requirements-dev.txt`, `scripts/smoke_test_deployment.py`, `tests/test_deployment.py` (12 checks).
- **API changes (additive only):** `GET /api/v1/ready` (checkpoint file present → 200, else 503, no paths); optional `ASTRAMED_PRELOAD_MODEL` (default false = previous lazy behaviour). Existing routes, response schemas, preprocessing, threshold 0.5, checkpoint selection and middleware are unchanged. The multimodal code is neither imported by the API nor in the image.
- **Verified (run here):** image `astramed-api:local` builds (1.58 GB); container reaches `healthy`; `/health` and `/ready` return 200; prediction smoke test 12/12 against the Compose container (also reported 12/12 by the project owner against the `docker run` container); checkpoint bind mount is read-only and its hash inside the container equals the host original and the backup (`77A82AAC...53ED26`); non-root user, read-only root filesystem, dropped capabilities, `no-new-privileges` and localhost-only port binding confirmed by `docker inspect`; `docker compose config`/`up`/`ps`/`logs`/`stop`/`start` work; in the project `.venv`: `test_deployment.py` 12/12, API foundation 10/10, explanations, integration, reliability (GPU memory growth 0.00 MiB) and runner-guard all passed.
- **Not verified:** the runner-guard test in the project owner's own environment (reported blocked by Windows Application Control for a scikit-learn DLL; not reproduced here); automatic restart after a host reboot; GPU inside Docker; throughput, latency, memory use; concurrent requests; long-running stability; arm64 and other operating systems; behaviour of the missing-checkpoint failure inside a container was reported by the owner, not re-run by the audit.
- **Notes:** `ASTRAMED_HOST` and `ASTRAMED_PORT` are validated settings but do not control the server (the port and address are hard-coded in the Dockerfile `CMD`, `HEALTHCHECK`, `EXPOSE` and the Compose mapping). The `docker run` and Compose containers both bind port 8000, so only one can run at a time. The base image tag `python:3.14-slim` floats (the image used Python 3.14.8; earlier host tests used 3.14.6).
- **Limitations:** no authentication, TLS or rate limiting; `/docs` enabled; synchronous inference (one request blocks others, including `/health`); single worker only; transitive dependencies unlocked; checkpoint provenance as documented above; not clinically validated.
