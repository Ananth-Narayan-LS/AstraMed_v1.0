# AstraMed — Verified Project Status

Research prototype. **Not clinically validated.** Last verified: 2026-10-11 (stabilization pass).

## Data and splits (verified by script)
- RSNA stage-2 train labels: 30,227 rows → 26,684 unique patients (6,012 positive, 20,672 negative; 9,555 boxes, positives only). No duplicate rows or conflicting labels. Every patient has a DICOM; all DICOMs are MONOCHROME2, 1024×1024, 8-bit, no rescale.
- Split: `src/data/split_data.py`, patient-level shuffle, 70/15/15, seed 42 → train 18,678 / val 4,002 / test 4,004 (test: 950 positive, 3,054 negative). Disjoint and deterministic (`tests/test_data_integrity.py`). **The split depends on the seed** — changing the seed changes the test set. The IDs are saved by `scripts/evaluate_model.py` to `outputs/evaluation/<experiment>/split_ids.json`.
- Preprocessing (training = inference): DICOM → per-image min-max → resize 512 → `AdvancedInputAdapter` (resize 224, 3 channels, ImageNet normalization).

## Model and checkpoint provenance (IMPORTANT)
- Active API model: ResNet50, `outputs/experiments/resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/checkpoints/best_model.pth`, SHA-256 `77a82aac41c7ee5d4d7b5a55964008e7da7dfb999aab602bc14de045fa53ed26`, epoch 1, stored loss 0.3677, loads with `strict=True`.
- **This checkpoint is from the full-data run (18,678 train patients, 2 epochs configured, best at epoch 1).** The `metadata.json` / `history.json` in the same folder, `reports/experiments/resnet50.json`, and `reports/experiments/model_comparison.md` describe an *earlier 1,000-sample smoke run* that used the same folder name. They do **not** describe this checkpoint. The full-data run's own metadata/history/log were not preserved. Do not cite the 1,000-sample numbers (ROC-AUC 0.8356) for the deployed model.
- The other four architectures and ViT in `model_comparison.md` are 1,000/300-sample, 2-epoch, single-seed runs: useful as a smoke comparison only. The baseline row's numbers come from the Sprint 4 report and were not recomputed.
- The runner now refuses to overwrite an existing checkpoint (`overwrite=True` required).

## Measured performance of the active checkpoint (`scripts/evaluate_model.py`, 2026-10-11)
Results: `outputs/evaluation/resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/evaluation.json`.

| Split | n | ROC-AUC | PR-AUC |
|---|---|---|---|
| Validation | 4,002 | 0.862 | 0.681 |
| Test (threshold 0.5) | 4,004 | 0.8775 | 0.7160 |

Test at 0.5: TN 2842, FP 212, FN 416, TP 534 → accuracy 0.843, precision 0.716, sensitivity 0.562, specificity 0.931, F1 0.630. These match the previously reported values. Prevalence is ~23.7%, so "always normal" scores ~76% accuracy; accuracy alone is not informative.

Validation-only threshold sweep (the deployed threshold is still 0.5, not changed): sensitivity/specificity at 0.40 = 0.69/0.86, 0.30 = 0.80/0.76, 0.25 = 0.84/0.72. Choosing a threshold is a methodology decision awaiting owner approval; it must be chosen on validation and applied once to test.

## Sprint status
| Sprint | Status | Evidence |
|---|---|---|
| 1–2 DICOM, preprocessing, labels | Verified | `tests/test_data_integrity.py` (7/7), header scan of all DICOMs |
| 3 / 3.5 EDA, split, Dataset, dashboard | Split/Dataset verified; notebooks and dashboard not re-executed | as above |
| 4 Baseline CNN | Implemented; baseline metrics not independently recomputed | `reports/baseline_model.md`, `models/baseline_cnn_checkpoint.pth` (different architecture, not used by API) |
| 5 Model factory / experiments | Implemented; comparison is small-sample, single-seed | `reports/experiments/` |
| 6 Explainability | Implemented; **generated with earlier small-sample checkpoints, not re-run on the current one**; Grad-CAM pointing-game 0.50, ViT attention 0.03 — faithfulness not established | `outputs/explainability/` |
| 7 API | Verified (see tests below) | `scripts/test_api_*.py` |
| 8 Multimodal / 9 Deployment | Not started | RSNA has no symptoms/history; no Dockerfile |

## Tests (run with the project `.venv`)
```
python scripts/test_api_foundation.py          # 10 checks
python scripts/test_api_explanations.py        # 6 checks
python scripts/test_api_integration.py --image <dcm>
python scripts/test_api_reliability.py --image <dcm>
python tests/test_data_integrity.py            # 7 checks
python tests/test_runner_guard.py              # 1 check
python scripts/evaluate_model.py --experiment <experiment_name>
```
`pytest` is not installed; every test file also runs as a plain script. Passing API tests show the service behaves consistently; they do not show clinical validity or that heatmaps are faithful.

## Known limitations
- Single seed, 1 effective epoch, no confidence intervals or calibration analysis.
- Default threshold 0.5 gives low sensitivity (0.56).
- `predict` is `async def` but runs blocking GPU work; one request blocks others (including `/health`). Safe for a single process; do not add threads without making Grad-CAM hook usage thread-safe.
- Only MONOCHROME2 8-bit DICOMs were seen in training; MONOCHROME1, rescaled, or compressed DICOM uploads are not specifically handled. PNG/JPEG uploads pass through 8-bit grayscale conversion.
- `src/api/services/explanation_services.py` (plural) is unused duplicate code; `src/explainability/input_pipeline.py` and `model_loader.py` are empty.
- ViT-B/16 is selectable in the API, but its only checkpoint is a collapsed smoke-test run.
- Several docs are empty placeholders (`ROADMAP.md`, `experiment_log.md`, `change_log.md`, ...).

## Obtaining / regenerating checkpoints
Checkpoints are git-ignored (`*.pth`, `outputs/experiments/**/checkpoints/`). The active ResNet50 file exists only on the original machine; back it up (e.g. GitHub Release asset or cloud storage) and verify with the SHA-256 above. To regenerate: `run_experiment(ExperimentConfig(model_name="resnet50", pretrained=True, batch_size=4, learning_rate=1e-4, epochs=2, seed=42), ...)` with no sample limits and a new `experiment_name`; results will be similar but not bit-identical.

## Before Sprint 8 / 9
1. Owner decision on threshold / research goal.
2. Re-run explainability evaluation on the active checkpoint.
3. Back up the checkpoint with its hash; pin dependencies.
4. Sprint 8 needs a dataset with symptoms/history; RSNA headers offer at most age/sex/view. Sprint 9 should follow items 1–3.
