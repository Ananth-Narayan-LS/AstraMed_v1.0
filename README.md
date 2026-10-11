## AstraMed — Explainable Pneumonia Detection (research prototype)

AstraMed classifies chest X-rays from the RSNA Pneumonia Detection Challenge (stage 2; 26,684 patients, label = lung opacity yes/no) with an ImageNet-pretrained ResNet50 and returns a Grad-CAM heatmap through a FastAPI service. Patients are split 70/15/15 (seed 42) into train / validation / test with no patient overlap.

Current verified state (details, evidence and caveats in [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)):

- Held-out test split (4,004 patients) at threshold 0.5: ROC-AUC 0.878, sensitivity 0.56, specificity 0.93. Historical independence of the test split cannot be guaranteed from the repository.
- The deployed threshold is 0.5; no validation-based threshold has been chosen yet. A validation-only comparison of thresholds (e.g. 0.30 and 0.50) is in the status document.
- Active checkpoint: `outputs/experiments/resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/checkpoints/best_model.pth` (git-ignored; SHA-256 in the status document). Its exact training provenance cannot be independently established: the original run's log and metadata were not preserved.
- The image-only ResNet50 pipeline is the completed baseline. Planned next step: multimodal research, subject to a defensible data-source decision (RSNA has no symptoms or clinical history).
- Sprints 1–7 are implemented; Sprint 8 (multimodal) and Sprint 9 (deployment) have not started.
- Grad-CAM heatmaps show where the model's score responded; they are not proof of disease location or reasoning. The overlay now blends only where the heatmap is active; the stored Sprint 6 gallery images still use the older whole-image tint. A small (25-image, validation-only, one-seed) weight-randomization check found the maps depend on the learned weights.
- Not clinically validated; do not use for medical decisions.

## AstraMed API — Local Setup

### Requirements

- Python version compatible with the project's installed PyTorch and other dependencies.
- Git.
- A trained AstraMed model checkpoint.
- Optional NVIDIA GPU with a compatible PyTorch CUDA installation.

### 1. Clone the repository

```powershell
git clone https://github.com/Ananth-Narayan-LS/AstraMed_v1.0.git
cd AstraMed_v1.0
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use the activation instructions for your system rather than changing system-wide security settings unnecessarily.

### 3. Install dependencies

Install the dependencies listed in the repository's dependency file. Install the appropriate PyTorch build for your hardware and operating system (a plain `pip install -r requirements.txt` gives the CPU build of PyTorch on Windows).

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install (`-e .`) is required: the scripts in `scripts/` import the `src` package and only find it when the project is installed (or when run from the repository root with `python -m`).

### 4. Configure the API

Copy `.env.example` to `.env` and edit the values if needed.

```powershell
Copy-Item .env.example .env
```

The `.env` file contains local configuration and should not be committed.

### 5. Place model checkpoints

The API expects trained checkpoints under:

```text
outputs/experiments/<experiment_name>/checkpoints/best_model.pth
```

The configured ResNet-50 checkpoint path is documented in `src/api/services/prediction_service.py`. Checkpoint files are model artifacts and may not be included in a fresh Git clone; obtain them from the project's approved artifact location.

### 6. Start the API

From the repository root:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

### 7. Run backend tests

```powershell
python scripts/test_api_foundation.py
python scripts/test_api_explanations.py
python scripts/test_api_integration.py --image "data\raw\rsna-pneumonia-detection-challenge\stage_2_train_images\0a0f91dc-6015-4342-b809-d19610854a21.dcm"
python scripts/test_api_reliability.py --image "data\raw\rsna-pneumonia-detection-challenge\stage_2_train_images\0a0f91dc-6015-4342-b809-d19610854a21.dcm"
python tests/test_data_integrity.py
python tests/test_runner_guard.py
python tests/test_request_id.py
python tests/test_overlay.py
```

The integration test requires the specified DICOM file and a compatible trained checkpoint.

### Model status and provenance

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for verified sprint status, checkpoint provenance, known limitations, and how to re-evaluate the model (`scripts/evaluate_model.py`).

### Scope and limitations

AstraMed is a research and educational project. Model predictions and heatmaps are not clinically validated and must not be used as a substitute for professional medical assessment. Screening metrics from small experiments are not evidence of clinical readiness.