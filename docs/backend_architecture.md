# AstraMed Backend Architecture — Sprint 7, M1

## Purpose

Create a versioned FastAPI backend contract for the AstraMed
chest X-ray pneumonia research project.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Basic application information |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/predict` | Prediction contract stub |

## Prediction request

The prediction endpoint accepts multipart form data:

- `image`: required uploaded file
- `model_name`: `resnet50` or `vit_b16`
- `explanation_method`: `gradcam` or `attention`
- `include_overlay`: boolean

The request options are validated by FastAPI.



## Error handling

The M1 prediction stub returns HTTP 501 with a structured
`ErrorResponse`.

Invalid form options are rejected by FastAPI with HTTP 422.

## Configuration

The API can be configured with these environment variables:

- `ASTRAMED_APP_NAME`
- `ASTRAMED_APP_VERSION`
- `ASTRAMED_HOST`
- `ASTRAMED_PORT`
- `ASTRAMED_LOG_LEVEL`
- `ASTRAMED_MAX_UPLOAD_MB`

The upload-size setting is reserved for M2; M1 does not yet
enforce file-size or file-content validation.

## Local execution

Run from the repository root:

`python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload`

Swagger: `http://127.0.0.1:8000/docs`

OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

Health: `http://127.0.0.1:8000/api/v1/health`

## M2 — Image upload and validation

Supported formats:
- PNG
- JPEG
- DICOM

Validation behavior:
- Enforce the configured maximum upload size.
- Validate actual image content rather than trusting the extension.
- Reject empty, corrupted, and unsupported files.
- Check that DICOM files contain image dimensions and pixel data.
- Return structured error responses with HTTP status codes.
- Do not save uploaded images to disk.

The successful M2 response reports the detected format, byte size,
dimensions, and request ID. It does not contain a prediction.

The API uses Pillow and pydicom locally. No external image-processing
service or API key is required.

Actual DICOM pixel decoding, preprocessing, pneumonia inference,
and explanation generation remain future milestones.

## M3 — Local prediction service

The API loads the existing ResNet-50 checkpoint from:

`outputs/experiments/resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/checkpoints/best_model.pth`

Inference pipeline:

1. Validate the uploaded image.
2. Decode PNG/JPEG or DICOM pixels into a grayscale NumPy array.
3. Apply `preprocess_image(..., target_size=(512, 512))`.
4. Convert to a tensor with shape `[1, 1, 512, 512]`.
5. Apply `AdvancedInputAdapter` to produce `[1, 3, 224, 224]`.
6. Load the local model checkpoint and compute logits.
7. Apply sigmoid to obtain the pneumonia probability.
8. Return the probability, predicted class, threshold, model metadata,
   request ID, and processing time.

The initial classification threshold is 0.5.

Inference is performed locally with PyTorch. CUDA is used when available;
otherwise, the service falls back to CPU.

The checkpoint is a limited screening experiment, not a clinically
validated model. Predictions must not be used as a medical diagnosis.

No paid external inference API is required.