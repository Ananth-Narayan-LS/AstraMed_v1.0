
import argparse
import gc
import mimetypes
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.services.prediction_service import (
    PredictionServiceError,
    _load_model_bundle,
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def send_prediction(client, image_path, model="resnet50", method="gradcam"):
    """Send one image to the real prediction API."""
    content_type = (
        mimetypes.guess_type(image_path.name)[0]
        or "application/octet-stream"
    )

    with image_path.open("rb") as file:
        return client.post(
            "/api/v1/predict",
            files={
                "image": (
                    image_path.name,
                    file,
                    content_type,
                )
            },
            data={
                "model_name": model,
                "explanation_method": method,
                "include_overlay": "true",
            },
        )


def test_success_response_schema(client, image_path):
    response = send_prediction(client, image_path)

    check(
        response.status_code == 200,
        f"Valid image failed: HTTP {response.status_code}: "
        f"{response.text[:1000]}",
    )

    body = response.json()

    # Check the top-level response structure.
    required_fields = {
        "request_id",
        "status",
        "prediction",
        "explanation",
        "model",
        "timing",
    }
    check(
        required_fields.issubset(body.keys()),
        f"Missing response fields: {required_fields - body.keys()}",
    )

    check(body["status"] == "success", "Unexpected response status.")
    check(isinstance(body["request_id"], str), "Invalid request ID.")

    prediction = body["prediction"]
    check(
        0.0 <= prediction["pneumonia_probability"] <= 1.0,
        "Probability is outside the range [0, 1].",
    )
    check(
        prediction["predicted_class"] in {"normal", "pneumonia"},
        "Unexpected predicted class.",
    )
    check(
        prediction["predicted_class"]
        == (
            "pneumonia"
            if prediction["pneumonia_probability"] >= prediction["threshold"]
            else "normal"
        ),
        "Predicted class does not match the returned threshold.",
    )

    explanation = body["explanation"]
    for key in (
        "original_image_base64",
        "heatmap_base64",
        "overlay_base64",
    ):
        check(
            isinstance(explanation.get(key), str) and explanation[key],
            f"Missing explanation field: {key}",
        )

    check(
        explanation["image_mime_type"] == "image/png",
        "Unexpected explanation MIME type.",
    )
    check(explanation["width"] > 0, "Invalid image width.")
    check(explanation["height"] > 0, "Invalid image height.")
    check(body["model"]["name"] == "resnet50", "Wrong model name.")
    check(body["timing"]["total_ms"] >= 0, "Invalid request timing.")

    print("PASS: Successful response schema and values.")


def test_unsupported_file(client, work_dir):
    # A text file is deliberately uploaded as an unsupported file type.
    bad_file = work_dir / "unsupported.txt"
    bad_file.write_text("This is not a chest X-ray.", encoding="utf-8")

    response = send_prediction(client, bad_file)

    check(
        response.status_code in {400, 415, 422},
        f"Unsupported file returned unexpected HTTP "
        f"{response.status_code}: {response.text[:1000]}",
    )
    check(
        isinstance(response.json(), dict),
        "Error response must be a JSON object.",
    )

    print(
        f"PASS: Unsupported file rejected with HTTP {response.status_code}."
    )


def test_corrupted_dicom(client, work_dir):
    # A .dcm filename with invalid content tests DICOM decoding failure.
    bad_file = work_dir / "corrupted.dcm"
    bad_file.write_bytes(b"This is not valid DICOM data.")

    response = send_prediction(client, bad_file)

    check(
        response.status_code in {400, 415, 422},
        f"Corrupted DICOM returned unexpected HTTP "
        f"{response.status_code}: {response.text[:1000]}",
    )
    check(
        isinstance(response.json(), dict),
        "Error response must be a JSON object.",
    )

    print(
        f"PASS: Corrupted DICOM rejected with HTTP {response.status_code}."
    )


def test_missing_checkpoint(client, image_path):
    # Clear the cache so the request must resolve the checkpoint again.
    _load_model_bundle.cache_clear()

    try:
        with patch(
            "src.api.services.prediction_service._resolve_checkpoint",
            side_effect=PredictionServiceError(
                "CHECKPOINT_NOT_FOUND",
                "Simulated missing checkpoint for reliability testing.",
                503,
            ),
        ):
            response = send_prediction(client, image_path)

        check(
            response.status_code == 503,
            f"Expected HTTP 503 for missing checkpoint, got "
            f"{response.status_code}: {response.text[:1000]}",
        )
        check(
            isinstance(response.json(), dict),
            "Checkpoint error response must be a JSON object.",
        )

        print("PASS: Missing checkpoint produces a controlled HTTP 503.")
    finally:
        # Ensure subsequent tests can load the real checkpoint.
        _load_model_bundle.cache_clear()


def test_repeated_requests_and_gpu_memory(client, image_path):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    # Warm up once so initial model loading isn't mistaken for a memory leak.
    warmup = send_prediction(client, image_path)
    check(
        warmup.status_code == 200,
        f"Warm-up request failed: {warmup.status_code}: "
        f"{warmup.text[:1000]}",
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        baseline_bytes = torch.cuda.memory_allocated()

    allocated_samples = []

    # Send three more requests using the same loaded model.
    for request_number in range(1, 4):
        response = send_prediction(client, image_path)

        check(
            response.status_code == 200,
            f"Repeated request {request_number} failed: "
            f"{response.status_code}: {response.text[:1000]}",
        )

        body = response.json()
        check(
            body["status"] == "success",
            f"Repeated request {request_number} returned failure status.",
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            allocated_samples.append(torch.cuda.memory_allocated())

        print(f"PASS: Repeated request {request_number}.")

    if torch.cuda.is_available():
        max_growth_bytes = max(allocated_samples) - baseline_bytes
        max_growth_mib = max_growth_bytes / (1024 ** 2)

        print(
            f"GPU allocated memory after warm-up: "
            f"{baseline_bytes / (1024 ** 2):.2f} MiB"
        )
        print(
            f"Maximum observed growth: {max_growth_mib:.2f} MiB"
        )

        # Allow small allocator/runtime variations while detecting
        # substantial retained allocations.
        check(
            max_growth_mib < 128,
            "GPU allocated memory grew by 128 MiB or more after warm-up. "
            "Investigate possible retained tensors or hooks.",
        )
        print("PASS: GPU allocated memory stayed within the test limit.")
    else:
        print(
            "SKIP: CUDA is unavailable; GPU memory test was not run."
        )


def main():
    parser = argparse.ArgumentParser(
        description="AstraMed API reliability tests."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a valid supported chest X-ray.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    check(image_path.is_file(), f"Image not found: {image_path}")

    work_dir = Path("outputs") / "api_reliability_temp"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("Starting AstraMed API reliability tests...")

    try:
        with TestClient(app) as client:
            test_success_response_schema(client, image_path)
            test_unsupported_file(client, work_dir)
            test_corrupted_dicom(client, work_dir)
            test_missing_checkpoint(client, image_path)
            test_repeated_requests_and_gpu_memory(client, image_path)
    finally:
        # Remove only the temporary files created by this test.
        for name in ("unsupported.txt", "corrupted.dcm"):
            path = work_dir / name
            if path.exists():
                path.unlink()

        if work_dir.exists() and not any(work_dir.iterdir()):
            work_dir.rmdir()

    print("\nAPI reliability tests completed successfully.")


if __name__ == "__main__":
    main()
