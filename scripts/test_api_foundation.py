from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.api.services.image_validation import (
    UploadValidationError,
    validate_uploaded_image,
)

client = TestClient(app)


def make_png() -> bytes:
    buffer = BytesIO()

    Image.new("RGB", (64, 64), color="white").save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def make_gif() -> bytes:
    buffer = BytesIO()

    Image.new("RGB", (32, 32), color="white").save(
        buffer,
        format="GIF",
    )

    return buffer.getvalue()


def test_root_and_health():
    root = client.get("/")
    health = client.get("/api/v1/health")

    assert root.status_code == 200
    assert root.json()["app"] == "AstraMed API"

    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_openapi_documents_m2_response():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    spec = response.json()
    predict_spec = spec["paths"]["/api/v1/predict"]["post"]

    success_schema = predict_spec["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert success_schema["$ref"] == (
        "#/components/schemas/InferenceResponse"
    )


def test_valid_png_upload():
    response = client.post(
        "/api/v1/predict",
        files={
            "image": (
                "chest_xray.png",
                make_png(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "success"
    assert body["request_id"]

    prediction = body["prediction"]

    assert prediction["predicted_class"] in {
        "pneumonia",
        "normal",
    }
    assert 0.0 <= prediction["pneumonia_probability"] <= 1.0
    assert prediction["threshold"] == 0.5

    assert body["model"]["name"] == "resnet50"
    assert body["model"]["version"].startswith("checkpoint-epoch-")
    assert body["timing"]["total_ms"] >= 0

def test_corrupted_upload_is_rejected():
    response = client.post(
        "/api/v1/predict",
        files={
            "image": (
                "broken.png",
                b"this is not a real image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["status"] == "error"
    assert response.json()["error"]["code"] == "INVALID_IMAGE_FILE"


def test_unsupported_image_format_is_rejected():
    response = client.post(
        "/api/v1/predict",
        files={
            "image": (
                "animation.gif",
                make_gif(),
                "image/gif",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == (
        "UNSUPPORTED_IMAGE_FORMAT"
    )


def test_empty_upload_is_rejected():
    response = client.post(
        "/api/v1/predict",
        files={
            "image": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_FILE"


def test_service_rejects_oversized_content():
    content = make_png()

    try:
        validate_uploaded_image(
            content=content,
            filename="test.png",
            max_bytes=8,
        )
    except UploadValidationError as exc:
        assert exc.status_code == 413
        assert exc.code == "FILE_TOO_LARGE"
    else:
        raise AssertionError("Oversized content was not rejected")


def run_tests():
    tests = [
        test_root_and_health,
        test_openapi_documents_m2_response,
        test_valid_png_upload,
        test_corrupted_upload_is_rejected,
        test_unsupported_image_format_is_rejected,
        test_empty_upload_is_rejected,
        test_service_rejects_oversized_content,
    ]

    failures = []

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL: {test.__name__}: {exc}")

    print(f"\nResults: {len(tests) - len(failures)}/{len(tests)} passed.")

    if failures:
        raise SystemExit(1)

    print("Sprint 7 M2 validation tests passed.")


if __name__ == "__main__":
    run_tests()
