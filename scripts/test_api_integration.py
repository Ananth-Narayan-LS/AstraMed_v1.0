
import argparse
import base64
import mimetypes
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def decode_and_check_png(encoded_image, image_name):
    """Verify that a Base64 image is a valid PNG payload."""
    check(
        isinstance(encoded_image, str) and len(encoded_image) > 0,
        f"{image_name} is missing or empty.",
    )

    try:
        image_bytes = base64.b64decode(
            encoded_image,
            validate=True,
        )
    except Exception as exc:
        raise AssertionError(
            f"{image_name} is not valid Base64."
        ) from exc

    png_signature = b"\x89PNG\r\n\x1a\n"

    check(
        image_bytes.startswith(png_signature),
        f"{image_name} is not a valid PNG payload.",
    )

    print(f"PASS: {image_name} is valid Base64-encoded PNG.")


def upload_image(client, image_path, model_name, method, overlay):
    content_type = (
        mimetypes.guess_type(image_path.name)[0]
        or "application/octet-stream"
    )

    with image_path.open("rb") as image_file:
        return client.post(
            "/api/v1/predict",
            files={
                "image": (
                    image_path.name,
                    image_file,
                    content_type,
                )
            },
            data={
                "model_name": model_name,
                "explanation_method": method,
                "include_overlay": str(overlay).lower(),
            },
        )


def test_successful_end_to_end_prediction(client, image_path):
    response = upload_image(
        client,
        image_path,
        model_name="resnet50",
        method="gradcam",
        overlay=True,
    )

    check(
        response.status_code == 200,
        f"Expected HTTP 200, got {response.status_code}: "
        f"{response.text[:1000]}",
    )

    body = response.json()

    check(body["status"] == "success", "Unexpected response status.")
    check(bool(body["request_id"]), "Missing request ID.")

    prediction = body["prediction"]
    probability = prediction["pneumonia_probability"]

    check(
        isinstance(probability, (int, float))
        and 0.0 <= probability <= 1.0,
        "Pneumonia probability must be between 0 and 1.",
    )

    check(
        prediction["predicted_class"] in {"normal", "pneumonia"},
        "Unexpected predicted class.",
    )

    check(
        prediction["threshold"] == 0.5,
        "Unexpected prediction threshold.",
    )

    explanation = body["explanation"]

    check(
        explanation["method"] == "gradcam",
        "Unexpected explanation method.",
    )

    decode_and_check_png(
        explanation["original_image_base64"],
        "Original image",
    )
    decode_and_check_png(
        explanation["heatmap_base64"],
        "Heatmap",
    )
    decode_and_check_png(
        explanation["overlay_base64"],
        "Overlay",
    )

    check(
        explanation["image_mime_type"] == "image/png",
        "Unexpected explanation image MIME type.",
    )

    check(explanation["width"] > 0, "Invalid image width.")
    check(explanation["height"] > 0, "Invalid image height.")

    check(
        body["model"]["name"] == "resnet50",
        "Unexpected model name.",
    )
    check(
        body["timing"]["total_ms"] >= 0,
        "Invalid timing value.",
    )

    print("PASS: End-to-end upload, prediction and explanation.")
    print(f"      Predicted class: {prediction['predicted_class']}")
    print(f"      Probability: {probability:.4f}")
    print(
        f"      Image dimensions: "
        f"{explanation['width']} x {explanation['height']}"
    )


def test_overlay_can_be_disabled(client, image_path):
    response = upload_image(
        client,
        image_path,
        model_name="resnet50",
        method="gradcam",
        overlay=False,
    )

    check(
        response.status_code == 200,
        f"Overlay-disabled request failed: "
        f"{response.status_code}: {response.text[:1000]}",
    )

    body = response.json()
    explanation = body["explanation"]

    check(
        bool(explanation["original_image_base64"]),
        "Original image is missing.",
    )
    check(
        bool(explanation["heatmap_base64"]),
        "Heatmap is missing.",
    )
    check(
        explanation["overlay_base64"] is None,
        "Overlay should be null when disabled.",
    )

    print("PASS: Overlay can be disabled without breaking the API.")


def test_incompatible_explanation_is_rejected(client, image_path):
    response = upload_image(
        client,
        image_path,
        model_name="resnet50",
        method="attention",
        overlay=True,
    )

    check(
        response.status_code == 422,
        f"Expected HTTP 422 for incompatible method, got "
        f"{response.status_code}: {response.text[:1000]}",
    )

    print("PASS: Incompatible model/explanation method is rejected.")


def main():
    parser = argparse.ArgumentParser(
        description="Run AstraMed prediction API integration tests."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a supported chest X-ray image.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()

    check(image_path.is_file(), f"Image file not found: {image_path}")

    print("Starting AstraMed API integration tests...")

    with TestClient(app) as client:
        test_successful_end_to_end_prediction(client, image_path)
        test_overlay_can_be_disabled(client, image_path)
        test_incompatible_explanation_is_rejected(client, image_path)

    print("\nAll API integration tests passed.")


if __name__ == "__main__":
    main()
