"""
Smoke-test a RUNNING AstraMed API (local process or Docker container) over real HTTP.

    python scripts/smoke_test_deployment.py --url http://127.0.0.1:8000 --image <file.dcm>

Needs httpx (requirements-dev.txt). It sends one prediction request and a few invalid ones;
it never writes files. Exit code 0 only if every check passes.
"""
import argparse
import sys
from pathlib import Path

import httpx

failures = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(("PASS: " if condition else "FAIL: ") + label + (f" ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", required=True, help="A DICOM/PNG/JPEG chest X-ray.")
    parser.add_argument("--expect-ready", choices=["yes", "no"], default="yes",
                        help="Whether the model checkpoint is expected to be mounted.")
    args = parser.parse_args()

    base = args.url.rstrip("/") + "/api/v1"
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return 2

    with httpx.Client(timeout=120) as client:
        r = client.get(f"{base}/health")
        check(r.status_code == 200 and r.json().get("status") == "ok", "GET /health returns ok", r.text)
        check("x-request-id" in r.headers, "health response carries X-Request-ID")

        r = client.get(f"{base}/ready")
        expected = 200 if args.expect_ready == "yes" else 503
        check(r.status_code == expected, f"GET /ready returns {expected}", f"{r.status_code} {r.text}")
        check("\\" not in r.text and "/app" not in r.text and "Users" not in r.text,
              "readiness reveals no paths")

        if args.expect_ready == "yes":
            with image_path.open("rb") as handle:
                r = client.post(f"{base}/predict", files={"image": (image_path.name, handle)},
                                headers={"X-Request-ID": "smoke-test-1"})
            body = r.json()
            check(r.status_code == 200 and body.get("status") == "success", "POST /predict succeeds", r.text[:200])
            check(r.headers.get("x-request-id") == "smoke-test-1" == body.get("request_id"),
                  "request ID is echoed in header and body")
            check(0.0 <= body["prediction"]["pneumonia_probability"] <= 1.0
                  and body["prediction"]["threshold"] == 0.5, "probability valid, threshold is 0.5")
            check(all(body["explanation"].get(k) for k in
                      ("original_image_base64", "heatmap_base64", "overlay_base64")), "explanation images present")
        else:
            with image_path.open("rb") as handle:
                r = client.post(f"{base}/predict", files={"image": (image_path.name, handle)})
            check(r.status_code == 503 and r.json()["error"]["code"] == "CHECKPOINT_NOT_FOUND",
                  "predict without checkpoint gives a controlled 503", r.text[:200])

        r = client.post(f"{base}/predict", files={"image": ("note.txt", b"not an image")})
        check(r.status_code == 400 and r.json()["status"] == "error", "invalid file rejected with 400", r.text[:200])
        check(r.headers.get("x-request-id") == r.json().get("request_id"), "error request ID matches header")
        check("\\" not in r.text and "/app" not in r.text and "Users" not in r.text, "error body reveals no paths")

        r = client.post(f"{base}/predict", files={"image": ("empty.png", b"")})
        check(r.status_code == 400, "empty upload rejected with 400", str(r.status_code))

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + '; '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
