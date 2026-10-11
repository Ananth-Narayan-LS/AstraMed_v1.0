"""
Request-ID regression tests. They do not need the model checkpoint.

Regression: RequestIDMiddleware was registered twice, so when the caller sent no
X-Request-ID header the response header and the body request_id differed.
"""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.main import app
from src.api.middleware import RequestIDMiddleware

client = TestClient(app)


def test_middleware_registered_once():
    names = [m.cls for m in app.user_middleware]
    assert names.count(RequestIDMiddleware) == 1


def test_generated_request_id_matches_header_and_body():
    # Rejected upload: reaches the route's error response without needing the model.
    response = client.post("/api/v1/predict", files={"image": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert response.headers["x-request-id"] == response.json()["request_id"]


def test_generated_ids_are_unique_and_supplied_id_is_kept():
    a = client.get("/api/v1/health").headers["x-request-id"]
    b = client.get("/api/v1/health").headers["x-request-id"]
    assert a != b
    assert client.get("/api/v1/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"


def test_invalid_supplied_id_is_replaced():
    response = client.get("/api/v1/health", headers={"X-Request-ID": "bad id with spaces!"})
    assert response.headers["x-request-id"] != "bad id with spaces!"


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS:", test.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
