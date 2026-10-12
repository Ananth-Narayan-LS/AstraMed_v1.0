"""
Deployment tests that need neither Docker nor the dataset. Run: python tests/test_deployment.py

They check (1) the static Docker/compose/requirements files against the real code,
(2) readiness, startup preload and configuration behaviour of the API, and
(3) that error responses do not reveal local paths.
Building and running the container itself is NOT covered here (see docs/DEPLOYMENT.md).
"""
import ast
import dataclasses
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.services import prediction_service as ps

IMPORT_TO_PACKAGE = {"PIL": "pillow", "dotenv": "python-dotenv", "yaml": "pyyaml"}
STDLIB = set(sys.stdlib_module_names)
NOT_IN_API_IMAGE = {"pandas", "sklearn", "scipy", "matplotlib", "streamlit", "cv2"}


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def requirement_names(text: str) -> dict:
    names = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            match = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==(.+)$", line)
            assert match, f"requirement is not pinned with ==: {line!r}"
            names[match.group(1).lower().replace("_", "-")] = match.group(3)
    return names


def src_import_closure(entry: str = "src.api.main") -> tuple[set, set]:
    """Follow imports of project modules starting at `entry`; return (modules, third-party roots)."""
    seen, third_party, todo = set(), set(), [entry]
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        path = ROOT / (module.replace(".", "/") + ".py")
        if not path.exists():
            path = ROOT / module.replace(".", "/") / "__init__.py"
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            for name in names:
                root = name.split(".")[0]
                if root == "src":
                    todo.append(name)
                elif root not in STDLIB:
                    third_party.add(root)
    return seen, third_party


# ------------------------------------------------------------------ static files
def test_requirements_api_is_pinned_and_covers_the_api_import_closure():
    pinned = requirement_names(read("requirements-api.txt"))
    modules, third_party = src_import_closure()
    needed = {IMPORT_TO_PACKAGE.get(name, name).lower() for name in third_party}
    assert not (needed & NOT_IN_API_IMAGE), f"API imports heavy packages: {needed & NOT_IN_API_IMAGE}"
    missing = needed - set(pinned)
    assert not missing, f"packages imported by the API but absent from requirements-api.txt: {missing}"
    assert "python-multipart" in pinned and "uvicorn" in pinned   # needed at run time, imported lazily
    assert not any(m.startswith("src.multimodal") for m in modules), "API must not import the multimodal code"


def test_dockerfile_matches_the_repository():
    text = read("Dockerfile")
    pinned = requirement_names(read("requirements-api.txt"))
    assert "COPY src/ ./src/" in text and "COPY requirements-api.txt" in text
    assert "src.api.main:app" in text and (ROOT / "src" / "api" / "main.py").exists()
    assert "USER astramed" in text and "HEALTHCHECK" in text and "/api/v1/health" in text
    assert f"torch=={pinned['torch']}" in text and f"torchvision=={pinned['torchvision']}" in text
    forbidden = (".pth", "outputs", "data/", "COPY . ", "ADD ", ".env")
    for line in text.splitlines():
        if line.startswith(("COPY", "ADD")):
            assert not any(token in line for token in forbidden), line
    assert "ASTRAMED_PRELOAD_MODEL" not in text or "ASTRAMED_PRELOAD_MODEL=true" not in text  # image default stays lazy


def test_dockerignore_keeps_data_weights_and_secrets_out_of_the_build_context():
    entries = set(read(".dockerignore").split())
    for required in (".git", ".venv", ".env", "data", "outputs", "models", "**/*.pth", "**/*.pt",
                     "**/*.dcm", "**/*.npy", "**/*.csv", "notebooks", "src/multimodal"):
        assert required in entries, f".dockerignore must exclude {required}"


def test_compose_mounts_checkpoint_read_only_and_binds_localhost():
    text = read("docker-compose.yml")
    experiment = ps.CHECKPOINT_PATHS["resnet50"].parents[1].name
    assert f"{experiment}/checkpoints:/app/outputs/experiments/{experiment}/checkpoints:ro" in text
    assert '"127.0.0.1:8000:8000"' in text and "read_only: true" in text
    assert "no-new-privileges" in text and "cap_drop" in text
    assert "ASTRAMED_PREDICTION_THRESHOLD: \"0.5\"" in text
    # the mounted path must be exactly where the code looks for the checkpoint
    assert str(ps.CHECKPOINT_PATHS["resnet50"].relative_to(ps.PROJECT_ROOT)).replace("\\", "/") == (
        f"outputs/experiments/{experiment}/checkpoints/best_model.pth")


def test_env_example_documents_every_setting_the_code_reads():
    documented = set(re.findall(r"^#?\s*(ASTRAMED_[A-Z_]+)=", read(".env.example"), re.M))
    used = set(re.findall(r"ASTRAMED_[A-Z_]+", read("src/api/config.py")))
    undocumented = {name for name in used if name not in documented and name not in (
        "ASTRAMED_API_TITLE", "ASTRAMED_API_VERSION")}  # legacy aliases
    assert not undocumented, f"settings missing from .env.example: {undocumented}"


# ------------------------------------------------------------------ runtime behaviour
class temporary_checkpoint_path:
    """Point the API at a different checkpoint path for the duration of a test."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.original = ps.CHECKPOINT_PATHS["resnet50"]
        ps.CHECKPOINT_PATHS["resnet50"] = self.path
        ps._load_model_bundle.cache_clear()

    def __exit__(self, *exc):
        ps.CHECKPOINT_PATHS["resnet50"] = self.original
        ps._load_model_bundle.cache_clear()


def test_health_is_independent_of_the_checkpoint_and_ready_is_not():
    with tempfile.TemporaryDirectory() as tmp, temporary_checkpoint_path(Path(tmp) / "absent.pth"):
        client = TestClient(api_main.app)
        assert client.get("/api/v1/health").status_code == 200
        response = client.get("/api/v1/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "not_ready", "model": "resnet50", "reason": "CHECKPOINT_NOT_FOUND"}
        assert tmp not in response.text and "Users" not in response.text and "\\" not in response.text
        present = Path(tmp) / "present.pth"
        present.write_bytes(b"not loaded by the readiness check")
        ps.CHECKPOINT_PATHS["resnet50"] = present
        response = client.get("/api/v1/ready")
        assert response.status_code == 200 and response.json() == {"status": "ready", "model": "resnet50"}


def test_predict_without_checkpoint_is_a_controlled_503_without_paths():
    with tempfile.TemporaryDirectory() as tmp, temporary_checkpoint_path(Path(tmp) / "absent.pth"):
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("L", (64, 64), 100).save(buffer, "PNG")
        client = TestClient(api_main.app)
        response = client.post("/api/v1/predict", files={"image": ("a.png", buffer.getvalue(), "image/png")})
        body = response.json()
        assert response.status_code == 503 and body["error"]["code"] == "CHECKPOINT_NOT_FOUND"
        assert tmp not in response.text and "Users" not in response.text
        assert response.headers["x-request-id"] == body["request_id"]


def test_invalid_upload_error_reveals_no_paths():
    client = TestClient(api_main.app)
    response = client.post("/api/v1/predict", files={"image": ("note.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert str(ROOT) not in response.text and "Users" not in response.text


def test_preload_fails_fast_when_checkpoint_is_missing_and_is_optional():
    with tempfile.TemporaryDirectory() as tmp, temporary_checkpoint_path(Path(tmp) / "absent.pth"):
        original = api_main.settings
        try:
            api_main.settings = dataclasses.replace(original, preload_model=True)
            try:
                with TestClient(api_main.app):
                    pass
            except RuntimeError as exc:
                assert "CHECKPOINT_NOT_FOUND" in str(exc) and tmp not in str(exc)
            else:
                raise AssertionError("startup must fail when the checkpoint is missing and preload is on")
            api_main.settings = dataclasses.replace(original, preload_model=False)
            with TestClient(api_main.app) as client:           # default: lazy, still starts
                assert client.get("/api/v1/health").status_code == 200
        finally:
            api_main.settings = original


def test_preload_loads_the_real_checkpoint_when_available():
    real = ps.CHECKPOINT_PATHS["resnet50"]
    if not real.is_file():
        print("SKIP: real checkpoint not present")
        return
    ps._load_model_bundle.cache_clear()
    original = api_main.settings
    try:
        api_main.settings = dataclasses.replace(original, preload_model=True)
        with TestClient(api_main.app) as client:
            assert client.get("/api/v1/ready").status_code == 200
        assert ps._load_model_bundle.cache_info().currsize == 1       # model really was loaded at startup
    finally:
        api_main.settings = original


def test_invalid_configuration_stops_startup_with_a_clear_message():
    for variable, value in (("ASTRAMED_PRELOAD_MODEL", "maybe"), ("ASTRAMED_PREDICTION_THRESHOLD", "7"),
                            ("ASTRAMED_DEFAULT_MODEL", "nonexistent"), ("ASTRAMED_MAX_UPLOAD_MB", "0")):
        env = {**os.environ, variable: value, "PYTHONPATH": str(ROOT)}
        result = subprocess.run([sys.executable, "-c", "import src.api.config"], env=env,
                                capture_output=True, text=True, cwd=tempfile.gettempdir())
        assert result.returncode != 0 and variable in result.stderr, (variable, result.stderr[-200:])


def test_default_threshold_and_model_are_unchanged():
    assert api_main.settings.prediction_threshold == 0.5 and api_main.settings.default_model == "resnet50"
    assert ps.PREDICTION_THRESHOLD == 0.5


if __name__ == "__main__":
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS:", test.__name__)
    print(f"{len(tests)}/{len(tests)} executed")
