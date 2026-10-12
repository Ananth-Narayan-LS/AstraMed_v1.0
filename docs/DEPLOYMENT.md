# AstraMed API — Docker deployment (Sprint 9)

Research prototype. **Not clinically validated; not for medical decisions.** This deploys the existing **image-only ResNet50** API unchanged (same preprocessing, same checkpoint selection, default threshold 0.5). The experimental Sprint 8 multimodal models are not part of the image and are not served.

## Verification status (read first)
Updated after the image was built and the container run on the development machine (Windows, Docker 29.8.2, Docker Compose v5.5.1, amd64).

| Item | Status |
|---|---|
| Image `astramed-api:local` builds | **Verified** (1.58 GB; Python 3.14.8 in the image; every installed package matches `requirements-api.txt`, torch/torchvision are the `+cpu` builds). |
| Container starts and reaches `healthy` | **Verified**, both with `docker run` and with Docker Compose. With `ASTRAMED_PRELOAD_MODEL=true` the log shows `Model resnet50 preloaded.` |
| `GET /api/v1/health` and `GET /api/v1/ready` | **Verified**: 200 `ok` and 200 `ready`. |
| Prediction smoke test (`scripts/smoke_test_deployment.py`, 12 checks) | **Verified** against the `docker run` container (reported by the project owner) and against the Docker Compose container (run by the audit, 12/12 passed). |
| Checkpoint mount | **Verified**: bind mount of the baseline `...resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/checkpoints` folder, `rw=false`; the file hashes to `77A82AAC...53ED26` inside the container, identical to the host original and the backup. Writes to `/app` and `/app/outputs/experiments` fail. |
| Non-root user (`astramed`), read-only root filesystem, `cap_drop: ALL`, `no-new-privileges`, tmpfs `/tmp`, port bound to `127.0.0.1` only | **Verified** with `docker inspect` (both start methods). |
| Docker Compose (`docker compose config`, `up -d --build`, `ps`, `logs`, `stop`/`start`) | **Verified**: config valid, container `astramed_v10-astramed-api-1` healthy, `docker compose stop` took 3 s and exited 0. Restart policy `unless-stopped` is set; automatic restart after a host reboot was **not** tested. |
| Missing checkpoint with preload on | **Reported** by the project owner (container fails at startup); also verified earlier in a host-side simulation (exit code 3). |
| Dependency set | **Verified**: pinned set installed in fresh Python 3.14.6 virtual environments and in the image; Linux CPython 3.14 wheels (incl. `uvloop`) exist. |
| `tests/test_deployment.py` (12), API foundation (10), explanations, integration, reliability, runner guard | **Verified** in the project `.venv` on 2026-10-12 (see below). |
| Runner-guard test in the project owner's own environment | **Not verified.** The owner reported it failing because Windows Application Control blocked a DLL needed by scikit-learn; this was not reproduced in the project `.venv` (it passed, exit code 0). The test imports `src/experiments/runner.py`, which imports `sklearn.metrics`, so it fails wherever scikit-learn cannot load. |
| GPU inside Docker, throughput/latency, memory use, concurrent requests, long-running stability, automatic restart, other CPU architectures (arm64) and other operating systems | **Not verified / not supported.** The image is CPU-only; the host GPU is not used by the container. |

## What the image contains
Only `src/` and the pinned runtime packages (CPU-only PyTorch). **No checkpoint, data, outputs, notebooks, tests, `.env` or secrets** (see `.dockerignore`; `src/multimodal` and `src/analysis` are excluded because the API does not import them — `tests/test_deployment.py` checks this against the real import graph). The checkpoint is mounted read-only at run time at the path the code already uses:
`/app/outputs/experiments/resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42/checkpoints/best_model.pth`.

## Endpoints
| Route | Purpose |
|---|---|
| `GET /api/v1/health` | Liveness. Always 200 if the process is up (used by the Docker `HEALTHCHECK`). |
| `GET /api/v1/ready` | **New, additive.** 200 `{"status":"ready","model":"resnet50"}` if the checkpoint file exists, else 503 `{"status":"not_ready","model":...,"reason":"CHECKPOINT_NOT_FOUND"}`. No paths are returned. |
| `POST /api/v1/predict` | Unchanged. |

## Configuration (environment variables; see `.env.example`)
`ASTRAMED_API_ENV`, `ASTRAMED_HOST`, `ASTRAMED_PORT`, `ASTRAMED_LOG_LEVEL`, `ASTRAMED_MAX_UPLOAD_MB`, `ASTRAMED_MAX_IMAGE_PIXELS`, `ASTRAMED_DEFAULT_MODEL`, `ASTRAMED_PREDICTION_THRESHOLD` (keep 0.5), and new `ASTRAMED_PRELOAD_MODEL` (default `false`; `true` loads the model at startup and **exits with an error** if the checkpoint is missing or invalid, instead of failing later per request). Invalid values stop startup with a message naming the variable. **`ASTRAMED_HOST` and `ASTRAMED_PORT` are informational only:** `src/api/config.py` reads and validates them, but nothing in the code uses `settings.host` or `settings.port` to start the server. The listening address and port come from the hard-coded `--host 0.0.0.0 --port 8000` in the Dockerfile `CMD`; `EXPOSE`, the `HEALTHCHECK` URL and the Compose port mapping (`127.0.0.1:8000:8000`) are hard-coded to 8000 too. To use another host port, change only the left side of the port mapping (for example `-p 127.0.0.1:9000:8000`). No secrets are required. Never bake a `.env` into the image; pass variables with `-e` or `--env-file`.

## Windows PowerShell commands
Run from the repository root. Requires Docker Desktop (not available when this was written).

```powershell
# 1. Build (installs pinned CPU-only dependencies; first build downloads several hundred MB)
docker build -t astramed-api:local .

# 2. Run, mounting the checkpoint read-only (it is never copied into the image)
$exp  = "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
$ckpt = "$PWD\outputs\experiments\$exp\checkpoints"
docker run -d --name astramed-api -p 127.0.0.1:8000:8000 `
  -e ASTRAMED_PRELOAD_MODEL=true `
  -v "${ckpt}:/app/outputs/experiments/$exp/checkpoints:ro" `
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges `
  astramed-api:local

# 3. Check it
docker ps
docker logs astramed-api                       # expect: "Model resnet50 preloaded."
docker inspect --format "{{.State.Health.Status}}" astramed-api   # healthy after ~40 s
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/ready

# 4. End-to-end smoke test over HTTP. Use the project virtual environment (it already has httpx);
#    plain `python` on PATH is the base interpreter and lacks httpx.
.\.venv\Scripts\python.exe scripts\smoke_test_deployment.py --url http://127.0.0.1:8000 `
  --image "data\raw\rsna-pneumonia-detection-challenge\stage_2_train_images\0a0f91dc-6015-4342-b809-d19610854a21.dcm"

# 5. Stop and remove
docker stop astramed-api; docker rm astramed-api
```
Docker Compose equivalent (same options, defined in `docker-compose.yml`; project name `astramed_v10`, container `astramed_v10-astramed-api-1`). Stop any container started with `docker run` first (`docker stop astramed-api`), because both bind port 8000:
```powershell
docker compose config                  # validate and print the resolved configuration
docker compose up -d --build           # build (cached) and start
docker compose ps                      # expect: Up ... (healthy)
docker compose logs --no-color --tail 20 astramed-api   # expect: Model resnet50 preloaded.
.\.venv\Scripts\python.exe scripts\smoke_test_deployment.py --url http://127.0.0.1:8000 --image "data\raw\rsna-pneumonia-detection-challenge\stage_2_train_images\0a0f91dc-6015-4342-b809-d19610854a21.dcm"
docker compose stop; docker compose start   # restart without recreating
docker compose down                    # remove the container and network (the image and the checkpoint are not touched)
```
A container started earlier with `docker run --name astramed-api ...` can be brought back with `docker start astramed-api` (only while the Compose container is stopped).

## Failure behaviour
| Situation | Behaviour |
|---|---|
| Checkpoint not mounted, preload off | Container starts; `/health` 200; `/ready` 503; `/predict` returns a controlled 503 `CHECKPOINT_NOT_FOUND` |
| Checkpoint not mounted, preload on | Process exits with an error in the log (exit code 3 in the tested simulation); with `restart: unless-stopped` it restarts repeatedly — fix the mount |
| Corrupt/mismatched checkpoint | Preload on: startup error; preload off: controlled 503 `MODEL_LOAD_FAILED` on first request |
| Invalid configuration value | Process exits immediately with a message naming the variable |
| Invalid, empty or oversized upload | 400 / 413 / 415 with the standard error body and request ID; no paths in the response |

## Local tests (no Docker needed)
```powershell
.\.venv\Scripts\python.exe tests\test_deployment.py     # 12 checks: Dockerfile/compose/.dockerignore vs the code, requirements coverage, readiness, preload, config errors, no path leaks
.\.venv\Scripts\python.exe scripts\test_api_foundation.py
.\.venv\Scripts\python.exe scripts\test_api_integration.py --image <dicom>   # --image is required
.\.venv\Scripts\python.exe scripts\test_api_reliability.py --image <dicom>   # --image is required; briefly writes and removes two temp files under outputs\api_reliability_temp
```

## Security and reproducibility limitations
- **No authentication, no TLS, no rate limiting.** Anyone who can reach the port can use the API. The compose file binds to `127.0.0.1` only; do not expose it publicly without a reverse proxy providing authentication and TLS. `/docs` and `/openapi.json` are enabled. Uploaded images are processed in memory and not stored; there is no PHI handling or audit logging.
- Inference is synchronous inside an `async` endpoint, so one request blocks others (including `/health`). Fine for a single user; not for concurrent traffic. Run one worker only (the model is cached per process).
- CPU inference only in the documented image. GPU in Docker (NVIDIA toolkit, CUDA wheels) is not set up and was not tested. Throughput, latency, memory use, concurrency and long-running behaviour were not measured.
- The base image tag `python:3.14-slim` floats; after your first successful build, pin its digest. Transitive dependencies of the pinned packages are not locked (no hash-pinned lockfile). The CPU PyTorch index is an external service.
- The checkpoint is a local artifact with documented, partly unrecoverable provenance (see `docs/PROJECT_STATUS.md`); the deployed model is the same research baseline, sensitivity 0.56 at threshold 0.5. Container logs on a startup failure include a Python traceback (container paths such as `/app/...`, no secrets).
