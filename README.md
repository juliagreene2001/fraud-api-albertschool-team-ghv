# Team GHV fraud-scoring API

A supplied XGBoost model scores payment transactions through FastAPI. Uvicorn runs the HTTP server.
The companion [GHV infrastructure](https://github.com/juliagreene2001/ai-cloud-infra-terraform-team-ghv)
configures private Cloud Run deployment and shared Terraform state.

## Run locally

Use Python 3.13 and uv. Obtain the three course Parquet files separately; Git tracks only their
checksum manifest. On macOS, XGBoost may require `brew install libomp`.

```bash
uv sync --frozen
(cd data && shasum -a 256 -c CHECKSUMS.sha256)
uv run --frozen pytest -q -ra
uv run --frozen uvicorn app:app --host 127.0.0.1 --port 8000
```

Use `sha256sum` instead of `shasum -a 256` on Linux. Runtime needs `customer_daily.parquet` and
`terminal_daily.parquet`; reference tests additionally need `replay.parquet`. The course files are
available in `gs://advanced-cloud-ai-models/data/` to authorized course accounts.

## Endpoints

| Method / path | Behavior |
| --- | --- |
| `GET /health` | Report loaded state and worker information |
| `POST /predict` | Score a transaction batch without recording counters |
| `POST /events` | Score a batch, then update that worker's in-memory counters |
| `GET /docs` | Interactive request/response documentation |

From another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @examples/transaction.json
```

The example uses synthetic course identifiers. Identifiers are strings to preserve leading zeros.
The model uses 20 inputs, including daily customer/terminal windows and delayed fraud labels.
It returns risk scores; threshold-based approval/blocking is a separate decision.

## Container

```bash
docker build --platform linux/amd64 -t team-ghv-fraud-api:lab .
docker run --rm -p 127.0.0.1:18000:8000 -v "$PWD/data:/app/data:ro" team-ghv-fraud-api:lab
```

The model is included in the image; historical data is external. The standard Dockerfile honors
`PORT`, defaulting to 8000. The advanced distroless variant fixes port 8000.

## Verification and measurements

- `scripts/check_http.py`: check live health and 500 reference predictions; `--private` also checks
  anonymous rejection and uses an identity token held in memory.
- `scripts/measure_http.py`: save raw client latency samples, p50/p99 and failures.
- `scripts/collect_cloud_metrics.py`: fetch service/revision metrics and matching startup logs.

Use the development environment installed by `uv sync --frozen`. HTTPS tools verify certificates
using certifi. Pull-request CI runs synthetic feature tests and lint without course data or cloud
credentials. Run the full data-dependent tests separately; skipped integration tests are reported.

## State and deployment limits

`/events` updates are per-worker memory, lost on restart and not shared across instances. It does not
deduplicate IDs, add unknown entities, persist labels, or extend its allocated horizon indefinitely.
The lab uses one worker and one XGBoost thread. Scaling stateful event processing needs additional design.

The optional manual deployment workflow is gated off by default. Activating it requires explicit
WIF/deployment-account settings and coordinated image ownership with Terraform. Ordinary PR checks
and merges do not deploy to Google Cloud. Raw results, local configuration and personal notes stay ignored.
