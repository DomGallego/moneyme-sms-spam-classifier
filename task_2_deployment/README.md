# Classification API

Run the selected ML, LLM, and decision classifiers in Docker. No host Python, GPU, dataset, or training is needed. Run commands **from the repository root**.

## Setup

Install Docker Engine with Compose, or Docker Desktop. `docker version` must show Client and Server. For WSL, enable Docker Desktop → Settings → Resources → WSL Integration for your distribution. The image targets Linux amd64; ARM requires emulation.

Copy [.env.example](../.env.example) to `.env` if it does not exist; otherwise edit the existing file:

```bash
if [ ! -f .env ]; then cp .env.example .env; fi
```

PowerShell: `if (-not (Test-Path .env)) { Copy-Item .env.example .env }`.

| Setting | Local ML only (default) | All three models |
|---|---|---|
| `ENABLE_HOSTED` | `false` | `true` |
| `OPENROUTER_API_KEY` | Blank | Your own key |
| `SERVICE_BUDGET_USD` | `0` | A positive cumulative cap, e.g. `0.10` |
| `MAX_INFLIGHT` | `4` | `4` |

Hosted mode sends SMS to OpenRouter and requires internet, model access, and credits. It enables remote inference; it does not publish the API or download models. Keep `.env` private. The initial Docker build also needs internet.

## Start

```bash
docker compose --env-file .env -f task_2_deployment/compose.yaml up --build -d
docker compose --env-file .env -f task_2_deployment/compose.yaml ps
```

Wait for healthy status, then open [interactive docs](http://localhost:8000/docs), [health](http://localhost:8000/health), or [OpenAPI](http://localhost:8000/openapi.json). Health reports the local model and enabled hosted groups without calling a provider. `/` has no route.

After changing `.env`, recreate the service:

```bash
docker compose --env-file .env -f task_2_deployment/compose.yaml up -d --force-recreate
```

Exported shell variables override Compose's `.env` values.

## Predict

Use `/docs`, curl, or the supplied client. All endpoints accept `{"text":"Your SMS"}` and return `{"label":"ham"}` or `{"label":"spam"}`.

```bash
curl -X POST http://localhost:8000/predict/ml \
  -H 'Content-Type: application/json' \
  -d '{"text":"Are we meeting for lunch?"}'
```

Change `/ml` to `/llm` or `/decision` for hosted inference. Run the client inside Docker without installing host Python:

```bash
# Local-only mode
docker compose --env-file .env -f task_2_deployment/compose.yaml exec classifier python task_2_deployment/sample_client.py --base-url http://127.0.0.1:8000 --models ml
# All three, with hosted mode enabled
docker compose --env-file .env -f task_2_deployment/compose.yaml exec classifier python task_2_deployment/sample_client.py --base-url http://127.0.0.1:8000
```

Add `--text "Your SMS"` or choose `--models decision`. The client prints model, HTTP status, and response; it exits nonzero on an HTTP error. With host dependencies installed, use `python task_2_deployment/sample_client.py --models ml`.

## Runtime controls

The local artifact checksum is checked at startup. Both hosted routes share a cumulative budget in the persistent `service-runtime` volume, separate from CLI/experiment ledgers. Reservations include uncertain costs and can stop calls before billed spend reaches the cap; this is not an account-wide provider limit.

`MAX_INFLIGHT` limits all prediction routes per API process. Provider timeouts and explicit errors apply; there is no automatic retry or fallback. Logs include group, status, and elapsed time without raw SMS or keys. See the [API contract](API.md) and [architecture](../docs/architecture.md) for details and selected model IDs.

## Troubleshooting and shutdown

| Symptom | Action |
|---|---|
| Docker unavailable or missing Server section | Start Docker Desktop/Engine; check WSL integration |
| Startup failure | Check logs, artifact/configuration, and hosted key/budget settings |
| Port 8000 occupied | Stop the other service or change the host port in `compose.yaml` |
| `422` | Send only a nonblank `text` string, at most 5,000 characters |
| `503` | Check hosted enablement, remaining budget, and concurrency |
| `502` / `504` | Check provider access, network, or timeout |

```bash
docker compose --env-file .env -f task_2_deployment/compose.yaml logs --tail=50 classifier
docker compose --env-file .env -f task_2_deployment/compose.yaml down
```

`down` preserves the runtime volume. Deleting it resets local accounting. Compose binds to localhost and the container runs as a non-root user.

## CI and deployment

[CI](../.github/workflows/ci.yml) defines locked installation, Ruff, tests, and a local-only Docker smoke check on pushes/PRs. It makes no paid calls. Run `python scripts/docker_smoke.py` for the local container check; [saved verification](../reports/committed_checkout_smoke.json) is separate from GitHub Actions results. Deployment uses Compose manually; there is no automated image publishing or hosted deployment.
