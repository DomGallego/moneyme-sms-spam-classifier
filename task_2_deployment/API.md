# SMS classification API

Base URL: `http://localhost:8000`. Interactive documentation: `/docs`. OpenAPI schema: `/openapi.json`.

Setup: [deployment guide](README.md).

## Prediction endpoints

| Method and path | Classifier |
|---|---|
| `POST /predict/ml` | Selected local supervised pipeline |
| `POST /predict/llm` | Selected hosted generative LLM |
| `POST /predict/decision` | Selected hosted decision model |

All endpoints accept the same JSON object:

```json
{"text":"Are we still meeting for lunch?"}
```

`text` must be nonblank, at most 5,000 characters. Extra fields are rejected; the API does not silently shorten text.

Success (`200`):

```json
{"label":"ham"}
```

`label` is `spam` or `ham`; spam is positive. The API classifies messages without blocking or forwarding them. Labels depend on the text and model version.

```bash
curl -X POST http://localhost:8000/predict/ml \
  -H 'Content-Type: application/json' \
  -d '{"text":"Are we still meeting for lunch?"}'
```

Change `/ml` to `/llm` or `/decision` for the other classifiers.

## Errors

Errors use FastAPI's `detail` field. No provider secret or raw upstream response is exposed.

| Status | Meaning |
|---|---|
| `422` | Invalid request, blank text, wrong type, oversized text, extra field |
| `502` | Upstream failure, malformed/invalid model response, or connection failure |
| `503` | Hosted route disabled, budget exhausted, or concurrency limit reached |
| `504` | Hosted inference timed out |

A failed call never becomes `ham`. Each inference makes one provider attempt, without automatic fallback. Client retries can incur another charge; unknown costs remain reserved after timeouts.

## Health and configuration

`GET /health` returns application liveness, selected local model name, and enabled hosted groups. It makes no paid call and does not establish provider availability.

Local-only mode is the default. Enable hosted routes with `ENABLE_HOSTED=true`, an `OPENROUTER_API_KEY`, and a positive `SERVICE_BUDGET_USD`. The persistent runtime ledger enforces the cumulative budget across requests/restarts. Use the same volume/ledger to retain accounting; deleting it resets local accounting and is not a provider-side account cap.

The service binds to localhost through Compose. Authentication, TLS, a gateway, provider-side account limits, and representative workload testing would be needed before exposing it publicly.
