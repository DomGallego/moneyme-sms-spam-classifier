# SMS classification API

Base URL: `http://localhost:8000`. Interactive documentation: `/docs`. OpenAPI schema: `/openapi.json`.

For installation, environment settings and commands to test all three classifiers on your own machine, follow the [Docker walkthrough](README.md).

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

`text` must be a string containing non-whitespace content, at most 5,000 characters. Extra fields are rejected. Text is not silently shortened by the API. Encoder pipelines may apply their documented token limit.

Success (`200`):

```json
{"label":"ham"}
```

`label` is always `spam` or `ham`. Spam is the positive class. The API returns a classification; it does not block or forward SMS. Exact examples are illustrative, not promised predictions for every model version.

```bash
curl -X POST http://localhost:8000/predict/ml \
  -H 'Content-Type: application/json' \
  -d '{"text":"Are we still meeting for lunch?"}'
```

Change `/ml` to `/llm` or `/decision` for the other classifiers. Or run:

```bash
python task_2_deployment/sample_client.py --text 'Your SMS here'
```

## Errors

Errors use FastAPI's `detail` field. No provider secret or raw upstream response is exposed.

| Status | Meaning |
|---|---|
| `422` | Invalid request, blank text, wrong type, oversized text, extra field |
| `502` | Upstream failure, malformed/invalid model response, or connection failure |
| `503` | Hosted route disabled, budget exhausted, or concurrency limit reached |
| `504` | Hosted inference timed out |

A failed model call never becomes a successful `ham` response. There is no automatic cross-model fallback. The initial implementation makes one provider attempt per inference; retrying is an explicit client decision and can incur another charge. The ledger reserves uncertain costs after timeouts.

## Health and configuration

`GET /health` returns application liveness, selected local model name, and enabled hosted groups. It makes no paid call and does not establish provider availability.

Local-only mode is the default. Enable hosted routes with `ENABLE_HOSTED=true`, an `OPENROUTER_API_KEY`, and a positive `SERVICE_BUDGET_USD`. The persistent runtime ledger enforces the cumulative budget across requests/restarts. Use the same volume/ledger to retain accounting; deleting it resets local accounting and is not a provider-side account cap.

The service binds to localhost through Compose. Authentication, TLS, a gateway, provider-side account limits, and representative workload testing would be needed before exposing it publicly.
