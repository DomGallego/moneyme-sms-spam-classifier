# SMS spam classifier

A reproducible SMS spam benchmark with a shared Python CLI and Dockerized FastAPI service. It compares **60 local ML configurations, seven generative LLMs, and seven specialized decision models**, plus an always-ham baseline.

Decision models are included to explore their quality, latency, and cost alongside conventional classifiers. Imbalance experiments cover class weights, validation-selected thresholds, LLM text augmentation, and SMOTE. Validation and test messages are never resampled.

## Results

Serving configurations were selected on validation data before their original test evaluation:

| Group | Selected model | Test spam F1 | Precision | Recall |
|---|---|---:|---:|---:|
| Local ML | Character TF-IDF + Logistic Regression | 0.9644 | 0.9919 | 0.9385 |
| LLM | `google/gemini-4-hydrogen`, few-shot | 0.9565 | 0.9837 | 0.9308 |
| Decision | `perplexity/pplx-decider-v1.3:free`, `choice` | 0.9883 | 1.0000 | 0.9769 |

All three cover all 1,034 test messages. See [results](reports/RESULTS.md) for costs, timing scopes, and limitations, or the [full comparison](reports/final_comparison.csv) for all 75 entries. The [training-treatment comparison](reports/ML_TREATMENTS.md) reuses an already-inspected test set and is exploratory. This historical public dataset does not establish current production performance.

## Quick start

Run from the repository root using Python 3.12. The selected local artifact is included; inference needs no training, dataset, or GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps .
python task_1_modelling/predict.py --model ml --text 'Are we meeting for lunch?'
```

Output: `{"label": "ham"}`. For hosted CLI calls, put your own `OPENROUTER_API_KEY` in `.env` and set a cumulative USD cap:

```bash
python task_1_modelling/predict.py --model llm --cap 0.05 --text 'Are we meeting for lunch?'
python task_1_modelling/predict.py --model decision --cap 0.05 --text 'Are we meeting for lunch?'
```

Hosted calls send SMS to OpenRouter. CLI accounting persists in `reports/service-budget.sqlite`; API and experiment budgets use separate ledgers. Keep `.env` private.

For Docker, follow the [deployment guide](task_2_deployment/README.md). The API accepts `{"text":"Your SMS"}` at `POST /predict/ml`, `/predict/llm`, and `/predict/decision`; successful responses contain a `spam` or `ham` label. Interactive documentation is at `/docs`.

## Documentation

| Document | Contents |
|---|---|
| [Modelling guide](task_1_modelling/README.md) | CLI usage, notebook, and experiment entry points |
| [Executed notebook](task_1_modelling/analysis.ipynb) | EDA, embedding projections, model charts, and error analysis |
| [Experiment protocol](docs/ml_experiments.md) | Splits, fitting boundaries, tuning, reproduction, and hardware |
| [Architecture](docs/architecture.md) | Shared inference, tracking, recovery, budgets, and deployment |
| [API reference](task_2_deployment/API.md) | Schemas, endpoints, errors, and configuration |

## Development

```bash
uv sync --frozen --extra dev
uv run --no-sync ruff check src tests scripts task_1_modelling task_2_deployment
uv run --no-sync ruff format --check src tests scripts task_1_modelling task_2_deployment
uv run --no-sync pytest -q
uv run --no-sync python scripts/docker_smoke.py
```

[GitHub Actions](.github/workflows/ci.yml) defines the same checks on pushes and pull requests, with no paid provider calls. Saved [Docker](reports/docker_smoke.json) and [committed-checkout](reports/committed_checkout_smoke.json) checks record local verification; they are separate from GitHub Actions results. Deployment is manual through Docker Compose.
