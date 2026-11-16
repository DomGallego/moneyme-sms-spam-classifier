# Modelling and classification

Compare supervised ML, generative LLMs, and specialized decision models. Spam is the positive class. Run commands **from the repository root**.

## Install

Python 3.12 is the tested version. Local prediction uses the included artifact and needs no training data or GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps .
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps .
```

Use `.venv\Scripts\python.exe` instead of `python` below on Windows. For Docker without host Python, see [deployment](../task_2_deployment/README.md).

## Predict

| Selector | Serving configuration | Runtime |
|---|---|---|
| `ml` | Character TF-IDF + Logistic Regression and saved threshold | Local CPU |
| `llm` | `google/gemini-4-hydrogen`, four-example few-shot | OpenRouter |
| `decision` | `perplexity/pplx-decider-v1.3:free`, `choice` and saved threshold | OpenRouter |

```bash
python task_1_modelling/predict.py --model ml --text 'Are we meeting for lunch?'
```

For hosted prediction, copy [.env.example](../.env.example) to `.env` **only if it does not already exist**, then set `OPENROUTER_API_KEY` to your own key:

```bash
python task_1_modelling/predict.py --model llm --cap 0.05 --text 'Are we meeting for lunch?'
python task_1_modelling/predict.py --model decision --cap 0.05 --text 'Are we meeting for lunch?'
```

Successful output is `{"label": "ham"}` or `{"label": "spam"}`. Blank text and inputs over 5,000 characters are rejected. Provider failures fail the command. Options: `python task_1_modelling/predict.py --help`.

Hosted calls send the SMS to OpenRouter. `--cap 0.05` sets a cumulative US$0.05 limit shared by both hosted CLI models in `reports/service-budget.sqlite`. Preserve that file; conservative reservations can stop calls before billed spend reaches the cap. This ledger is separate from API/experiment accounting. `ENABLE_HOSTED` and `SERVICE_BUDGET_USD` control the API, not the CLI. No HTTP server is needed.

## Notebook and experiments

Open [analysis.ipynb](analysis.ipynb) for dataset statistics, class/length plots, training embedding projections, model comparisons, and errors. It includes precision, recall, accuracy, F1, coverage, latency, and cost charts. Saved outputs require no key or computation.

To execute the notebook against saved reports, without training or paid requests:

```bash
uv sync --frozen --extra analysis --extra dev
uv run --no-sync python scripts/execute_notebook.py
```

| Entry point | Purpose |
|---|---|
| [prepare_data.py](prepare_data.py) | Parse, validate, deduplicate, group families, and create partitions |
| [train.py](train.py) | Classical ML search; full treatments/encoders use the protocol below |
| [evaluate.py](evaluate.py) | Final local evaluation; refuses to overwrite an existing final run |
| [Shared package](../src/moneyme_sms_spam_classifier/) | Implementations shared by CLI and API |

The [experiment protocol](../docs/ml_experiments.md) contains full reproduction steps, search budgets, and fitting boundaries. It covers 21 original-data, 21 text-augmentation, and 18 SMOTE configurations, plus seven LLMs with zero-/few-shot variants and seven decision models with `choice`/`noul` variants. One variant per hosted model proceeds to final evaluation; the always-ham baseline is separate.

Read [results](../reports/RESULTS.md), [treatment effects](../reports/ML_TREATMENTS.md), and [tracking/architecture](../docs/architecture.md). The treatment comparison uses an already-inspected test set and is exploratory. Serving selections remain separately frozen.
