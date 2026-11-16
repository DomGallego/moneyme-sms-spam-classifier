# Assessment deliverable checklist

Checked against the supplied MoneyMe Technical Challenge guide on 16 November 2026, with committed-checkout verification on 17 November 2026. The technical implementation is committed; repository publication and email submission remain separate steps.

## Task 1: modelling

| Requirement | Implementation | Evidence |
|---|---|---|
| Explore the supplied dataset and show basic statistics | [Executed notebook](../task_1_modelling/analysis.ipynb): provenance, counts, imbalance, splits, message lengths and exploratory figures | Ten executed code cells, zero errors; [final audit](../reports/runs/imbalance/final_audit.json) |
| Build an ML spam/ham classifier | Shared inference in [inference.py](../src/moneyme_sms_spam_classifier/inference.py); selected character TF-IDF + Logistic Regression artifact in `artifacts/release/model.joblib` | Checksummed release; local CLI invocation and Docker inference verified |
| Build an LLM spam/ham classifier | Hosted generative adapter in [hosted.py](../src/moneyme_sms_spam_classifier/hosted.py), with selection in `configs/hosted_frozen.json` | Full saved validation/test records; [live API evidence](../reports/live_container_smoke.json) |
| Provide a standalone command-line Python program | [predict.py](../task_1_modelling/predict.py) supports `--model ml`, `llm`, and `decision`; shared CLI also installed as `sms-spam` | Installation, commands and hosted spending controls in the [Task 1 README](../task_1_modelling/README.md) |
| Present classifier performance | Notebook, [results](../reports/RESULTS.md), [full comparison](../reports/final_comparison.csv), and [ML treatment report](../reports/ML_TREATMENTS.md) | Quality, confusion counts, coverage, latency, costs and uncertainty; evaluation provenance is disclosed |

The guide does not require every model to occupy its own Python file. The standalone program delegates to reusable modules shared with the API. The decision-model group, 60-configuration local matrix, and broader diagnostics exceed the minimum two-classifier requirement.

## Task 2: deployment

| Requirement | Implementation | Evidence |
|---|---|---|
| Inference application serving ML and LLM APIs | [FastAPI application](../src/moneyme_sms_spam_classifier/api.py), entry point [app.py](../task_2_deployment/app.py) | `/predict/ml`, `/predict/llm`, and the decision route accept SMS text and return a spam/ham label; no training runs in the API |
| Document API calls | [API documentation](../task_2_deployment/API.md), plus generated `/docs` and `/openapi.json` | Input/output examples, validation, errors and configuration documented |
| Sample Python API client that prints responses | [sample_client.py](../task_2_deployment/sample_client.py) | Used in the budgeted live container check; exits nonzero on requested endpoint failures |
| Modular, readable implementation | Package modules separate data, features, training treatments, evaluation, hosted adapters and inference | Shared CLI/API inference, tests and [architecture diagrams](architecture.md) |
| Controlled deployable environment | [Dockerfile](../task_2_deployment/Dockerfile), [Compose configuration](../task_2_deployment/compose.yaml), pinned requirements and `uv.lock` | [Docker smoke test](../reports/docker_smoke.json) and [Git-visible-files rehearsal](../reports/clean_checkout_smoke.json) passed; non-root execution |
| README explaining startup | Root [README](../README.md) and [deployment README](../task_2_deployment/README.md) | Install/build/start, API calls, hosted activation, logs and shutdown documented |

## General requirements and operational conditions

- Python 3.12 is the tested target and satisfies the guide's Python 3.10-or-later requirement.
- Runtime libraries are pinned in `requirements.txt`; modelling/notebook dependencies are in `requirements-analysis.txt`, with the optional isolated CPU PyTorch runtime in `requirements-encoder-cpu.txt`. The resolved primary environment is in `uv.lock`.
- The selected local artifact is included and serves without raw training data, model downloads, or GPU access.
- Hosted models are explicitly allowed by the guide. The [all-three-classifier walkthrough](../task_2_deployment/README.md) configures the reviewer's own OpenRouter key, internet access, `ENABLE_HOSTED=true` and a positive `SERVICE_BUDGET_USD`. Hosted routes return 503 when disabled or when a request exceeds the available budget; enabling hosted mode without a key prevents startup. Credentials are not supplied in the repository.
- The container target is Linux amd64; ARM hosts require Docker's amd64 emulation. Internet is needed for the initial image build and hosted inference.
- The guide asks for APIs reviewers can run in their environment. It does not require a publicly hosted service or a cloud deployment.
- Verification includes 27 passing automated tests, lint/format checks, ten executed notebook cells, container checks, preserved artifact hashes, fold-isolation checks and a Git-visible secret scan. See [final audit](../reports/runs/imbalance/final_audit.json).
- The [repository audit](../reports/repository_audit.json) records the subsequent review of all Git-eligible files, documentation links and diagrams, notebook execution, saved prediction metrics, lockfile consistency, cost accounting, and local/clean-copy deployment checks. It made no new paid calls.
- The [committed-checkout check](../reports/committed_checkout_smoke.json) verified an archive of revision `348ed4b` containing only tracked files. Its tests passed (26 passed, one expected raw-dataset check skipped), and its Docker smoke test exercised local inference, both disabled hosted routes, the Python client and non-root execution. The subsequent verification commit changes evidence and submission-status documentation only.
- Live hosted-route evidence and local-only container evidence identify their respective tested image IDs. The serving selections are preserved separately from the exploratory ML treatment comparison.
- Treatment results use an already-inspected historical test set and are explicitly exploratory; they are not a fresh independent production-performance claim.

## Submission steps still pending

The implementation, reports, notebook, requirements and release artifact are committed in focused groups. The committed-source deployment rehearsal has passed.

1. Push the commits to the intended GitHub repository, confirm reviewers can access the submitted revision, and inspect the GitHub Actions result after publication.
2. Email the repository link to all three addresses specified in the guide: `xiaofeng@moneyme.com.au`, `carlo.ngo@moneyme.com.au`, and `jenvia.domingo@moneyme.com.au`.

No repository push, reviewer invitation, or email submission is performed by this checklist review.
