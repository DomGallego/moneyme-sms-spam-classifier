# Implementation status

Implementation and the agreed end-to-end verification are complete.

The serving selections were frozen using development/validation evidence before their original held-out test. The 60-configuration ML treatment benchmark uses the same already-inspected test set and is explicitly exploratory. See ML_TREATMENTS.md for treatment results and evaluation provenance. These are historical public-dataset results, not production claims.

## Selected systems

| Role | Model | Test spam F1 | Precision | Recall | Coverage | p95 ms | Reported USD / 1,000 SMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Local ML | logistic_char | 0.9644 | 0.9919 | 0.9385 | 100.00% | 5.86 | infrastructure unmeasured |
| llm | google/gemini-4-hydrogen | 0.9565 | 0.9837 | 0.9308 | 100.00% | 1684.12 | 1.3806 |
| decision | perplexity/pplx-decider-v1.3:free | 0.9883 | 1.0000 | 0.9769 | 100.00% | 960.13 | 0.0000 |

All systems use the same 1,034 test SMS (904 ham, 130 spam). Spam is positive. Local p95 measures warm sequential full-pipeline inference over 100 test messages; hosted p95 measures complete HTTP requests at concurrency 8. These are different runtime scopes, not isolated hardware benchmarks. API/container measurements are separate.

The local artifact is character TF-IDF + Logistic Regression, about 252 KiB. Frozen embeddings and full encoder fine-tuning were included in the experiment; greater model complexity did not automatically improve held-out performance in the bounded searches. The LLM selection follows validation F1 rather than minimum price. The decision selection follows validation F1 subject to coverage eligibility. Free-tier pricing is temporary, not an established production price.

## Full comparison and diagnostics

- `final_comparison.csv`: 60 local treatment configurations, dummy baseline, seven LLMs, seven decision models; 75 rows. Evaluation-scope fields distinguish exploratory treatment results from original preselected-system evidence.
- `ML_TREATMENTS.md` and `ml_treatment_comparison.csv`: fitting treatments, per-treatment leaders and full development/test comparison. No automatic serving-release replacement.
- `ml_treatment_paired_differences.csv`: exploratory paired family-bootstrap changes relative to each pipeline's original-data treatment.
- `paired_selected_comparisons.csv`: paired family-bootstrap F1 differences for the preselected systems; intervals can overlap zero. No multiple-comparison correction.
- `hosted_test_uncertainty.csv`: family-bootstrap F1 intervals conditional on successful responses.
- `hosted_operating_points.csv`: validation-selected operating points evaluated on test.
- `hosted_calibration_bins.csv`: probability bins with counts; never assume vendor calibration transfers.
- `error_slices.csv`: overlapping descriptive slices with sample/class support and failed spam/ham counts.
- `local_test_errors.csv` and `hosted_test_errors.csv`: inspection examples; no retuning on these errors.
- `runs/`: effective configurations, manifests, search histories, per-message predictions and provider usage.

## Cost accounting

User-authorized cap: **US$10.00** total, including a US$1 pilot. Provider-reported charges: **US$5.162992**. Conservative accounted total: **US$5.274930**, including retained reservations for **167** attempts with unknown charges.

`cost_attempts.csv` exports sanitized attempt-level accounting; `cost_ledger_summary.csv` separates pilot, development, validation, test, augmentation generation, robustness and service checks. Unknown charges are not zero. Augmentation cost is a one-time training-data cost, not per-prediction classifier cost. Projections per million messages assume the same message/prompt lengths, cache behavior, retry policy and prices. Local API charges are zero but infrastructure/electricity costs are unmeasured. Pretraining costs are outside this assessment for pretrained systems.

## Reliability and verification

- nace-ai/drex-v2.0: test coverage 95.65%; failures {"http_429": 45}. Conditional classification metrics must be read with coverage and correct/attempted accuracy.
- liquid/d1-ultra: test coverage 99.03%; failures {"http_504": 10}. Conditional classification metrics must be read with coverage and correct/attempted accuracy.
- google/gemini-decision-1: test coverage 99.81%; failures {"http_400": 2}. Conditional classification metrics must be read with coverage and correct/attempted accuracy.
- openai/gpt-6-luna-decisions: test coverage 99.81%; failures {"http_502": 2}. Conditional classification metrics must be read with coverage and correct/attempted accuracy.
- Automated suite: 27 tests, zero failures/errors; see `tests.xml`. A pinned Starlette TestClient deprecation warning is non-blocking.
- Real Docker local-only smoke test and a Git-visible-files rehearsal passed without secrets, raw data, or local caches.
- Live container checks exercised all three prediction routes, the sample Python client, and OpenAPI. Live calls used the same authorized budget ledger.
- Stability checks repeated 16 validation messages three times per hosted champion; separate handwritten instruction-override examples are diagnostics, not a production benchmark. See `robustness_summary.json`.

## Limits and next steps

The small historical English corpus cannot establish performance on current MoneyMe traffic. Some conversational-looking messages and jokes carry spam labels, so label/task ambiguity matters. Public pretrained contamination cannot be ruled out, especially when benchmark scores approach perfection. Conservative family grouping misses some paraphrases. High-precision and calibration estimates have limited support. Hosted models and provider routing can change behind stable names.

Next production step: collect representative reviewed labels, establish the actual blocking/review policy and error costs, test temporal generalization, load-test the API, monitor quality/latency/cost/version changes, and retain rollback artifacts. No repository publication, email submission, or public hosting was performed by this implementation.
