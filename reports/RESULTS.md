# Results

Serving selections were frozen on development/validation before their original test. The 60-configuration [treatment comparison](ML_TREATMENTS.md) reuses that already-inspected test and is exploratory. These historical public-dataset scores do not establish production performance.

## Selected systems

| Role | Model | Test spam F1 | Precision | Recall | Coverage | p95 ms | Reported USD / 1,000 SMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Local ML | logistic_char | 0.9644 | 0.9919 | 0.9385 | 100.00% | 5.86 | infrastructure unmeasured |
| llm | google/gemini-4-hydrogen | 0.9565 | 0.9837 | 0.9308 | 100.00% | 1684.12 | 1.3806 |
| decision | perplexity/pplx-decider-v1.3:free | 0.9883 | 1.0000 | 0.9769 | 100.00% | 960.13 | 0.0000 |

Test: 1,034 SMS (904 ham, 130 spam); spam is positive. Local p95 is warm sequential full-pipeline inference over 100 messages. Hosted p95 includes HTTP/provider time at concurrency 8. These scopes differ; container timings are separate.

The local artifact is character TF-IDF + Logistic Regression, about 252 KiB. Selections use validation F1, with coverage eligibility for decision models. Reported free-tier pricing may change; local infrastructure costs are unmeasured.

## Detailed results

| Files | Contents |
|---|---|
| [final_comparison.csv](final_comparison.csv) | 75 entries: 60 local configurations, baseline, seven LLMs, seven decision models; evaluation-scope fields distinguish original and exploratory results |
| [ML_TREATMENTS.md](ML_TREATMENTS.md), `ml_treatment_comparison.csv` | Treatment results and validation leaders |
| `ml_treatment_paired_differences.csv`, `paired_selected_comparisons.csv` | Paired family-bootstrap F1 differences; exploratory intervals without multiple-comparison correction |
| `hosted_test_uncertainty.csv`, `hosted_operating_points.csv`, `hosted_calibration_bins.csv` | Successful-response F1 intervals, selected thresholds, calibration bins/counts |
| `error_slices.csv`, `local_test_errors.csv`, `hosted_test_errors.csv` | Overlapping descriptive slices and errors; no retuning on these examples |
| [runs/](runs/) | Configurations, manifests, predictions, search records, and provider usage |

## Cost

Experiment cap: **US$10.00**, including a US$1 pilot. Provider-reported: **US$5.162992**. Conservative accounted total: **US$5.274930**, including reservations for **167** attempts with unknown charges.

`cost_attempts.csv` and `cost_ledger_summary.csv` separate attempt/stage accounting. Unknown charges are not zero. Augmentation is a one-time training-data expense. Cost projections assume recorded prices, message/prompt lengths, caches, and retry policy. Local electricity/infrastructure and pretrained-model training costs are unmeasured.

## Reliability

Hosted classification metrics are conditional on successful responses; read them with coverage and correct/attempted accuracy. Models with test failures:

- `nace-ai/drex-v2.0`: coverage 95.65%; failures {"http_429": 45}.
- `liquid/d1-ultra`: coverage 99.03%; failures {"http_504": 10}.
- `google/gemini-decision-1`: coverage 99.81%; failures {"http_400": 2}.
- `openai/gpt-6-luna-decisions`: coverage 99.81%; failures {"http_502": 2}.

Saved verification: 27 automated tests with zero failures/errors ([tests.xml](tests.xml)); [local Docker](docker_smoke.json), [committed-checkout](committed_checkout_smoke.json), and [live API](live_container_smoke.json) checks. These are local records, not GitHub Actions results.

`robustness_summary.json` records three repeats of 16 validation messages per hosted champion and handwritten instruction-override examples. These are diagnostics, not a production benchmark.

## Limitations and next steps

The historical English corpus has ambiguous labels and may overlap pretrained-model data. Family grouping can miss paraphrases; high-precision/calibration estimates have limited support. Hosted weights/routing can change. Generated training labels are inherited without independent human verification; treatment results reuse an inspected test set.

Next: representative reviewed labels and a fresh temporal holdout, explicit error costs/operating policy, API load tests, quality/latency/cost monitoring, challenger evaluation, and rollback artifacts.
