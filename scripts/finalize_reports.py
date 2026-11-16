"""Finalize artifacts after all evaluation and live verification; makes no API calls."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from moneyme_sms_spam_classifier.common import read_json
from moneyme_sms_spam_classifier.reporting import (
    comparison,
    enrich_hosted_test,
    error_slices,
    generate,
    hosted_figures,
    paired_selected_comparisons,
)

hosted = read_json("reports/runs/test/results.json")
local = read_json("reports/runs/local_test/results.json")
if len(hosted) != 14 or len(local) != 22:
    raise ValueError("Final evaluation is incomplete")
if Path("configs/imbalance_experiment.json").exists():
    from moneyme_sms_spam_classifier.treatment_reporting import generate_treatment_report

    treatment_status = generate_treatment_report()
    if not treatment_status["complete"]:
        raise ValueError("All 60 ML treatment configurations must finish before final reporting")
for required in [
    "reports/live_container_smoke.json",
    "reports/clean_checkout_smoke.json",
    "reports/robustness_summary.json",
    "reports/tests.xml",
]:
    if not Path(required).exists():
        raise ValueError(f"Verification evidence missing: {required}")
enrich_hosted_test()
paired_selected_comparisons()
error_slices()
generate()
hosted_figures()
table = comparison("test", include_treatments=False)
selection = read_json("configs/local_selection.json")
frozen = read_json("configs/hosted_frozen.json")
cost = read_json("reports/cost_summary.json")
if cost["accounted_usd"] > cost["authorized_usd"] + 1e-8:
    raise ValueError("Accounted spend exceeds authorized cap")
roles = {
    "Local ML": selection["candidate"],
    **{group: cfg["model"] for group, cfg in frozen["champions"].items()},
}
lines = [
    "# Results",
    "",
    "Serving selections were frozen on development/validation before their original test. The 60-configuration [treatment comparison](ML_TREATMENTS.md) reuses that already-inspected test and is exploratory. These historical public-dataset scores do not establish production performance.",
    "",
    "## Selected systems",
    "",
    "| Role | Model | Test spam F1 | Precision | Recall | Coverage | p95 ms | Reported USD / 1,000 SMS |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for role, model in roles.items():
    row = table[table.candidate == model].iloc[0]
    price = (
        f"{row.usd_per_1000_attempted:.4f}" if role != "Local ML" else "infrastructure unmeasured"
    )
    lines.append(
        f"| {role} | {model} | {row.spam_f1:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.coverage:.2%} | {row.p95_ms:.2f} | {price} |"
    )
lines += [
    "",
    "Test: 1,034 SMS (904 ham, 130 spam); spam is positive. Local p95 is warm sequential full-pipeline inference over 100 messages. Hosted p95 includes HTTP/provider time at concurrency 8. These scopes differ; container timings are separate.",
    "",
    "The local artifact is character TF-IDF + Logistic Regression, about 252 KiB. Selections use validation F1, with coverage eligibility for decision models. Reported free-tier pricing may change; local infrastructure costs are unmeasured.",
    "",
    "## Detailed results",
    "",
    "| Files | Contents |",
    "|---|---|",
    "| [final_comparison.csv](final_comparison.csv) | 75 entries: 60 local configurations, baseline, seven LLMs, seven decision models; evaluation-scope fields distinguish original and exploratory results |",
    "| [ML_TREATMENTS.md](ML_TREATMENTS.md), `ml_treatment_comparison.csv` | Treatment results and validation leaders |",
    "| `ml_treatment_paired_differences.csv`, `paired_selected_comparisons.csv` | Paired family-bootstrap F1 differences; exploratory intervals without multiple-comparison correction |",
    "| `hosted_test_uncertainty.csv`, `hosted_operating_points.csv`, `hosted_calibration_bins.csv` | Successful-response F1 intervals, selected thresholds, calibration bins/counts |",
    "| `error_slices.csv`, `local_test_errors.csv`, `hosted_test_errors.csv` | Overlapping descriptive slices and errors; no retuning on these examples |",
    "| [runs/](runs/) | Configurations, manifests, predictions, search records, and provider usage |",
    "",
    "## Cost",
    "",
    f"Experiment cap: **US${cost['authorized_usd']:.2f}**, including a US$1 pilot. Provider-reported: **US${cost['reported_usd']:.6f}**. Conservative accounted total: **US${cost['accounted_usd']:.6f}**, including reservations for **{cost['unknown_attempts']}** attempts with unknown charges.",
    "",
    "`cost_attempts.csv` and `cost_ledger_summary.csv` separate attempt/stage accounting. Unknown charges are not zero. Augmentation is a one-time training-data expense. Cost projections assume recorded prices, message/prompt lengths, caches, and retry policy. Local electricity/infrastructure and pretrained-model training costs are unmeasured.",
    "",
    "## Reliability",
    "",
    "Hosted classification metrics are conditional on successful responses; read them with coverage and correct/attempted accuracy. Models with test failures:",
    "",
]
for result in hosted:
    if result["failures"]:
        lines.append(
            f"- `{result['candidate']}`: coverage {result['metrics']['coverage']:.2%}; failures {json.dumps(result['failures'])}."
        )
tests = ET.parse("reports/tests.xml").getroot()
suites = list(tests.iter("testsuite"))
total = sum(int(s.attrib.get("tests", 0)) for s in suites)
failures = sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites)
if failures:
    raise ValueError("Test evidence includes failures")
lines += [
    "",
    f"Saved verification: {total} automated tests with zero failures/errors ([tests.xml](tests.xml)); [local Docker](docker_smoke.json), [committed-checkout](committed_checkout_smoke.json), and [live API](live_container_smoke.json) checks. These are local records, not GitHub Actions results.",
    "",
    "`robustness_summary.json` records three repeats of 16 validation messages per hosted champion and handwritten instruction-override examples. These are diagnostics, not a production benchmark.",
    "",
    "## Limitations and next steps",
    "",
    "The historical English corpus has ambiguous labels and may overlap pretrained-model data. Family grouping can miss paraphrases; high-precision/calibration estimates have limited support. Hosted weights/routing can change. Generated training labels are inherited without independent human verification; treatment results reuse an inspected test set.",
    "",
    "Next: representative reviewed labels and a fresh temporal holdout, explicit error costs/operating policy, API load tests, quality/latency/cost monitoring, challenger evaluation, and rollback artifacts.",
    "",
]
Path("reports/RESULTS.md").write_text("\n".join(lines))
subprocess.run([sys.executable, "scripts/build_notebook.py"], check=True)
subprocess.run([sys.executable, "scripts/execute_notebook.py"], check=True)
print("Final reports and executed notebook written.")
