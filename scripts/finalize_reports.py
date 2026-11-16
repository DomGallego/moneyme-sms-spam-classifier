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
    "# Final results and implementation evidence",
    "",
    "The serving selections were frozen using development/validation evidence before their original held-out test. The 60-configuration ML treatment benchmark uses the same already-inspected test set and is explicitly exploratory. See ML_TREATMENTS.md for treatment results and evaluation provenance. These are historical public-dataset results, not production claims.",
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
    "All systems use the same 1,034 test SMS (904 ham, 130 spam). Spam is positive. Local p95 measures warm sequential full-pipeline inference over 100 test messages; hosted p95 measures complete HTTP requests at concurrency 8. These are different runtime scopes, not isolated hardware benchmarks. API/container measurements are separate.",
    "",
    "The local artifact is character TF-IDF + Logistic Regression, about 252 KiB. Frozen embeddings and full encoder fine-tuning were included in the experiment; greater model complexity did not automatically improve held-out performance in the bounded searches. The LLM selection follows validation F1 rather than minimum price. The decision selection follows validation F1 subject to coverage eligibility. Free-tier pricing is temporary, not an established production price.",
    "",
    "## Full comparison and diagnostics",
    "",
    "- `final_comparison.csv`: 60 local treatment configurations, dummy baseline, seven LLMs, seven decision models; 75 rows. Evaluation-scope fields distinguish exploratory treatment results from original preselected-system evidence.",
    "- `ML_TREATMENTS.md` and `ml_treatment_comparison.csv`: fitting treatments, per-treatment leaders and full development/test comparison. No automatic serving-release replacement.",
    "- `ml_treatment_paired_differences.csv`: exploratory paired family-bootstrap changes relative to each pipeline's original-data treatment.",
    "- `paired_selected_comparisons.csv`: paired family-bootstrap F1 differences for the preselected systems; intervals can overlap zero. No multiple-comparison correction.",
    "- `hosted_test_uncertainty.csv`: family-bootstrap F1 intervals conditional on successful responses.",
    "- `hosted_operating_points.csv`: validation-selected operating points evaluated on test.",
    "- `hosted_calibration_bins.csv`: probability bins with counts; never assume vendor calibration transfers.",
    "- `error_slices.csv`: overlapping descriptive slices with sample/class support and failed spam/ham counts.",
    "- `local_test_errors.csv` and `hosted_test_errors.csv`: inspection examples; no retuning on these errors.",
    "- `runs/`: effective configurations, manifests, search histories, per-message predictions and provider usage.",
    "",
    "## Cost accounting",
    "",
    f"User-authorized cap: **US${cost['authorized_usd']:.2f}** total, including a US$1 pilot. Provider-reported charges: **US${cost['reported_usd']:.6f}**. Conservative accounted total: **US${cost['accounted_usd']:.6f}**, including retained reservations for **{cost['unknown_attempts']}** attempts with unknown charges.",
    "",
    "`cost_attempts.csv` exports sanitized attempt-level accounting; `cost_ledger_summary.csv` separates pilot, development, validation, test, augmentation generation, robustness and service checks. Unknown charges are not zero. Augmentation cost is a one-time training-data cost, not per-prediction classifier cost. Projections per million messages assume the same message/prompt lengths, cache behavior, retry policy and prices. Local API charges are zero but infrastructure/electricity costs are unmeasured. Pretraining costs are outside this assessment for pretrained systems.",
    "",
    "## Reliability and verification",
    "",
]
for result in hosted:
    if result["failures"]:
        lines.append(
            f"- {result['candidate']}: test coverage {result['metrics']['coverage']:.2%}; failures {json.dumps(result['failures'])}. Conditional classification metrics must be read with coverage and correct/attempted accuracy."
        )
tests = ET.parse("reports/tests.xml").getroot()
suites = list(tests.iter("testsuite"))
total = sum(int(s.attrib.get("tests", 0)) for s in suites)
failures = sum(int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites)
if failures:
    raise ValueError("Test evidence includes failures")
lines += [
    f"- Automated suite: {total} tests, zero failures/errors; see `tests.xml`. A pinned Starlette TestClient deprecation warning is non-blocking.",
    "- Real Docker local-only smoke test and a Git-visible-files rehearsal passed without secrets, raw data, or local caches.",
    "- Live container checks exercised all three prediction routes, the sample Python client, and OpenAPI. Live calls used the same authorized budget ledger.",
    "- Stability checks repeated 16 validation messages three times per hosted champion; separate handwritten instruction-override examples are diagnostics, not a production benchmark. See `robustness_summary.json`.",
    "",
    "## Limits and next steps",
    "",
    "The small historical English corpus cannot establish performance on current MoneyMe traffic. Some conversational-looking messages and jokes carry spam labels, so label/task ambiguity matters. Public pretrained contamination cannot be ruled out, especially when benchmark scores approach perfection. Conservative family grouping misses some paraphrases. High-precision and calibration estimates have limited support. Hosted models and provider routing can change behind stable names.",
    "",
    "Next production step: collect representative reviewed labels, establish the actual blocking/review policy and error costs, test temporal generalization, load-test the API, monitor quality/latency/cost/version changes, and retain rollback artifacts. No repository publication, email submission, or public hosting was performed by this implementation.",
    "",
]
Path("reports/RESULTS.md").write_text("\n".join(lines))
subprocess.run([sys.executable, "scripts/build_notebook.py"], check=True)
subprocess.run([sys.executable, "scripts/execute_notebook.py"], check=True)
Path("reports/implementation_status.md").write_text(
    "# Implementation status\n\nImplementation and the agreed end-to-end verification are complete.\n\n"
    + "\n".join(lines[2:])
)
print("Final reports and executed notebook written.")
