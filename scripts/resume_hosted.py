"""Resume the current phase without reselecting configurations after test begins."""

from pathlib import Path

from moneyme_sms_spam_classifier.common import read_json, write_json
from moneyme_sms_spam_classifier.experiments import freeze_hosted, run_hosted

cap = read_json("configs/budget.json")["total_cap_usd"]
selection = read_json("configs/hosted_frozen.json")
validation_path = Path("reports/runs/validation/results.json")
complete = validation_path.exists() and len(read_json(validation_path)) == 14
if not complete or selection["selection_stage"] != "validation":
    if list(Path("reports/runs/test").glob("*/requests.jsonl")):
        raise ValueError("Test has begun: do not reselect configurations")
    run_hosted(stage="validation", limit=0, cap=cap, workers=8, selected=True)
    selection = freeze_hosted("validation")
snapshot = Path("reports/frozen_before_test.json")
if snapshot.exists():
    if read_json(snapshot) != selection:
        raise ValueError("Hosted selections changed after test freeze")
else:
    write_json(snapshot, selection)
run_hosted(stage="test", limit=0, cap=cap, workers=8, selected=True)
print("All hosted final evaluations complete.", flush=True)
