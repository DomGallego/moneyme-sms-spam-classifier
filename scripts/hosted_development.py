from moneyme_sms_spam_classifier.common import read_json, write_json
from moneyme_sms_spam_classifier.experiments import freeze_hosted, run_hosted
from moneyme_sms_spam_classifier.hosted import Budget

pilot = read_json("reports/runs/pilot/results.json")
per_message = sum((r["cost_usd"] or 0) / r["record_count"] for r in pilot)
forecast = {
    "authorized_total_usd": 10,
    "pilot_cap_usd": 1,
    "pilot_reported_usd": sum(r["cost_usd"] or 0 for r in pilot),
    "accounted_usd_at_forecast": Budget(cap=10).total(),
    "development_96_all_variants_estimate_usd": per_message * 96,
    "validation_and_test_single_variant_estimate_usd": per_message / 2 * (1035 + 1034),
    "note": "Pilot-based estimate only; final variants, reasoning and caches can change charges. Hard cumulative cap remains $10.",
}
write_json("reports/cost_forecast.json", forecast)
print(forecast, flush=True)
run_hosted(stage="development", limit=96, cap=10, workers=8)
freeze_hosted("development")
run_hosted(stage="validation", limit=0, cap=10, workers=8, selected=True)
freeze_hosted("validation")
print("Hosted configurations frozen; final test requires an explicit separate command.", flush=True)
