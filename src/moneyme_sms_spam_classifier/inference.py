"""Shared inference used by both CLI and HTTP handlers."""

from pathlib import Path

import joblib

from .common import fingerprint, read_json
from .evaluation import score_model
from .hosted import Budget, HostedError, OpenRouterClassifier


class LocalClassifier:
    def __init__(self, selection="configs/local_selection.json"):
        selected = read_json(selection)
        path = Path(selected["artifact"])
        expected = selected.get("sha256")
        if expected and fingerprint(path.read_bytes()) != expected:
            raise ValueError("Model artifact checksum mismatch")
        # Load trusted repository/release artifacts only; pickle is executable code.
        self.bundle = joblib.load(path)
        self.candidate = self.bundle["candidate"]

    def predict(self, text):
        scores, _ = score_model(self.bundle["model"], [text])
        return "spam" if scores[0] >= self.bundle["threshold"] else "ham"


def hosted_classifier(group, cap=0, ledger="reports/service-budget.sqlite"):
    frozen = read_json("configs/hosted_frozen.json")
    config = frozen["champions"][group]
    catalog = {m["id"]: m for m in read_json("reports/catalog.json")["data"]}
    return OpenRouterClassifier(config, Budget(ledger, cap), catalog[config["model"]])


def hosted_label(adapter, text):
    result = adapter.predict(text, {"stage": "service"})
    if result["prediction"] < 0:
        raise HostedError(result["status"], 504 if result["status"] == "timeout" else 502)
    if adapter.config["group"] == "decision":
        return "spam" if result["score"] >= adapter.config.get("threshold", 0.5) else "ham"
    return "spam" if result["prediction"] else "ham"
