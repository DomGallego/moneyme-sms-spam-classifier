"""Metrics and validation-only threshold selection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def metrics(y, predictions, scores=None, probabilities=False):
    y, predictions = np.asarray(y), np.asarray(predictions)
    valid = predictions >= 0
    result = {
        "attempted": len(y),
        "completed": int(valid.sum()),
        "coverage": float(valid.mean()),
        "end_to_end_accuracy": float(np.sum(predictions == y) / len(y)),
    }
    if not valid.any():
        return result
    yt, yp = y[valid], predictions[valid]
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    result.update(
        {
            "accuracy": accuracy_score(yt, yp),
            "spam_f1": f1_score(yt, yp, zero_division=0),
            "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "false_positive_rate": float(fp / max(tn + fp, 1)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "ham_support": int((yt == 0).sum()),
            "spam_support": int((yt == 1).sum()),
        }
    )
    if scores is not None:
        s = np.asarray(scores, dtype=float)[valid]
        if np.isfinite(s).all() and len(np.unique(yt)) == 2:
            result["average_precision"] = average_precision_score(yt, s)
            if probabilities:
                result["brier"] = brier_score_loss(yt, s)
                result["log_loss"] = log_loss(yt, np.clip(s, 1e-7, 1 - 1e-7), labels=[0, 1])
    return result


def choose_threshold(y, scores, precision_target=0.98):
    # Fixed target chosen before inspecting validation results; exploratory, not a business SLA.
    scores = np.asarray(scores)
    thresholds = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    entries = []
    for threshold in thresholds:
        pred = scores >= threshold
        entries.append(
            (
                float(threshold),
                f1_score(y, pred, zero_division=0),
                precision_score(y, pred, zero_division=0),
                recall_score(y, pred, zero_division=0),
                int(pred.sum()),
            )
        )
    best = max(entries, key=lambda e: (e[1], e[2], e[0]))
    eligible = [e for e in entries if e[2] >= precision_target and e[4] >= 10]
    high = max(eligible, key=lambda e: (e[3], e[2])) if eligible else None

    def asdict(e):
        return (
            dict(zip(["threshold", "f1", "precision", "recall", "predicted_spam"], e))
            if e
            else None
        )

    return {
        "primary": asdict(best),
        "high_precision": asdict(high),
        "precision_target": precision_target,
        "minimum_positive_predictions": 10,
        "note": "Validation estimates only; 10 flags is not strong statistical support for 98% precision.",
    }


def score_model(model, texts):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, list(model.classes_).index(1)], True
    return model.decision_function(texts), False


def latency_summary(milliseconds):
    a = np.asarray(milliseconds, dtype=float)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "count": len(a),
    }
