"""Cross-validation search; test records are never passed to training or selection."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .common import fingerprint, manifest, read_json, write_json
from .data import load
from .evaluation import choose_threshold, metrics, score_model
from .features import representation


def classifier(name):
    weights = [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 4}]
    regularization = np.logspace(-2, 2, 9).tolist()
    if name == "mnb":
        return MultinomialNB(), {"alpha": np.logspace(-3, 1, 12).tolist()}
    if name == "cnb":
        return ComplementNB(), {"alpha": np.logspace(-3, 1, 12).tolist(), "norm": [True, False]}
    if name == "logistic":
        return LogisticRegression(max_iter=1500, random_state=42), {
            "C": regularization,
            "class_weight": weights,
        }
    if name == "svm":
        return LinearSVC(max_iter=5000, random_state=42), {
            "C": regularization,
            "class_weight": weights,
        }
    if name == "ridge":
        return RidgeClassifier(solver="lsqr"), {"alpha": regularization, "class_weight": weights}
    if name == "forest":
        return RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=42), {
            "max_depth": [8, 20, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.3],
            "class_weight": [None, "balanced"],
        }
    if name == "xgboost":
        return XGBClassifier(
            n_jobs=1, random_state=42, tree_method="hist", eval_metric="logloss"
        ), {
            "n_estimators": [100, 200],
            "max_depth": [2, 4, 6],
            "learning_rate": [0.03, 0.1],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "scale_pos_weight": [1, 2, 4, 6],
        }
    raise ValueError(name)


EXPERIMENTS = [
    ("mnb", "counts"),
    ("mnb", "word"),
    ("cnb", "word"),
    ("logistic", "word"),
    ("svm", "word"),
    ("ridge", "word"),
    ("logistic", "char"),
    ("svm", "char"),
    ("logistic", "combined"),
    ("svm", "combined"),
    ("logistic", "engineered"),
    ("forest", "engineered"),
    ("xgboost", "engineered"),
    ("forest", "word"),
    ("xgboost", "word"),
    ("logistic", "word_engineered"),
]


def search_space(name, rep):
    """Shared search definition for equal-budget treatment comparisons."""
    _, parameters = classifier(name)
    space = {f"classifier__{key}": values for key, values in parameters.items()}
    if rep in {"word", "counts"}:
        space.update({"features__ngram_range": [(1, 1), (1, 2)], "features__min_df": [1, 2, 3]})
        if rep == "word":
            space["features__sublinear_tf"] = [True, False]
    elif rep == "char":
        space.update(
            {"features__ngram_range": [(2, 4), (3, 5), (3, 6)], "features__min_df": [1, 2]}
        )
    elif rep in {"combined", "word_engineered"}:
        space["features__word__min_df"] = [1, 2, 3]
    return space


def persist_candidate(name, model, train, val, cv_score, elapsed, params, output, kind="classical"):
    scores, probability = score_model(model, val.text.tolist())
    thresholds = choose_threshold(val.target, scores)
    threshold = thresholds["primary"]["threshold"]
    pred = (scores >= threshold).astype(int)
    result = {
        "candidate": name,
        "group": "ml",
        "kind": kind,
        "cv_f1": float(cv_score) if cv_score is not None else None,
        "training_seconds": elapsed,
        "parameters": params,
        "probabilities": probability,
        "thresholds": thresholds,
        "validation": metrics(val.target, pred, scores, probability),
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "threshold": threshold, "probabilities": probability, "candidate": name},
        output / "model.joblib",
        compress=3,
    )
    result["artifact_bytes"] = (output / "model.joblib").stat().st_size
    write_json(output / "result.json", result)
    pd.DataFrame({"id": val.id, "target": val.target, "score": scores, "prediction": pred}).to_csv(
        output / "validation_predictions.csv", index=False
    )
    return result


def train_classical(trials=20, jobs=2, only=None):
    df = load()
    train, val = df[df.split == "train"], df[df.split == "validation"]
    root = Path("artifacts/models")
    from datetime import UTC, datetime

    report = Path("reports/runs/classical") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    report.mkdir(parents=True, exist_ok=True)
    write_json(
        report / "manifest.json",
        manifest(trials=trials, cv_folds=5, splits=read_json("artifacts/splits/manifest.json")),
    )
    results = []
    for name, rep in EXPERIMENTS:
        candidate = f"{name}_{rep}"
        if only and candidate not in only:
            continue
        estimator, _ = classifier(name)
        pipeline = Pipeline([("features", representation(rep)), ("classifier", estimator)])
        space = search_space(name, rep)
        search = RandomizedSearchCV(
            pipeline,
            space,
            n_iter=trials,
            scoring="f1",
            cv=PredefinedSplit(train.fold.to_numpy()),
            n_jobs=jobs,
            random_state=42,
            error_score="raise",
            return_train_score=True,
        )
        started = time.perf_counter()
        search.fit(train.text.tolist(), train.target)
        output = root / candidate
        result = persist_candidate(
            candidate,
            search.best_estimator_,
            train,
            val,
            search.best_score_,
            time.perf_counter() - started,
            search.best_params_,
            output,
        )
        pd.DataFrame(search.cv_results_).to_csv(output / "search_results.csv", index=False)
        write_json(report / candidate / "result.json", result)
        pd.DataFrame(search.cv_results_).to_csv(
            report / candidate / "search_results.csv", index=False
        )
        results.append(result)
        print(
            f"{candidate}: CV F1={search.best_score_:.4f}, validation F1={result['validation']['spam_f1']:.4f}",
            flush=True,
        )
        write_json(report / "results.json", results)
    dummy = DummyClassifier(strategy="constant", constant=0).fit(
        np.zeros((len(train), 1)), train.target
    )
    write_json(
        report / "dummy.json",
        {"validation": metrics(val.target, dummy.predict(np.zeros((len(val), 1))))},
    )
    return results


def freeze_local():
    candidates = [read_json(p) for p in Path("artifacts/models").glob("*/result.json")]
    if not candidates:
        raise ValueError("No trained candidates")
    best = max(candidates, key=lambda r: (r["validation"]["spam_f1"], r["validation"]["precision"]))
    import shutil

    source = Path("artifacts/models") / best["candidate"] / "model.joblib"
    release = Path("artifacts/release")
    release.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(source)
    if best["kind"] in {"encoder", "frozen_embeddings"}:
        model = bundle["model"]
        assets = release / "encoder"
        shutil.copytree(model.model_path, assets, dirs_exist_ok=True)
        model.model_path = str(assets)
    joblib.dump(bundle, release / "model.joblib", compress=3)
    selected = {
        "candidate": best["candidate"],
        "kind": best["kind"],
        "artifact": "artifacts/release/model.joblib",
        "sha256": fingerprint((release / "model.joblib").read_bytes()),
        "selection_basis": "validation spam F1, then precision; frozen before test evaluation",
    }
    write_json("configs/local_selection.json", selected)
    print(selected)
    return selected
