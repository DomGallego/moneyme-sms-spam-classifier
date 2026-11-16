"""Final local evaluation and regeneration of saved-result reports (no hosted calls)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import fingerprint, manifest, read_json, write_json
from .data import load
from .evaluation import latency_summary, metrics, score_model


def bootstrap_f1(frame, repetitions=500):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(42)
    if frame.empty:
        return {
            "f1_ci_low": None,
            "f1_ci_high": None,
            "bootstrap_repetitions": 0,
            "unit": "template family",
        }
    groups = frame.group.unique()
    members = {group: np.flatnonzero(frame.group.to_numpy() == group) for group in groups}
    estimates = []
    for _ in range(repetitions):
        indices = np.concatenate(
            [members[g] for g in rng.choice(groups, len(groups), replace=True)]
        )
        sample = frame.iloc[indices]
        estimates.append(f1_score(sample.target, sample.prediction, zero_division=0))
    return {
        "f1_ci_low": float(np.percentile(estimates, 2.5)),
        "f1_ci_high": float(np.percentile(estimates, 97.5)),
        "bootstrap_repetitions": repetitions,
        "unit": "template family",
    }


def finalize_local():
    if not Path("configs/local_selection.json").exists():
        raise ValueError("Freeze local selection before final evaluation")
    output = Path("reports/runs/local_test")
    results = read_json(output / "results.json") if (output / "results.json").exists() else []
    completed = {r["candidate"]: r for r in results}
    if "always_ham" in completed:
        raise ValueError(
            "Final local evaluation is complete; regenerate reports instead of retesting"
        )
    df = load()
    test = df[df.split == "test"]
    if not (output / "manifest.json").exists():
        write_json(
            output / "manifest.json",
            manifest(
                selection=read_json("configs/local_selection.json"),
                split=read_json("artifacts/splits/manifest.json"),
                record_ids=test.id.tolist(),
            ),
        )
    frozen = read_json(output / "manifest.json")
    if frozen["selection"] != read_json("configs/local_selection.json"):
        raise ValueError("Cannot resume final evaluation with a changed selection")
    for path in sorted(Path("artifacts/models").glob("*/result.json")):
        development = read_json(path)
        candidate = development["candidate"]
        if candidate in completed:
            if completed[candidate]["artifact_sha256"] != fingerprint(
                (path.parent / "model.joblib").read_bytes()
            ):
                raise ValueError(f"Frozen artifact changed for {candidate}")
            continue
        bundle = joblib.load(path.parent / "model.joblib")
        scores, probability = score_model(bundle["model"], test.text.tolist())
        predictions = (scores >= bundle["threshold"]).astype(int)
        pred = pd.DataFrame(
            {
                "id": test.id,
                "group": test.group,
                "target": test.target,
                "prediction": predictions,
                "score": scores,
            }
        )
        pred.to_csv(output / f"{candidate}_predictions.csv", index=False)
        # Warm sequential per-message latency; record cold loading separately in deployment checks.
        score_model(bundle["model"], [test.text.iloc[0]])
        timings = []
        for text in test.text.iloc[:100]:
            started = time.perf_counter()
            score_model(bundle["model"], [text])
            timings.append((time.perf_counter() - started) * 1000)
        result = {
            "candidate": candidate,
            "group": "ml",
            "kind": development["kind"],
            "stage": "test",
            "metrics": metrics(test.target, predictions, scores, probability),
            "latency": latency_summary(timings),
            "cost_usd": None,
            "cost_note": "Local compute: no API charge; infrastructure cost not measured.",
            "threshold": bundle["threshold"],
            "probabilities": probability,
            "artifact_sha256": fingerprint((path.parent / "model.joblib").read_bytes()),
            "uncertainty": bootstrap_f1(pred),
        }
        high = development["thresholds"].get("high_precision")
        result["high_precision_metrics"] = (
            metrics(test.target, (scores >= high["threshold"]).astype(int)) if high else None
        )
        results.append(result)
        write_json(output / "results.json", results)
        print("Tested local candidate", candidate, flush=True)
    dummy = {
        "candidate": "always_ham",
        "group": "baseline",
        "stage": "test",
        "metrics": metrics(test.target, np.zeros(len(test), dtype=int)),
        "latency": {},
        "cost_usd": None,
    }
    results.append(dummy)
    write_json(output / "results.json", results)


def comparison(stage="test", include_treatments=True):
    results = []
    if stage == "test":
        for filename in ["reports/runs/local_test/results.json", "reports/runs/test/results.json"]:
            if Path(filename).exists():
                results.extend(read_json(filename))
    else:
        results.extend(
            {
                "candidate": r["candidate"],
                "group": "ml",
                "variant": r["kind"],
                "metrics": r["validation"],
                "latency": {},
                "cost_usd": None,
            }
            for p in Path("artifacts/models").glob("*/result.json")
            for r in [read_json(p)]
        )
        filename = "reports/runs/validation/results.json"
        if not Path(filename).exists():
            filename = "reports/runs/development/results.json"
        if Path(filename).exists():
            results.extend(read_json(filename))
    treatment_path = Path("reports/runs/imbalance") / (
        "test_results.json" if stage == "test" else "validation_results.json"
    )
    if include_treatments and treatment_path.exists():
        treatment_results = read_json(treatment_path)
        if len(treatment_results) == 60:
            results = [row for row in results if row["group"] != "ml"]
            if stage == "test":
                results.extend(treatment_results)
            else:
                results.extend(
                    {
                        **row,
                        "metrics": row["validation"],
                        "variant": row["treatment"],
                        "latency": {},
                        "stage": "validation",
                    }
                    for row in treatment_results
                )
    rows = []
    for result in results:
        m = result["metrics"]
        cost, attempts = result.get("cost_usd"), m["attempted"]
        runtime = {"measurement_concurrency": 1, "latency_scope": "local_pipeline"}
        if result["group"] in {"llm", "decision"}:
            actual_stage = result.get("stage", stage)
            stage_manifest = Path("reports/runs") / actual_stage / "manifest.json"
            runtime = {
                "measurement_concurrency": read_json(stage_manifest)["workers"],
                "latency_scope": "hosted_HTTP_request",
            }
            journal = stage_manifest.parent / result["configuration_id"] / "requests.jsonl"
            if journal.exists():
                requests = pd.read_json(journal, lines=True)
                if {"started_at_unix", "finished_at_unix"}.issubset(
                    requests.columns
                ) and requests.started_at_unix.notna().all():
                    seconds = requests.finished_at_unix.max() - requests.started_at_unix.min()
                    runtime["observed_messages_per_second"] = (
                        len(requests) / seconds if seconds > 0 else None
                    )
        rows.append(
            {
                "candidate": result["candidate"],
                "group": result["group"],
                "variant": result.get("variant", result.get("kind", "default")),
                "stage": stage,
                "source_stage": result.get("stage", stage),
                "evaluation_scope": result.get(
                    "evaluation_scope", "Preselected historical benchmark"
                ),
                **runtime,
                **m,
                **result.get("latency", {}),
                "cost_usd": cost,
                "usd_per_1000_attempted": cost / attempts * 1000 if cost is not None else None,
                "usd_per_1000_completed": cost / m["completed"] * 1000
                if cost is not None and m["completed"]
                else None,
                "projected_reported_usd_per_million": cost / attempts * 1_000_000
                if cost is not None
                else None,
                "cost_projection_note": "Same workload/cache/pricing; excludes unknown charges and local infrastructure.",
                "unknown_cost_attempts": result.get("unknown_cost_attempts", 0),
            }
        )
    return pd.DataFrame(rows)


def comparison_metric_figures(stage="test"):
    """Compare individual metrics within model groups using saved results only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = comparison(stage)
    if table.empty:
        return
    figures = Path("reports/figures")
    figures.mkdir(parents=True, exist_ok=True)
    groups = [
        ("Local models and baseline", ["ml", "baseline"]),
        ("Hosted LLMs", ["llm"]),
        ("Hosted decision models", ["decision"]),
    ]
    for metric, title in [
        ("precision", "Spam precision"),
        ("recall", "Spam recall"),
        ("accuracy", "Accuracy"),
    ]:
        local_count = int(table.group.isin(["ml", "baseline"]).sum())
        fig, axes = plt.subplots(1, 3, figsize=(22, max(11, local_count * 0.32)), sharex=True)
        for ax, (group_title, members) in zip(axes, groups, strict=True):
            rows = table[table.group.isin(members)].sort_values(metric)
            labels = rows.candidate + ":" + rows.variant
            if members != ["ml", "baseline"]:
                labels = labels + rows.coverage.map(lambda v: f" [coverage {v:.1%}]")
            bars = ax.barh(labels, rows[metric])
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            ax.set(xlim=(0, 1.10), xlabel=title, title=group_title)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        fig.suptitle(
            f"{stage.title()} {title.lower()} — hosted metrics conditional on successful responses\n"
            "Read with coverage; zero predicted spam gives precision 0 by reporting convention."
            + (
                "\nLocal treatment results use previously inspected test data: exploratory."
                if stage == "test" and table.evaluation_scope.str.contains("Exploratory").any()
                else ""
            )
        )
        fig.tight_layout()
        fig.savefig(figures / f"{stage}_{metric}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def generate():
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.calibration import calibration_curve
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import PrecisionRecallDisplay

    from .features import SMSFeatures

    df = load()
    train = df[df.split == "train"]
    figures = Path("reports/figures")
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def save(name):
        plt.tight_layout()
        plt.savefig(figures / f"{name}.png", dpi=140, bbox_inches="tight")
        plt.close()

    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df.label.value_counts().plot.bar(ax=axes[0], title="Cleaned class counts")
    for label in ["ham", "spam"]:
        sns.ecdfplot(df[df.label == label].text.str.len(), label=label, ax=axes[1])
    axes[1].set(title="Message length distribution", xlabel="Characters")
    axes[1].legend()
    save("class_and_length")
    plt.figure(figsize=(8, 4))
    for label in ["ham", "spam"]:
        sns.ecdfplot(train[train.label == label].text.str.split().str.len(), label=label)
    plt.xlabel("Words per training message")
    plt.title("Training word-count distribution by class")
    plt.legend()
    save("word_length")
    values = SMSFeatures().transform(train.text)
    feature_frame = pd.DataFrame(values, columns=SMSFeatures.names)
    feature_frame["label"] = train.label.to_numpy()
    feature_frame.groupby("label").mean().T.plot.bar(subplots=True, figsize=(10, 13), legend=False)
    save("engineered_features")
    for label in ["ham", "spam"]:
        vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3)
        X = vectorizer.fit_transform(train[train.label == label].text)
        counts = np.asarray(X.sum(axis=0)).ravel()
        indices = counts.argsort()[-15:]
        plt.figure(figsize=(8, 5))
        plt.barh(vectorizer.get_feature_names_out()[indices], counts[indices])
        plt.title(f"Training {label}: frequent words / phrases")
        save(f"ngrams_{label}")
    if Path("artifacts/splits/embeddings.npy").exists():
        X = np.load("artifacts/splits/embeddings.npy")[train.index]
        projection = PCA(n_components=2, random_state=42).fit_transform(X)
        plt.figure(figsize=(8, 6))
        for label in ["ham", "spam"]:
            mask = train.label.to_numpy() == label
            plt.scatter(*projection[mask].T, s=10, alpha=0.35, label=label)
        plt.legend()
        plt.title("Training sentence embeddings: PCA (descriptive projection)")
        save("embeddings_pca")
        try:
            import umap

            projection = umap.UMAP(
                random_state=42, n_neighbors=15, min_dist=0.1, n_jobs=1, compatibility_layout=True
            ).fit_transform(X)
            plt.figure(figsize=(8, 6))
            for label in ["ham", "spam"]:
                mask = train.label.to_numpy() == label
                plt.scatter(*projection[mask].T, s=10, alpha=0.35, label=label)
            plt.legend()
            plt.title("Training embeddings: UMAP (distances distorted)")
            save("embeddings_umap")
        except ImportError:
            pass
    for stage in ["validation", "test"]:
        table = comparison(stage)
        if table.empty:
            continue
        table.to_csv(f"reports/{stage}_comparison.csv", index=False)
        if stage == "test":
            table.to_csv("reports/final_comparison.csv", index=False)
        comparison_metric_figures(stage)
        plt.figure(figsize=(11, max(6, len(table) * 0.25)))
        ordered = table.sort_values("spam_f1")
        labels = ordered.candidate + ":" + ordered.variant
        labels += np.where(
            ordered.group.isin(["llm", "decision"]),
            ordered.coverage.map(lambda value: f" [coverage {value:.1%}]"),
            "",
        )
        plt.barh(labels, ordered.spam_f1)
        plt.xlim(0, 1.02)
        plt.title(
            f"{stage.title()} spam F1 — hosted scores conditional on successful responses"
            + (
                "\nLocal treatments: exploratory results on previously inspected test data"
                if stage == "test" and table.evaluation_scope.str.contains("Exploratory").any()
                else ""
            )
        )
        save(f"{stage}_f1")
        hosted = table[table.group.isin(["llm", "decision"])]
        if len(hosted):
            plt.figure(figsize=(9, 6))
            for _, row in hosted.iterrows():
                plt.scatter(
                    row.usd_per_1000_attempted, row.spam_f1, label=f"{row.candidate}:{row.variant}"
                )
            plt.xlabel("Reported USD per 1,000 attempted SMS (unknown charges excluded)")
            plt.ylabel("Spam F1")
            plt.legend(fontsize=6, bbox_to_anchor=(1, 1))
            plt.title(stage.title() + " quality versus hosted cost")
            save(f"{stage}_cost")
            plt.figure(figsize=(9, 6))
            for _, row in hosted.iterrows():
                plt.scatter(row.p95_ms, row.spam_f1, label=f"{row.candidate}:{row.variant}")
            plt.xlabel("p95 request latency, ms")
            plt.ylabel("Spam F1")
            plt.legend(fontsize=6, bbox_to_anchor=(1, 1))
            save(f"{stage}_latency")
    local_results = Path("reports/runs/local_test/results.json")
    if local_results.exists():
        selection = read_json("configs/local_selection.json")["candidate"]
        rows = pd.read_csv(f"reports/runs/local_test/{selection}_predictions.csv")
        result = next(r for r in read_json(local_results) if r["candidate"] == selection)
        _fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        m = result["metrics"]
        sns.heatmap(
            [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
            annot=True,
            fmt="d",
            ax=axes[0],
            xticklabels=["ham", "spam"],
            yticklabels=["ham", "spam"],
        )
        axes[0].set(xlabel="Predicted", ylabel="Actual", title=selection)
        PrecisionRecallDisplay.from_predictions(rows.target, rows.score, ax=axes[1])
        if result["probabilities"]:
            actual, predicted = calibration_curve(
                rows.target, rows.score, n_bins=8, strategy="quantile"
            )
            axes[2].plot(predicted, actual, marker="o")
            axes[2].plot([0, 1], [0, 1], "--")
            axes[2].set(
                xlabel="Mean predicted spam probability",
                ylabel="Observed spam fraction",
                title="Reliability (quantile bins)",
            )
        else:
            axes[2].text(0.1, 0.5, "Uncalibrated scores: reliability plot not applicable")
        save("selected_local_diagnostics")
        errors = rows[rows.target != rows.prediction].merge(df[["id", "text"]], on="id")
        errors.to_csv("reports/local_test_errors.csv", index=False)
    if Path("reports/budget.sqlite").exists():
        with sqlite3.connect("reports/budget.sqlite") as db:
            ledger = pd.read_sql_query("SELECT * FROM attempts", db)
        meta = pd.DataFrame(ledger.metadata.map(json.loads).tolist())
        ledger["stage"] = meta.get("stage")
        ledger["model"] = meta.get("model_requested")
        ledger["accounted_usd"] = ledger.cost.fillna(ledger.reserved)
        summary = ledger.groupby(["stage", "model"], dropna=False).agg(
            attempts=("id", "count"),
            reported_usd=("cost", lambda s: s.sum(min_count=1)),
            accounted_usd=("accounted_usd", "sum"),
            unknown_charges=("cost", lambda s: s.isna().sum()),
        )
        audit = ledger[["id", "cost", "reserved", "accounted_usd"]].rename(
            columns={"id": "attempt_id", "cost": "reported_usd", "reserved": "reserved_usd"}
        )
        for column in [
            "stage",
            "model_requested",
            "configuration_id",
            "id",
            "status",
            "request_id",
            "model_returned",
            "provider",
            "started_at_unix",
            "finished_at_unix",
        ]:
            if column in meta:
                audit["record_id" if column == "id" else column] = meta[column]
        usage = meta["usage"].map(lambda item: item if isinstance(item, dict) else {})
        audit["input_tokens"] = usage.map(
            lambda item: item.get("prompt_tokens", item.get("input_tokens"))
        )
        audit["output_tokens"] = usage.map(
            lambda item: item.get("completion_tokens", item.get("output_tokens"))
        )
        audit["reasoning_tokens"] = usage.map(
            lambda item: (item.get("completion_tokens_details") or {}).get("reasoning_tokens")
        )
        audit["cached_tokens"] = usage.map(
            lambda item: (item.get("prompt_tokens_details") or {}).get("cached_tokens")
        )
        token_totals = audit.groupby(["stage", "model_requested"], dropna=False)[
            ["input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens"]
        ].sum(min_count=1)
        token_totals.index.names = ["stage", "model"]
        summary.join(token_totals).to_csv("reports/cost_ledger_summary.csv")
        audit.to_csv("reports/cost_attempts.csv", index=False)
        write_json(
            "reports/cost_summary.json",
            {
                "reported_usd": float(ledger.cost.sum()),
                "accounted_usd": float(ledger.accounted_usd.sum()),
                "unknown_attempts": int(ledger.cost.isna().sum()),
                "authorized_usd": read_json("configs/budget.json")["total_cap_usd"],
                "note": "Accounted total includes retained reservations for unknown charges.",
            },
        )
    print("Reports regenerated from saved data; no hosted requests.")


def enrich_hosted_test():
    """Additional diagnostics from saved test responses; no new inference or selection."""
    root = Path("reports/runs/test")
    if not (root / "results.json").exists():
        return
    df = load().query("split == 'test'")
    uncertainty, operating, errors, calibration = [], [], [], []
    validation = read_json("reports/runs/validation/results.json")
    for result in read_json(root / "results.json"):
        config = result["configuration"]
        folder = root / result["configuration_id"]
        frame = pd.read_json(folder / "requests.jsonl", lines=True)
        frame = frame.merge(
            df[["id", "group", "text"]].rename(columns={"group": "family"}), on="id"
        )
        prediction = frame.prediction.to_numpy()
        if result["group"] == "decision":
            prediction = np.where(
                prediction >= 0, frame.score.to_numpy() >= config.get("threshold", 0.5), -1
            ).astype(int)
        frame["prediction"] = prediction
        good = frame[frame.prediction >= 0].copy()
        good["group"] = good.family
        interval = bootstrap_f1(good)
        uncertainty.append(
            {
                "candidate": result["candidate"],
                "coverage": result["metrics"]["coverage"],
                "interval_scope": "F1 conditional on successful responses",
                **interval,
            }
        )
        frame[["id", "family", "target", "prediction", "score", "status"]].to_csv(
            folder / "predictions.csv", index=False
        )
        for row in frame[frame.prediction >= 0].query("target != prediction").to_dict("records"):
            errors.append(
                {
                    "candidate": result["candidate"],
                    "variant": result["variant"],
                    "id": row["id"],
                    "target": row["target"],
                    "prediction": row["prediction"],
                    "text": row["text"],
                }
            )
        if result["group"] == "decision" and not good.empty:
            # Quantile bins with explicit support; ties can produce fewer bins.
            good["bin"] = pd.qcut(good.score, 8, duplicates="drop")
            for interval_bin, rows in good.groupby("bin", observed=True):
                calibration.append(
                    {
                        "candidate": result["candidate"],
                        "bin": str(interval_bin),
                        "count": len(rows),
                        "mean_probability": rows.score.mean(),
                        "observed_spam_fraction": rows.target.mean(),
                    }
                )
            val = next(v for v in validation if v["candidate"] == result["candidate"])
            high = (val.get("thresholds") or {}).get("high_precision")
            if high:
                high_pred = np.where(prediction >= 0, frame.score >= high["threshold"], -1).astype(
                    int
                )
                operating.append(
                    {
                        "candidate": result["candidate"],
                        "point": "higher_precision",
                        "threshold": high["threshold"],
                        **metrics(frame.target, high_pred),
                    }
                )
        operating.append(
            {
                "candidate": result["candidate"],
                "point": "primary",
                "threshold": config.get("threshold"),
                **result["metrics"],
            }
        )
    pd.DataFrame(uncertainty).to_csv("reports/hosted_test_uncertainty.csv", index=False)
    pd.DataFrame(operating).to_csv("reports/hosted_operating_points.csv", index=False)
    pd.DataFrame(errors).to_csv("reports/hosted_test_errors.csv", index=False)
    pd.DataFrame(calibration).to_csv("reports/hosted_calibration_bins.csv", index=False)


def error_slices():
    """Descriptive test slices; never used for threshold/prompt/model selection."""
    from .data import URL

    df = load().query("split == 'test'").set_index("id")
    length = df.text.str.len()
    digits = df.text.map(lambda text: sum(c.isdigit() for c in text) / max(len(text), 1))
    slices = {
        "all": pd.Series(True, index=df.index),
        "short_le_40_chars": length <= 40,
        "long_gt_160_chars": length > 160,
        "contains_url": df.text.map(lambda text: bool(URL.search(text))),
        "contains_digit": df.text.str.contains(r"\d"),
        "digit_fraction_gt_10pct": digits > 0.1,
    }
    candidates = []
    treatment_results = Path("reports/runs/imbalance/test_results.json")
    if treatment_results.exists() and len(read_json(treatment_results)) == 60:
        for p in Path("reports/runs/imbalance/test").glob("*/predictions.csv"):
            candidates.append((p.parent.name, pd.read_csv(p).set_index("id")))
    else:
        for p in Path("reports/runs/local_test").glob("*_predictions.csv"):
            candidates.append(
                (p.name.removesuffix("_predictions.csv"), pd.read_csv(p).set_index("id"))
            )
    for p in Path("reports/runs/test").glob("*/predictions.csv"):
        config = read_json(p.parent / "configuration.json")
        candidates.append((config["model"], pd.read_csv(p).set_index("id")))
    output = []
    for candidate, predictions in candidates:
        for name, mask in slices.items():
            ids = df.index[mask]
            part = predictions.loc[predictions.index.intersection(ids)]
            if part.empty:
                continue
            summary = metrics(part.target, part.prediction)
            output.append(
                {
                    "candidate": candidate,
                    "slice": name,
                    "attempted_spam": int(part.target.sum()),
                    "failed_spam": int(((part.target == 1) & (part.prediction < 0)).sum()),
                    "failed_ham": int(((part.target == 0) & (part.prediction < 0)).sum()),
                    **summary,
                }
            )
    pd.DataFrame(output).to_csv("reports/error_slices.csv", index=False)


def hosted_figures():
    """Visual diagnostics for the frozen hosted selections and decision probabilities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    root = Path("reports/runs/test")
    results = read_json(root / "results.json")
    frozen = read_json("configs/hosted_frozen.json")
    _fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, group in zip(axes, ["llm", "decision"]):
        selected = frozen["champions"][group]["model"]
        result = next(r for r in results if r["candidate"] == selected)
        m = result["metrics"]
        sns.heatmap(
            [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
            annot=True,
            fmt="d",
            xticklabels=["ham", "spam"],
            yticklabels=["ham", "spam"],
            ax=ax,
        )
        ax.set(
            title=f"{selected}\ncoverage={m['coverage']:.1%}", xlabel="Predicted", ylabel="Actual"
        )
    plt.tight_layout()
    plt.savefig("reports/figures/hosted_confusion.png", dpi=140, bbox_inches="tight")
    plt.close()
    bins = pd.read_csv("reports/hosted_calibration_bins.csv")
    models = [r["candidate"] for r in results if r["group"] == "decision"]
    _fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, model in zip(axes.flat, models):
        rows = bins[bins.candidate == model].sort_values("mean_probability")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.plot(rows.mean_probability, rows.observed_spam_fraction, marker="o")
        for row in rows.itertuples():
            ax.annotate(
                f"n={row.count}", (row.mean_probability, row.observed_spam_fraction), fontsize=6
            )
        ax.set(
            title=model,
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
            xlabel="Mean P(spam)",
            ylabel="Observed spam fraction",
        )
    for ax in list(axes.flat)[len(models) :]:
        ax.axis("off")
    plt.suptitle(
        "Decision-model reliability: quantile bins; ties reduce bin count; successful responses only"
    )
    plt.tight_layout()
    plt.savefig("reports/figures/decision_reliability.png", dpi=140, bbox_inches="tight")
    plt.close()


def paired_selected_comparisons(repetitions=500):
    """Paired family bootstrap of F1 differences for preselected systems."""
    from itertools import combinations

    from sklearn.metrics import f1_score

    df = load().query("split == 'test'").set_index("id")
    local = read_json("configs/local_selection.json")["candidate"]
    predictions = {
        "ml": pd.read_csv(f"reports/runs/local_test/{local}_predictions.csv").set_index("id")
    }
    results = read_json("reports/runs/test/results.json")
    frozen = read_json("configs/hosted_frozen.json")
    for group, config in frozen["champions"].items():
        result = next(r for r in results if r["candidate"] == config["model"])
        predictions[group] = pd.read_csv(
            f"reports/runs/test/{result['configuration_id']}/predictions.csv"
        ).set_index("id")
    output = []
    for left, right in combinations(predictions, 2):
        merged = (
            df[["group", "target"]]
            .join(predictions[left][["prediction"]].rename(columns={"prediction": "left"}))
            .join(predictions[right][["prediction"]].rename(columns={"prediction": "right"}))
        )
        merged = merged[(merged.left >= 0) & (merged.right >= 0)].reset_index()
        if merged.empty or merged.target.nunique() < 2:
            output.append(
                {
                    "left": left,
                    "right": right,
                    "common_successes": len(merged),
                    "common_success_coverage": len(merged) / len(df),
                    "delta_f1_left_minus_right": None,
                    "ci_low": None,
                    "ci_high": None,
                    "note": "Insufficient common class support for paired F1.",
                }
            )
            continue
        groups = merged.group.unique()
        members = {g: np.flatnonzero(merged.group.to_numpy() == g) for g in groups}
        rng = np.random.default_rng(42)
        differences = []
        for _ in range(repetitions):
            indices = np.concatenate(
                [members[g] for g in rng.choice(groups, len(groups), replace=True)]
            )
            sample = merged.iloc[indices]
            differences.append(
                f1_score(sample.target, sample.left, zero_division=0)
                - f1_score(sample.target, sample.right, zero_division=0)
            )
        output.append(
            {
                "left": left,
                "right": right,
                "common_successes": len(merged),
                "common_success_coverage": len(merged) / len(df),
                "delta_f1_left_minus_right": f1_score(merged.target, merged.left)
                - f1_score(merged.target, merged.right),
                "ci_low": float(np.percentile(differences, 2.5)),
                "ci_high": float(np.percentile(differences, 97.5)),
                "bootstrap_repetitions": repetitions,
                "unit": "template family",
                "note": "Exploratory paired intervals; no multiple-comparison correction.",
            }
        )
    pd.DataFrame(output).to_csv("reports/paired_selected_comparisons.csv", index=False)
