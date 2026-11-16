"""Saved-result treatment summaries, costs and exploratory paired comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_json, write_json
from .imbalance import ROOT, refresh_validation


def augmentation_quality_figures():
    """Training-only distribution checks; these do not verify synthetic labels."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .augmentation import POOL
    from .data import load
    from .features import SMSFeatures
    from .sampling import augmented_training

    train = load().query("split == 'train'")
    pool = pd.read_json(POOL, lines=True)
    fitted = augmented_training(train, pool)
    synthetic = fitted[fitted.parent_id.notna()]
    sources = [
        ("Original ham", train.query("target == 0")),
        ("Original spam", train.query("target == 1")),
        ("Selected generated spam", synthetic),
    ]
    frames = []
    for name, frame in sources:
        values = pd.DataFrame(SMSFeatures().transform(frame.text), columns=SMSFeatures.names)
        values["source"] = name
        values["non_ascii_fraction"] = frame.text.map(
            lambda text: sum(ord(char) > 127 for char in text) / max(len(text), 1)
        ).to_numpy()
        frames.append(values)
    values = pd.concat(frames, ignore_index=True)
    values.groupby("source").agg(["mean", "median"]).to_csv(
        "reports/augmentation_feature_summary.csv"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feature in zip(
        axes.flat,
        ["characters", "digit_fraction", "upper_fraction", "non_ascii_fraction"],
        strict=True,
    ):
        sns.ecdfplot(data=values, x=feature, hue="source", ax=ax)
        ax.set(title=feature, ylabel="Cumulative fraction")
    fig.suptitle(
        "Training-only augmentation distribution checks — differences can expose generator shortcuts"
    )
    fig.tight_layout()
    fig.savefig("reports/figures/augmentation_distributions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    rejected = read_json("reports/runs/augmentation/rejected.json")
    summary = {
        "original_spam": int((train.target == 1).sum()),
        "accepted_pool": len(pool),
        "selected_full_training": len(synthetic),
        "pool_parent_coverage": int(pool.parent_id.nunique()),
        "selected_parent_coverage": int(synthetic.parent_id.nunique()),
        "maximum_selected_variants_per_parent": int(synthetic.groupby("parent_id").size().max()),
        "rejection_reasons": pd.Series([row["reason"] for row in rejected])
        .value_counts()
        .to_dict(),
        "scope": "Integrity and distribution diagnostics only; inherited labels are not independently verified.",
    }
    write_json("reports/augmentation_quality_summary.json", summary)


def paired_interval(left, right, repetitions=500):
    """Family-bootstrap F1(right)-F1(left), preserving pairing and multiplicity."""
    if (
        left.id.tolist() != right.id.tolist()
        or not left.target.equals(right.target)
        or not left.group.equals(right.group)
    ):
        raise ValueError("Predictions must have identical ordered IDs and targets")
    groups = pd.factorize(left.group)[0]
    count = int(groups.max()) + 1
    contributions = []
    for frame in [left, right]:
        target, prediction = frame.target.to_numpy(), frame.prediction.to_numpy()
        contributions.append(
            np.stack(
                [
                    np.bincount(groups, weights=(target == 1) & (prediction == 1), minlength=count),
                    np.bincount(groups, weights=(target == 0) & (prediction == 1), minlength=count),
                    np.bincount(groups, weights=(target == 1) & (prediction == 0), minlength=count),
                ],
                axis=1,
            )
        )

    def f1(values):
        tp, fp, fn = values
        return 2 * tp / max(2 * tp + fp + fn, 1)

    rng, deltas = np.random.default_rng(42), []
    for _ in range(repetitions):
        sample = rng.integers(0, count, size=count)
        deltas.append(
            f1(contributions[1][sample].sum(axis=0)) - f1(contributions[0][sample].sum(axis=0))
        )
    return {
        "delta_f1_ci_low": float(np.percentile(deltas, 2.5)),
        "delta_f1_ci_high": float(np.percentile(deltas, 97.5)),
        "repetitions": repetitions,
        "unit": "template family",
        "scope": "Exploratory; no multiple-comparison correction",
    }


def generate_treatment_report():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    validation = refresh_validation()
    test_path = ROOT / "test_results.json"
    test = read_json(test_path) if test_path.exists() else []
    rows = []
    for stage, results in [("validation", validation), ("test", test)]:
        for result in results:
            measure = result["validation"] if stage == "validation" else result["metrics"]
            rows.append(
                {
                    "stage": stage,
                    "candidate": result["candidate"],
                    "base_candidate": result["base_candidate"],
                    "treatment": result["treatment"],
                    **measure,
                    **result.get("uncertainty", {}),
                    "cv_f1": result.get("cv_f1"),
                    "training_seconds": result.get("training_seconds"),
                    "training_device": result.get("training_device", "cpu"),
                    "training_precision": result.get("training_precision"),
                    "evaluation_scope": "Development selection"
                    if stage == "validation"
                    else read_json("configs/imbalance_experiment.json")["evaluation_scope"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv("reports/ml_treatment_comparison.csv", index=False)
    figures = Path("reports/figures")
    figures.mkdir(parents=True, exist_ok=True)
    augmentation_quality_figures()
    for stage in table.stage.unique():
        subset = table[table.stage == stage]
        fig, axes = plt.subplots(1, 4, figsize=(19, 10))
        for ax, metric in zip(axes, ["spam_f1", "precision", "recall", "accuracy"], strict=True):
            pivot = subset.pivot(
                index="base_candidate", columns="treatment", values=metric
            ).reindex(columns=["original", "text_aug", "smote"])
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                vmin=0,
                vmax=1,
                cmap="YlGnBu",
                ax=ax,
                cbar=metric == "accuracy",
                annot_kws={"size": 8},
            )
            ax.set(
                title=metric,
                xlabel="Training treatment",
                ylabel="Pipeline" if metric == "spam_f1" else "",
            )
            if metric != "spam_f1":
                ax.set_yticklabels([])
        fig.suptitle(
            f"{stage.title()} ML treatment comparison — missing cells are inapplicable or pending\n"
            + (
                "Existing inspected test set: exploratory evidence"
                if stage == "test"
                else "Original validation messages only"
            )
        )
        fig.tight_layout()
        fig.savefig(figures / f"ml_treatments_{stage}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    deltas, errors = [], []
    by_candidate = {result["candidate"]: result for result in test}
    for result in test:
        predictions = pd.read_csv(ROOT / "test" / result["candidate"] / "predictions.csv")
        wrong = predictions[predictions.target != predictions.prediction].copy()
        wrong["candidate"] = result["candidate"]
        wrong["treatment"] = result["treatment"]
        errors.append(wrong)
        if result["treatment"] == "original":
            continue
        baseline_name = f"{result['base_candidate']}__original"
        if baseline_name not in by_candidate:
            continue
        baseline = by_candidate[baseline_name]
        left = pd.read_csv(ROOT / "test" / baseline_name / "predictions.csv")
        right = pd.read_csv(ROOT / "test" / result["candidate"] / "predictions.csv")
        deltas.append(
            {
                "candidate": result["candidate"],
                "base_candidate": result["base_candidate"],
                "treatment": result["treatment"],
                "delta_f1": result["metrics"]["spam_f1"] - baseline["metrics"]["spam_f1"],
                **paired_interval(left, right),
            }
        )
    pd.DataFrame(deltas).to_csv("reports/ml_treatment_paired_differences.csv", index=False)
    if errors:
        from .data import load

        pd.concat(errors, ignore_index=True).merge(load()[["id", "text"]], on="id").to_csv(
            "reports/ml_treatment_errors.csv", index=False
        )
    status = {
        "expected_configurations": 60,
        "validation_complete": len(validation),
        "test_complete": len(test),
        "complete": len(validation) == len(test) == 60,
        "generation": read_json("reports/runs/augmentation/summary.json"),
        "evaluation_scope": read_json("configs/imbalance_experiment.json")["evaluation_scope"],
    }
    write_json(ROOT / "status.json", status)
    lines = [
        "# ML treatment benchmark",
        "",
        f"Validation complete: {len(validation)}/60. Exploratory test complete: {len(test)}/60.",
        "",
        "The 60 configurations compare original fitting data, LLM-generated training SMS, and fold-local SMOTE. See `docs/ml_experiments.md` for the complete protocol.",
        "",
        "## Per-treatment validation leaders",
        "",
        "| Treatment | Pipeline | Validation spam F1 |",
        "|---|---|---:|",
    ]
    for treatment, candidates in table[table.stage == "validation"].groupby("treatment"):
        leader = candidates.sort_values(["spam_f1", "precision"], ascending=False).iloc[0]
        lines.append(f"| {treatment} | {leader.base_candidate} | {leader.spam_f1:.4f} |")
    effects = []
    lines += [
        "",
        "## Matched treatment effects",
        "",
        "| Stage | Treatment | Paired pipelines | Higher F1 | Equal F1 | Lower F1 | Median F1 change |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for stage, candidates in table.groupby("stage"):
        pivot = candidates.pivot(index="base_candidate", columns="treatment", values="spam_f1")
        for treatment in ["text_aug", "smote"]:
            if treatment not in pivot or "original" not in pivot:
                continue
            delta = (pivot[treatment] - pivot.original).dropna()
            if delta.empty:
                continue
            row = {
                "stage": stage,
                "treatment": treatment,
                "paired_pipelines": len(delta),
                "higher_f1": int((delta > 1e-12).sum()),
                "equal_f1": int((delta.abs() <= 1e-12).sum()),
                "lower_f1": int((delta < -1e-12).sum()),
                "median_delta_f1": float(delta.median()),
            }
            effects.append(row)
            lines.append(
                f"| {stage} | {treatment} | {len(delta)} | {row['higher_f1']} | {row['equal_f1']} | {row['lower_f1']} | {row['median_delta_f1']:+.4f} |"
            )
    pd.DataFrame(effects).to_csv("reports/ml_treatment_effect_summary.csv", index=False)
    lines += [
        "",
        "These counts summarize this fixed set of heterogeneous pipelines. They are descriptive, not independent replications or a general probability that a treatment helps. Training wall times can include concurrent work and are not isolated hardware comparisons.",
    ]
    lines += [
        "",
        "`ml_treatment_comparison.csv` contains every completed result. `ml_treatment_paired_differences.csv` gives paired family-bootstrap differences from each pipeline's original-data arm. These are exploratory intervals without multiple-comparison correction. Read metric differences with counts, uncertainty, training cost, and label-quality limitations.",
        "",
        "All validation/test messages remain original. The existing test has been inspected, so its treatment results are exploratory rather than a fresh independent estimate. The original serving selections and artifacts remain separately frozen. No API release is changed automatically.",
        "",
    ]
    if len(test) == 60:
        frozen = read_json(ROOT / "frozen_before_test.json")
        champion = next(
            result for result in test if result["candidate"] == frozen["validation_champion"]
        )
        lines += [
            f"Validation-selected treatment champion: `{frozen['validation_champion']}`. This selection preceded treatment test prediction, but the existing test's prior inspection remains a limitation.",
            f"Its exploratory test spam F1 is {champion['metrics']['spam_f1']:.4f}, precision {champion['metrics']['precision']:.4f}, recall {champion['metrics']['recall']:.4f}.",
            "",
        ]
    Path("reports/ML_TREATMENTS.md").write_text("\n".join(lines))
    return status
