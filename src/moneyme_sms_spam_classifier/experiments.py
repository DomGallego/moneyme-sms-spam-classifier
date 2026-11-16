"""Budget-controlled hosted experiments, resumable by complete configuration hash."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import fingerprint, manifest, read_journal, read_json, write_json
from .data import load
from .evaluation import choose_threshold, latency_summary, metrics
from .hosted import Budget, BudgetExceeded, OpenRouterClassifier


def sample_partition(df, split, limit):
    part = df[df.split == split]
    if limit and limit < len(part):
        _, indices = train_test_split(
            np.arange(len(part)), test_size=limit, stratify=part.target, random_state=42
        )
        part = part.iloc[sorted(indices)]
    return part


def configurations():
    candidates = read_json("configs/candidates.json")
    train = load().query("split == 'train'")
    examples = []
    for label in ["ham", "spam"]:
        examples.extend(
            train[train.label == label]
            .sample(2, random_state=42)[["id", "text", "label"]]
            .to_dict("records")
        )
    configs = []
    for model in candidates["llm"]:
        for mode in ["zero_shot", "few_shot"]:
            configs.append(
                {
                    "group": "llm",
                    "model": model,
                    "variant": mode,
                    "max_tokens": 2048,
                    "examples": examples if mode == "few_shot" else [],
                }
            )
    for model in candidates["decision"]:
        for primitive in ["choice", "noul"]:
            configs.append(
                {"group": "decision", "model": model, "variant": primitive, "primitive": primitive}
            )
    return configs


def configuration_id(config):
    return fingerprint(
        json.dumps(
            {
                "config": config,
                "prompt": Path("prompts/spam_zero_shot_v1.txt").read_text(),
                "criteria": read_json("configs/decision_criteria.json"),
            },
            sort_keys=True,
        )
    )[:20]


def run_hosted(stage="pilot", limit=8, cap=1.0, workers=4, selected=False):
    df = load()
    split = "test" if stage == "test" else "validation"
    part = sample_partition(df, split, limit)
    catalog = {m["id"]: m for m in read_json("reports/catalog.json")["data"]}
    configs = (
        read_json("configs/hosted_frozen.json")["configurations"]
        if selected or stage == "test"
        else configurations()
    )
    if stage == "test":
        frozen = read_json("configs/hosted_frozen.json")
        if frozen["selection_stage"] != "validation":
            raise ValueError("Freeze full validation selections before test")
        snapshot = Path("reports/frozen_before_test.json")
        if snapshot.exists() and read_json(snapshot) != frozen:
            raise ValueError("Test selections differ from the frozen snapshot")
        # A changed prompt/criteria file changes the configuration hash. Reject
        # resuming a test stage under new definitions instead of silently paying again.
        for saved in Path("reports/runs/test").glob("*/configuration.json"):
            if configuration_id(read_json(saved)) != saved.parent.name:
                raise ValueError("Prompt or criteria changed after test began")
    budget = Budget(cap=cap)
    root = Path("reports/runs") / stage
    root.mkdir(parents=True, exist_ok=True)
    run_manifest = manifest(
        stage=stage,
        split=split,
        limit=limit,
        record_ids=part.id.tolist(),
        cap_usd=cap,
        workers=workers,
        split_fingerprint=read_json("artifacts/splits/manifest.json")["dataset_sha256"],
    )
    if (root / "manifest.json").exists():
        original = read_json(root / "manifest.json")
        if (
            original["record_ids"] != run_manifest["record_ids"]
            or original["split_fingerprint"] != run_manifest["split_fingerprint"]
        ):
            raise ValueError("Cannot resume a stage with different records or splits")
        write_json(root / "resume_manifest.json", run_manifest)
    else:
        write_json(root / "manifest.json", run_manifest)
    summaries = []
    for config in configs:
        name = configuration_id(config)
        output = root / name
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "configuration.json", config)
        rows = []
        journal = output / "requests.jsonl"
        if journal.exists():
            rows = read_journal(journal)
        complete = {r["id"] for r in rows}
        targets = dict(zip(part.id, part.target))
        # Recover settled results if a restart interrupted the journal append.
        for result in budget.recover(stage, name):
            record_id = result.get("id")
            if record_id in targets and record_id not in complete and "prediction" in result:
                recovered = {
                    **result,
                    "target": int(targets[record_id]),
                    "recovered_from_ledger": True,
                }
                rows.append(recovered)
                complete.add(record_id)
                with journal.open("a") as file:
                    file.write(json.dumps(recovered, allow_nan=False) + "\n")
        pending = [row for row in part.to_dict("records") if row["id"] not in complete]
        adapter = OpenRouterClassifier(config, budget, catalog[config["model"]])

        def call(row, adapter=adapter, name=name):
            result = adapter.predict(
                row["text"],
                {"id": row["id"], "split": split, "stage": stage, "configuration_id": name},
            )
            return {**result, "id": row["id"], "target": row["target"]}

        try:
            # Fixed-size batches bound in-flight reservations and allow stopping on budget exhaustion.
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0, len(pending), workers):
                    futures = [pool.submit(call, row) for row in pending[start : start + workers]]
                    for future in futures:
                        result = future.result()
                        rows.append(result)
                        with journal.open("a") as file:
                            file.write(json.dumps(result, allow_nan=False) + "\n")
                    if (start // workers) % 10 == 0:
                        print(
                            stage,
                            config["model"],
                            config["variant"],
                            f"{len(rows)}/{len(part)}",
                            f"ledger=${budget.total():.5f}",
                            flush=True,
                        )
        except BudgetExceeded:
            write_json(
                root / "budget_stop.json",
                {"accounted_usd": budget.total(), "cap": cap, "candidate": config["model"]},
            )
            raise
        finally:
            adapter.close()
        current = [r for r in rows if r["id"] in set(part.id)]
        frame = pd.DataFrame(current)
        if len(frame) != len(part):
            raise ValueError("Incomplete candidate partition")
        score = frame.score.to_numpy(dtype=float)
        prediction = frame.prediction.to_numpy()
        thresholds = None
        if (
            config["group"] == "decision"
            and np.isfinite(score[prediction >= 0]).all()
            and (prediction >= 0).any()
        ):
            if split != "test":
                good = prediction >= 0
                thresholds = choose_threshold(frame.target[good], score[good])
                threshold = thresholds["primary"]["threshold"]
            else:
                threshold = config.get("threshold", 0.5)
            prediction = np.where(prediction >= 0, score >= threshold, -1).astype(int)
        reported_cost = frame.cost_usd.sum(min_count=1)
        result = {
            "candidate": config["model"],
            "group": config["group"],
            "variant": config["variant"],
            "configuration_id": name,
            "configuration": config,
            "thresholds": thresholds,
            "metrics": metrics(
                frame.target,
                prediction,
                score if config["group"] == "decision" else None,
                config["group"] == "decision",
            ),
            "latency": latency_summary(frame.latency_ms),
            "cost_usd": None if pd.isna(reported_cost) else float(reported_cost),
            "unknown_cost_attempts": int(frame.cost_usd.isna().sum()),
            "failures": frame[frame.prediction < 0].status.value_counts().to_dict(),
            "record_count": len(frame),
            "stage": stage,
        }
        write_json(output / "result.json", result)
        summaries.append(result)
        write_json(root / "results.json", summaries)
        print("Finished", config["model"], config["variant"], result["metrics"], flush=True)
    return summaries


def freeze_hosted(stage="development"):
    results = read_json(f"reports/runs/{stage}/results.json")
    configs = []
    for model in (
        read_json("configs/candidates.json")["llm"]
        + read_json("configs/candidates.json")["decision"]
    ):
        evaluated = [
            r for r in results if r["candidate"] == model and r["metrics"]["completed"] > 0
        ]
        valid = [r for r in evaluated if r["metrics"]["coverage"] >= 0.95]
        if not valid:
            # Keep operationally unreliable candidates in the scientific comparison;
            # exclude them from deployment champion eligibility below.
            if not evaluated:
                raise ValueError(f"No successful predictions for {model}; investigate access")
            valid = evaluated
        best = max(
            valid,
            key=lambda r: (
                r["metrics"].get("spam_f1", 0),
                r["metrics"]["coverage"],
                -r["latency"]["p50_ms"],
            ),
        )
        config = best["configuration"]
        if best["thresholds"]:
            config["threshold"] = best["thresholds"]["primary"]["threshold"]
        configs.append(config)
    champions = {}
    for group in ["llm", "decision"]:
        eligible = [r for r in results if r["group"] == group and r["metrics"]["coverage"] >= 0.95]
        best = max(
            eligible, key=lambda r: (r["metrics"].get("spam_f1", 0), -r["latency"]["p50_ms"])
        )
        champions[group] = next(c for c in configs if c["model"] == best["candidate"])
    frozen = {
        "configurations": configs,
        "champions": champions,
        "selection_stage": stage,
        "selection_basis": "validation F1 with >=95% coverage; latency tie break",
    }
    write_json("configs/hosted_frozen.json", frozen)
    return frozen
