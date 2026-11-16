"""Equal-budget ML treatment experiments, frozen before exploratory test comparison."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .augmentation import POOL
from .common import fingerprint, manifest, read_json, write_json
from .data import load
from .evaluation import latency_summary, metrics, score_model
from .features import representation
from .sampling import FrozenEmbeddingTable, TreatmentClassifier, augmented_training
from .training import EXPERIMENTS, classifier, persist_candidate, search_space

ROOT = Path("reports/runs/imbalance")
MODELS = Path("artifacts/imbalance/models")
EMBEDDINGS = Path("artifacts/imbalance/embeddings")


def initialize():
    config = read_json("configs/imbalance_experiment.json")
    if config["cv_folds"] != 5 or config["sampling_strategy"] != 1.0:
        raise ValueError("This protocol requires five fixed folds and equal fitted class counts")
    identity = {
        "config": config,
        "dataset_sha256": read_json("artifacts/splits/manifest.json")["dataset_sha256"],
    }
    path = ROOT / "manifest.json"
    if path.exists():
        if read_json(path)["definition"] != identity:
            raise ValueError("Treatment experiment definition changed; archive before restarting")
    else:
        write_json(
            path,
            manifest(
                definition=identity,
                release_selection=read_json("configs/local_selection.json"),
                evaluation_scope=config["evaluation_scope"],
                baseline_protocol="Equal 20-trial searches for all numeric treatments; historical refinements are separate evidence.",
            ),
        )
    return config


def prepare_embedding_table():
    import torch
    from sentence_transformers import SentenceTransformer

    from .semantic import EMBEDDER

    definition = {
        "dataset": read_json("artifacts/splits/manifest.json")["dataset_sha256"],
        "pool": fingerprint(POOL.read_bytes()),
    }
    path = EMBEDDINGS / "manifest.json"
    if path.exists():
        if read_json(path)["definition"] != definition:
            raise ValueError("Frozen embedding table inputs changed")
        return
    df = pd.concat([load(), pd.read_json(POOL, lines=True)], ignore_index=True).drop_duplicates(
        "id"
    )
    revision = read_json("configs/encoder_revisions.json")[EMBEDDER]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(revision["path"], device=device, local_files_only=True)
    values = encoder.encode(
        df.text.tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS / "vectors.npy", values)
    write_json(EMBEDDINGS / "ids.json", df.id.tolist())
    write_json(
        path,
        manifest(
            definition=definition,
            revision=revision["revision"],
            device=device,
            note="Frozen encoder only; no fit or normalization learned from evaluation corpus/labels.",
        ),
    )


def assert_pool_ready():
    pool_sha = fingerprint(POOL.read_bytes())
    frozen = POOL.parent / "frozen.json"
    if frozen.exists() and read_json(frozen)["pool_sha256"] != pool_sha:
        raise ValueError("Frozen augmentation pool changed")
    train = load().query("split == 'train'")
    pool = pd.read_json(POOL, lines=True)
    pools = [("full", train)] + [(f"fold_{fold}", train[train.fold != fold]) for fold in range(5)]
    evidence = {}
    membership = {}
    for name, frame in pools:
        balanced = augmented_training(frame, pool)
        counts = balanced.target.value_counts().to_dict()
        if counts[0] != counts[1]:
            raise ValueError("Augmentation balancing failed")
        evidence[name] = {
            "original_rows": len(frame),
            "fitted_rows": len(balanced),
            "class_counts": counts,
        }
        membership[name] = {
            "original_ids": frame.id.tolist(),
            "generated_ids": balanced.loc[balanced.parent_id.notna(), "id"].tolist(),
            "generated_parent_ids": balanced.loc[balanced.parent_id.notna(), "parent_id"].tolist(),
        }
    write_json(ROOT / "augmentation_fit_counts.json", evidence)
    write_json(ROOT / "augmentation_fit_membership.json", membership)
    return pool_sha


def train_numeric(treatments=("original", "text_aug", "smote"), only=None):
    from .semantic import EMBEDDER, EmbeddingClassifier

    config = initialize()
    df = load()
    train, val = df.query("split == 'train'"), df.query("split == 'validation'")
    candidates = EXPERIMENTS + [("logistic", "embeddings"), ("xgboost", "embeddings")]
    for treatment in treatments:
        if treatment not in config["treatments"]:
            raise ValueError("Unknown treatment")
        pool_sha = assert_pool_ready() if treatment == "text_aug" else None
        for algorithm, rep in candidates:
            base = f"{algorithm}_{rep}"
            name = f"{base}__{treatment}"
            if only and base not in only:
                continue
            dest = MODELS / name
            definition = {
                "treatment": treatment,
                "base_candidate": base,
                "pool_sha256": pool_sha,
                "trials": config["search_trials"],
                "seed": config["seed"],
            }
            if (dest / "result.json").exists() and (dest / "treatment.json").exists():
                if read_json(dest / "treatment.json") != definition:
                    raise ValueError(f"Changed cached treatment: {name}")
                continue
            estimator, parameters = classifier(algorithm)
            if rep == "embeddings":
                if not (EMBEDDINGS / "manifest.json").exists():
                    prepare_embedding_table()
                features = FrozenEmbeddingTable()
                space = {f"classifier__{key}": value for key, value in parameters.items()}
            else:
                features, space = representation(rep), search_space(algorithm, rep)
            pipeline = Pipeline([("features", features), ("classifier", estimator)])
            wrapped = TreatmentClassifier(
                pipeline, treatment=treatment, neighbors=config["smote_neighbors"]
            )
            search = RandomizedSearchCV(
                wrapped,
                {f"estimator__{key}": values for key, values in space.items()},
                n_iter=config["search_trials"],
                scoring="f1",
                cv=PredefinedSplit(train.fold.to_numpy()),
                n_jobs=config["jobs"],
                random_state=config["seed"],
                error_score="raise",
                return_train_score=True,
            )
            write_json(
                dest / "manifest.json",
                manifest(
                    definition=definition,
                    space=space,
                    training_ids=train.id.tolist(),
                    validation_ids=val.id.tolist(),
                ),
            )
            print(f"Training {name}: {config['search_trials']} trials × five folds", flush=True)
            started = time.perf_counter()
            search.fit(train.text.tolist(), train.target.to_numpy())
            fitted = search.best_estimator_.model_
            if rep == "embeddings":
                revision = read_json("configs/encoder_revisions.json")[EMBEDDER]
                fitted = EmbeddingClassifier(
                    fitted.named_steps["classifier"], revision["path"], revision["revision"]
                )
            result = persist_candidate(
                name,
                fitted,
                train,
                val,
                search.best_score_,
                time.perf_counter() - started,
                {
                    key.removeprefix("estimator__"): value
                    for key, value in search.best_params_.items()
                },
                dest,
                kind="frozen_embeddings" if rep == "embeddings" else "classical",
            )
            result.update(
                base_candidate=base,
                treatment=treatment,
                fitted_class_counts=search.best_estimator_.output_class_counts_,
                model_sha256=fingerprint((dest / "model.joblib").read_bytes()),
            )
            write_json(dest / "result.json", result)
            write_json(dest / "treatment.json", definition)
            pd.DataFrame(search.cv_results_).to_csv(dest / "search_results.csv", index=False)
            print(name, "validation F1", result["validation"]["spam_f1"], flush=True)
            refresh_validation()


def train_encoders(treatments=("original", "text_aug")):
    import shutil

    from .semantic import train_encoder

    initialize()
    train = load().query("split == 'train'")
    runtime_path = Path("configs/encoder_runtime.json")
    cpu_settings = set(read_json(runtime_path)["cpu_settings"]) if runtime_path.exists() else set()
    settings = [("head", 1e-3), ("full", 2e-5), ("full", 5e-5)]
    for treatment in treatments:
        if treatment not in {"original", "text_aug"}:
            raise ValueError("DistilBERT does not accept ordinary SMOTE")
        pool_sha = assert_pool_ready() if treatment == "text_aug" else None
        suffix = f"__{treatment}"
        names = [
            f"distilbert_{mode}_{rate:g}{suffix}"
            for mode, rate in [("head", 1e-3), ("full", 2e-5), ("full", 5e-5)]
        ]
        frame = augmented_training(train) if treatment == "text_aug" else train
        recover_encoder_checkpoints(treatment, frame, pool_sha)
        for name in names:
            folder = MODELS / name
            if name.removesuffix(suffix) not in cpu_settings or not folder.exists():
                continue
            device_path = folder / "training_manifest.json"
            recorded_device = read_json(device_path).get("device") if device_path.exists() else None
            if recorded_device != "cpu":
                archive = MODELS.parent / "gpu_archive" / name
                if archive.exists():
                    raise ValueError(f"Encoder archive already exists: {archive}")
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(folder), str(archive))
                history = ROOT / "encoder" / treatment / "history.json"
                if history.exists():
                    shutil.copy2(history, history.with_name("gpu_history.json"))
        for requested_device in ["auto", "cpu"]:
            missing = [
                (mode, rate)
                for mode, rate in settings
                if ((f"distilbert_{mode}_{rate:g}" in cpu_settings) == (requested_device == "cpu"))
                and not (MODELS / f"distilbert_{mode}_{rate:g}{suffix}" / "result.json").exists()
            ]
            if missing:
                train_encoder(
                    training_frame=frame,
                    report_root=str(ROOT / "encoder" / treatment),
                    model_root=str(MODELS),
                    suffix=suffix,
                    settings=missing,
                    training_device=requested_device,
                    resume=True,
                )
        for name in names:
            path = MODELS / name / "result.json"
            result = read_json(path)
            result.update(
                base_candidate=name.removesuffix(suffix),
                treatment=treatment,
                fitted_class_counts=np.bincount(frame.target, minlength=2).tolist(),
                pool_sha256=pool_sha,
                model_sha256=fingerprint((path.parent / "model.joblib").read_bytes()),
            )
            write_json(path, result)
        refresh_validation()


def recover_encoder_checkpoints(treatment, frame, pool_sha):
    """Recover fully trained settings if a driver stalls after checkpoint persistence."""
    from .semantic import ENCODER, EncoderClassifier

    history_path = ROOT / "encoder" / treatment / "history.json"
    if not history_path.exists():
        return
    history = read_json(history_path)
    revision = read_json("configs/encoder_revisions.json")[ENCODER]
    val = load().query("split == 'validation'")
    for mode, rate in [("head", 1e-3), ("full", 2e-5), ("full", 5e-5)]:
        base = f"distilbert_{mode}_{rate:g}"
        name = f"{base}__{treatment}"
        folder = MODELS / name
        checkpoint = folder / "checkpoint"
        records = [row for row in history if row["candidate"] == name]
        if {row["epoch"] for row in records} != {1, 2, 3} or not (
            checkpoint / "config.json"
        ).exists():
            continue
        path = folder / "result.json"
        if path.exists():
            result = read_json(path)
            if (
                "treatment" in result
                and result.get("training_device")
                and result.get("training_precision")
            ):
                continue
        else:
            best = max(records, key=lambda row: row["metrics"]["spam_f1"])
            result = persist_candidate(
                name,
                EncoderClassifier(str(checkpoint), 128),
                frame,
                val,
                None,
                None,
                {
                    "learning_rate": rate,
                    "mode": mode,
                    "best_epoch": best["epoch"],
                    "epochs": 3,
                    "revision": revision["revision"],
                },
                folder,
                kind="encoder",
            )
            result["recovery_note"] = (
                "All three training epochs and best checkpoint persisted; post-training driver stall prevented wall-time persistence. Training time is unknown, not zero."
            )
        result.update(
            base_candidate=base,
            treatment=treatment,
            fitted_class_counts=np.bincount(frame.target, minlength=2).tolist(),
            pool_sha256=pool_sha,
            model_sha256=fingerprint((folder / "model.joblib").read_bytes()),
        )
        device_path = folder / "training_manifest.json"
        run_path = ROOT / "encoder" / treatment / "manifest.json"
        device = (
            read_json(device_path).get("device")
            if device_path.exists()
            else read_json(run_path).get("device")
            if run_path.exists()
            else "unknown"
        )
        result.setdefault("training_device", device)
        result.setdefault(
            "training_precision",
            "autocast_float16" if device == "cuda" else "float32" if device == "cpu" else None,
        )
        write_json(path, result)


def refresh_validation():
    results = [read_json(path) for path in sorted(MODELS.glob("*/result.json"))]
    results = [result for result in results if "base_candidate" in result and "treatment" in result]
    for result in results:
        if result["kind"] != "encoder":
            continue
        folder = MODELS / result["candidate"]
        path = folder / "training_manifest.json"
        if not path.exists():
            path = ROOT / "encoder" / result["treatment"] / "manifest.json"
        if path.exists():
            execution = read_json(path)
            device = execution.get("device")
            if not result.get("training_device"):
                result["training_device"] = device
            if not result.get("training_precision"):
                result["training_precision"] = "autocast_float16" if device == "cuda" else "float32"
            result.setdefault("dynamic_padding", execution.get("dynamic_padding", False))
            result.setdefault(
                "length_grouped_batches", execution.get("length_grouped_batches", False)
            )
            write_json(folder / "result.json", result)
    write_json(ROOT / "validation_results.json", results)
    return results


def audit_search_budgets(results, trials):
    """Require identical sampled settings and complete CV evidence across treatments."""
    comparisons, score_entries = {}, 0
    numeric = [result for result in results if result["kind"] != "encoder"]
    for result in numeric:
        history = pd.read_csv(MODELS / result["candidate"] / "search_results.csv")
        columns = [f"split{fold}_test_score" for fold in range(5)]
        if len(history) != trials or not np.isfinite(history[columns].to_numpy()).all():
            raise ValueError(f"Incomplete CV history: {result['candidate']}")
        score_entries += len(history) * len(columns)
        comparisons.setdefault(result["base_candidate"], {})[result["treatment"]] = (
            history.params.tolist()
        )
        expected_counts = [2710, 392] if result["treatment"] == "original" else [2710, 2710]
        if result["fitted_class_counts"] != expected_counts:
            raise ValueError(f"Unexpected fitting class counts: {result['candidate']}")
    for base, arms in comparisons.items():
        if set(arms) != {"original", "text_aug", "smote"} or not (
            arms["original"] == arms["text_aug"] == arms["smote"]
        ):
            raise ValueError(f"Unmatched treatment search budgets/settings: {base}")
    evidence = {
        "numeric_configurations": len(numeric),
        "base_pipelines": len(comparisons),
        "trials_per_configuration": trials,
        "folds": 5,
        "finite_cv_score_entries": score_entries,
        "identical_sampled_settings_across_treatments": True,
    }
    write_json(ROOT / "search_budget_audit.json", evidence)
    return evidence


def freeze_treatments():
    config = initialize()
    results = refresh_validation()
    if len(results) != 60:
        raise ValueError(f"Require all 60 configurations before freezing; have {len(results)}")
    audit_search_budgets(results, config["search_trials"])
    encoder_pairs = {}
    for result in results:
        if result["kind"] == "encoder":
            policy = {
                "device": result.get("training_device"),
                "precision": result.get("training_precision"),
                "dynamic_padding": result.get("dynamic_padding", False),
                "length_grouped_batches": result.get("length_grouped_batches", False),
            }
            encoder_pairs.setdefault(result["base_candidate"], {})[result["treatment"]] = policy
    for name, arms in encoder_pairs.items():
        if set(arms) != {"original", "text_aug"} or arms["original"] != arms["text_aug"]:
            raise ValueError(f"Unmatched encoder execution policy: {name}")
    write_json(ROOT / "encoder_pair_audit.json", encoder_pairs)
    frozen = {
        "config": config,
        "pool_sha256": assert_pool_ready(),
        "evaluation_scope": config["evaluation_scope"],
        "configurations": [],
        "release_selection": read_json("configs/local_selection.json"),
        "encoder_runtime": read_json("configs/encoder_runtime.json")
        if Path("configs/encoder_runtime.json").exists()
        else {},
        "inference_policy": {
            "neural_bulk_metric_threads": 6,
            "warm_latency_threads": 1,
            "latency_samples": 100,
            "latency_concurrency": 1,
        },
    }
    for result in results:
        path = MODELS / result["candidate"] / "model.joblib"
        frozen["configurations"].append(
            {
                "candidate": result["candidate"],
                "sha256": fingerprint(path.read_bytes()),
                "threshold": result["thresholds"]["primary"]["threshold"],
                "treatment": result["treatment"],
                "training_device": result.get("training_device", "cpu"),
                "training_precision": result.get("training_precision"),
                "dynamic_padding": result.get("dynamic_padding", False),
                "length_grouped_batches": result.get("length_grouped_batches", False),
            }
        )
        if result["kind"] == "encoder":
            frozen["configurations"][-1]["checkpoint_files"] = {
                str(asset): fingerprint(asset.read_bytes())
                for asset in sorted((path.parent / "checkpoint").glob("*"))
                if asset.is_file()
            }
    best = max(results, key=lambda r: (r["validation"]["spam_f1"], r["validation"]["precision"]))
    frozen["validation_champion"] = best["candidate"]
    path = ROOT / "frozen_before_test.json"
    if path.exists():
        if read_json(path) != frozen:
            raise ValueError("Frozen treatment experiment changed")
    else:
        write_json(path, frozen)
    return frozen


def evaluate_treatments():
    from .reporting import bootstrap_f1

    frozen = freeze_treatments()
    df = load().query("split == 'test'")
    path = ROOT / "test_results.json"
    results = read_json(path) if path.exists() else []
    done = {row["candidate"] for row in results}
    for item in frozen["configurations"]:
        name = item["candidate"]
        model_path = MODELS / name / "model.joblib"
        if fingerprint(model_path.read_bytes()) != item["sha256"]:
            raise ValueError(f"Frozen model changed: {name}")
        for asset, expected in item.get("checkpoint_files", {}).items():
            if fingerprint(Path(asset).read_bytes()) != expected:
                raise ValueError(f"Frozen checkpoint changed: {asset}")
        if name in done:
            continue
        development = read_json(model_path.parent / "result.json")
        bundle = joblib.load(model_path)
        neural = development["kind"] in {"encoder", "frozen_embeddings"}
        if neural:
            import torch

            torch.set_num_threads(6)
        scores, probability = score_model(bundle["model"], df.text.tolist())
        prediction = (scores >= item["threshold"]).astype(int)
        predictions = pd.DataFrame(
            {
                "id": df.id,
                "group": df.group,
                "target": df.target,
                "prediction": prediction,
                "score": scores,
            }
        )
        output = ROOT / "test" / name
        output.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output / "predictions.csv", index=False)
        timings = []
        if neural:
            torch.set_num_threads(1)
        score_model(bundle["model"], [df.text.iloc[0]])
        for text in df.text.iloc[:100]:
            start = time.perf_counter()
            score_model(bundle["model"], [text])
            timings.append((time.perf_counter() - start) * 1000)
        result = {
            "candidate": name,
            "base_candidate": development["base_candidate"],
            "group": "ml",
            "kind": development["kind"],
            "training_device": development.get("training_device", "cpu"),
            "training_precision": development.get("training_precision"),
            "training_seconds": development.get("training_seconds"),
            "treatment": item["treatment"],
            "variant": item["treatment"],
            "stage": "test",
            "evaluation_scope": frozen["evaluation_scope"],
            "metrics": metrics(df.target, prediction, scores, probability),
            "latency": latency_summary(timings),
            "bulk_metric_threads": 6 if neural else 1,
            "warm_latency_threads": 1,
            "uncertainty": bootstrap_f1(predictions),
            "threshold": item["threshold"],
            "artifact_sha256": item["sha256"],
            "cost_usd": None,
        }
        write_json(output / "result.json", result)
        results.append(result)
        write_json(path, results)
        print("Exploratory test", name, result["metrics"]["spam_f1"], flush=True)
    return results
