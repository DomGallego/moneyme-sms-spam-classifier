"""Unified CLI; shared implementation with notebook and HTTP service."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--data", type=Path, default=Path("data/sms+spam+collection/SMSSpamCollection")
    )
    train = commands.add_parser("train")
    train.add_argument("--trials", type=int, default=20)
    train.add_argument("--jobs", type=int, default=2)
    train.add_argument("--only", nargs="+")
    commands.add_parser("embeddings")
    commands.add_parser("encoder")
    commands.add_parser("catalog")
    commands.add_parser("freeze-local")
    commands.add_parser("freeze-hosted")
    hosted = commands.add_parser("hosted")
    hosted.add_argument(
        "--stage", choices=["pilot", "development", "validation", "test"], default="pilot"
    )
    hosted.add_argument("--limit", type=int, default=8, help="0 means full partition")
    hosted.add_argument("--cap", type=float, required=True, help="Cumulative cap across runs, USD")
    hosted.add_argument("--workers", type=int, default=4)
    hosted.add_argument("--selected", action="store_true")
    predict = commands.add_parser("predict")
    predict.add_argument("--model", choices=["ml", "llm", "decision"], default="ml")
    predict.add_argument("--text", required=True)
    predict.add_argument("--cap", type=float, default=0)
    commands.add_parser("report")
    commands.add_parser("finalize")
    augment = commands.add_parser("augment")
    augment.add_argument("--limit", type=int, default=12)
    augment.add_argument("--rounds", type=int, default=1)
    treatments = commands.add_parser("treatments")
    treatments.add_argument(
        "stage", choices=["embeddings", "numeric", "encoders", "freeze", "evaluate", "report"]
    )
    treatments.add_argument("--treatments", nargs="+")
    args = parser.parse_args()
    result = None
    if args.command == "prepare":
        from .data import prepare

        result = prepare(args.data)
    elif args.command == "train":
        from .training import train_classical

        train_classical(args.trials, args.jobs, args.only)
    elif args.command == "embeddings":
        from .semantic import train_embeddings

        train_embeddings()
    elif args.command == "encoder":
        from .semantic import train_encoder

        train_encoder()
    elif args.command == "catalog":
        from .hosted import fetch_catalog

        result = fetch_catalog()
    elif args.command == "freeze-local":
        from .training import freeze_local

        result = freeze_local()
    elif args.command == "freeze-hosted":
        from .experiments import freeze_hosted

        result = freeze_hosted()
    elif args.command == "hosted":
        from .experiments import run_hosted

        run_hosted(args.stage, args.limit, args.cap, args.workers, args.selected)
    elif args.command == "predict":
        from .inference import LocalClassifier, hosted_classifier, hosted_label

        if not args.text.strip() or len(args.text) > 5000:
            parser.error("text must contain 1–5000 nonempty characters")
        if args.model == "ml":
            label = LocalClassifier().predict(args.text)
        else:
            adapter = hosted_classifier(args.model, args.cap)
            try:
                label = hosted_label(adapter, args.text)
            finally:
                adapter.close()
        result = {"label": label}
    elif args.command == "report":
        from .reporting import generate

        generate()
    elif args.command == "finalize":
        from .reporting import finalize_local

        finalize_local()
    elif args.command == "augment":
        from .augmentation import generate_pool

        result = generate_pool(args.limit, args.rounds)
    elif args.command == "treatments":
        from .imbalance import (
            evaluate_treatments,
            freeze_treatments,
            prepare_embedding_table,
            train_encoders,
            train_numeric,
        )

        if args.stage == "embeddings":
            prepare_embedding_table()
        elif args.stage == "numeric":
            train_numeric(args.treatments or ("original", "text_aug", "smote"))
        elif args.stage == "encoders":
            train_encoders(args.treatments or ("original", "text_aug"))
        elif args.stage == "freeze":
            result = freeze_treatments()
        elif args.stage == "evaluate":
            evaluate_treatments()
        else:
            from .treatment_reporting import generate_treatment_report

            result = generate_treatment_report()
    if result is not None:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
