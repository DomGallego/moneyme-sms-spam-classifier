"""Run reproducible, resumable ML training treatments; hosted classifiers are untouched."""

import argparse

from moneyme_sms_spam_classifier.imbalance import (
    assert_pool_ready,
    evaluate_treatments,
    freeze_treatments,
    prepare_embedding_table,
    train_encoders,
    train_numeric,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["embeddings", "numeric", "encoders", "freeze", "evaluate"])
parser.add_argument("--treatments", nargs="+", default=None)
parser.add_argument("--only", nargs="+", default=None)
args = parser.parse_args()
if args.stage == "embeddings":
    assert_pool_ready()
    prepare_embedding_table()
elif args.stage == "numeric":
    train_numeric(treatments=args.treatments or ("original", "text_aug", "smote"), only=args.only)
elif args.stage == "encoders":
    train_encoders(treatments=args.treatments or ("original", "text_aug"))
elif args.stage == "freeze":
    freeze_treatments()
else:
    evaluate_treatments()
