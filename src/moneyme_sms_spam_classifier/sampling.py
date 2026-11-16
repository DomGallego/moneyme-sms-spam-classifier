"""Training-only treatments with parent-safe augmentation and fold-local SMOTE."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .augmentation import POOL
from .common import fingerprint, read_json


def augmented_training(frame, pool=None, seed=42):
    """Balance using only descendants of the supplied fitting rows; never sample originals."""
    pool = pd.read_json(POOL, lines=True) if pool is None else pool.copy()
    parents = frame.set_index("id")
    eligible = pool[pool.parent_id.isin(parents.index)].copy()
    if not eligible.empty:
        if not eligible.parent_id.map(parents.target).eq(1).all():
            raise ValueError("An augmentation parent is not spam")
        if not eligible.group.eq(eligible.parent_id.map(parents.group)).all():
            raise ValueError("An augmentation has a different family from its parent")
        if not eligible.target.eq(1).all():
            raise ValueError("An augmentation label changed")
    needed = int((frame.target == 0).sum() - (frame.target == 1).sum())
    if needed <= 0:
        return frame.copy()
    eligible = eligible.drop_duplicates("text")
    if len(eligible) < needed:
        raise ValueError(f"Insufficient eligible augmentation: need {needed}, have {len(eligible)}")
    # Round-robin parent coverage avoids letting prolific parents dominate.
    eligible = eligible.sample(frac=1, random_state=seed)
    eligible["parent_rank"] = eligible.groupby("parent_id").cumcount()
    eligible = eligible.sort_values("parent_rank", kind="stable").head(needed)
    return pd.concat([frame, eligible], ignore_index=True)


@lru_cache(maxsize=4)
def embedding_table(root):
    root = Path(root)
    identifiers = read_json(root / "ids.json")
    vectors = np.load(root / "vectors.npy", mmap_mode="r")
    return {identifier: index for index, identifier in enumerate(identifiers)}, vectors


class FrozenEmbeddingTable(TransformerMixin, BaseEstimator):
    """Precomputed frozen features; no fit on held-out labels or corpus statistics."""

    def __init__(self, root="artifacts/imbalance/embeddings"):
        self.root = root

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        indices, vectors = embedding_table(self.root)
        return np.asarray(vectors[[indices[fingerprint(text)] for text in X]])


class TreatmentClassifier(ClassifierMixin, BaseEstimator):
    """Fit-only treatment around a feature/classifier pipeline; prediction never resamples."""

    def __init__(
        self,
        estimator,
        treatment="original",
        pool_path=str(POOL),
        dataset_path="artifacts/splits/dataset.jsonl",
        neighbors=5,
        random_state=42,
    ):
        self.estimator = estimator
        self.treatment = treatment
        self.pool_path = pool_path
        self.dataset_path = dataset_path
        self.neighbors = neighbors
        self.random_state = random_state

    def fit(self, X, y):
        texts, targets = list(X), np.asarray(y)
        self.input_rows_ = len(texts)
        if self.treatment == "text_aug":
            dataset = pd.read_json(self.dataset_path, lines=True).set_index("id")
            frame = dataset.loc[[fingerprint(text) for text in texts]].reset_index()
            if not frame.split.eq("train").all() or not np.array_equal(frame.target, targets):
                raise ValueError("Augmentation fit requires matching original training rows")
            balanced = augmented_training(
                frame, pd.read_json(self.pool_path, lines=True), self.random_state
            )
            texts, targets = balanced.text.tolist(), balanced.target.to_numpy()
        self.model_ = clone(self.estimator)
        if self.treatment == "smote":
            from imblearn.over_sampling import SMOTE

            features = self.model_.named_steps["features"]
            classifier = self.model_.named_steps["classifier"]
            vectors = features.fit_transform(texts, targets)
            vectors, targets = SMOTE(
                random_state=self.random_state, k_neighbors=self.neighbors
            ).fit_resample(vectors, targets)
            classifier.fit(vectors, targets)
            self.model_ = Pipeline([("features", features), ("classifier", classifier)])
        elif self.treatment in {"original", "text_aug"}:
            self.model_.fit(texts, targets)
        else:
            raise ValueError(f"Unknown treatment: {self.treatment}")
        self.output_class_counts_ = np.bincount(targets.astype(int), minlength=2).tolist()
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.model_.predict(X)

    def decision_function(self, X):
        check_is_fitted(self)
        return self.model_.decision_function(X)
