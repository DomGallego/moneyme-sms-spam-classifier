"""Pickleable text feature transforms shared by training and serving."""

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler


class SMSFeatures(BaseEstimator, TransformerMixin):
    names = (
        "characters",
        "words",
        "digit_fraction",
        "upper_fraction",
        "urls",
        "currency",
        "exclamations",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(
            [
                [
                    len(t),
                    len(t.split()),
                    sum(c.isdigit() for c in t) / max(len(t), 1),
                    sum(c.isupper() for c in t) / max(len(t), 1),
                    len(re.findall(r"https?://|www\.", t, re.IGNORECASE)),
                    len(re.findall(r"[£$€]", t)),
                    t.count("!"),
                ]
                for t in X
            ],
            dtype=np.float64,
        )


def representation(name):
    if name == "counts":
        return CountVectorizer(ngram_range=(1, 1), max_features=30000)
    if name == "word":
        return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30000)
    if name == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000)
    if name == "engineered":
        return Pipeline([("extract", SMSFeatures()), ("scale", MaxAbsScaler())])
    if name == "combined":
        return FeatureUnion([("word", representation("word")), ("char", representation("char"))])
    if name == "word_engineered":
        return FeatureUnion(
            [("word", representation("word")), ("engineered", representation("engineered"))]
        )
    raise ValueError(f"Unknown representation: {name}")
