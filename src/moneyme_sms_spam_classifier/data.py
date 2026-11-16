"""Strict parsing and conservative template grouping; no label-based grouping."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .common import fingerprint, write_json

DEFAULT_DATA = Path("data/sms+spam+collection/SMSSpamCollection")
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def parse(path: Path) -> pd.DataFrame:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"Empty record at line {line_no}")
        parts = line.split("\t", 1)
        if len(parts) != 2 or parts[0] not in {"spam", "ham"} or not parts[1].strip():
            raise ValueError(f"Invalid record at line {line_no}")
        label, text = parts
        records.append(
            {
                "id": fingerprint(text),
                "source_line": line_no,
                "label": label,
                "target": int(label == "spam"),
                "text": text,
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Dataset is empty")
    conflicts = df.groupby("text").label.nunique().gt(1)
    if conflicts.any():
        raise ValueError(f"Conflicting labels for {int(conflicts.sum())} exact texts")
    return df


def template_key(text: str) -> str:
    # Short generic conversational messages remain independent. Long messages must
    # retain >=8 lexical words; only URLs and digit sequences are masked.
    normalized = " ".join(text.casefold().split())
    if len(normalized) >= 50 and len(re.findall(r"[a-z]+", normalized)) >= 8:
        normalized = URL.sub("<url>", normalized)
        normalized = re.sub(r"\d+", "<number>", normalized)
    return fingerprint(normalized)


def prepare(path: Path = DEFAULT_DATA, output: Path = Path("artifacts/splits"), seed=42):
    raw = parse(path)
    df = raw.drop_duplicates("text").copy().reset_index(drop=True)
    df["group"] = df.text.map(template_key)
    # One 5-fold grouped partition produces train/validation/test at ~60/20/20.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    df["split"] = "train"
    partitions = list(splitter.split(df, df.target, df.group))
    df.loc[partitions[0][1], "split"] = "test"
    df.loc[partitions[1][1], "split"] = "validation"
    df["fold"] = -1
    train = df[df.split == "train"]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    for fold, (_, idx) in enumerate(cv.split(train, train.target, train.group)):
        df.loc[train.index[idx], "fold"] = fold
    output.mkdir(parents=True, exist_ok=True)
    df.to_json(output / "dataset.jsonl", orient="records", lines=True, force_ascii=False)
    # Reviewable template groups, including single-source labels for human inspection.
    families = df[df.group.duplicated(keep=False)].sort_values("group")
    families[["id", "group", "label", "text", "split"]].to_csv(output / "families.csv", index=False)
    summary = {
        "raw_sha256": fingerprint(path.read_bytes()),
        "seed": seed,
        "raw_rows": len(raw),
        "raw_labels": raw.label.value_counts().to_dict(),
        "unique_rows": len(df),
        "duplicates_removed": len(raw) - len(df),
        "groups": df.group.nunique(),
        "grouped_rows": len(families),
        "grouping_rule": "casefold/whitespace; long >=8-word messages mask URLs/digits",
        "split_counts": pd.crosstab(df.split, df.label).to_dict(orient="index"),
        "dataset_sha256": fingerprint((output / "dataset.jsonl").read_bytes()),
    }
    write_json(output / "manifest.json", summary)
    assert_disjoint(df)
    return summary


def assert_disjoint(df: pd.DataFrame):
    if df.groupby("group").split.nunique().max() != 1:
        raise ValueError("Family leakage across partitions")
    train = df[df.split == "train"]
    if train.groupby("group").fold.nunique().max() != 1:
        raise ValueError("Family leakage across CV folds")
    for _, part in df.groupby("split"):
        if set(part.target) != {0, 1}:
            raise ValueError("Every split must contain both labels")
    if not np.isfinite(df.target).all():
        raise ValueError("Invalid targets")


def load(path: Path = Path("artifacts/splits")) -> pd.DataFrame:
    from .common import read_json

    expected = read_json(path / "manifest.json")["dataset_sha256"]
    if fingerprint((path / "dataset.jsonl").read_bytes()) != expected:
        raise ValueError("Prepared dataset fingerprint mismatch; rerun preparation intentionally")
    df = pd.read_json(path / "dataset.jsonl", lines=True)
    assert_disjoint(df)
    return df
