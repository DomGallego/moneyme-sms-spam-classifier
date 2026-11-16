import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from moneyme_sms_spam_classifier.augmentation import rejection_reason
from moneyme_sms_spam_classifier.common import fingerprint
from moneyme_sms_spam_classifier.sampling import TreatmentClassifier, augmented_training


def records():
    texts = [
        f"ordinary lunch meeting {word}"
        for word in ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    ]
    texts += [f"claim free prize {word}" for word in ["one", "two", "three", "four"]]
    return pd.DataFrame(
        {
            "text": texts,
            "id": [fingerprint(text) for text in texts],
            "group": [f"family_{i}" for i in range(12)],
            "target": [0] * 8 + [1] * 4,
            "split": "train",
            "fold": [0, 1] * 6,
        }
    )


def pool_for(frame):
    return pd.DataFrame(
        [
            {
                "id": fingerprint(f"new offer {index} {parent['id']}"),
                "parent_id": parent["id"],
                "group": parent["group"],
                "target": 1,
                "text": f"new offer {index} {parent['id']}",
            }
            for parent in frame.query("target == 1").to_dict("records")
            for index in range(3)
        ]
    )


def test_augmentation_excludes_heldout_parents_and_preserves_family():
    frame = records()
    pool = pool_for(frame)
    fitting = frame[frame.fold == 0]
    balanced = augmented_training(fitting, pool)
    generated = balanced[balanced.parent_id.notna()]
    assert set(generated.parent_id) <= set(fitting.id)
    assert not set(generated.parent_id) & set(frame[frame.fold == 1].id)
    assert balanced.target.value_counts().to_dict() == {0: 4, 1: 4}
    assert generated.group.eq(generated.parent_id.map(fitting.set_index("id").group)).all()
    corrupt = pool.copy()
    corrupt.loc[corrupt.parent_id.isin(fitting.id), "group"] = "wrong"
    with pytest.raises(ValueError, match="family"):
        augmented_training(fitting, corrupt)


def test_augmentation_requires_sufficient_pool():
    frame = records()
    with pytest.raises(ValueError, match="Insufficient"):
        augmented_training(frame, pool_for(frame).head(1))


def test_smote_is_fit_only_and_vectorizer_does_not_learn_prediction_text():
    frame = records()
    model = TreatmentClassifier(
        Pipeline([("features", TfidfVectorizer()), ("classifier", LogisticRegression())]),
        treatment="smote",
        neighbors=2,
    )
    model.fit(frame.text, frame.target)
    assert model.input_rows_ == 12
    assert model.output_class_counts_ == [8, 8]
    assert len(model.predict(["heldoutonlytoken"])) == 1
    assert "heldoutonlytoken" not in model.model_.named_steps["features"].vocabulary_


def test_text_augmentation_works_inside_randomized_cv(tmp_path):
    frame = records()
    dataset, pool = tmp_path / "data.jsonl", tmp_path / "pool.jsonl"
    frame.to_json(dataset, orient="records", lines=True)
    pool_for(frame).to_json(pool, orient="records", lines=True)
    model = TreatmentClassifier(
        Pipeline([("features", TfidfVectorizer()), ("classifier", LogisticRegression())]),
        treatment="text_aug",
        dataset_path=str(dataset),
        pool_path=str(pool),
    )
    search = RandomizedSearchCV(
        model,
        {"estimator__classifier__C": [1, 10]},
        n_iter=2,
        cv=PredefinedSplit(frame.fold),
        scoring="f1",
        error_score="raise",
    )
    search.fit(frame.text.tolist(), frame.target.to_numpy())
    assert np.isfinite(search.best_score_)
    assert search.best_estimator_.output_class_counts_ == [8, 8]
    assert search.predict(["lunch tomorrow"]).shape == (1,)


def test_augmentation_integrity_filters():
    parent = {"text": "Claim a free prize by calling 123456 today", "group": "family"}
    assert rejection_reason(parent["text"], parent, {parent["text"].casefold()}, {}) == "duplicate"
    assert (
        rejection_reason("Call 999999 now to claim a free prize", parent, set(), {})
        == "new_contact_or_number"
    )
    assert rejection_reason("Call 123456 today for your free prize", parent, set(), {}) is None


def test_augmentation_parser_rejects_non_text_and_extra_fields():
    from moneyme_sms_spam_classifier.augmentation import AugmentationGenerator
    from moneyme_sms_spam_classifier.hosted import HostedError

    adapter = object.__new__(AugmentationGenerator)
    adapter.config = {"variants_per_request": 6}

    def envelope(value):
        return {"choices": [{"message": {"content": json.dumps(value)}}]}

    assert adapter.parse_response(envelope({"variants": ["hello"]})) == {"variants": ["hello"]}
    with pytest.raises(HostedError):
        adapter.parse_response(envelope({"variants": [123]}))
    with pytest.raises(HostedError):
        adapter.parse_response(envelope({"variants": [], "label": "spam"}))


def test_paired_treatment_interval_preserves_identity_and_pairing():
    from moneyme_sms_spam_classifier.treatment_reporting import paired_interval

    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "group": ["g1", "g1", "g2", "g3"],
            "target": [0, 1, 1, 0],
            "prediction": [0, 1, 0, 0],
        }
    )
    interval = paired_interval(frame, frame.copy(), repetitions=50)
    assert interval["delta_f1_ci_low"] == interval["delta_f1_ci_high"] == 0
    with pytest.raises(ValueError, match="identical"):
        paired_interval(frame, frame.iloc[::-1], repetitions=50)


def test_encoder_recovery_requires_complete_epochs_and_selects_best(monkeypatch, tmp_path):
    from moneyme_sms_spam_classifier import imbalance, semantic
    from moneyme_sms_spam_classifier.common import write_json

    monkeypatch.setattr(imbalance, "ROOT", tmp_path / "reports")
    monkeypatch.setattr(imbalance, "MODELS", tmp_path / "models")
    frame = records()
    monkeypatch.setattr(imbalance, "load", lambda: frame.assign(split="validation"))
    monkeypatch.setattr(semantic, "EncoderClassifier", lambda *_args: object())
    name = "distilbert_full_5e-05__original"
    folder = imbalance.MODELS / name
    write_json(folder / "checkpoint" / "config.json", {})
    history = [
        {"candidate": name, "epoch": i, "metrics": {"spam_f1": value}}
        for i, value in [(1, 0.8), (2, 0.9), (3, 0.85)]
    ]
    path = imbalance.ROOT / "encoder" / "original" / "history.json"
    calls = []

    def persist(candidate, model, train, val, cv, elapsed, parameters, output, kind):
        calls.append(parameters)
        assert elapsed is None  # Recovery does not invent an elapsed training time.
        (output / "model.joblib").write_bytes(b"test checkpoint reference")
        return {"candidate": candidate, "parameters": parameters}

    monkeypatch.setattr(imbalance, "persist_candidate", persist)
    write_json(path, history[:2])
    imbalance.recover_encoder_checkpoints("original", frame, None)
    assert calls == []
    write_json(path, history)
    imbalance.recover_encoder_checkpoints("original", frame, None)
    result = json.loads((folder / "result.json").read_text())
    assert result["parameters"]["best_epoch"] == 2
    assert result["treatment"] == "original"
    assert "unknown" in result["recovery_note"]


def test_frozen_augmentation_reuses_without_api_and_rejects_definition_change(
    monkeypatch, tmp_path
):
    from moneyme_sms_spam_classifier import augmentation
    from moneyme_sms_spam_classifier.common import write_json

    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("frozen instructions")
    config = {"prompt": str(prompt), "model": "test-generator"}
    write_json("configs/imbalance_experiment.json", config)
    pool = tmp_path / "pool" / "pool.jsonl"
    pool.parent.mkdir()
    pool.write_text('{"text":"cached variant"}\n')
    monkeypatch.setattr(augmentation, "POOL", pool)
    monkeypatch.setattr(augmentation, "ROOT", tmp_path / "report")
    identity = fingerprint(
        json.dumps({**config, "group": "augmentation"}, sort_keys=True) + prompt.read_text()
    )
    write_json(
        pool.parent / "frozen.json",
        {"pool_sha256": fingerprint(pool.read_bytes()), "generation_configuration_id": identity},
    )
    write_json(augmentation.ROOT / "summary.json", {"accepted": 1})

    def unexpected_call(*_args, **_kwargs):
        pytest.fail("A frozen pool must not instantiate an API adapter")

    monkeypatch.setattr(augmentation, "AugmentationGenerator", unexpected_call)
    assert augmentation.generate_pool() == {"accepted": 1}
    prompt.write_text("different instructions")
    with pytest.raises(ValueError, match="definition changed"):
        augmentation.generate_pool()
