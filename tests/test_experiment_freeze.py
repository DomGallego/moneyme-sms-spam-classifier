"""Scientific/budget guards must fail before constructing a paid provider client."""

import pandas as pd
import pytest

from moneyme_sms_spam_classifier import experiments
from moneyme_sms_spam_classifier.common import write_json


def setup_files(tmp_path, monkeypatch, stage="validation"):
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "prompts/spam_zero_shot_v1.txt"
    prompt.parent.mkdir()
    prompt.write_text("Original classification definition")
    write_json("configs/decision_criteria.json", {"spam": "unsolicited", "ham": "legitimate"})
    config = {"group": "llm", "model": "unit/model", "variant": "zero_shot"}
    frozen = {"selection_stage": stage, "configurations": [config], "champions": {}}
    write_json("configs/hosted_frozen.json", frozen)
    write_json("reports/catalog.json", {"data": [{"id": "unit/model"}]})
    frame = pd.DataFrame(
        {"id": ["a", "b"], "text": ["hello", "prize"], "target": [0, 1], "split": ["test", "test"]}
    )
    monkeypatch.setattr(experiments, "load", lambda: frame)

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not construct provider before freeze validation")

    monkeypatch.setattr(experiments, "OpenRouterClassifier", forbidden)
    return config, prompt


def test_test_phase_requires_full_validation_freeze(tmp_path, monkeypatch):
    setup_files(tmp_path, monkeypatch, stage="development")
    with pytest.raises(ValueError, match="full validation"):
        experiments.run_hosted(stage="test", limit=0, cap=1)
    assert not (tmp_path / "reports/budget.sqlite").exists()


def test_resuming_test_rejects_changed_prompt_before_paid_calls(tmp_path, monkeypatch):
    config, prompt = setup_files(tmp_path, monkeypatch)
    identity = experiments.configuration_id(config)
    write_json(f"reports/runs/test/{identity}/configuration.json", config)
    prompt.write_text("Changed classification definition")
    with pytest.raises(ValueError, match="Prompt or criteria changed"):
        experiments.run_hosted(stage="test", limit=0, cap=1)
    assert not (tmp_path / "reports/budget.sqlite").exists()
