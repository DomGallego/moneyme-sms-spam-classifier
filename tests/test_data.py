from pathlib import Path

import pytest

from moneyme_sms_spam_classifier.data import assert_disjoint, load, parse, prepare, template_key


def test_parser_first_tab_and_conflicting_labels(tmp_path):
    source = tmp_path / "sms"
    source.write_text("ham\thello\tthere\nspam\tclaim prize\n")
    assert parse(source).iloc[0].text == "hello\tthere"
    source.write_text("ham\tsame\nspam\tsame\n")
    with pytest.raises(ValueError, match="Conflicting"):
        parse(source)


def test_family_rules_are_label_independent():
    first = "Congratulations you have won a special prize today call 01234567890"
    second = "Congratulations you have won a special prize today call 09876543210"
    assert template_key(first) == template_key(second)
    assert template_key("meet at 5") != template_key("meet at 6")


def test_real_splits_no_leakage_and_reproducible(tmp_path):
    data = Path("data/sms+spam+collection/SMSSpamCollection")
    if not data.exists():
        pytest.skip("Original dataset not distributed in CI")
    a, b = tmp_path / "a", tmp_path / "b"
    assert prepare(data, a) == prepare(data, b)
    frame = load(a)
    assert len(frame) == 5171
    assert_disjoint(frame)
    (a / "dataset.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="fingerprint"):
        load(a)


def test_persisted_release_splits_are_disjoint():
    frame = load()
    assert_disjoint(frame)
    assert len(frame) == 5171
    assert set(frame.query("split == 'train'").fold) == {0, 1, 2, 3, 4}
    assert frame.id.is_unique
