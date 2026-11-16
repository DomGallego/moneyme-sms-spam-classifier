import pytest

from moneyme_sms_spam_classifier.evaluation import choose_threshold, metrics


def test_failures_not_hidden_and_dummy_spam_f1_zero():
    result = metrics([0, 1, 1], [0, -1, 1])
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["accuracy"] == 1
    assert result["end_to_end_accuracy"] == pytest.approx(2 / 3)
    assert metrics([0, 1], [0, 0])["spam_f1"] == 0


def test_high_precision_target_can_be_unattainable():
    thresholds = choose_threshold([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])
    assert thresholds["high_precision"] is None
