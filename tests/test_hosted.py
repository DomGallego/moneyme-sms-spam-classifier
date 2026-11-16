import concurrent.futures

import httpx
import pytest

from moneyme_sms_spam_classifier.hosted import (
    Budget,
    BudgetExceeded,
    HostedError,
    OpenRouterClassifier,
    parse_prediction,
)


def test_budget_atomic_reservations_unknown_cost_and_restart(tmp_path):
    path = tmp_path / "budget.sqlite"
    budget = Budget(path, cap=1)

    def reserve(_):
        try:
            return budget.reserve(0.4, {})
        except BudgetExceeded:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        keys = [k for k in pool.map(reserve, range(4)) if k]
    assert len(keys) == 2
    assert Budget(path, 1).total() == pytest.approx(0.8)
    budget.settle(keys[0], None, {})
    assert budget.total() == pytest.approx(0.8)
    budget.settle(keys[1], 0.1, {})
    assert budget.total() == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded):
        budget.reserve(0.6, {})


def test_strict_label_and_probability_parsing():
    with pytest.raises(HostedError):
        parse_prediction(
            {"choices": [{"message": {"content": '{"label":"spam","explanation":"x"}'}}]}, "llm"
        )
    with pytest.raises(HostedError):
        parse_prediction(
            {"answers": {"classification": {"noul": float("nan")}}}, "decision", "noul"
        )
    assert (
        parse_prediction({"answers": {"classification": {"noul": 0.8}}}, "decision", "noul")[
            "prediction"
        ]
        == 1
    )


def test_invalid_output_is_billed_but_not_ham(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder")
    body = {"choices": [{"message": {"content": "not valid JSON"}}], "usage": {"cost": 0.001}}
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))
    budget = Budget(tmp_path / "budget.sqlite", 1)
    adapter = OpenRouterClassifier(
        {"group": "llm", "model": "test"},
        budget,
        {"pricing": {"prompt": "0.000001", "completion": "0.000001"}},
        client,
    )
    result = adapter.predict("a message")
    assert result["prediction"] == -1
    assert result["status"] == "invalid_output"
    assert budget.total() == pytest.approx(0.001)


def test_timeout_keeps_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder")

    def fail(request):
        raise httpx.ReadTimeout("test", request=request)

    budget = Budget(tmp_path / "budget.sqlite", 1)
    adapter = OpenRouterClassifier(
        {"group": "decision", "model": "test"},
        budget,
        {"pricing": {"prompt": "0.000001", "completion": "0"}},
        httpx.Client(transport=httpx.MockTransport(fail)),
    )
    result = adapter.predict("a message")
    assert result["status"] == "timeout"
    assert budget.total() > 0


def test_concurrent_settlement_and_recovery(tmp_path):
    budget = Budget(tmp_path / "budget.sqlite", cap=1)

    def operation(index):
        key = budget.reserve(0.01, {"stage": "unit", "configuration_id": "config"})
        budget.settle(
            key,
            0.001,
            {"stage": "unit", "configuration_id": "config", "id": str(index), "prediction": 1},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(operation, range(64)))
    assert budget.total() == pytest.approx(0.064)
    assert len(budget.recover("unit", "config")) == 64


def test_interrupted_journal_and_atomic_metadata(tmp_path):
    from moneyme_sms_spam_classifier.common import read_journal, read_json, write_json

    journal = tmp_path / "requests.jsonl"
    journal.write_text('{"id":"complete"}\n{"id":"incomplete')
    assert read_journal(journal) == [{"id": "complete"}]
    assert journal.read_text().endswith("\n")
    assert journal.with_suffix(".jsonl.interrupted").exists()
    output = tmp_path / "config.json"
    write_json(output, {"model": "first"})
    write_json(output, {"model": "second"})
    assert read_json(output) == {"model": "second"}
    assert output.stat().st_mode & 0o004  # Readable by the non-root image user.


def test_malformed_cost_keeps_reservation_and_records_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder")
    body = {"choices": [{"message": {"content": '{"label":"ham"}'}}], "usage": {"cost": "invalid"}}
    budget = Budget(tmp_path / "budget.sqlite", 1)
    adapter = OpenRouterClassifier(
        {"group": "llm", "model": "unit"},
        budget,
        {"pricing": {"prompt": "0.000001", "completion": "0.000001"}},
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))),
    )
    result = adapter.predict("hello")
    assert result["status"] == "invalid_cost_metadata"
    assert result["prediction"] == -1
    assert result["cost_usd"] is None
    assert budget.total() > 0
