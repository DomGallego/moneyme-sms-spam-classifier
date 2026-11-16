from fastapi.testclient import TestClient

from moneyme_sms_spam_classifier.api import create_app


class Local:
    candidate = "test-local"

    def predict(self, text):
        return "spam" if "prize" in text else "ham"


def test_contract_validation_and_disabled_hosted():
    with TestClient(create_app(local=Local(), hosted={})) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict/ml", json={"text": "claim prize"}).json() == {"label": "spam"}
        for body in [{"text": " "}, {"text": 123}, {"text": "a" * 5001}, {"text": "x", "extra": 1}]:
            assert client.post("/predict/ml", json=body).status_code == 422
        assert client.post("/predict/llm", json={"text": "hello"}).status_code == 503
        assert client.post("/predict/decision", json={"text": "hello"}).status_code == 503


class Hosted:
    def __init__(self):
        self.config = {"group": "decision", "threshold": 0.8}

    def predict(self, text, metadata=None):
        return {"prediction": 1, "score": 0.7, "status": "ok"}

    def close(self):
        pass


def test_decision_api_uses_frozen_threshold():
    with TestClient(create_app(local=Local(), hosted={"decision": Hosted()})) as client:
        assert client.post("/predict/decision", json={"text": "message"}).json() == {"label": "ham"}


def test_upstream_timeout_is_not_a_successful_ham():
    class Timeout(Hosted):
        def predict(self, text, metadata=None):
            return {"prediction": -1, "score": None, "status": "timeout"}

    with TestClient(create_app(local=Local(), hosted={"decision": Timeout()})) as client:
        response = client.post("/predict/decision", json={"text": "message"})
        assert response.status_code == 504
        assert "label" not in response.json()
