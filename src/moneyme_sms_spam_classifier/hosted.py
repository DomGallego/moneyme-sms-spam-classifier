"""OpenRouter adapters with strict parsing and a persistent, fail-closed budget ledger.

Requests may run concurrently. SQLite transactions reserve worst-case cost before
network I/O; unknown charges remain reserved across process restarts.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import httpx
from dotenv import load_dotenv

from .common import fingerprint, read_json, write_json

BASE_URL = "https://openrouter.ai/api"


class HostedError(RuntimeError):
    def __init__(self, category, status=502):
        super().__init__(category)
        self.category, self.status = category, status


class BudgetExceeded(HostedError):
    def __init__(self):
        super().__init__("budget_exhausted", 503)


class Budget:
    _locks: ClassVar[dict[str, threading.RLock]] = {}

    def __init__(self, path="reports/budget.sqlite", cap=0.0):
        self.path, self.cap = Path(path), float(cap)
        if not math.isfinite(self.cap) or self.cap < 0:
            raise ValueError("Budget must be finite and nonnegative")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mutex = self._locks.setdefault(str(self.path.resolve()), threading.RLock())
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, reserved REAL NOT NULL, cost REAL, metadata TEXT)"
            )

    @contextmanager
    def connect(self):
        # Serialize short in-process transactions and always close connections.
        # WAL avoids reader/writer contention across independent report processes.
        with self._mutex:
            db = sqlite3.connect(self.path, timeout=60)
            try:
                with db:
                    yield db
            finally:
                db.close()

    def recover(self, stage, configuration_id):
        with self.connect() as db:
            records = db.execute(
                "SELECT metadata FROM attempts WHERE json_extract(metadata, '$.stage')=? "
                "AND json_extract(metadata, '$.configuration_id')=?",
                (stage, configuration_id),
            ).fetchall()
        return [json.loads(row[0]) for row in records]

    def total(self):
        with self.connect() as db:
            return db.execute(
                "SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM attempts"
            ).fetchone()[0]

    def reserve(self, amount, metadata):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Invalid reservation")
        key = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute(
                "SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM attempts"
            ).fetchone()[0]
            if total + amount > self.cap:
                raise BudgetExceeded()
            db.execute(
                "INSERT INTO attempts VALUES (?,?,NULL,?)", (key, amount, json.dumps(metadata))
            )
        return key

    def settle(self, key, cost, metadata):
        # Missing usage after timeout/malformed response retains the reservation.
        if cost is not None and (not math.isfinite(float(cost)) or float(cost) < 0):
            raise ValueError("Invalid returned cost")
        with self.connect() as db:
            db.execute(
                "UPDATE attempts SET cost=?, metadata=? WHERE id=?",
                (cost, json.dumps(metadata), key),
            )


def fetch_catalog(output="reports/catalog.json"):
    with httpx.Client(timeout=60) as client:
        response = client.get(f"{BASE_URL}/v1/models", params={"output_modalities": "all"})
        response.raise_for_status()
    catalog = response.json()
    write_json(output, catalog)
    candidates = read_json("configs/candidates.json")
    models = {m["id"]: m for m in catalog["data"]}
    status = {
        group: {
            name: {
                "listed": name in models,
                "pricing": models.get(name, {}).get("pricing"),
                "endpoints": models.get(name, {}).get("endpoints"),
                "reasoning": models.get(name, {}).get("reasoning"),
            }
            for name in names
        }
        for group, names in candidates.items()
    }
    write_json("reports/candidate_availability.json", status)
    return status


def validate_probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HostedError("invalid_probability")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise HostedError("invalid_probability")
    return float(value)


def parse_prediction(body, group, primitive="choice"):
    if group == "llm":
        try:
            content = body["choices"][0]["message"]["content"]
            data = json.loads(content)
            if (
                not isinstance(data, dict)
                or set(data) != {"label"}
                or data["label"] not in {"spam", "ham"}
            ):
                raise ValueError()
            return {"prediction": int(data["label"] == "spam"), "score": None}
        except (KeyError, IndexError, TypeError, ValueError):
            raise HostedError("invalid_output") from None
    try:
        answer = body["answers"]["classification"]
        if primitive == "noul":
            score = validate_probability(answer["noul"])
            return {"prediction": int(score >= 0.5), "score": score}
        probabilities = answer["probabilities"]
        if set(probabilities) != {"spam", "ham"}:
            raise HostedError("invalid_probability")
        score, ham = (
            validate_probability(probabilities["spam"]),
            validate_probability(probabilities["ham"]),
        )
        if abs(score + ham - 1) > 1e-3 or answer["choice"] not in {"spam", "ham"}:
            raise HostedError("invalid_output")
        return {"prediction": int(answer["choice"] == "spam"), "score": score}
    except (KeyError, TypeError, ValueError):
        raise HostedError("invalid_output") from None


class OpenRouterClassifier:
    def __init__(self, config, budget, catalog, client=None):
        load_dotenv()
        self.key = os.getenv("OPENROUTER_API_KEY", "")
        if not self.key:
            raise HostedError("missing_openrouter_key", 503)
        self.config, self.budget, self.catalog = config, budget, catalog
        self.client = client or httpx.Client(timeout=httpx.Timeout(60, connect=10))

    def close(self):
        self.client.close()

    def parse_response(self, body):
        return parse_prediction(body, self.config["group"], self.config.get("primitive", "choice"))

    def payload(self, text):
        cfg = self.config
        if cfg["group"] == "llm":
            system = Path("prompts/spam_zero_shot_v1.txt").read_text()
            messages = [{"role": "system", "content": system}]
            for example in cfg.get("examples", []):
                messages.extend(
                    [
                        {"role": "user", "content": example["text"]},
                        {"role": "assistant", "content": json.dumps({"label": example["label"]})},
                    ]
                )
            messages.append({"role": "user", "content": text})
            payload = {
                "model": cfg["model"],
                "messages": messages,
                "max_tokens": cfg.get("max_tokens", 2048),
            }
            parameters = self.catalog.get("supported_parameters", [])
            if "temperature" in parameters:
                payload["temperature"] = 0
            if "response_format" in parameters:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sms_classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"label": {"type": "string", "enum": ["spam", "ham"]}},
                            "required": ["label"],
                            "additionalProperties": False,
                        },
                    },
                }
            reasoning = self.catalog.get("reasoning") or {}
            if "reasoning" in parameters:
                if reasoning.get("mandatory"):
                    efforts = reasoning.get("supported_efforts", [])
                    payload["reasoning"] = {
                        "effort": "low"
                        if "low" in efforts
                        else reasoning.get("default_effort", "high")
                    }
                else:
                    payload["reasoning"] = {"enabled": False}
            return "/v1/chat/completions", payload
        criteria = read_json("configs/decision_criteria.json")
        primitive = cfg.get("primitive", "choice")
        question = {"type": primitive, "instructions": criteria["instructions"]}
        if primitive == "noul":
            question["instructions"] += " Is this SMS spam?"
            question["criteria"] = {
                "true": criteria["criteria"]["spam"],
                "false": criteria["criteria"]["ham"],
            }
        else:
            question["criteria"] = criteria["criteria"]
        return "/alpha/decisions", {
            "model": cfg["model"],
            "state": text,
            "questions": {"classification": question},
        }

    def predict(self, text, metadata=None):
        endpoint, payload = self.payload(text)
        serialized = json.dumps(payload, ensure_ascii=False)
        pricing = self.catalog.get("pricing") or {}
        if "prompt" not in pricing or "completion" not in pricing:
            raise HostedError("unknown_pricing", 503)
        # UTF-8 byte count overestimates ordinary tokenizer inputs; extra allowance
        # covers provider wrappers and decision schemas. No unbounded tools are enabled.
        input_bound = len(serialized.encode()) + 8192
        output_bound = payload.get("max_tokens", 8192)
        reservation = input_bound * float(pricing["prompt"]) + output_bound * max(
            float(pricing["completion"]), float(pricing.get("internal_reasoning", 0))
        )
        meta = {
            **(metadata or {}),
            "model_requested": payload["model"],
            "group": self.config["group"],
            "request_hash": fingerprint(serialized),
            "configuration": {k: v for k, v in self.config.items() if k != "examples"},
        }
        attempt = self.budget.reserve(reservation, meta)
        started = time.perf_counter()
        wall_started = time.time()
        body, cost = {}, None
        try:
            response = self.client.post(
                BASE_URL + endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "X-Title": "SMS Spam Classifier",
                },
            )
            if response.status_code >= 400:
                raise HostedError(
                    f"http_{response.status_code}", 504 if response.status_code == 504 else 502
                )
            body = response.json()
            if not isinstance(body, dict):
                body = {}
                raise HostedError("invalid_response_envelope")
            usage = body.get("usage") or {}
            if not isinstance(usage, dict):
                raise HostedError("invalid_usage_envelope")
            cost = usage.get("cost")
            if cost is not None:
                try:
                    cost = float(cost)
                except (TypeError, ValueError):
                    cost = None
                    raise HostedError("invalid_cost_metadata") from None
                if not math.isfinite(cost) or cost < 0:
                    cost = None
                    raise HostedError("invalid_cost_metadata")
            result = self.parse_response(body)
            result["status"] = "ok"
        except httpx.ConnectError:
            cost = 0.0  # Connection establishment failed before sending the request.
            result = {"prediction": -1, "score": None, "status": "connection_not_sent"}
        except httpx.TimeoutException:
            result = {"prediction": -1, "score": None, "status": "timeout"}
        except (httpx.HTTPError, ValueError):
            result = {"prediction": -1, "score": None, "status": "transport_or_json_error"}
        except HostedError as exc:
            result = {"prediction": -1, "score": None, "status": exc.category}
        result.update(
            {
                "attempt_id": attempt,
                "started_at_unix": wall_started,
                "finished_at_unix": time.time(),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "cost_usd": cost,
                "reserved_usd": reservation,
                "usage": body.get("usage", {}),
                "model_returned": body.get("model"),
                "provider": body.get("provider"),
                "request_id": body.get("id"),
                **meta,
            }
        )
        # Do not save raw prompts, messages, API errors, or credentials in the ledger.
        self.budget.settle(attempt, cost, result)
        if cost is not None and cost > reservation + 1e-8:
            raise HostedError("provider_charge_exceeded_reservation", 503)
        return result
