"""Inference-only FastAPI application. No training or model downloads at startup."""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .hosted import HostedError
from .inference import LocalClassifier, hosted_classifier, hosted_label

logger = logging.getLogger("sms_spam")


class SMSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(
        min_length=1, max_length=5000, description="SMS text; maximum 5,000 characters"
    )

    @field_validator("text")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("SMS must contain non-whitespace text")
        return value


class Prediction(BaseModel):
    label: Literal["spam", "ham"]


def create_app(local=None, hosted=None):
    @asynccontextmanager
    async def lifespan(app):
        load_dotenv()
        app.state.local = local or LocalClassifier(
            os.getenv("LOCAL_SELECTION", "configs/local_selection.json")
        )
        app.state.hosted = hosted if hosted is not None else {}
        enabled = os.getenv("ENABLE_HOSTED", "false").lower() == "true"
        if hosted is None and enabled:
            cap = float(os.getenv("SERVICE_BUDGET_USD", "0"))
            for group in ["llm", "decision"]:
                app.state.hosted[group] = hosted_classifier(
                    group, cap, os.getenv("SERVICE_LEDGER", "reports/service-budget.sqlite")
                )
        app.state.semaphore = threading.BoundedSemaphore(int(os.getenv("MAX_INFLIGHT", "4")))
        yield
        for adapter in app.state.hosted.values():
            adapter.close()

    app = FastAPI(
        title="SMS Spam Classifier",
        version="0.1.0",
        lifespan=lifespan,
        description="Local ML and hosted LLM/decision classifiers. Spam is positive.",
    )

    @app.get("/health", tags=["operations"])
    def health():
        return {
            "status": "ok",
            "local_model": app.state.local.candidate,
            "hosted_enabled": sorted(app.state.hosted),
        }

    def classify(group, request):
        if not app.state.semaphore.acquire(blocking=False):
            raise HTTPException(503, "Inference capacity reached; retry later")
        started = time.perf_counter()
        status = "ok"
        try:
            if group == "ml":
                label = app.state.local.predict(request.text)
            else:
                adapter = app.state.hosted.get(group)
                if adapter is None:
                    raise HostedError("hosted_classifier_disabled", 503)
                label = hosted_label(adapter, request.text)
            return Prediction(label=label)
        except HostedError as exc:
            status = exc.category
            raise HTTPException(exc.status, exc.category) from None
        finally:
            app.state.semaphore.release()
            logger.info(
                "group=%s status=%s latency_ms=%.2f",
                group,
                status,
                (time.perf_counter() - started) * 1000,
            )

    @app.post("/predict/ml", response_model=Prediction, tags=["prediction"])
    def predict_ml(request: SMSRequest):
        return classify("ml", request)

    @app.post("/predict/llm", response_model=Prediction, tags=["prediction"])
    def predict_llm(request: SMSRequest):
        return classify("llm", request)

    @app.post("/predict/decision", response_model=Prediction, tags=["prediction"])
    def predict_decision(request: SMSRequest):
        return classify("decision", request)

    return app
