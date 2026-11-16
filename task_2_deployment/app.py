"""ASGI entry point: uvicorn task_2_deployment.app:app."""

from moneyme_sms_spam_classifier.api import create_app

app = create_app()
