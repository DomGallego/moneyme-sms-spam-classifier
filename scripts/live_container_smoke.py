"""Opt-in live API check using the SAME authorized experiment budget ledger.

Requires completed full validation/test and an .env key. No secrets are printed or
included in the image. Container UID matches the host ledger owner for this check.
"""

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx

from moneyme_sms_spam_classifier.common import read_json, write_json

frozen = read_json("configs/hosted_frozen.json")
if frozen["selection_stage"] != "validation":
    raise ValueError("Complete full validation and freeze hosted selections first")
if len(read_json("reports/runs/test/results.json")) != 14:
    raise ValueError("Finish the hosted test run before the live container check")
cap = read_json("configs/budget.json")["total_cap_usd"]
image, name = "moneyme-sms:assessment", "moneyme-sms-live-smoke"
subprocess.run(
    [
        "docker",
        "build",
        "--platform",
        "linux/amd64",
        "-f",
        "task_2_deployment/Dockerfile",
        "-t",
        image,
        ".",
    ],
    check=True,
    timeout=300,
)
subprocess.run(
    [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--env-file",
        ".env",
        "--env",
        "ENABLE_HOSTED=true",
        "--env",
        f"SERVICE_BUDGET_USD={cap}",
        "--env",
        "SERVICE_LEDGER=/app/runtime/budget.sqlite",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{Path('reports').resolve()}:/app/runtime",
        "-p",
        "127.0.0.1:18002:8000",
        image,
    ],
    check=True,
    timeout=30,
)
try:
    evidence = {"routes": {}, "cap_usd": cap, "shared_ledger": "reports/budget.sqlite"}
    with httpx.Client(timeout=90) as client:
        for _ in range(30):
            try:
                response = client.get("http://127.0.0.1:18002/health", timeout=3)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Container not ready")
        for group in ["ml", "llm", "decision"]:
            response = client.post(
                f"http://127.0.0.1:18002/predict/{group}",
                json={"text": "Are we still meeting for lunch?"},
            )
            response.raise_for_status()
            assert response.json()["label"] in {"spam", "ham"}
            evidence["routes"][group] = {
                "status": response.status_code,
                "response": response.json(),
            }
        evidence["openapi_status"] = client.get("http://127.0.0.1:18002/openapi.json").status_code
    subprocess.run(
        [
            sys.executable,
            "task_2_deployment/sample_client.py",
            "--base-url",
            "http://127.0.0.1:18002",
        ],
        check=True,
        timeout=180,
    )
    with sqlite3.connect("reports/budget.sqlite") as db:
        service = db.execute(
            "SELECT COUNT(*),SUM(cost),SUM(COALESCE(cost,reserved)) FROM attempts WHERE json_extract(metadata,'$.stage')='service'"
        ).fetchone()
    (
        evidence["service_attempts"],
        evidence["service_reported_usd"],
        evidence["service_accounted_usd"],
    ) = service
    evidence["image"] = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
    ).strip()
    write_json("reports/live_container_smoke.json", evidence)
    print(evidence)
finally:
    subprocess.run(["docker", "stop", name], check=False, timeout=30)
    subprocess.run(["docker", "rm", name], check=False, timeout=30)
