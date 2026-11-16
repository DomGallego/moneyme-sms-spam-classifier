"""Build/run/check/stop a local-only container. Does not spend OpenRouter credits."""

import json
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np

name = "moneyme-sms-smoke"
image = "moneyme-sms:assessment"
build_started = time.perf_counter()
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
build_seconds = time.perf_counter() - build_started
startup_started = time.perf_counter()
subprocess.run(
    ["docker", "run", "-d", "--name", name, "-p", "127.0.0.1:18000:8000", image],
    check=True,
    timeout=30,
)
try:
    with httpx.Client(timeout=5) as client:
        for _ in range(30):
            try:
                response = client.get("http://127.0.0.1:18000/health")
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Container did not become healthy")
        startup_seconds = time.perf_counter() - startup_started
        texts = [
            "Are we still meeting for lunch?",
            "WIN a free prize! Call now to claim your reward.",
        ]
        latencies = []
        for index in range(101):
            started = time.perf_counter()
            response = client.post(
                "http://127.0.0.1:18000/predict/ml", json={"text": texts[index % 2]}
            )
            response.raise_for_status()
            assert response.json()["label"] in {"spam", "ham"}
            if index:
                latencies.append((time.perf_counter() - started) * 1000)
        for group in ["llm", "decision"]:
            assert (
                client.post(
                    f"http://127.0.0.1:18000/predict/{group}", json={"text": "hello"}
                ).status_code
                == 503
            )
        subprocess.run(
            [
                "docker",
                "exec",
                name,
                "python",
                "task_2_deployment/sample_client.py",
                "--base-url",
                "http://127.0.0.1:8000",
                "--models",
                "ml",
            ],
            check=True,
            timeout=30,
        )
        evidence = {
            "health": "passed",
            "local_prediction": response.json(),
            "hosted_disabled": "passed",
            "sample_client": "passed",
            "build_seconds": build_seconds,
            "startup_seconds": startup_seconds,
            "http_warm_p50_ms": float(np.percentile(latencies, 50)),
            "http_warm_p95_ms": float(np.percentile(latencies, 95)),
            "sequential_requests": len(latencies),
            "concurrency": 1,
            "timing_note": "Two smoke-test SMS, warm HTTP connections; not a workload benchmark.",
            "image": subprocess.check_output(
                ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
            ).strip(),
        }
        evidence["runtime_uid"] = int(
            subprocess.check_output(["docker", "exec", name, "id", "-u"], text=True)
        )
        assert evidence["runtime_uid"] != 0
        evidence["container_resources"] = json.loads(
            subprocess.check_output(
                ["docker", "stats", name, "--no-stream", "--format", "{{json .}}"],
                text=True,
                timeout=15,
            )
        )
        Path("reports/docker_smoke.json").write_text(json.dumps(evidence, indent=2))
        print(evidence)
finally:
    subprocess.run(["docker", "stop", name], check=False, timeout=30)
    subprocess.run(["docker", "rm", name], check=False, timeout=30)
