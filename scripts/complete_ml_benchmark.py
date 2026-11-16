"""Resume the 60-configuration protocol from persisted artifacts, without duplicate generation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from moneyme_sms_spam_classifier.common import fingerprint, read_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--skip-docker", action="store_true")
args = parser.parse_args()
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(Path(".cache/jupyter/runtime").resolve()))
for path, expected in read_json("reports/runs/imbalance/preserved_artifacts.json").items():
    if fingerprint(Path(path).read_bytes()) != expected:
        raise ValueError(f"Preserved serving/hosted/data artifact changed: {path}")
print("Preparing frozen embedding cache", flush=True)
subprocess.run([sys.executable, "scripts/run_imbalance.py", "embeddings"], check=True)
print("Running numeric searches and encoder training independently", flush=True)
encoder_process = subprocess.Popen([sys.executable, "scripts/run_imbalance.py", "encoders"])
try:
    subprocess.run([sys.executable, "scripts/run_imbalance.py", "numeric"], check=True)
finally:
    encoder_status = encoder_process.wait()
if encoder_status:
    raise subprocess.CalledProcessError(encoder_status, "encoder training")
steps = [
    ["scripts/run_imbalance.py", "freeze"],
    ["scripts/run_imbalance.py", "evaluate"],
    ["-m", "pytest", "-q", "--junitxml=reports/tests.xml"],
]
if not args.skip_docker:
    steps += [["scripts/docker_smoke.py"], ["scripts/clean_checkout_smoke.py"]]
steps += [
    ["scripts/finalize_reports.py"],
    ["-m", "ruff", "check", "src", "tests", "scripts", "task_1_modelling", "task_2_deployment"],
    [
        "-m",
        "ruff",
        "format",
        "--check",
        "src",
        "tests",
        "scripts",
        "task_1_modelling",
        "task_2_deployment",
    ],
]
for step in steps:
    print("Running", " ".join(step), flush=True)
    subprocess.run([sys.executable, *step], check=True)
print("ML benchmark and verification complete.", flush=True)
