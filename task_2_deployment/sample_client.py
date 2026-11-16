"""Call every classifier: python task_2_deployment/sample_client.py --base-url URL."""

import argparse

import httpx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://localhost:8000")
parser.add_argument("--text", default="WIN a free prize! Call now to claim your reward.")
parser.add_argument(
    "--models", nargs="+", choices=["ml", "llm", "decision"], default=["ml", "llm", "decision"]
)
args = parser.parse_args()
failed = False
with httpx.Client(timeout=90) as client:
    for model in args.models:
        response = client.post(f"{args.base_url}/predict/{model}", json={"text": args.text})
        print(model, response.status_code, response.json())
        failed |= response.is_error
if failed:
    raise SystemExit(1)
