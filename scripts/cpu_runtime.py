"""Execute a Python script with the isolated, pinned CPU-only PyTorch build."""

import os
import subprocess
import sys
from pathlib import Path

root = Path(".cache/cpu_torch").resolve()
if not (root / "torch" / "__init__.py").exists():
    raise SystemExit(
        "Install requirements-encoder-cpu.txt into .cache/cpu_torch with --no-deps first."
    )
if len(sys.argv) < 2:
    raise SystemExit("Usage: python scripts/cpu_runtime.py SCRIPT [ARGS...]")
environment = os.environ.copy()
environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
environment["CUDA_VISIBLE_DEVICES"] = ""
subprocess.run([sys.executable, "-u", *sys.argv[1:]], env=environment, check=True)
