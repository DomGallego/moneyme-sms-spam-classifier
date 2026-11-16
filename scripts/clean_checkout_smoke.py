"""Rehearse deployment using only Git-visible files, excluding secrets/caches/raw data."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path.cwd()
files = subprocess.check_output(
    ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
).split(b"\0")
with tempfile.TemporaryDirectory(prefix="moneyme-review-") as directory:
    checkout = Path(directory)
    copied = 0
    for entry in files:
        if not entry:
            continue
        relative = Path(entry.decode())
        source = root / relative
        if not source.is_file():
            continue
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    assert not (checkout / ".env").exists()
    assert not (checkout / "data").exists()
    assert (checkout / "artifacts/release/model.joblib").exists()
    subprocess.run([sys.executable, "scripts/docker_smoke.py"], cwd=checkout, check=True)
    evidence = json.loads((checkout / "reports/docker_smoke.json").read_text())
    evidence.update(
        {
            "git_visible_files_copied": copied,
            "secrets_present": False,
            "raw_dataset_present": False,
            "note": "Uncommitted working tree mirrored using Git-visible paths, not a committed clone.",
        }
    )
    (root / "reports/clean_checkout_smoke.json").write_text(json.dumps(evidence, indent=2))
    print(evidence)
