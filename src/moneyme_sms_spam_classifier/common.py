"""Small persistence helpers shared by experiment commands."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def fingerprint(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def manifest(**extra) -> dict:
    import importlib.metadata
    import platform

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unknown", True
    packages = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "pandas", "scikit-learn", "httpx", "fastapi")
    }
    sources = {str(p): fingerprint(p.read_bytes()) for p in sorted(Path("src").rglob("*.py"))}
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "source_hashes": sources,
        **extra,
    }


def read_journal(path: str | Path) -> list[dict]:
    """Preserve a partial final line after interruption; never ignore interior corruption."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            path.with_suffix(path.suffix + ".interrupted").write_text(line)
            break
    normalized = "".join(json.dumps(record, allow_nan=False) + "\n" for record in records)
    # Ensure an interrupted trailing record cannot corrupt the next append.
    if path.read_text() != normalized:
        path.write_text(normalized)
    return records
