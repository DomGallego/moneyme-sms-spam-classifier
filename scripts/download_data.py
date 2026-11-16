"""Download the public UCI corpus and verify the recorded dataset fingerprint."""

import io
import zipfile
from pathlib import Path

import httpx

from moneyme_sms_spam_classifier.common import fingerprint

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
response = httpx.get(url, timeout=60, follow_redirects=True)
response.raise_for_status()
expected = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
    name = next(name for name in archive.namelist() if name.endswith("SMSSpamCollection"))
    content = archive.read(name)
    if fingerprint(content) != expected:
        raise ValueError(
            "UCI file differs from the recorded dataset fingerprint; verify the source version before reproducing results"
        )
    target = Path("data/sms+spam+collection/SMSSpamCollection")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
print("Dataset downloaded and fingerprint verified")
