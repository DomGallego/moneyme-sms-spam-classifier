"""Fetch the exact recorded encoder revisions and update machine-local cache paths."""

from huggingface_hub import snapshot_download

from moneyme_sms_spam_classifier.common import read_json, write_json

records = read_json("configs/encoder_revisions.json")
for model, details in records.items():
    details["path"] = snapshot_download(
        model,
        revision=details["revision"],
        allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
    )
    print(model, details["revision"])
write_json("configs/encoder_revisions.json", records)
