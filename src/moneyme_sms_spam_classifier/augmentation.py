"""Budgeted, resumable, parent-traceable generation of training-only SMS variants."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .common import fingerprint, manifest, read_journal, read_json, write_json
from .data import load, template_key
from .hosted import Budget, HostedError, OpenRouterClassifier

ROOT = Path("reports/runs/augmentation")
POOL = Path("artifacts/augmentation/pool.jsonl")


class AugmentationGenerator(OpenRouterClassifier):
    def payload(self, text):
        config = self.config
        schema = {
            "type": "object",
            "properties": {
                "variants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": config["variants_per_request"],
                }
            },
            "required": ["variants"],
            "additionalProperties": False,
        }
        payload = {
            "model": config["model"],
            "max_tokens": config["max_tokens"],
            "temperature": config["generation_temperature"],
            "messages": [
                {"role": "system", "content": Path(config["prompt"]).read_text()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"original_sms": text, "maximum_variants": config["variants_per_request"]}
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "sms_variants", "strict": True, "schema": schema},
            },
            "reasoning": {"enabled": False},
        }
        return "/v1/chat/completions", payload

    def parse_response(self, body):
        try:
            result = json.loads(body["choices"][0]["message"]["content"])
            variants = result["variants"]
            if set(result) != {"variants"} or not isinstance(variants, list):
                raise ValueError()
            if len(variants) > self.config["variants_per_request"] or not all(
                isinstance(value, str) for value in variants
            ):
                raise ValueError()
            return {"variants": variants}
        except (KeyError, TypeError, IndexError, ValueError):
            raise HostedError("invalid_augmentation_output") from None


def normalized(text):
    return " ".join(text.casefold().split())


def rejection_reason(text, parent, seen, original_templates):
    if not isinstance(text, str) or not text.strip():
        return "empty"
    text = text.strip()
    if not max(8, len(parent["text"]) * 0.4) <= len(text) <= max(80, len(parent["text"]) * 2):
        return "length"
    if normalized(text) in seen:
        return "duplicate"
    if (
        template_key(text) in original_templates
        and original_templates[template_key(text)] != parent["group"]
    ):
        return "other_family_template"
    # Copying source contact details is allowed; inventing new long digit sequences/URLs is not.
    for pattern in [r"\d{3,}", r"(?:https?://|www\.)\S+"]:
        if set(re.findall(pattern, text, re.IGNORECASE)) - set(
            re.findall(pattern, parent["text"], re.IGNORECASE)
        ):
            return "new_contact_or_number"
    return None


def generate_pool(limit=0, rounds=1):
    config = read_json("configs/imbalance_experiment.json")
    config["group"] = "augmentation"
    identity = fingerprint(json.dumps(config, sort_keys=True) + Path(config["prompt"]).read_text())
    frozen = POOL.parent / "frozen.json"
    if frozen.exists():
        definition = read_json(frozen)
        if fingerprint(POOL.read_bytes()) != definition["pool_sha256"]:
            raise ValueError("Frozen augmentation pool changed")
        if definition["generation_configuration_id"] != identity:
            raise ValueError("Frozen augmentation definition changed; archive before regenerating")
        return read_json(ROOT / "summary.json")
    df = load()
    train = df.query("split == 'train'")
    parents = train.query("target == 1").sample(frac=1, random_state=config["seed"])
    if limit:
        parents = parents.head(limit)
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if (
            old["configuration_id"] != identity
            or old["dataset_sha256"]
            != read_json("artifacts/splits/manifest.json")["dataset_sha256"]
        ):
            raise ValueError("Augmentation definition changed; archive the run before changing it")
    else:
        write_json(
            manifest_path,
            manifest(
                configuration_id=identity,
                config=config,
                dataset_sha256=read_json("artifacts/splits/manifest.json")["dataset_sha256"],
                training_parent_ids=train.query("target == 1").id.tolist(),
                quality_scope="Automated integrity checks plus assistant-reviewed pilot; no independent human label audit.",
            ),
        )
    budget = Budget(cap=read_json("configs/budget.json")["total_cap_usd"])
    catalog = {row["id"]: row for row in read_json("reports/catalog.json")["data"]}
    adapter = AugmentationGenerator(config, budget, catalog[config["model"]])
    journal = ROOT / "requests.jsonl"
    requests = {row["id"]: row for row in read_journal(journal)}
    for row in budget.recover("augmentation", identity):
        if row.get("id") not in requests:
            with journal.open("a") as file:
                file.write(json.dumps(row) + "\n")
            requests[row["id"]] = row
    try:
        for round_index in range(rounds):
            for parent in parents.to_dict("records"):
                key = f"{parent['id']}:{round_index}"
                if key in requests:
                    continue
                result = adapter.predict(
                    parent["text"],
                    metadata={
                        "stage": "augmentation",
                        "configuration_id": identity,
                        "id": key,
                        "parent_id": parent["id"],
                        "family_id": parent["group"],
                        "round": round_index,
                    },
                )
                with journal.open("a") as file:
                    file.write(json.dumps(result) + "\n")
                requests[key] = result
                if len(requests) % 20 == 0:
                    print(
                        f"Generation requests: {len(requests)}; accounted USD {budget.total():.4f}",
                        flush=True,
                    )
    finally:
        adapter.close()
    # Integrity-only exact-match exclusion: no evaluation text or labels reach the generator.
    seen = set(df.text.map(normalized))
    templates = dict(zip(train.text.map(template_key), train.group))
    by_id = train.set_index("id").to_dict("index")
    accepted, rejected = [], []
    for request in requests.values():
        parent = by_id[request["parent_id"]]
        for index, text in enumerate(request.get("variants", [])):
            reason = rejection_reason(text, parent, seen, templates)
            if reason:
                rejected.append({"parent_id": request["parent_id"], "text": text, "reason": reason})
                continue
            text = text.strip()
            seen.add(normalized(text))
            accepted.append(
                {
                    "id": fingerprint(text),
                    "parent_id": request["parent_id"],
                    "group": parent["group"],
                    "parent_fold": parent["fold"],
                    "text": text,
                    "target": 1,
                    "label": "spam",
                    "generator": config["model"],
                    "configuration_id": identity,
                    "attempt_id": request["attempt_id"],
                    "variant_index": index,
                }
            )
    POOL.parent.mkdir(parents=True, exist_ok=True)
    POOL.write_text("".join(json.dumps(row) + "\n" for row in accepted))
    write_json(ROOT / "rejected.json", rejected)
    summary = {
        "requests": len(requests),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "covered_parents": len({r["parent_id"] for r in accepted}),
        "pool_sha256": fingerprint(POOL.read_bytes()),
        "accounted_total_usd": budget.total(),
        "configuration_id": identity,
    }
    write_json(ROOT / "summary.json", summary)
    print(summary, flush=True)
    return summary
