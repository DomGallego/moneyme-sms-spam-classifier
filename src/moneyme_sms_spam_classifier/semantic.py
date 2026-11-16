"""Frozen embeddings and supervised DistilBERT; optional analysis dependencies."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from .common import manifest, read_json, write_json
from .data import load
from .evaluation import metrics
from .training import classifier, persist_candidate

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER = "distilbert/distilbert-base-uncased"


class EmbeddingClassifier:
    def __init__(self, classifier, model_path, revision):
        self.classifier, self.model_path, self.revision = classifier, model_path, revision
        self.classes_ = np.array([0, 1])
        self._encoder = None

    def __getstate__(self):
        return {**self.__dict__, "_encoder": None}

    def encode(self, texts):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self.model_path, device="cpu", local_files_only=True
            )
        return self._encoder.encode(
            list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )

    def predict_proba(self, texts):
        return self.classifier.predict_proba(self.encode(texts))


class EncoderClassifier:
    def __init__(self, model_path, max_length=128):
        self.model_path, self.max_length = model_path, max_length
        self.classes_ = np.array([0, 1])
        self._model, self._tokenizer = None, None

    def __getstate__(self):
        return {**self.__dict__, "_model": None, "_tokenizer": None}

    def predict_proba(self, texts):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, local_files_only=True
            ).eval()
        output = []
        for start in range(0, len(texts), 16):
            inputs = self._tokenizer(
                list(texts)[start : start + 16],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                output.append(self._model(**inputs).logits.softmax(-1).numpy())
        return np.concatenate(output)


def train_embeddings(trials=20):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    df = load()
    revision = read_json("configs/encoder_revisions.json")[EMBEDDER]
    encoder = SentenceTransformer(
        revision["path"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    )
    started = time.perf_counter()
    X = encoder.encode(
        df.text.tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )
    np.save("artifacts/splits/embeddings.npy", X)
    embedding_time = time.perf_counter() - started
    train, val = df[df.split == "train"], df[df.split == "validation"]
    output = Path("reports/runs/embeddings")
    write_json(
        output / "manifest.json",
        manifest(
            model=EMBEDDER,
            revision=revision["revision"],
            extraction_seconds=embedding_time,
            normalized=True,
            dimensions=X.shape[1],
            max_length=encoder.max_seq_length,
        ),
    )
    results = []
    for name in ["logistic", "xgboost"]:
        estimator, space = classifier(name)
        search = RandomizedSearchCV(
            estimator,
            space,
            n_iter=trials,
            scoring="f1",
            n_jobs=2,
            cv=PredefinedSplit(train.fold),
            random_state=42,
            error_score="raise",
        )
        started = time.perf_counter()
        search.fit(X[train.index], train.target)
        wrapper = EmbeddingClassifier(
            search.best_estimator_, revision["path"], revision["revision"]
        )
        candidate = f"{name}_embeddings"
        dest = Path("artifacts/models") / candidate
        result = persist_candidate(
            candidate,
            wrapper,
            train,
            val,
            search.best_score_,
            time.perf_counter() - started,
            search.best_params_,
            dest,
            kind="frozen_embeddings",
        )
        import pandas as pd

        pd.DataFrame(search.cv_results_).to_csv(dest / "search_results.csv", index=False)
        results.append(result)
        print(candidate, result["validation"], flush=True)
    write_json(output / "results.json", results)


def train_encoder(
    epochs=3,
    learning_rates=(2e-5, 5e-5),
    max_length=128,
    training_frame=None,
    report_root="reports/runs/encoder",
    model_root="artifacts/models",
    suffix="",
    settings=None,
    training_device="auto",
    resume=False,
):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
    from transformers.trainer_pt_utils import LengthGroupedSampler

    torch.set_num_threads(2)
    set_seed(42)
    df = load()
    train, val = df[df.split == "train"], df[df.split == "validation"]
    if training_frame is not None:
        train = training_frame.reset_index(drop=True)
    revision = read_json("configs/encoder_revisions.json")[ENCODER]
    cuda_available = torch.cuda.is_available() if training_device != "cpu" else None
    device = torch.device(
        ("cuda" if cuda_available else "cpu") if training_device == "auto" else training_device
    )
    torch.set_num_threads(6 if device.type == "cpu" else 2)
    dynamic_padding = device.type == "cpu"
    tokenizer = AutoTokenizer.from_pretrained(revision["path"], local_files_only=True)
    tokenized = tokenizer(
        train.text.tolist(),
        truncation=True,
        max_length=max_length,
        padding=not dynamic_padding,
        return_tensors=None if dynamic_padding else "pt",
    )
    val_tokens = tokenizer(
        val.text.tolist(),
        truncation=True,
        max_length=max_length,
        padding=not dynamic_padding,
        return_tensors=None if dynamic_padding else "pt",
    )
    lengths = [len(x) for x in tokenizer(train.text.tolist(), truncation=False)["input_ids"]]
    root = Path(report_root)
    write_json(
        root
        / (
            "manifest.json" if not (root / "manifest.json").exists() else f"execution_{device}.json"
        ),
        manifest(
            model=ENCODER,
            revision=revision["revision"],
            device=str(device),
            cpu_threads=torch.get_num_threads(),
            torch_version=torch.__version__,
            torch_cuda_build=torch.version.cuda,
            dynamic_padding=dynamic_padding,
            length_grouped_batches=dynamic_padding,
            cuda_available=cuda_available,
            max_length=max_length,
            batch_size=8,
            accumulation=2,
            epochs=epochs,
            learning_rates=list(learning_rates),
            train_truncated_fraction=float(np.mean(np.asarray(lengths) > max_length)),
        ),
    )
    trials = (
        settings
        if settings is not None
        else [("head", 1e-3)] + [("full", lr) for lr in learning_rates]
    )
    history_path = root / "history.json"
    history = read_json(history_path) if resume and history_path.exists() else []
    for mode, lr in trials:
        set_seed(42)
        candidate = f"distilbert_{mode}_{lr:g}{suffix}"
        history = [row for row in history if row["candidate"] != candidate]
        folder = Path(model_root) / candidate
        write_json(
            folder / "training_manifest.json",
            manifest(
                device=str(device),
                torch_version=torch.__version__,
                torch_cuda_build=torch.version.cuda,
                precision="autocast_float16" if device.type == "cuda" else "float32",
                dynamic_padding=dynamic_padding,
                length_grouped_batches=dynamic_padding,
                seed=42,
                epochs=epochs,
                learning_rate=lr,
                mode=mode,
                training_rows=len(train),
            ),
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            revision["path"],
            num_labels=2,
            id2label={0: "ham", 1: "spam"},
            label2id={"ham": 0, "spam": 1},
            local_files_only=True,
        )
        if mode == "head":
            for param in model.distilbert.parameters():
                param.requires_grad = False
        model.to(device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        generator = torch.Generator().manual_seed(42)
        sampler = (
            LengthGroupedSampler(
                batch_size=8,
                lengths=[min(length, max_length) for length in lengths],
                generator=generator,
            )
            if dynamic_padding
            else None
        )
        loader = DataLoader(
            list(range(len(train))),
            batch_size=8,
            shuffle=not dynamic_padding,
            sampler=sampler,
            generator=generator,
        )
        best_f1, started, best_epoch = -1, time.perf_counter(), None
        checkpoint = folder / "checkpoint"
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            for step, indices in enumerate(loader):
                if dynamic_padding:
                    batch = [
                        {key: values[index] for key, values in tokenized.items()}
                        for index in indices.tolist()
                    ]
                    inputs = tokenizer.pad(batch, padding=True, return_tensors="pt")
                else:
                    inputs = {k: v[indices].to(device) for k, v in tokenized.items()}
                labels = torch.tensor(train.target.to_numpy()[indices.numpy()], device=device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    loss = model(**inputs, labels=labels).loss / 2
                scaler.scale(loss).backward()
                if (step + 1) % 2 == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            model.eval()
            probabilities = []
            with torch.inference_mode():
                for start in range(0, len(val), 16):
                    if dynamic_padding:
                        batch = [
                            {key: values[index] for key, values in val_tokens.items()}
                            for index in range(start, min(start + 16, len(val)))
                        ]
                        inputs = tokenizer.pad(batch, padding=True, return_tensors="pt")
                    else:
                        inputs = {
                            k: v[start : start + 16].to(device) for k, v in val_tokens.items()
                        }
                    probabilities.extend(
                        model(**inputs).logits.softmax(-1)[:, 1].cpu().numpy().tolist()
                    )
            score = metrics(
                val.target, (np.asarray(probabilities) >= 0.5).astype(int), probabilities, True
            )
            history.append(
                {
                    "candidate": candidate,
                    "epoch": epoch + 1,
                    "metrics": score,
                    "device": str(device),
                    "precision": "autocast_float16" if device.type == "cuda" else "float32",
                }
            )
            write_json(root / "history.json", history)
            print(candidate, "epoch", epoch + 1, score["spam_f1"], flush=True)
            if score["spam_f1"] > best_f1:
                best_f1, best_epoch = score["spam_f1"], epoch + 1
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
        elapsed = time.perf_counter() - started
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        wrapper = EncoderClassifier(str(checkpoint), max_length)
        result = persist_candidate(
            candidate,
            wrapper,
            train,
            val,
            None,
            elapsed,
            {
                "learning_rate": lr,
                "mode": mode,
                "best_epoch": best_epoch,
                "epochs": epochs,
                "revision": revision["revision"],
            },
            folder,
            kind="encoder",
        )
        result["training_device"] = str(device)
        result["training_precision"] = "autocast_float16" if device.type == "cuda" else "float32"
        result["dynamic_padding"] = dynamic_padding
        result["length_grouped_batches"] = dynamic_padding
        write_json(folder / "result.json", result)
