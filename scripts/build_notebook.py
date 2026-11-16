"""Build the narrative notebook from saved reports; execution never calls paid APIs."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# SMS spam classification: experiment and engineering report

Compare 60 local ML configurations, seven generative LLMs, and seven specialized decision models on the SMS Spam Collection. Decision models explore quality, latency, and cost alongside conventional approaches.

Imbalance handling includes grouped stratification, class-weight searches, validation-selected thresholds, and separate training-only text augmentation/SMOTE treatments. Validation and test messages remain original.

**Spam is positive.** Configurations and thresholds are selected using development data and frozen before their test predictions. The existing test set was already inspected before the treatment experiment, so treatment results are explicitly exploratory, not fresh independent estimates. The separately frozen serving release retains its original selection provenance. This notebook reads saved experiment artifacts and makes **no paid API requests**. Run commands and deployment instructions are in the repository README. Model IDs and prices reflect the recorded execution date, not a permanent catalog guarantee."""),
    code("""from pathlib import Path
import os
# Support execution from either the repository root or notebook directory.
if not Path("artifacts/splits/manifest.json").exists():
    os.chdir("..")
import pandas as pd
from IPython.display import Image, Markdown, display
from moneyme_sms_spam_classifier.common import read_json
from moneyme_sms_spam_classifier.data import load
from moneyme_sms_spam_classifier.reporting import comparison
pd.set_option("display.max_columns", 25)
pd.set_option("display.max_rows", 100)
df = load()
train = df[df.split == "train"]
def figure(name):
    path = Path("reports/figures") / (name + ".png")
    if path.exists():
        display(Image(filename=str(path)))
    else:
        display(Markdown("Figure pending: `" + name + "`"))"""),
    md("""## Data provenance and integrity

Provenance records where the data came from and what happened to it. The corpus is the historical English-language SMS Spam Collection, credited to Almeida, Gómez Hidalgo, and Yamakami (2011); see the supplied readme and [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

We parse each line at its first tab, validate the label and text, and fingerprint the raw file with SHA-256. Exact duplicates are removed. Conservative template groups mask URLs and digits only in sufficiently long messages; casing/whitespace variants are grouped too. Families stay within one partition and CV fold. These rules catch some near duplicates, not every possible paraphrase. Review `artifacts/splits/families.csv`.

The dataset is historical, from multiple sources, and not chronologically sorted. It cannot establish quality on current Australian financial-services traffic. Public pretraining exposure also cannot be excluded."""),
    code("""display(pd.Series(read_json("artifacts/splits/manifest.json"), dtype=object).to_frame("value"))
display(pd.crosstab(df.split, df.label))
display(train.groupby("label").text.apply(lambda s: s.str.len().describe()).unstack())
figure("class_and_length")"""),
    md("""## Exploratory signals

Length, digit density, capitalization, URLs, and currency mentions are descriptive signals, not hard spam rules. For example, legitimate messages can contain URLs and money. Frequent words/phrases below use training data only. No test messages are used to design model features or prompts."""),
    code("""figure("word_length")
figure("engineered_features")
figure("ngrams_ham")
figure("ngrams_spam")
display(train.groupby("label").head(3)[["label", "text"]])"""),
    md("""## Embedding visualization

A frozen sentence encoder maps each message to a contextual vector. PCA/UMAP project **training** vectors into two dimensions, colored by label. Apparent clusters are exploratory: projection changes distances, overlap is expected, and separation is not evidence of test accuracy. Classifiers use full embeddings, not these two plotting coordinates."""),
    code("""figure("embeddings_pca")
figure("embeddings_umap")
if Path("reports/runs/embeddings/manifest.json").exists():
    display(pd.Series(read_json("reports/runs/embeddings/manifest.json"), dtype=object))"""),
    md("""## Evaluation and operating policy

The cleaned corpus is split approximately 60/20/20 into train/validation/test, with five fixed group-aware CV folds in training. Vectorizers and fitted transformations learn only within training folds. Class weighting is calculated from training. Validation selects thresholds, prompts, and checkpoints; the test set is used after freezing configurations.

The dummy predicts ham for every message. Its accuracy can be high while spam recall and F1 are zero.

- Precision = TP / (TP + FP): reliability of spam flags.
- Recall = TP / (TP + FN): spam caught.
- F1 = 2TP / (2TP + FP + FN): initial selection metric.
- False positive rate = FP / (FP + TN): legitimate messages incorrectly flagged.
- Accuracy = (TP + TN) / N: useful alongside class-specific measures.

We also explore a 98% precision target on validation with at least 10 positive predictions. This is an illustrative operating point, not an established service guarantee. If unsupported or unattainable, report it. The API classifies; it does not block messages.

Invalid hosted responses count as failures, not ham. Coverage and correct/attempted accuracy accompany successful-response metrics."""),
    md("""## ML methods and tuning

Representation and classifier are separate choices. Compare word counts, word TF-IDF, character TF-IDF, combined word/character features, engineered SMS features, and frozen embeddings. Classifiers include Multinomial NB, Complement NB, Logistic Regression, Linear SVM, Ridge, Random Forest, and XGBoost.

The 54 numeric pipeline/treatment configurations each receive a 20-trial randomized search using the same five training folds and per-pipeline spaces across treatments. The six DistilBERT configurations compare three training settings on original and augmented text. Search histories are beside model artifacts. Threshold optimization is separate from parameter fitting. The historical serving selection has its own search/refinement manifests, separate from this equal-budget comparison.

Frozen embedding models train only the downstream classifier. Head-only DistilBERT freezes its encoder; full fine-tuning updates encoder and head. Head-only training is not LoRA: LoRA learns low-rank internal adapters while original weights remain frozen."""),
    code("""model_root = Path("artifacts/imbalance/models")
if not model_root.exists():
    model_root = Path("artifacts/models")
local = [read_json(p) for p in sorted(model_root.glob("*/result.json"))]
display(pd.DataFrame([{"candidate": r["candidate"], "kind": r["kind"], "cv_f1": r["cv_f1"],
    "validation_f1": r["validation"]["spam_f1"], "precision": r["validation"]["precision"],
    "recall": r["validation"]["recall"], "training_s": r["training_seconds"],
    "parameters": r["parameters"]} for r in local]))
if Path("reports/runs/encoder/history.json").exists():
    display(pd.DataFrame(read_json("reports/runs/encoder/history.json")))"""),
    md("""## Training treatments and fold boundaries

The matrix contains 16 classical pipelines and two frozen MiniLM pipelines × three treatments, plus three DistilBERT settings × two treatments = 60 configurations. The always-ham baseline is separate.

`original` uses real fitting messages only. `text_aug` uses Luna-generated, label-preserving variants of fitting-fold spam parents before feature extraction. `smote` interpolates fitting-fold numerical vectors after feature extraction. These are separate arms; ordinary SMOTE does not apply to DistilBERT token sequences.

The final fitting target is 2,710 ham and 2,710 spam; augmentation requires 2,318 accepted additions. Each CV fitting partition has its own target. Generated descendants inherit parent family IDs and cannot enter fitting when their parent family is held out. The original held-out folds, validation and test are never resampled. Class weights are searched explicitly; balanced weighting recomputes from effective fitting counts.

Generated labels are inherited, not independently human-verified. Integrity filters cannot rule out semantic drift, ambiguous labels, repetition, or generator-style shortcuts. The pilot record documents the review method; cost and parent/configuration provenance are persisted.

The head-only and 0.00002 encoder settings use CUDA/float16 autocast. The 0.00005 setting uses isolated CPU-only PyTorch/float32 with dynamic padding and seeded length-grouped batches for **both** treatments. Backend, precision and batching are matched within each setting, but differ across settings; this is not a controlled learning-rate-only ablation. Per-model manifests record actual execution. Full details: `docs/ml_experiments.md`."""),
    code("""if Path("reports/runs/augmentation/summary.json").exists():
    display(read_json("reports/runs/augmentation/summary.json"))
if Path("reports/runs/augmentation/pilot_review.json").exists():
    display(read_json("reports/runs/augmentation/pilot_review.json"))
if Path("reports/augmentation_quality_summary.json").exists():
    display(read_json("reports/augmentation_quality_summary.json"))
figure("augmentation_distributions")
if Path("reports/runs/imbalance/augmentation_fit_counts.json").exists():
    display(pd.DataFrame(read_json("reports/runs/imbalance/augmentation_fit_counts.json")).T)
if Path("reports/runs/imbalance/search_budget_audit.json").exists():
    display(read_json("reports/runs/imbalance/search_budget_audit.json"))
if Path("reports/runs/imbalance/encoder_pair_audit.json").exists():
    audit = read_json("reports/runs/imbalance/encoder_pair_audit.json")
    display(pd.DataFrame([{"setting": name, "treatment": treatment, **policy}
        for name, arms in audit.items() for treatment, policy in arms.items()]))
for history in sorted(Path("reports/runs/imbalance/encoder").glob("*/history.json")):
    display(Markdown("Encoder history: " + history.parent.name))
    display(pd.DataFrame(read_json(history)))
if Path("reports/ml_treatment_comparison.csv").exists():
    display(pd.read_csv("reports/ml_treatment_comparison.csv"))
figure("ml_treatments_validation")
figure("ml_treatments_test")
if Path("reports/ml_treatment_paired_differences.csv").exists():
    display(pd.read_csv("reports/ml_treatment_paired_differences.csv"))
if Path("reports/ml_treatment_effect_summary.csv").exists():
    display(pd.read_csv("reports/ml_treatment_effect_summary.csv"))
    display(Markdown("Treatment win/tie/loss counts are descriptive across these fixed pipelines, not independent replications."))"""),
    md("""## Hosted generative and decision models

Seven LLMs compare zero-shot and fixed four-example few-shot prompts. Examples come from training only. Supported reasoning/output controls are logged. Seven decision models compare explicit `choice` with binary `noul`, using equivalent class definitions. Validation picks one configuration per model. For decision probabilities, validation selects an operating threshold.

Typed output is not a guarantee of correctness or calibration. Luna's generative and decision interfaces are related, so they are not independent underlying model families. Microsoft's decision model is Qwen-derived and may update behind a stable API. Catalog availability is verified separately from actual successful inference.

A budget ledger reserves conservative costs before requests and retains unknown charges. The experiment cap is US$10 total, including a US$1 pilot. Pilot accuracy is not a meaningful ranking; development selection and final evaluation use larger partitions."""),
    code("""display(read_json("configs/candidates.json"))
if Path("reports/candidate_availability.json").exists():
    available = read_json("reports/candidate_availability.json")
    display(pd.DataFrame([{"group": group, "model": model, "listed": info["listed"], "pricing": info["pricing"]}
        for group, models in available.items() for model, info in models.items()]))
for stage in ["development", "validation"]:
    path = Path("reports/runs") / stage / "results.json"
    if path.exists():
        display(pd.DataFrame([{"model": r["candidate"], "variant": r["variant"],
            "n": r["record_count"], "f1": r["metrics"].get("spam_f1"),
            "coverage": r["metrics"]["coverage"], "cost_usd": r["cost_usd"],
            "p95_ms": r["latency"]["p95_ms"]} for r in read_json(path)]))"""),
    md("""## Final held-out comparison

The table below is test performance only. Selections are recorded in configuration files before test execution. Report raw counts, class support, F1, precision/recall, average precision, calibration where applicable, latency, coverage, and cost. SVM/Ridge decision scores are not calibrated probabilities. Brier score is the mean squared error between P(spam) and the 0/1 label; lower is better. Reliability plots compare mean predicted probability with observed spam fraction and display bin counts. Local inference latency includes preprocessing; embedding inference includes encoding. Hosted latency includes network overhead.

p50 is the median; p95 is the time within which 95% of measured requests finish. Local timings use warm sequential requests on the recorded machine; hosted experiments state concurrency. These are practical system measurements, not isolated hardware-speed comparisons.

Separate precision, recall, and accuracy charts compare local models/baseline, LLMs, and decision models in three panels on a common scale. Hosted classification metrics are conditional on successful responses; labels show coverage so unavailable predictions are visible. The always-ham baseline has zero spam recall despite high accuracy. Zero predicted spam is reported as precision zero by convention."""),
    code("""test_results = comparison("test")
if test_results.empty:
    display(Markdown("Final test evaluation is pending; no development results are presented as test results."))
else:
    display(test_results.sort_values(["group", "spam_f1"], ascending=[True, False]))
figure("test_f1")
figure("test_precision")
figure("test_recall")
figure("test_accuracy")
figure("selected_local_diagnostics")
figure("hosted_confusion")
figure("decision_reliability")
if Path("reports/paired_selected_comparisons.csv").exists():
    display(pd.read_csv("reports/paired_selected_comparisons.csv"))
if Path("reports/hosted_test_uncertainty.csv").exists():
    display(pd.read_csv("reports/hosted_test_uncertainty.csv"))
if Path("reports/hosted_operating_points.csv").exists():
    display(pd.read_csv("reports/hosted_operating_points.csv"))"""),
    md("""## Cost and quality tradeoffs

Separate development spend from final evaluation. Report cost per 1,000 attempted and successfully completed SMS, unknown charges, cache/token usage, and projected per-million cost under explicit assumptions. Include retries and failures. Unknown cost is not zero; free promotional pricing may change. Reasoning tokens can already be included in output accounting, so do not double count them.

Local compute has no provider API charge, but hardware, electricity, and hosting are not free. Without a measured infrastructure pricing model, local dollar cost is reported as unmeasured."""),
    code("""if Path("reports/cost_summary.json").exists():
    display(read_json("reports/cost_summary.json"))
    display(pd.read_csv("reports/cost_ledger_summary.csv"))
figure("test_cost")
figure("test_latency")"""),
    md("""## Error analysis, selection, and deployment

Inspect errors after freezing models; this analysis must not feed further tuning on the same test set. False positives represent legitimate SMS flagged, while false negatives represent missed spam. Repeated-template, short-message, URL, digit, and capitalization slices are descriptive; small slices do not support broad claims.

Deploy one selected local ML pipeline, one selected hosted LLM, and one selected decision model. The FastAPI application runs inference only, with validated requests, explicit errors, bounded concurrency/timeouts, and budget controls. Hosted inference requires an OpenRouter key. Docker packages dependencies and the local artifact; hosted inference still requires internet access.

Some misses can reflect conversational-looking spam or jokes labelled spam in the source corpus. Keep the supplied labels unchanged, discuss ambiguous examples, and avoid relabelling test errors to improve reported scores.

Future monitoring needs representative labelled traffic, error/latency/cost signals, version tracking, challenger evaluation, and rollback. Distribution drift alone does not prove accuracy loss."""),
    code("""if Path("configs/local_selection.json").exists():
    display(read_json("configs/local_selection.json"))
if Path("configs/hosted_frozen.json").exists():
    frozen = read_json("configs/hosted_frozen.json")
    display(pd.DataFrame([{"group": group, "model": cfg["model"], "variant": cfg["variant"],
        "threshold": cfg.get("threshold"), "selection_stage": frozen["selection_stage"]}
        for group, cfg in frozen["champions"].items()]))
if Path("reports/local_test_errors.csv").exists():
    errors = pd.read_csv("reports/local_test_errors.csv")
    display(errors.head(20))
    display(Markdown(f"Local champion errors: {len(errors)}. Full examples saved in reports/local_test_errors.csv."))
if Path("reports/error_slices.csv").exists():
    slices = pd.read_csv("reports/error_slices.csv")
    selected = read_json("configs/local_selection.json")["candidate"]
    if (slices.candidate == selected + "__original").any():
        selected += "__original"
    display(slices[slices.candidate == selected])
    display(Markdown("Slices are descriptive, overlap, and can have very few spam examples. Do not interpret no-spam-slice F1 as a useful quality estimate."))"""),
    md("""## Reproduction and limitations

See the [modelling guide](README.md) for CLI usage, the [experiment protocol](../docs/ml_experiments.md) for reproduction and evaluation boundaries, and [results](../reports/RESULTS.md) for findings. The [architecture](../docs/architecture.md) covers shared inference, tracking, and deployment.

Results are specific to the recorded split, dataset, code/configuration hashes, hardware, provider routing, and execution date. This small historical public corpus cannot establish modern production performance or exclude pretrained contamination. Hosted inference can change behind stable names. Training family grouping is conservative and may miss paraphrases. Calibration and high-precision estimates have limited sample support.

The repository contains saved results so reading this notebook need not spend credits. Regenerating hosted experiments requires an explicit budget and key."""),
]
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nbf.write(nb, "task_1_modelling/analysis.ipynb")
# The documented development environment includes Ruff; keep generated cells clean.
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "ruff", "check", "task_1_modelling/analysis.ipynb", "--fix"], check=True
)
subprocess.run(
    [sys.executable, "-m", "ruff", "format", "task_1_modelling/analysis.ipynb"], check=True
)
print("Notebook written")
