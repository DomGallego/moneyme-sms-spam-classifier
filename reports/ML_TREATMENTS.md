# ML treatment benchmark

Validation complete: 60/60. Exploratory test complete: 60/60.

The 60 configurations compare original fitting data, LLM-generated training SMS, and fold-local SMOTE. See `docs/ml_experiments.md` for the complete protocol.

## Per-treatment validation leaders

| Treatment | Pipeline | Validation spam F1 |
|---|---|---:|
| original | logistic_char | 0.9690 |
| smote | logistic_char | 0.9653 |
| text_aug | svm_char | 0.9653 |

## Matched treatment effects

| Stage | Treatment | Paired pipelines | Higher F1 | Equal F1 | Lower F1 | Median F1 change |
|---|---|---:|---:|---:|---:|---:|
| test | text_aug | 21 | 8 | 3 | 10 | +0.0000 |
| test | smote | 18 | 5 | 1 | 12 | -0.0041 |
| validation | text_aug | 21 | 10 | 1 | 10 | +0.0000 |
| validation | smote | 18 | 4 | 2 | 12 | -0.0040 |

These counts summarize this fixed set of heterogeneous pipelines. They are descriptive, not independent replications or a general probability that a treatment helps. Training wall times can include concurrent work and are not isolated hardware comparisons.

`ml_treatment_comparison.csv` contains every completed result. `ml_treatment_paired_differences.csv` gives paired family-bootstrap differences from each pipeline's original-data arm. These are exploratory intervals without multiple-comparison correction. Read metric differences with counts, uncertainty, training cost, and label-quality limitations.

All validation/test messages remain original. The existing test has been inspected, so its treatment results are exploratory rather than a fresh independent estimate. The original serving selections and artifacts remain separately frozen. No API release is changed automatically.

Validation-selected treatment champion: `logistic_char__original`. This selection preceded treatment test prediction, but the existing test's prior inspection remains a limitation.
Its exploratory test spam F1 is 0.9644, precision 0.9919, recall 0.9385.
