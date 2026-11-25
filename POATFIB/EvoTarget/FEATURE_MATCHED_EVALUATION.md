# STS AFib feature-matched hospital test

Use the feature-matched split to measure transfer to unseen hospitals with a similar patient mix. Use the 2017 cohort to measure temporal transfer, including the changes in practice and documentation that actually occurred. The two tests answer different questions.

## Cohorts

| Role | Cases | Hospitals | Purpose |
| --- | ---: | ---: | --- |
| `train` | 244,881 | 862 | Model development in 2014–2016 |
| `matched_hospital_test` | 61,262 | 224 | Test on unseen hospitals with comparable pre-incision features |
| `temporal_2017` | 61,552 | 1,003 | Test on cases that occurred later |

All 224 matched-test hospitals are absent from training. The matched test contains about 20% of the cases in each development year. Of the 1,003 hospitals with cases in 2017, 790 also have cases in the new matched training set. Join the new roles by `row_id`. Development rows assigned to training retain their original hospital CV fold. Both test roles have `cv_fold=-1`. Split Parquet (`data/feature_matched_splits.parquet`) · [Assignment receipt](reports/feature_matched_split_receipt.json)

## How the split was selected

1. Define the reference population as all labeled-eligible 2014–2016 cases. Use all **207 audited raw pre-incision predictors** and read zero outcome values. Use hospital IDs solely to allocate whole hospitals and year solely to balance enrollment across years.
2. Encode clinical profiles using exact codes for discrete columns and percentile bins for continuous columns, with bins for the tails. Give missingness its own level. Balance the distributions for each variable and a two-bandwidth kernel approximation of the joint profiles. The joint comparison helps detect combinations of measurements and missingness that marginal balancing can miss.
3. Draw 256 random whole-hospital test sets near 20% of the reference cases. Use their typical discrepancies to put marginal, joint, and year balance on comparable scales. Refine eight strong starting allocations with seeded hospital swaps, with a hard test size of 19.5%–20.5%. The objective weights are 45% marginal feature balance, 40% joint balance, 10% year balance, and 5% size. The search produces a reproducible low-discrepancy allocation.
4. Check the fixed assignment with different percentile cuts and an independent random-feature basis. Compare against another 256 random hospital splits and the 2017 cohort. Check all 58 candidate transformations separately after selection and train a hospital-grouped cohort classifier as a diagnostic for remaining joint shift.

The method adapts covariate-balanced sampling to whole hospitals. The joint distance uses a random Fourier approximation to maximum mean discrepancy. [Balanced sampling](https://doi.org/10.1093/biomet/91.4.893) · [Kernel two-sample comparison](https://jmlr.org/papers/v13/gretton12a.html) · [Random Fourier features](https://papers.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) · Selection code (`scripts/05_feature_balanced_split.py`)

## Balance on the selected cohort

Total variation (TV) is half the sum of absolute differences in bin shares. Lower values indicate closer distributions. The kernel distance summarizes differences across joint profiles on a common basis.

| Comparison | Mean TV across 207 raw features | 95th percentile feature TV | Largest feature TV | Joint kernel distance |
| --- | ---: | ---: | ---: | ---: |
| Matched test vs. its training cohort | 0.45 pp | 1.22 pp | 2.38 pp | 0.00725 |
| Random hospital splits, median of 256 | 1.18 pp | 3.32 pp | 11.02 pp | 0.01739 |
| 2017 vs. all 2014–2016 cases | 1.39 pp | 4.79 pp | 6.10 pp | 0.01938 |

The matched set had smaller mean TV and joint distance than the fifth percentile of the random reference. Its largest raw gaps were albumin (2.38 pp), region (1.95 pp), and proximal LAD stenosis (1.79 pp). The largest 2017 gaps included availability of ventricular dimensions (6.10 pp), LV systolic dimension (6.07 pp), and LV diastolic dimension (6.06 pp).

Across the 58 optional transformations, mean TV was 0.82 pp for the matched contrast and 1.85 pp for the temporal contrast. The three-fold cohort classifier had mean AUC 0.449 for matched, 0.499 for a typical random split, and 0.588 for 2017. The below-chance score on matched indicates rankings reversed on held-out hospital folds. Use the measured distribution distances for the balance comparison. [Full balance results](reports/feature_matched_split_receipt.json) · [Candidate-layer diagnostics](reports/candidate_feature_layer_receipt.json)

The matched cohort measures transfer at similar observed case mix. The 2017 results quantify performance under the observed shift. A deployment evaluation can report both with hospital-clustered confidence intervals. The earlier Batch 1 screen used all 2014–2016 outcomes, so its results are prior evidence on the cases that now belong to the matched test. For new comparisons, do all preprocessing, feature selection, and tuning inside the `train` folds and freeze decisions before scoring either test role. A wholly untouched confirmatory study requires a new cohort. [Clinical prediction model evaluation](https://www.bmj.com/content/384/bmj-2023-074819)

## Switch predictor views

From the foundation directory, use the same rows, labels, and CV assignments in each view.

```python
from scripts.training_data import load_training_data

X, y, fold = load_training_data(features="raw")
X_candidates, y, fold = load_training_data(features="candidates")
X_subset, y, fold = load_training_data(
    features="candidates",
    derived_columns=["egfr_ckdepi_2021", "combined_pulmonary_deficit"],
)
X_published, y, fold = load_training_data(features="published")
```

The raw view has 207 features. The candidate view adds up to 58 audited row-deterministic transformations from the candidate layer. The published view reflects the promotion rules. In Batch 1 none of the ten shortlisted candidates achieved the required development gain, so the published view currently has the same columns as the raw view. The 2017 baseline ROC-AUC was 0.674429. `split="original"` gives the original training cohort. Training loader (`scripts/training_data.py`) · [Score report](reports/feature_screening_leaderboard.md)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/05_feature_balanced_split.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/training_data.py
OPENBLAS_NUM_THREADS=1 python -m pytest -q scripts/tests/test_feature_matched_training.py
```
