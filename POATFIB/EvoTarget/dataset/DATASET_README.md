# Clean Modular Dual-Scale STS ACSD v2.81 POAF Dataset (`dataset/`)

This directory contains two hospital-disjoint benchmarks with toggleable pre-incision features, derived from 367,695 labeled STS ACSD v2.81 isolated-CABG procedures (`PRIMARY_TARGET_COtAFib`, 0/1).

| Scale (`scale=...`) | Train Rows (Hospitals) | Test Rows (Unseen Hospitals) | Hospital Overlap | Train POAF | Test POAF | Primary Use Case |
|---|---:|---:|---:|---:|---:|---|
| **`full`** (94.2% / 5.8%) | **346,454** (1,017 hospitals, 5 folds) | **21,241** (74 unseen hospitals, 5,160 POAF events) | **0.0%** | 24.30% | 24.29% | Primary retrospective evaluation at observed prevalence |
| **`small_50k_10k`** (50/50 balanced) | **50,000** (1,017 hospitals, 5,000 per class per fold) | **10,000** (74 unseen hospitals, 5,000 per class, subset of `full/test`) | **0.0%** | **50.00%** | **50.00%** | Matched training budget and balanced-sample sensitivity |

The full partition was chosen using covariates and outcome counts. The [label-blind hospital match](../FEATURE_MATCHED_EVALUATION.md) is a separate design.

## Pre-incision feature groups
- **Raw pre-incision predictors (`207` columns)**: Exact canonical STS ACSD v2.81 names (NO prefix; `Region` preserved as native 9-level pandas `category`). Zero `OUTPUT_VARIABLE_STS.md`, zero post-incision, zero `HospID`/`DataYear`/`Surgorder`.
- **Augmented features (`aug_*` prefix)**:
  - `aug_b1__*`: Batch-1 pure row-deterministic clinical features (58 candidates).
  - `aug_b2a__*`: Track-2A pure row-deterministic clinical features (24 candidates).
  - `aug_b2b__*` contains 10 hospital-grouped out-of-fold auxiliary risk features. Each scale fits these features within its own development rows.

## Quick Start (`dataset_loader.py`)

```python
import sys
sys.path.insert(0, "dataset")
from dataset_loader import load_sts_dataset, load_inner_oof_folds, list_feature_groups

# 1. Raw pre-incision predictors on the full 94.2/5.8 split
ds_full_raw = load_sts_dataset(scale="full", include_groups=["raw_all"])

# 2. Raw + kept deterministic clinical features (exclude nested OOF priors)
ds_full_clin = load_sts_dataset(scale="full", include_groups=["raw_all", "aug_clinical_kept"])

# 3. Raw + all kept augmented features (clinical + nested OOF priors)
ds_full_aug = load_sts_dataset(scale="full", include_groups=["raw_all", "aug_all_kept"])

# 4. Switch seamlessly to the exact 50/50 balanced 50k / 10k dataset!
ds_small_raw = load_sts_dataset(scale="small_50k_10k", include_groups=["raw_all"])
ds_small_aug = load_sts_dataset(scale="small_50k_10k", include_groups=["raw_all", "aug_all_kept"])

# 5. Fine-grained toggles (drop a family, drop a feature, or opt-in any tested candidate)
ds_custom = load_sts_dataset(
    scale="small_50k_10k",
    include_groups=["raw_all", "aug_all_kept"],
    exclude_groups=["aug_b2b_kept"],
    include_features=["aug_b1__score_poaf_mariscalco", "aug_b1__egfr_ckdepi_2021"],
)
```

## Ready-to-Use CSV Exports (`PRIMARY_TARGET_COtAFib` = Final Rightmost Column)

Both scales include clean, ML-ready CSVs (`NO row_id`, `NO auxiliary targets`, `PRIMARY_TARGET_COtAFib` strictly in the **final last rightmost column**):
- **Full (`94.2% / 5.8%`)**: `dataset/full/train.csv` and `dataset/full/test.csv`
- **Small (`50k / 10k`, exact 50/50)**: `dataset/small_50k_10k/train.csv` and `dataset/small_50k_10k/test.csv`

```python
import pandas as pd

train_df = pd.read_csv("dataset/small_50k_10k/train.csv")
assert train_df.columns[-1] == "PRIMARY_TARGET_COtAFib"
feature_cols = list(train_df.columns[:-1])
raw_only = [c for c in feature_cols if not c.startswith("aug_")]
aug_only = [c for c in feature_cols if c.startswith("aug_")]
```

## Bayes Prior-Shift Recalibration for `small_50k_10k`

The sampler balances the outcome and matches feature profiles within classes. The logit offset uses the full development cohort's 24.30% event rate and assumes prior probability shift. The shift preserves the score ranking on a fixed set of rows. Calibration and performance at observed prevalence are evaluated on `full/test`. The class-ratio weights below use the parent holdout class proportions for a descriptive comparison on the nested 10k slice.

```python
import numpy as np
from sklearn.metrics import log_loss, average_precision_score

ds = load_sts_dataset(scale="small_50k_10k", include_groups=["raw_all", "aug_all_kept"])
# Suppose p_bal are predicted probabilities from a model trained on ds.X_train (50/50)
p_clip = np.clip(p_bal, 1e-6, 1.0 - 1e-6)
logit_nat = np.log(p_clip / (1.0 - p_clip)) - ds.bayes_logit_offset
p_nat = 1.0 / (1.0 + np.exp(-logit_nat))

# Class-ratio-weighted log loss and average precision on the balanced test slice.
nat_ll = log_loss(ds.y_test, p_nat, sample_weight=ds.sample_weight_test_natural_prior)
nat_pr = average_precision_score(ds.y_test, p_nat, sample_weight=ds.sample_weight_test_natural_prior)
```

## Embedded Stress Slices (`ds.stress_masks_test`)
- `stress_2017_unseen_hosp` selects 2017 cases at unseen hospitals. Development includes 2017 cases from other hospitals.
- `stress_rare_clinical_tails`: Top-5% Mahalanobis leverage + extreme clinical phenotypes (shock, dialysis, HDEF <30%, age ≥80, severe MR, AS, inotropes).
- `stress_denovo_no_preop_afib` selects rows with no recorded preoperative AF history.
