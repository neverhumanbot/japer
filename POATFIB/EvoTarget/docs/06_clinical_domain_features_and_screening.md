# 06. Clinical Domain Feature Engineering & Empirical Screening (92 Candidates Across 3 Tracks)

- **Paper Section**: *Methods — Domain-Expert Feature Engineering* & *Results — Large-Sample Feature Saturation vs Clinical Scores*
- **Core Claim**: We systematically engineered and screened **92 strictly pre-incision clinical features** across **three orthogonal tracks** (58 classic STS/POAF scores & ratios in Batch 1; 24 nonlinear/physiological interactions in Track 2A; 10 strictly nested Hospital-Grouped OOF multi-task auxiliary complication risk priors in Track 2B). Across $N = 306{,}143$ development patients (2014–2016, 5-fold Hospital-Grouped CV), only **1 feature (`diabetes_age_decoupled`)** cleared strict forward selection (`CatBoost +0.000144`, `LightGBM +0.000242`), and none improved sealed 2017 holdout ROC-AUC (`-0.00020`, 95% CI `[-0.00046, +0.00006]`). Meanwhile, **native one-hot handling of `Region` (`one_hot_max_size=16`) + 5-fold GBDT ensembling** lifted raw 2017 ROC-AUC from `0.67443` to **`0.67828` (`+0.00385`)** — demonstrating a fundamental large-$N$ tabular ML insight: on >300k high-resolution STS patients, depth-6 GBDTs already saturate classical clinical risk scores and multi-task complication priors directly from the 207 clean pre-incision predictors.
- **Evidence & Source Artifacts**: `features/registry.json`, `features/batch2_registry.json`, `features/tested/*.json`, `features/batch2_tested/*.json`, `reports/feature_screening_leaderboard.md`, `reports/batch2_leaderboard.md`, `reports/batch2_summary.json`.
- **Status**: `Verified`

---

## 1. Three-Track Clinical Feature Taxonomy (92 Audited Candidates)

Every candidate feature carries an individual JSON metadata card (`features/tested/*.json` and `features/batch2_tested/*.json`) verifying formula, clinical rationale, literature citation, training distribution, screening deltas, and zero leakage (`test_batch2_contract.py` **PASSED 100%**):

| Track | Count | Clinical Families | Computation | Promoted on 306k CV |
|---|---:|---|---|---:|
| **Batch 1** (`aug_b1__*`) | `58` | Validated POAF scores (Mariscalco POAF, Burgos COM-AF, CHA₂DS₂-VASc, HATCH, Mathew), CKD-EPI 2021 eGFR, Cockcroft-Gault, ALBI hepatic score, WHO anemia, Leaman coronary jeopardy, biatrial remodeling, beta-blocker withdrawal, pulmonary deficit | Pure row-deterministic | `0 / 58` (Codex Batch-1 audit) |
| **Track 2A** (`aug_b2a__*`) | `24` | Orthogonal physiological/nonlinear terms: `diabetes_age_decoupled`, `hfpef_diastolic_substrate_score`, `prior_revasc_scar_burden`, `bsa_height_excess_residual`, `height_sex_standardized_z`, `age_gompertz_atrial_hazard`, `epidemiologic_age_race_sex_bsa`, ischemic vs degenerative MR etiology, calcific AS, cardiorenal-anemia triad, warfarin-free INR, DO₂ proxy, antithrombotic load, BB contraindication × age | Pure row-deterministic | **`1 / 24`** (`diabetes_age_decoupled`) |
| **Track 2B** (`aug_b2b__*`) | `10` | Strictly nested Hospital-Grouped OOF multi-task risk logits predicting all 8 clean STS complications strictly from pre-incision inputs ($C(5,3)=10$ inner + $C(5,4)=5$ outer folds) + 2 multi-organ physiological cascades (`oof_reconstructed_sts_morb_mort`, `oof_intrathoracic_inflam_cascade`) | Nested Hospital-Grouped OOF | `0 / 10` |

---

## 2. Track-2B Multi-Task Auxiliary Complication Predictability (Pre-Incision Only)

Even though adding nested auxiliary risk logits did not add incremental POAF AUC above the 207 raw predictors, the pre-incision auxiliary models themselves achieved strong cross-hospital discrimination across 1,086 centers (`n = 306,143`):

| Nested OOF Feature | Predicted STS Auxiliary Complication | 5-Fold Hospital-Grouped OOF ROC-AUC | Univariate POAF AUC | Incremental POAF CB ΔAUC |
|---|---|---:|---:|---:|
| `oof_logit_crenfail` | Postoperative Renal Failure | **`0.83654`** | `0.5694` | `-0.000102` |
| `oof_logit_mortalty` | Operative / 30-Day Mortality | **`0.80476`** | `0.5758` | `-0.000437` |
| `oof_logit_cpvntlng` | Prolonged Ventilation (>24 h) | **`0.76533`** | `0.5453` | `-0.000001` |
| `oof_logit_cotarrst` | Postoperative Cardiac Arrest | **`0.74593`** | `0.5608` | `-0.000316` |
| `oof_logit_csepsis` | Postoperative Sepsis | **`0.73133`** | `0.5638` | `-0.000664` |
| `oof_logit_cppneum` | Postoperative Pneumonia | **`0.72391`** | `0.5420` | `-0.000247` |
| `oof_logit_cpleff` | Pleural Effusion Requiring Drainage | **`0.68445`** | `0.5624` | `-0.000081` |
| `oof_logit_cvte` | Venous Thromboembolism (DVT/PE) | **`0.64589`** | `0.5626` | `-0.000611` |

---

## 3. Track-2A Notable Individual Signals & Sealed 2017 Holdout Verification

Top Track-2A candidates on 5-fold Hospital-Grouped CV (`2014–2016`, `B1` raw baseline: CatBoost `0.677947`, LightGBM `0.677398`):
- `diabetes_age_decoupled` (**Promoted**): `uAUC = 0.6440`, global CatBoost rank **#3**, CB `ΔAUC = +0.000144` (3/5 folds), LGB `ΔAUC = +0.000242` (4/5 folds).
- `hfpef_diastolic_substrate_score` (Stage-1 Pass): `uAUC = 0.5869`, rank **#17**, CB `ΔAUC = +0.000141` (4/5 folds), LGB `ΔAUC = +0.000247`.
- `prior_revasc_scar_burden` (Stage-1 Pass): CB `ΔAUC = +0.000070` (3/5 folds), LGB `ΔAUC = +0.000214`.
- `B1 + All 24 Track-2A` block: LightGBM gained **`+0.000721`** (`0.677398 → 0.678119`), whereas CatBoost (`one_hot_max_size=16`) shifted `-0.000138`.

### Sealed 2017 Temporal Holdout Verification (`n = 61,552`, `reports/batch2_summary.json`)

| Model | Feature Set | 5f CV OOF AUC | Sealed 2017 ROC-AUC | Sealed 2017 PR-AUC | Sealed 2017 LogLoss | *De Novo* 2017 AUC | Paired Bootstrap Δ vs B1 [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|
| CatBoost (5-bag) | `B1` (207 raw) | `0.678124` | **`0.677761`** | `0.388151` | `0.522711` | `0.678014` | — |
| CatBoost (5-bag) | `B1 + Track-2A` | `0.678067` | `0.677566` | `0.388622` | `0.522736` | `0.677831` | `-0.000195 [-0.000461, +0.000060]` |
| LightGBM (5-bag) | `B1` (207 raw) | `0.677542` | **`0.678096`** | `0.387987` | `0.522627` | `0.678402` | — |
| LightGBM (5-bag) | `B1 + Track-2A` | `0.677867` | `0.678037` | `0.388145` | `0.522620` | `0.678368` | `-0.000059 [-0.000421, +0.000302]` |
| **50/50 CB+LGB Blend** | **`B1` (207 raw)** | **`0.678445`** | **`0.678283`** | **`0.388613`** | **`0.522508`** | **`0.678562`** | — |
| 50/50 CB+LGB Blend | `B1 + Track-2A` | `0.678567` | `0.678138` | `0.388970` | `0.522528` | `0.678436` | `-0.000145 [-0.000385, +0.000098]` |

---

## 4. Scientific Interpretation for Manuscript Discussion

1. **Why Classic Clinical Scores & Multi-Task OOF Priors Plateau at $N > 300\text{k}$**: POAF has high irreducible physiological stochasticity (intraoperative pericardial inflammation, autonomic surge, electrolyte flux — all strictly unobservable pre-incision). With 306k patients and 207 rich raw STS predictors, modern oblivious/leaf-wise trees already approximate smooth nonlinear interactions and multi-complication risk manifolds directly.
2. **All 92 Candidates Preserved for Small-Sample & Foundation-Model Studies**: Because domain-engineered inductive biases often behave differently at smaller sample sizes ($N = 50\text{k}$) or inside tabular foundation models (TabPFN, neural nets, linear/GAM baselines), all 92 audited `aug_*` features are preserved cleanly inside `all_aug_candidates.parquet` and benchmarked head-to-head on `full` (`346k`) vs `small_50k_10k` (`50k`) in [07_modular_dataset_and_toggle_benchmarks.md](07_modular_dataset_and_toggle_benchmarks.md).
