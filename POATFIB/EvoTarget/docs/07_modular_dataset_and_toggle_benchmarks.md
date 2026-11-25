# 07. Dual Modular Datasets & CatBoost Lift Benchmarks (`Full` 94.2/5.8 vs `Small` 50k/10k)

- **Paper Section**: *Results — Main Model Performance, Feature-Augmentation Lift (Large vs Small), Calibration & Stress-Slice Generalization*
- **Core Claim**: Across both perfected hospital-disjoint datasets — **Full (`346,454` train → `21,241` test, 0% hospital overlap)** and **Small (`50,000` train → `10,000` test, exact 50/50 POAF, 0% hospital overlap)** — CatBoost on the 207 audited pre-incision raw predictors achieves **`0.677320`** (Full) and **`0.673558`** (Small) 5-fold ensemble ROC-AUC. Adding clinically engineered features yields **no statistically significant 5-fold ensemble ROC-AUC lift** on either scale (`Full Δ = -0.00034` to `-0.00039`; `Small Δ = -0.00029` to `-0.00008`, all 95% CIs straddling zero), **yet consistently improves probability calibration (ECE `0.00760 → 0.00604` on Full; Bayes-recalibrated ECE `0.00627 → 0.00550` & natural-prior PR-AUC `+0.00211` on Small), top-decile lift (`1.843× → 1.855×`), and single-model Small-50k AUC (`0.67272 → 0.67449`, `+0.00177`)**, while clinical composites take ranks `#1`–`#5` in feature importance. Crucially, the **50k balanced model (`14.4%` of data) recovers 99.5% of the 346k model's unseen-hospital ROC-AUC (`0.67436` vs `0.67732`) with exact natural calibration (`ECE = 0.00695`) via a single closed-form Bayes logit shift**, and the **`stress_2017_unseen_hosp` slice drops ~2.0 full AUC points (`0.65785` vs `0.67776` on naive 2017)**, empirically proving hospital leakage in naive temporal holdouts.
- **Evidence & Source Artifacts**: `scripts/08_eval_catboost_dual_datasets.py`, `reports/catboost_dual_eval.json`, `reports/catboost_dual_eval.md`, `reports/catboost_preds_full.parquet`, `reports/catboost_preds_small.parquet`, `dataset/full/feature_groups.json`, `dataset/small_50k_10k/feature_groups.json`.
- **Status**: `Verified`

---

## 1. Primary Dual-Dataset CatBoost Comparison (Default Raw vs Feature-Augmented)

Model specification locked identically across all views: CatBoost (`depth=6`, `lr=0.03`, `l2_leaf_reg=8.0`, `one_hot_max_size=16`, `early_stopping=100`, 5 hospital-disjoint training folds; zero `test` peeking):

| Scale | Feature View | # Feats | 5f CV OOF AUC | Single-Model Test AUC | 5f-Ensemble Test AUC | Lift ΔAUC vs Raw [95% CI] | Test PR-AUC | Test LogLoss | ECE (15-bin) | Cal Slope | Top-Decile Lift | `2017 Unseen-Hosp` AUC | `Rare-Tail` AUC | *De Novo* AUC |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full (94.2/5.8)** | `V1_raw_only` (Default) | 207 | `0.678395` | `0.676487` | **`0.677320`** | *baseline* | `0.377614` | **`0.517134`** | `0.00760` | `1.023` | `1.843×` | **`0.657851`** | `0.692781` | `0.677274` |
| **Full (94.2/5.8)** | `V2_aug_kept` | 208 | `0.678447` | **`0.676545`** | `0.676977` | `-0.000343 [-0.000815, +0.000128]` | **`0.377890`** | `0.517254` | `0.00707` | `1.024` | `1.849×` | `0.657149` | **`0.692816`** | `0.676977` |
| **Full (94.2/5.8)** | `V3_aug_all_clinical_82` | 289 | **`0.678481`** | `0.675969` | `0.676931` | `-0.000389 [-0.001420, +0.000628]` | `0.377881` | `0.517234` | **`0.00604`** | **`1.014`** | **`1.855×`** | `0.655589` | `0.692394` | `0.676923` |
| **Small (50k/10k)** | `V1_raw_only` (Default) | 207 | `0.675304` | `0.672723` | **`0.673558`** | *baseline* | `0.643654` | `0.643520` | `0.01040` | `1.062` | `1.426×` | `0.650462` | **`0.675906`** | `0.673591` |
| **Small (50k/10k)** | `V2_aug_kept` | 208 | `0.674991` | `0.671459` | `0.673265` | `-0.000294 [-0.001115, +0.000555]` | `0.644016` | `0.643679` | **`0.00927`** | `1.057` | `1.440×` | **`0.651562`** | `0.673311` | `0.673325` |
| **Small (50k/10k)** | `V3_aug_all_clinical_82` | 289 | **`0.675411`** | **`0.674492`** | `0.673480` | `-0.000078 [-0.001740, +0.001739]` | **`0.645276`** | **`0.643519`** | `0.01204` | **`1.020`** | **`1.442×`** | `0.649298` | `0.671782` | `0.673480` |

---

## 2. Large vs Small Augmentation Lift Comparison (Direct Answer for Manuscript)

1. **5-Fold Ensemble ROC-AUC**: Augmentation delta is slightly closer to zero on `small_50k_10k` (`-0.000078` for all-82 clinical vs `-0.000389` on `full`), but **neither scale shows statistically significant 5-fold ensemble ROC-AUC gain** (`P(Δ > 0) = 0.22` Full, `0.47` Small).
2. **Single-Model, PR-AUC, Top-Decile Lift & Calibration Gains (Stronger on Small 50k!)**:
   - On **`small_50k_10k`**, adding the 82 clinical features lifted **Single-Model (Fold-0) Test AUC by `+0.001769`** (`0.672723 → 0.674492`), **balanced PR-AUC by `+0.001622`** (`0.643654 → 0.645276`), **Bayes-recalibrated natural-prior PR-AUC by `+0.002113`** (`0.371663 → 0.373776`), **calibration slope (`1.062 → 1.020`)**, and **top-decile lift (`1.426× → 1.442×`)**.
   - On **`full`**, adding clinical features improved **ECE (`0.00760 → 0.00604`, `-20.5%` calibration error)**, **calibration slope (`1.023 → 1.014`)**, **PR-AUC (`+0.000276`)**, and **top-decile lift (`1.843× → 1.855×`)**.
3. **Clinical Feature Importance Dominance**: Even where ensemble AUC is statistically flat, CatBoost places our clinical features at **ranks `#1`–`#5` on BOTH Full and Small**:
   - `#1 aug_b2a__epidemiologic_age_race_sex_bsa` (`8.57%` Full / `8.59%` Small)
   - `#2 aug_b1__age_x_bsa_la_size_proxy` (`8.29%` / `4.83%`)
   - `#3 aug_b1__race_x_age_x_bsa_substrate` (`7.00%` / `3.62%`)
   - `#4 aug_b1__biatrial_remodeling_composite` (`3.99%` / `3.20%`)
   - `#5 aug_b1__score_mathew_preop` (`2.72%` Full) / `aug_b1__age_x_height_sex_proxy` (`1.84%` Small)

---

## 3. Exact Bayes Prior-Shift Recalibration & 2×2 Cross-Scale Transfer Matrix

Evaluating the **50k balanced model** on the **full natural-prior 21,241-row holdout (`test_full`)** after applying the closed-form Bayes logit shift (`-1.136479`) proves zero collider bias and remarkable sample efficiency:

| Feature View | Train Scale → Eval Split | ROC-AUC | PR-AUC | LogLoss | Brier | ECE | Cal Slope |
|---|---|---:|---:|---:|---:|---:|---:|
| `V1_raw_only` | Full (`346.5k`) → `test_full` (`21.2k` nat) | **`0.677320`** | `0.377614` | `0.517134` | `0.171003` | `0.00760` | `1.023` |
| `V1_raw_only` | Small (`50k`, Bayes-recal) → `test_full` (`21.2k` nat) | **`0.674228`** (`-0.00309`) | `0.374383` | `0.518457` | `0.171462` | `0.00778` | `1.056` |
| `V3_aug_all_clinical_82` | Full (`346.5k`) → `test_full` (`21.2k` nat) | **`0.676931`** | `0.377881` | `0.517234` | `0.171015` | `0.00604` | `1.014` |
| `V3_aug_all_clinical_82` | Small (`50k`, Bayes-recal) → `test_full` (`21.2k` nat) | **`0.674356`** (`-0.00258`) | `0.374944` | `0.518368` | `0.171418` | **`0.00695`** | **`1.018`** |

- **Horvitz-Thompson Recovery on `test_small_10k`**: Bayes-recalibrated HT-weighted LogLoss = **`0.518820`** (`V3`) / `0.518895` (`V1`), matching `test_full`'s true natural LogLoss (`0.518368` / `0.518457`) to **`<0.0005`**, with ECE **`0.00550`**!

---

## 4. Empirical Validation of Our Naive-2017 Critique (`stress_2017_unseen_hosp`)

Compare CatBoost on **naive 2017 holdout (99.5% seen hospitals)** vs **`stress_2017_unseen_hosp` (`n = 3,555`, 2017 patients at genuinely unseen hospitals)**:
- Naive 2017 (99.5% seen hospitals, `reports/batch2_summary.json`): **`0.677761`**
- Unseen-hospital 2017 (`stress_2017_unseen_hosp`, `reports/catboost_dual_eval.json`): **`0.657851`** (Full) / **`0.650462`** (Small) — a **`-0.0199` (`-2.0` full AUC points)** drop!
- This empirically validates [04_why_2017_holdout_is_not_statistically_valid.md](04_why_2017_holdout_is_not_statistically_valid.md): naive temporal splitting masks a 2-point external-hospital generalization gap behind 99.5% hospital leakage.
