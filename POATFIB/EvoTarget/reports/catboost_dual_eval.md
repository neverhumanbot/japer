# CatBoost Dual-Dataset Evaluation Report (`full` 94.4/5.6 & `small_50k_10k` Exact 50/50)

- **Generated (UTC)**: `2026-11-25T09:02:58.933047+00:00` (Elapsed `104.3s`)
- **Model**: 5-fold Hospital-Disjoint CatBoost ensemble (`depth=6`, `lr=0.03`, `l2=8.0`, `one_hot_max_size=16`, `early_stopping=100`)

## 1. Primary Results Across Both Perfected Datasets & Feature Toggles (Default Raw vs Augmented)

| Dataset Scale | Feature Toggle | # Feats | 5f CV OOF AUC | Single-Model Test AUC | 5f-Ensemble Test AUC | Test PR-AUC | Test LogLoss | Test Brier | ECE (15-bin) | Cal Slope | Top-Decile Lift | 2017 Unseen-Hosp AUC | Rare-Tail AUC | *De Novo* AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full (94.2/5.8)** | `V1_raw_only` | 207 | 0.678395 | 0.676487 | **0.677320** | 0.377614 | 0.517134 | 0.171003 | 0.00760 | 1.023 | 1.843x | 0.657851 | 0.692781 | 0.677274 |
| **Full (94.2/5.8)** | `V2_aug_kept` | 208 | 0.678447 | 0.676545 | **0.676977** | 0.377890 | 0.517254 | 0.171033 | 0.00707 | 1.024 | 1.849x | 0.657149 | 0.692816 | 0.676977 |
| **Full (94.2/5.8)** | `V3_aug_all_clinical_82` | 289 | 0.678481 | 0.675969 | **0.676931** | 0.377881 | 0.517234 | 0.171015 | 0.00604 | 1.014 | 1.855x | 0.655589 | 0.692394 | 0.676923 |
| **Small (50k/10k 50/50)** | `V1_raw_only` | 207 | 0.675304 | 0.672723 | **0.673558** | 0.643654 | 0.643520 | 0.226375 | 0.01040 | 1.062 | 1.426x | 0.650462 | 0.675906 | 0.673591 |
| **Small (50k/10k 50/50)** | `V2_aug_kept` | 208 | 0.674991 | 0.671459 | **0.673265** | 0.644016 | 0.643679 | 0.226444 | 0.00927 | 1.057 | 1.440x | 0.651562 | 0.673311 | 0.673325 |
| **Small (50k/10k 50/50)** | `V3_aug_all_clinical_82` | 289 | 0.675411 | 0.674492 | **0.673480** | 0.645276 | 0.643519 | 0.226383 | 0.01204 | 1.020 | 1.442x | 0.649298 | 0.671782 | 0.673480 |

## 2. Bayes Prior-Shift Recalibration & Horvitz-Thompson Natural-Prior Recovery (`small_50k_10k`)

| Feature Toggle | Balanced Test LogLoss | Bayes-Recal HT-Weighted LogLoss | Bayes-Recal HT-Weighted PR-AUC | Bayes-Recal HT-Weighted Brier | Bayes-Recal ECE |
|---|---:|---:|---:|---:|---:|
| `V1_raw_only` | 0.643520 | **0.518895** | 0.371663 | 0.171636 | 0.00627 |
| `V2_aug_kept` | 0.643679 | **0.518977** | 0.372032 | 0.171643 | 0.00632 |
| `V3_aug_all_clinical_82` | 0.643519 | **0.518820** | 0.373776 | 0.171563 | 0.00550 |

## 3. 2×2 Cross-Scale Transfer Matrix (Sample Efficiency on Identical Unseen Hospitals)

| Feature View | Train Scale → Eval Split | ROC-AUC | PR-AUC | LogLoss | Brier | ECE | Cal Slope |
|---|---|---:|---:|---:|---:|---:|---:|
| `V1_raw_only` | Full (347k) → `test_full` (20.8k nat) | **0.677320** | 0.377614 | 0.517134 | 0.171003 | 0.00760 | 1.023 |
| `V1_raw_only` | Small (50k, Bayes-recal) → `test_full` (20.8k nat) | **0.674228** | 0.374383 | 0.518457 | 0.171462 | 0.00778 | 1.056 |
| `V1_raw_only` | Full (347k) → `test_small_10k` (10k 50/50) | **0.675442** | 0.645236 | 0.782577 | 0.285532 | 0.23877 | 1.020 |
| `V1_raw_only` | Small (50k) → `test_small_10k` (10k 50/50) | **0.673558** | 0.643654 | 0.643520 | 0.226375 | 0.01040 | 1.062 |
| `V3_aug_all_clinical_82` | Full (347k) → `test_full` (20.8k nat) | **0.676931** | 0.377881 | 0.517234 | 0.171015 | 0.00604 | 1.014 |
| `V3_aug_all_clinical_82` | Small (50k, Bayes-recal) → `test_full` (20.8k nat) | **0.674356** | 0.374944 | 0.518368 | 0.171418 | 0.00695 | 1.018 |
| `V3_aug_all_clinical_82` | Full (347k) → `test_small_10k` (10k 50/50) | **0.675147** | 0.645992 | 0.782476 | 0.285296 | 0.23849 | 1.011 |
| `V3_aug_all_clinical_82` | Small (50k) → `test_small_10k` (10k 50/50) | **0.673480** | 0.645276 | 0.643519 | 0.226383 | 0.01204 | 1.020 |

## 4. Paired Bootstrap 95% Confidence Intervals (1,000 Resamples)

| Scale | Slice | Comparison | Δ ROC-AUC | 95% CI | P(Δ > 0) |
|---|---|---|---:|---|---:|
| `full` | `test_primary` | `V2_aug_kept_vs_V1_raw` | **-0.000343** | [-0.000815, +0.000128] | 0.075 |
| `full` | `test_primary` | `V3_all_clin82_vs_V1_raw` | **-0.000389** | [-0.001420, +0.000628] | 0.221 |
| `full` | `test_primary` | `V3_all_clin82_vs_V2_kept` | **-0.000047** | [-0.001027, +0.000974] | 0.463 |
| `full` | `stress_2017_unseen_hosp` | `V2_aug_kept_vs_V1_raw` | **-0.000702** | [-0.001918, +0.000504] | 0.122 |
| `full` | `stress_2017_unseen_hosp` | `V3_all_clin82_vs_V1_raw` | **-0.002262** | [-0.004982, +0.000696] | 0.068 |
| `full` | `stress_rare_clinical_tails` | `V2_aug_kept_vs_V1_raw` | **+0.000035** | [-0.002175, +0.002152] | 0.493 |
| `full` | `stress_denovo_no_preop_afib` | `V2_aug_kept_vs_V1_raw` | **-0.000297** | [-0.000744, +0.000163] | 0.117 |
| `small_50k_10k` | `test_primary` | `V2_aug_kept_vs_V1_raw` | **-0.000294** | [-0.001115, +0.000555] | 0.252 |
| `small_50k_10k` | `test_primary` | `V3_all_clin82_vs_V1_raw` | **-0.000078** | [-0.001740, +0.001739] | 0.469 |
| `small_50k_10k` | `test_primary` | `V3_all_clin82_vs_V2_kept` | **+0.000215** | [-0.001554, +0.002095] | 0.614 |
| `small_50k_10k` | `stress_2017_unseen_hosp` | `V2_aug_kept_vs_V1_raw` | **+0.001101** | [-0.001240, +0.003315] | 0.841 |
| `small_50k_10k` | `stress_2017_unseen_hosp` | `V3_all_clin82_vs_V1_raw` | **-0.001164** | [-0.005791, +0.003696] | 0.314 |
| `small_50k_10k` | `stress_rare_clinical_tails` | `V2_aug_kept_vs_V1_raw` | **-0.002595** | [-0.006095, +0.001354] | 0.089 |
| `small_50k_10k` | `stress_denovo_no_preop_afib` | `V2_aug_kept_vs_V1_raw` | **-0.000266** | [-0.001156, +0.000547] | 0.246 |
