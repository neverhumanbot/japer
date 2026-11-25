# Tabular Foundation Models Benchmark on `SMALL 50k/10k` (Exact 50/50, Hospital-Disjoint)

- **Generated (UTC)**: `2026-11-25T12:44:04.221109+00:00`
- **Hardware**: NVIDIA A100-PCIE-40GB (`xbabe8`, offline `HF_HUB_OFFLINE=1`) for Tabular FMs; CPU cluster (`xbabe0`–`xbabe3`) for CatBoost 5-fold ensembles.
- **Protocol**: Zero test-label leakage (`y_test` never staged to GPU); exact 5 hospital-disjoint `cv_fold` shards (`10,000` rows = `5,000` pos / `5,000` neg each) for all `5x10k` bagged models; identical metric suite & 1,000-resample paired bootstrap (`seed=42`) as `08_eval_catboost_dual_datasets.py`.

## 1. Primary Side-by-Side Comparison (`test_small_10k`, Exact 50/50, 74 Unseen Hospitals)

| View (#F) | Model | Test ROC-AUC | Δ vs CB Raw | V3-V1 Lift | PR-AUC | LogLoss | Brier | ECE (15-bin) | Cal Slope | Top-10% Lift | 2017 Unseen | Rare Tails | *De Novo* | Cross Full-21k AUC | ms/pt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V1_raw_only` (207) | **CatBoost (5f Ens, CPU)** | **0.673558** | +0.000000 | base | 0.643654 | 0.643520 | 0.226375 | 0.01040 | 1.062 | 1.426x | 0.650462 | 0.675906 | 0.673591 | 0.674228 | — |
| `V1_raw_only` (207) | **TabPFN-3.5 (50k Native e8)** | **0.673334** | -0.000224 | base | 0.644278 | 0.643115 | 0.226319 | 0.00975 | 1.004 | 1.448x | 0.650647 | 0.676473 | 0.673235 | 0.673755 | 23.37 |
| `V1_raw_only` (207) | **TabPFN-3.5 (5x10k Bag e40)** | **0.672071** | -0.001487 | base | 0.642109 | 0.643985 | 0.226680 | 0.01182 | 1.042 | 1.434x | 0.649465 | 0.670238 | 0.671955 | 0.672504 | 19.30 |
| `V1_raw_only` (207) | **KumoTabular-L (5x10k e40)** | **0.670748** | -0.002810 | base | 0.640518 | 0.644730 | 0.226995 | 0.01159 | 1.049 | 1.416x | 0.649533 | 0.669359 | 0.670594 | 0.671474 | 22.54 |
| `V1_raw_only` (207) | **TabFM v1.0 (5x10k e80)** | **0.670434** | -0.003124 | base | 0.641460 | 0.644920 | 0.227112 | 0.01234 | 0.975 | 1.420x | 0.648496 | 0.667373 | 0.670339 | 0.671150 | 76.63 |
| `V1_raw_only` (207) | **FM-3 Equal Blend (3 FMs)** | **0.671347** | -0.002211 | base | 0.641742 | 0.644325 | 0.226830 | 0.01134 | 1.025 | 1.410x | 0.649321 | 0.669164 | 0.671224 | 0.671985 | — |
| `V1_raw_only` (207) | **2-Paradigm (CB+TabPFN-Nat)** | **0.674369** | +0.000811 | base | 0.645019 | 0.642780 | 0.226096 | 0.01022 | 1.046 | 1.432x | 0.651309 | 0.677385 | 0.674311 | 0.674908 | — |
| `V1_raw_only` (207) | **4-Paradigm (CB+Nat+Km+TF)** | **0.673007** | -0.000551 | base | 0.643350 | 0.643478 | 0.226426 | 0.00991 | 1.036 | 1.424x | 0.650772 | 0.673562 | 0.672912 | 0.673627 | — |
| `V1_raw_only` (207) | **4-Paradigm Blend (CB+3FM)** | **0.672474** | -0.001084 | base | 0.642738 | 0.643817 | 0.226573 | 0.00977 | 1.042 | 1.416x | 0.650186 | 0.671930 | 0.672381 | 0.673118 | — |
| `V3_aug_all_clinical_82` (289) | **CatBoost (5f Ens, CPU)** | **0.673480** | -0.000078 | -0.000078 | 0.645276 | 0.643519 | 0.226383 | 0.01204 | 1.020 | 1.442x | 0.649298 | 0.671782 | 0.673480 | 0.674356 | — |
| `V3_aug_all_clinical_82` (289) | **TabPFN-3.5 (50k Native e8)** | **0.673118** | -0.000440 | -0.000215 | 0.642671 | 0.643386 | 0.226384 | 0.00894 | 1.017 | 1.420x | 0.649960 | 0.675260 | 0.673089 | 0.673789 | 29.16 |
| `V3_aug_all_clinical_82` (289) | **TabPFN-3.5 (5x10k Bag e40)** | **0.671487** | -0.002071 | -0.000584 | 0.641510 | 0.644477 | 0.226864 | 0.00931 | 1.041 | 1.400x | 0.648966 | 0.671675 | 0.671383 | 0.672138 | 25.19 |
| `V3_aug_all_clinical_82` (289) | **KumoTabular-L (5x10k e40)** | **0.670050** | -0.003508 | -0.000698 | 0.640730 | 0.645374 | 0.227266 | 0.01053 | 1.040 | 1.408x | 0.648273 | 0.668266 | 0.669946 | 0.670703 | 28.11 |
| `V3_aug_all_clinical_82` (289) | **TabFM v1.0 (5x10k e80)** | **0.670668** | -0.002890 | +0.000234 | 0.641428 | 0.645030 | 0.227130 | 0.00744 | 0.985 | 1.406x | 0.649128 | 0.669698 | 0.670556 | 0.671329 | 99.69 |
| `V3_aug_all_clinical_82` (289) | **FM-3 Equal Blend (3 FMs)** | **0.671030** | -0.002528 | -0.000317 | 0.641616 | 0.644768 | 0.227000 | 0.00655 | 1.025 | 1.410x | 0.648912 | 0.670228 | 0.670923 | 0.671671 | — |
| `V3_aug_all_clinical_82` (289) | **2-Paradigm (CB+TabPFN-Nat)** | **0.673943** | +0.000385 | -0.000426 | 0.644508 | 0.643068 | 0.226204 | 0.00913 | 1.028 | 1.430x | 0.650098 | 0.674483 | 0.673919 | 0.674720 | — |
| `V3_aug_all_clinical_82` (289) | **4-Paradigm (CB+Nat+Km+TF)** | **0.672711** | -0.000847 | -0.000296 | 0.643431 | 0.643796 | 0.226547 | 0.00860 | 1.028 | 1.418x | 0.650026 | 0.672619 | 0.672643 | 0.673421 | — |
| `V3_aug_all_clinical_82` (289) | **4-Paradigm Blend (CB+3FM)** | **0.672075** | -0.001483 | -0.000399 | 0.642848 | 0.644203 | 0.226729 | 0.00787 | 1.030 | 1.412x | 0.649504 | 0.671414 | 0.671988 | 0.672761 | — |

## 2. Bayes Prior-Shift Recalibration & Horvitz-Thompson Natural-Prior Recovery + Cross-Scale (`test_full` 21,241)

| View | Model | Balanced LL | HT-Nat LL | HT-Nat PR-AUC | HT-Nat Brier | HT-Nat ECE | Cross Full-21k AUC | Cross Full-21k LL | Cross Full-21k ECE | Peak VRAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V1_raw_only` | CatBoost (5f Ens, CPU) | 0.643520 | **0.518895** | 0.371663 | 0.171636 | 0.00627 | **0.674228** | 0.518457 | 0.00778 | CPU |
| `V1_raw_only` | TabPFN-3.5 (50k Native e8) | 0.643115 | **0.518646** | 0.372459 | 0.171586 | 0.00735 | **0.673755** | 0.518443 | 0.00683 | 7.65 |
| `V1_raw_only` | TabPFN-3.5 (5x10k Bag e40) | 0.643985 | **0.519278** | 0.369799 | 0.171806 | 0.00527 | **0.672504** | 0.519055 | 0.00593 | 8.04 |
| `V1_raw_only` | KumoTabular-L (5x10k e40) | 0.644730 | **0.519851** | 0.368430 | 0.172032 | 0.00851 | **0.671474** | 0.519515 | 0.00702 | 6.84 |
| `V1_raw_only` | TabFM v1.0 (5x10k e80) | 0.644920 | **0.520021** | 0.369570 | 0.172111 | 0.00840 | **0.671150** | 0.519728 | 0.00901 | 22.15 |
| `V1_raw_only` | FM-3 Equal Blend (3 FMs) | 0.644325 | **0.519547** | 0.369799 | 0.171918 | 0.00808 | **0.671985** | 0.519263 | 0.00727 | CPU |
| `V1_raw_only` | 2-Paradigm (CB+TabPFN-Nat) | 0.642780 | **0.518372** | 0.373226 | 0.171466 | 0.00613 | **0.674908** | 0.518052 | 0.00677 | CPU |
| `V1_raw_only` | 4-Paradigm (CB+Nat+Km+TF) | 0.643478 | **0.518907** | 0.371361 | 0.171674 | 0.00637 | **0.673627** | 0.518582 | 0.00622 | CPU |
| `V1_raw_only` | 4-Paradigm Blend (CB+3FM) | 0.643817 | **0.519152** | 0.370688 | 0.171759 | 0.00524 | **0.673118** | 0.518820 | 0.00627 | CPU |
| `V3_aug_all_clinical_82` | CatBoost (5f Ens, CPU) | 0.643519 | **0.518820** | 0.373776 | 0.171563 | 0.00550 | **0.674356** | 0.518368 | 0.00695 | CPU |
| `V3_aug_all_clinical_82` | TabPFN-3.5 (50k Native e8) | 0.643386 | **0.518880** | 0.370317 | 0.171689 | 0.00474 | **0.673789** | 0.518510 | 0.00488 | 7.70 |
| `V3_aug_all_clinical_82` | TabPFN-3.5 (5x10k Bag e40) | 0.644477 | **0.519608** | 0.369256 | 0.171903 | 0.00442 | **0.672138** | 0.519246 | 0.00515 | 7.71 |
| `V3_aug_all_clinical_82` | KumoTabular-L (5x10k e40) | 0.645374 | **0.520241** | 0.368845 | 0.172114 | 0.00732 | **0.670703** | 0.519880 | 0.00617 | 7.37 |
| `V3_aug_all_clinical_82` | TabFM v1.0 (5x10k e80) | 0.645030 | **0.520020** | 0.369414 | 0.172051 | 0.00456 | **0.671329** | 0.519617 | 0.00524 | 28.96 |
| `V3_aug_all_clinical_82` | FM-3 Equal Blend (3 FMs) | 0.644768 | **0.519813** | 0.369713 | 0.171970 | 0.00699 | **0.671671** | 0.519435 | 0.00618 | CPU |
| `V3_aug_all_clinical_82` | 2-Paradigm (CB+TabPFN-Nat) | 0.643068 | **0.518567** | 0.372535 | 0.171523 | 0.00613 | **0.674720** | 0.518157 | 0.00489 | CPU |
| `V3_aug_all_clinical_82` | 4-Paradigm (CB+Nat+Km+TF) | 0.643796 | **0.519096** | 0.371586 | 0.171710 | 0.00699 | **0.673421** | 0.518701 | 0.00372 | CPU |
| `V3_aug_all_clinical_82` | 4-Paradigm Blend (CB+3FM) | 0.644203 | **0.519376** | 0.370949 | 0.171798 | 0.00781 | **0.672761** | 0.518978 | 0.00572 | CPU |

## 3. Single-Shard (`1x10k`) vs Full-50k Bagged (`5x10k`) Scaling Ablation

| View | Model | 1x10k (Shard-0) AUC | 5x10k Bagged AUC | Δ Bagging | 1x10k LL | 5x10k LL |
|---|---|---:|---:|---:|---:|---:|
| `V1_raw_only` | TabPFN-3.5 | 0.668713 | **0.672071** | +0.003358 | 0.645660 | 0.643985 |
| `V1_raw_only` | KumoTabular-Large | 0.666751 | **0.670748** | +0.003997 | 0.646946 | 0.644730 |
| `V1_raw_only` | TabFM v1.0 Ensemble | 0.666511 | **0.670434** | +0.003923 | 0.647417 | 0.644920 |
| `V3_aug_all_clinical_82` | TabPFN-3.5 | 0.668529 | **0.671487** | +0.002958 | 0.645986 | 0.644477 |
| `V3_aug_all_clinical_82` | KumoTabular-Large | 0.667045 | **0.670050** | +0.003005 | 0.647109 | 0.645374 |
| `V3_aug_all_clinical_82` | TabFM v1.0 Ensemble | 0.667376 | **0.670668** | +0.003292 | 0.647124 | 0.645030 |

## 4. Paired Bootstrap 95% CIs (1,000 Resamples, `test_small_10k`)

| Comparison Type | View | Model | Δ ROC-AUC | 95% CI | P(Δ > 0) |
|---|---|---|---:|---|---:|
| vs CatBoost Small (V1_raw_only) | `V1_raw_only` | TabPFN-3.5 (50k Native e8) | **-0.000224** | [-0.002666, +0.002378] | 0.469 |
| vs CatBoost Small (V1_raw_only) | `V1_raw_only` | TabPFN-3.5 (5x10k Bag e40) | **-0.001487** | [-0.003960, +0.000986] | 0.121 |
| vs CatBoost Small (V1_raw_only) | `V1_raw_only` | KumoTabular-L (5x10k e40) | **-0.002810** | [-0.005130, -0.000382] | 0.011 |
| vs CatBoost Small (V1_raw_only) | `V1_raw_only` | TabFM v1.0 (5x10k e80) | **-0.003125** | [-0.005580, -0.000738] | 0.003 |
| vs CatBoost Small (V1_raw_only) | `V1_raw_only` | FM-3 Equal Blend (3 FMs) | **-0.002211** | [-0.004467, +0.000154] | 0.035 |
| vs CatBoost Small (V1_raw_only) | `V1_raw_only` | 2-Paradigm (CB+TabPFN-Nat) | **+0.000811** | [-0.000425, +0.002142] | 0.908 |
| vs CatBoost Small (V1_raw_only) | `V1_raw_only` | 4-Paradigm (CB+Nat+Km+TF) | **-0.000551** | [-0.002198, +0.001183] | 0.275 |
| vs CatBoost Small (V1_raw_only) | `V1_raw_only` | 4-Paradigm Blend (CB+3FM) | **-0.001084** | [-0.002804, +0.000681] | 0.121 |
| vs CatBoost Small (V3_aug_all_clinical_82) | `V3_aug_all_clinical_82` | TabPFN-3.5 (50k Native e8) | **-0.000362** | [-0.002198, +0.001632] | 0.385 |
| vs CatBoost Small (V3_aug_all_clinical_82) | `V3_aug_all_clinical_82` | TabPFN-3.5 (5x10k Bag e40) | **-0.001993** | [-0.004057, +0.000030] | 0.027 |
| vs CatBoost Small (V3_aug_all_clinical_82) | `V3_aug_all_clinical_82` | KumoTabular-L (5x10k e40) | **-0.003430** | [-0.005602, -0.001354] | 0.001 |
| vs CatBoost Small (V3_aug_all_clinical_82) | `V3_aug_all_clinical_82` | TabFM v1.0 (5x10k e80) | **-0.002812** | [-0.004787, -0.000902] | 0.002 |
| vs CatBoost Small (V3_aug_all_clinical_82) | `V3_aug_all_clinical_82` | FM-3 Equal Blend (3 FMs) | **-0.002450** | [-0.004307, -0.000563] | 0.005 |
| vs CatBoost Small (V3_aug_all_clinical_82) | `V3_aug_all_clinical_82` | 2-Paradigm (CB+TabPFN-Nat) | **+0.000463** | [-0.000468, +0.001516] | 0.830 |
| vs CatBoost Small (V3_aug_all_clinical_82) | `V3_aug_all_clinical_82` | 4-Paradigm (CB+Nat+Km+TF) | **-0.000769** | [-0.002092, +0.000567] | 0.121 |
| vs CatBoost Small (V3_aug_all_clinical_82) | `V3_aug_all_clinical_82` | 4-Paradigm Blend (CB+3FM) | **-0.001405** | [-0.002793, +0.000073] | 0.026 |
| Augmentation Lift (V3-V1) | V3 vs V1 | CatBoost (5f Ens, CPU) | **-0.000078** | [-0.001740, +0.001739] | 0.469 |
| Augmentation Lift (V3-V1) | V3 vs V1 | TabPFN-3.5 (50k Native e8) | **-0.000215** | [-0.002044, +0.001430] | 0.414 |
| Augmentation Lift (V3-V1) | V3 vs V1 | TabPFN-3.5 (5x10k Bag e40) | **-0.000584** | [-0.002029, +0.000914] | 0.210 |
| Augmentation Lift (V3-V1) | V3 vs V1 | KumoTabular-L (5x10k e40) | **-0.000698** | [-0.002023, +0.000624] | 0.142 |
| Augmentation Lift (V3-V1) | V3 vs V1 | TabFM v1.0 (5x10k e80) | **+0.000234** | [-0.001352, +0.001713] | 0.612 |
| Augmentation Lift (V3-V1) | V3 vs V1 | FM-3 Equal Blend (3 FMs) | **-0.000317** | [-0.001688, +0.000966] | 0.296 |
| Augmentation Lift (V3-V1) | V3 vs V1 | 2-Paradigm (CB+TabPFN-Nat) | **-0.000426** | [-0.001768, +0.000974] | 0.273 |
| Augmentation Lift (V3-V1) | V3 vs V1 | 4-Paradigm (CB+Nat+Km+TF) | **-0.000296** | [-0.001529, +0.000897] | 0.313 |
| Augmentation Lift (V3-V1) | V3 vs V1 | 4-Paradigm Blend (CB+3FM) | **-0.000399** | [-0.001687, +0.000890] | 0.256 |
