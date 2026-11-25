# JAIN Ultra Out-of-Sample Evaluation on Generation-20 Evolved Targets

- **Generated UTC**: `2026-11-25T17:46:15.626055+00:00`
- **Features**: Strictly `V1_raw_only` (207 raw pre-incision predictors → 215 numeric columns with `Region` one-hot encoded)
- **Zero Test Leakage**: Verified — test feature CSVs (`test_small_10k` [10,000 rows], `test_full_21k` [21,241 rows], 74 unseen hospitals) contain **ZERO target column**.

## Sealed Unseen-Hospital Leaderboard (Sorted by `test_full_21k` ROC-AUC)

| Rank | Model | Engine | `test_10k` AUC | `test_full` AUC | `test_full` PR-AUC | `test_full` Top-10% Lift | `test_full` Platt ECE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **CB_CE_Gen20_P2B2** | CatBoost 1.2.10 (5-fold CV) | `0.673989` | `0.674973` | `0.373908` | `1.8218x` | `0.23282` |
| 2 | **CB_CE_Gen12_P2B3** | CatBoost 1.2.10 (5-fold CV) | `0.674033` | `0.674863` | `0.374677` | `1.8450x` | `0.23273` |
| 3 | **CB_Cls_Binary** | CatBoost 1.2.10 (5-fold CV) | `0.673273` | `0.674083` | `0.373151` | `1.8237x` | `0.23375` |
| 4 | **CB_Reg_Binary** | CatBoost 1.2.10 (5-fold CV) | `0.673391` | `0.674022` | `0.373409` | `1.8199x` | `0.23355` |
| 5 | **JAIN_FitAuto_Reg_Gen20_P2B2** | JAIN (fit_auto) | `0.672012` | `0.672681` | `0.370970` | `1.8354x` | `0.23405` |
| 6 | **JAIN_FitAuto_Reg_Gen12_P2B3** | JAIN (fit_auto) | `0.671588` | `0.672188` | `0.370425` | `1.8121x` | `0.23439` |
| 7 | **JAIN_FitAuto_Reg_BinaryControl** | JAIN (fit_auto) | `0.668210` | `0.668096` | `0.365035` | `1.7753x` | `0.23329` |
| 8 | **JAIN_FitAuto_Bin_BinaryControl** | JAIN (fit_auto) | `0.492119` | `0.491068` | `0.236337` | `0.9109x` | `0.25712` |

## Paired Bootstrap 95% CIs (1,000 Resamples, Sealed Unseen Hospitals)

| Candidate | vs. Baseline | `test_10k` ΔAUC [95% CI] (`P>0`) | `test_full` ΔAUC [95% CI] (`P>0`) |
|---|---|---|---|
| `JAIN_FitAuto_Reg_Gen20_P2B2` | `JAIN_FitAuto_Reg_BinaryControl` | `+0.003801` [+0.000448, +0.006871] (`0.990`) | `+0.004585` [+0.001874, +0.007007] (`0.999`) |
| `JAIN_FitAuto_Reg_Gen20_P2B2` | `CB_Cls_Binary` | `-0.001261` [-0.003373, +0.000765] (`0.128`) | `-0.001401` [-0.003122, +0.000355] (`0.046`) |
| `JAIN_FitAuto_Reg_Gen20_P2B2` | `CB_Reg_Binary` | `-0.001379` [-0.003575, +0.000790] (`0.110`) | `-0.001341` [-0.003110, +0.000507] (`0.069`) |
| `JAIN_FitAuto_Reg_Gen20_P2B2` | `CB_CE_Gen20_P2B2` | `-0.001978` [-0.004143, +0.000024] (`0.029`) | `-0.002292` [-0.003948, -0.000679] (`0.001`) |
| `JAIN_FitAuto_Reg_Gen12_P2B3` | `JAIN_FitAuto_Reg_BinaryControl` | `+0.003378` [+0.000167, +0.006420] (`0.984`) | `+0.004091` [+0.001503, +0.006387] (`0.999`) |
| `JAIN_FitAuto_Reg_Gen12_P2B3` | `CB_Cls_Binary` | `-0.001685` [-0.003883, +0.000536] (`0.068`) | `-0.001895` [-0.003627, -0.000138] (`0.016`) |
| `JAIN_FitAuto_Reg_Gen12_P2B3` | `CB_Reg_Binary` | `-0.001803` [-0.004149, +0.000417] (`0.062`) | `-0.001834` [-0.003624, +0.000031] (`0.027`) |
| `JAIN_FitAuto_Reg_Gen12_P2B3` | `CB_CE_Gen20_P2B2` | `-0.002401` [-0.004563, -0.000309] (`0.014`) | `-0.002786` [-0.004476, -0.001134] (`0.000`) |
| `JAIN_FitAuto_Reg_BinaryControl` | `CB_Cls_Binary` | `-0.005063` [-0.007788, -0.002188] (`0.000`) | `-0.005987` [-0.008037, -0.003606] (`0.000`) |
| `JAIN_FitAuto_Reg_BinaryControl` | `CB_Reg_Binary` | `-0.005180` [-0.008109, -0.002471] (`0.000`) | `-0.005926` [-0.008035, -0.003614] (`0.000`) |
| `JAIN_FitAuto_Reg_BinaryControl` | `CB_CE_Gen20_P2B2` | `-0.005779` [-0.008900, -0.002743] (`0.000`) | `-0.006877` [-0.009106, -0.004431] (`0.000`) |
| `JAIN_FitAuto_Bin_BinaryControl` | `JAIN_FitAuto_Reg_BinaryControl` | `-0.176092` [-0.191767, -0.161729] (`0.000`) | `-0.177029` [-0.188822, -0.164389] (`0.000`) |
| `JAIN_FitAuto_Bin_BinaryControl` | `CB_Cls_Binary` | `-0.181154` [-0.196888, -0.165689] (`0.000`) | `-0.183015` [-0.194587, -0.169948] (`0.000`) |
| `JAIN_FitAuto_Bin_BinaryControl` | `CB_Reg_Binary` | `-0.181272` [-0.197142, -0.166113] (`0.000`) | `-0.182954` [-0.194554, -0.169655] (`0.000`) |
| `JAIN_FitAuto_Bin_BinaryControl` | `CB_CE_Gen20_P2B2` | `-0.181871` [-0.197881, -0.166522] (`0.000`) | `-0.183906` [-0.195517, -0.171295] (`0.000`) |

## Discovered Programs & Selected Native Methods

### `JAIN_FitAuto_Reg_Gen20_P2B2` (JAIN (fit_auto))
- Wall time: `1252.33s`
- Selected method / DSL: `rust.elasticnet`

### `JAIN_FitAuto_Reg_Gen12_P2B3` (JAIN (fit_auto))
- Wall time: `1309.32s`
- Selected method / DSL: `rust.elasticnet`

### `JAIN_FitAuto_Reg_BinaryControl` (JAIN (fit_auto))
- Wall time: `985.85s`
- Selected method / DSL: `rust.histgbm`

### `JAIN_FitAuto_Bin_BinaryControl` (JAIN (fit_auto))
- Wall time: `2442.33s`
- Selected method / DSL: `rust.logit_ridge`
