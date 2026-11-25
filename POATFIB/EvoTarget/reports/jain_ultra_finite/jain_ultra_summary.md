# JAIN Ultra Out-of-Sample Evaluation on Generation-20 Evolved Targets

- **Generated UTC**: `2026-11-25T17:29:30.908930+00:00`
- **Features**: Strictly `V1_raw_only` (207 raw pre-incision predictors → 215 numeric columns with `Region` one-hot encoded)
- **Zero Test Leakage**: Verified — test feature CSVs (`test_small_10k` [10,000 rows], `test_full_21k` [21,241 rows], 74 unseen hospitals) contain **ZERO target column**.

## Sealed Unseen-Hospital Leaderboard (Sorted by `test_full_21k` ROC-AUC)

| Rank | Model | Engine | `test_10k` AUC | `test_full` AUC | `test_full` PR-AUC | `test_full` Top-10% Lift | `test_full` Platt ECE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **CB_CE_Gen20_P2B2** | CatBoost 1.2.10 (5-fold CV) | `0.674038` | `0.674633` | `0.373553` | `1.8063x` | `0.23421` |
| 2 | **Blend_50_50_JAIN_Ultra_Gen20_+_CB_Gen20** | Rank-50/50 Fixed Blend | `0.673587` | `0.674305` | `0.372698` | `1.8179x` | `0.25708` |
| 3 | **CB_CE_Gen12_P2B3** | CatBoost 1.2.10 (5-fold CV) | `0.673596` | `0.674243` | `0.373680` | `1.8218x` | `0.23416` |
| 4 | **Blend_50_50_JAIN_Ultra_Gen12_+_CB_Gen12** | Rank-50/50 Fixed Blend | `0.673426` | `0.674156` | `0.373348` | `1.8043x` | `0.25708` |
| 5 | **CB_Reg_Binary** | CatBoost 1.2.10 (5-fold CV) | `0.672876` | `0.673850` | `0.373661` | `1.7966x` | `0.23502` |
| 6 | **CB_Cls_Binary** | CatBoost 1.2.10 (5-fold CV) | `0.673191` | `0.673694` | `0.373187` | `1.8160x` | `0.23501` |
| 7 | **JAIN_Ultra_Reg_Gen12_P2B3** | JAIN (ultra_run) | `0.672701` | `0.673475` | `0.372327` | `1.7888x` | `0.23237` |
| 8 | **JAIN_Ultra_Reg_Gen20_P2B2** | JAIN (ultra_run) | `0.672565` | `0.673412` | `0.371164` | `1.8121x` | `0.23220` |
| 9 | **JAIN_Ultra_Reg_BinaryControl** | JAIN (ultra_run) | `0.669553` | `0.671757` | `0.371076` | `1.8276x` | `0.23084` |

## Paired Bootstrap 95% CIs (1,000 Resamples, Sealed Unseen Hospitals)

| Candidate | vs. Baseline | `test_10k` ΔAUC [95% CI] (`P>0`) | `test_full` ΔAUC [95% CI] (`P>0`) |
|---|---|---|---|
| `JAIN_Ultra_Reg_Gen20_P2B2` | `JAIN_Ultra_Reg_BinaryControl` | `+0.003011` [+0.000081, +0.006016] (`0.980`) | `+0.001655` [-0.000496, +0.003908] (`0.935`) |
| `JAIN_Ultra_Reg_Gen20_P2B2` | `CB_Cls_Binary` | `-0.000627` [-0.002207, +0.000865] (`0.211`) | `-0.000283` [-0.001460, +0.000962] (`0.339`) |
| `JAIN_Ultra_Reg_Gen20_P2B2` | `CB_Reg_Binary` | `-0.000311` [-0.001946, +0.001200] (`0.344`) | `-0.000439` [-0.001728, +0.000941] (`0.239`) |
| `JAIN_Ultra_Reg_Gen20_P2B2` | `CB_CE_Gen20_P2B2` | `-0.001473` [-0.002775, -0.000162] (`0.014`) | `-0.001222` [-0.002207, -0.000176] (`0.013`) |
| `JAIN_Ultra_Reg_Gen12_P2B3` | `JAIN_Ultra_Reg_BinaryControl` | `+0.003147` [+0.000450, +0.005889] (`0.990`) | `+0.001718` [-0.000322, +0.003863] (`0.941`) |
| `JAIN_Ultra_Reg_Gen12_P2B3` | `CB_Cls_Binary` | `-0.000491` [-0.002193, +0.001273] (`0.284`) | `-0.000220` [-0.001500, +0.001123] (`0.395`) |
| `JAIN_Ultra_Reg_Gen12_P2B3` | `CB_Reg_Binary` | `-0.000176` [-0.001832, +0.001636] (`0.401`) | `-0.000376` [-0.001699, +0.000968] (`0.315`) |
| `JAIN_Ultra_Reg_Gen12_P2B3` | `CB_CE_Gen20_P2B2` | `-0.001338` [-0.002980, +0.000306] (`0.053`) | `-0.001159` [-0.002464, +0.000165] (`0.044`) |
| `JAIN_Ultra_Reg_BinaryControl` | `CB_Cls_Binary` | `-0.003638` [-0.006421, -0.000789] (`0.009`) | `-0.001938` [-0.004150, +0.000227] (`0.037`) |
| `JAIN_Ultra_Reg_BinaryControl` | `CB_Reg_Binary` | `-0.003323` [-0.005977, -0.000571] (`0.009`) | `-0.002093` [-0.004227, +0.000051] (`0.028`) |
| `JAIN_Ultra_Reg_BinaryControl` | `CB_CE_Gen20_P2B2` | `-0.004485` [-0.007251, -0.001464] (`0.003`) | `-0.002876` [-0.005237, -0.000477] (`0.011`) |
| `Blend_50_50_JAIN_Ultra_Gen20_+_CB_Gen20` | `JAIN_Ultra_Reg_BinaryControl` | `+0.004034` [+0.001172, +0.006915] (`0.997`) | `+0.002548` [+0.000324, +0.004734] (`0.986`) |
| `Blend_50_50_JAIN_Ultra_Gen20_+_CB_Gen20` | `CB_Cls_Binary` | `+0.000396` [-0.000842, +0.001576] (`0.741`) | `+0.000610` [-0.000367, +0.001603] (`0.888`) |
| `Blend_50_50_JAIN_Ultra_Gen20_+_CB_Gen20` | `CB_Reg_Binary` | `+0.000711` [-0.000515, +0.001930] (`0.868`) | `+0.000455` [-0.000649, +0.001525] (`0.820`) |
| `Blend_50_50_JAIN_Ultra_Gen20_+_CB_Gen20` | `CB_CE_Gen20_P2B2` | `-0.000451` [-0.001107, +0.000208] (`0.096`) | `-0.000328` [-0.000824, +0.000191] (`0.104`) |
| `Blend_50_50_JAIN_Ultra_Gen12_+_CB_Gen12` | `JAIN_Ultra_Reg_BinaryControl` | `+0.003873` [+0.001001, +0.006717] (`0.996`) | `+0.002399` [+0.000300, +0.004565] (`0.990`) |
| `Blend_50_50_JAIN_Ultra_Gen12_+_CB_Gen12` | `CB_Cls_Binary` | `+0.000235` [-0.000980, +0.001541] (`0.639`) | `+0.000461` [-0.000423, +0.001451] (`0.829`) |
| `Blend_50_50_JAIN_Ultra_Gen12_+_CB_Gen12` | `CB_Reg_Binary` | `+0.000550` [-0.000658, +0.001859] (`0.774`) | `+0.000306` [-0.000692, +0.001298] (`0.746`) |
| `Blend_50_50_JAIN_Ultra_Gen12_+_CB_Gen12` | `CB_CE_Gen20_P2B2` | `-0.000612` [-0.001751, +0.000555] (`0.148`) | `-0.000477` [-0.001388, +0.000461] (`0.153`) |

## Discovered Programs & Selected Native Methods

### `JAIN_Ultra_Reg_Gen12_P2B3` (JAIN (ultra_run))
- Wall time: `1879.86s`
- Selected method / DSL: `alg_00000006`

### `JAIN_Ultra_Reg_Gen20_P2B2` (JAIN (ultra_run))
- Wall time: `1896.85s`
- Selected method / DSL: `alg_00000042`

### `JAIN_Ultra_Reg_BinaryControl` (JAIN (ultra_run))
- Wall time: `1943.25s`
- Selected method / DSL: `alg_00000005`
