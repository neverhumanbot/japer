# Phase 2 High-Capacity Whole-Hospital Multi-Family Target Solver — Summary Report

- **Generated (UTC)**: `2026-11-25T16:05:48+00:00`
- **Strict Contract**: ALL evaluated final models take ONLY `V1_raw_only` (207 raw pre-incision predictors); ZERO test-label leakage.

## 1. Phase 2 Evolutionary Tracks (Whole-Hospital Shuffling Across 1,017 Hospitals)

| Track | Solver | Lock | Total Fits | Fits/s | Final OOS r | Final Orig r | Final Unseen-Hosp AUC | Best V1 5f CV (Gen) |
|---|---|---|---:|---:|---:|---:|---:|---|
| `P2B1_MultiFamily_HospShuffle_V3_L2_Consensus` | `stochastic_consensus` | `L2_confident_core` | 11,520 | 7.4 | 0.9144 | 0.6311 | 0.67622 | **0.677928** (g16) |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO` | `memetic_pso` | `L2_confident_core` | 11,520 | 7.7 | 0.8415 | 0.7469 | 0.67854 | **0.678305** (g20) |
| `P2B3_MultiFamily_HospShuffle_V3_L0_PureRisk` | `stochastic_consensus` | `L0_unlocked` | 11,520 | 7.5 | 0.9816 | 0.4709 | 0.67381 | **0.677342** (g12) |
| `P2C_LUPI_MultiFamily_HospShuffle_V3_L2_Consensus` | `stochastic_consensus` | `L2_confident_core` | 11,520 | 7.6 | 0.9183 | 0.6267 | 0.67613 | **0.677921** (g20) |

## 2. Sealed Unseen-Hospital Leaderboard (`test_small_10k` & `test_full_21k`, Ranked by `test_10k` AUC)

| Model (`V1_raw_only` 207 Features) | 5f OOF | `test_10k` AUC | Δ vs Cls (`P>0`) | Δ vs Reg (`P>0`) | `test_full` AUC | Δ Full vs Cls (`P>0`) | Δ Full vs Reg (`P>0`) | Platt ECE | 2017 Unseen |
|---|---:|---:|---|---|---:|---|---|---:|---:|
| `P2B3_MultiFamily_HospShuffle_V3_L0_PureRisk[best_canary|cross_entropy]` | 0.678776 | **0.674515** | +0.00127 (0.987) | +0.00085 (0.948) | **0.675340** | +0.00128 (0.996) | +0.00100 (0.991) | 0.01060 | 0.65049 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[best_canary|cross_entropy]` | 0.680061 | **0.674366** | +0.00112 (0.959) | +0.00070 (0.860) | **0.675221** | +0.00116 (0.991) | +0.00088 (0.966) | 0.01334 | 0.64948 |
| `P2B3_MultiFamily_HospShuffle_V3_L0_PureRisk[best_canary|logit_rmse]` | 0.679141 | **0.674338** | +0.00109 (0.978) | +0.00067 (0.902) | **0.675272** | +0.00122 (0.994) | +0.00093 (0.980) | 0.01060 | 0.65033 |
| `P2B1_MultiFamily_HospShuffle_V3_L2_Consensus[best_canary|cross_entropy]` | 0.679890 | **0.674320** | +0.00108 (0.958) | +0.00066 (0.855) | **0.675215** | +0.00116 (0.991) | +0.00087 (0.971) | 0.01332 | 0.64912 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[best_canary|rmse]` | 0.679924 | **0.674287** | +0.00104 (0.977) | +0.00062 (0.881) | **0.675141** | +0.00108 (0.996) | +0.00080 (0.980) | 0.01403 | 0.64997 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[best_canary|cross_entropy|d7_l2_16]` | 0.680404 | **0.674227** | +0.00098 (0.936) | +0.00056 (0.807) | **0.675181** | +0.00112 (0.985) | +0.00084 (0.960) | 0.01450 | 0.65000 |
| `P2B3_MultiFamily_HospShuffle_V3_L0_PureRisk[best_canary|rmse]` | 0.678558 | **0.674198** | +0.00095 (0.954) | +0.00053 (0.841) | **0.675126** | +0.00107 (0.992) | +0.00078 (0.966) | 0.01182 | 0.65003 |
| `P2C_LUPI_MultiFamily_HospShuffle_V3_L2_Consensus[best_canary|cross_entropy]` | 0.679761 | **0.674003** | +0.00076 (0.849) | +0.00034 (0.678) | **0.674813** | +0.00076 (0.922) | +0.00047 (0.801) | 0.01489 | 0.64917 |
| `P2C_LUPI_MultiFamily_HospShuffle_V3_L2_Consensus[best_canary|rmse]` | 0.679460 | **0.673977** | +0.00073 (0.869) | +0.00031 (0.687) | **0.674750** | +0.00069 (0.929) | +0.00041 (0.808) | 0.01458 | 0.64969 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[ewma_final|rmse]` | 0.679779 | **0.673970** | +0.00072 (0.889) | +0.00031 (0.698) | **0.674688** | +0.00063 (0.934) | +0.00035 (0.795) | 0.01540 | 0.64906 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[best_canary|logit_rmse]` | 0.680447 | **0.673966** | +0.00072 (0.840) | +0.00030 (0.657) | **0.674785** | +0.00073 (0.927) | +0.00044 (0.785) | 0.01764 | 0.64898 |
| `P2B1_MultiFamily_HospShuffle_V3_L2_Consensus[best_canary|rmse]` | 0.679411 | **0.673919** | +0.00067 (0.897) | +0.00026 (0.683) | **0.674787** | +0.00073 (0.966) | +0.00044 (0.864) | 0.01670 | 0.64942 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[ewma_final|logit_rmse]` | 0.680124 | **0.673894** | +0.00065 (0.807) | +0.00023 (0.607) | **0.674626** | +0.00057 (0.862) | +0.00028 (0.698) | 0.02008 | 0.64894 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[ewma_final|cross_entropy|d7_l2_16]` | 0.680377 | **0.673885** | +0.00064 (0.824) | +0.00022 (0.621) | **0.674868** | +0.00081 (0.954) | +0.00053 (0.852) | 0.01612 | 0.64903 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[ewma_final|logit_rmse|d7_l2_16]` | 0.680135 | **0.673884** | +0.00064 (0.803) | +0.00022 (0.603) | **0.674730** | +0.00067 (0.890) | +0.00039 (0.746) | 0.02037 | 0.64898 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[ewma_final|cross_entropy]` | 0.680265 | **0.673758** | +0.00051 (0.782) | +0.00010 (0.561) | **0.674624** | +0.00057 (0.889) | +0.00028 (0.727) | 0.01893 | 0.64833 |
| `P2B1_MultiFamily_HospShuffle_V3_L2_Consensus[best_canary|logit_rmse]` | 0.680224 | **0.673749** | +0.00050 (0.767) | +0.00009 (0.537) | **0.674562** | +0.00051 (0.855) | +0.00022 (0.683) | 0.01969 | 0.64873 |
| `Baseline_CatBoostRegressor_Binary` | 0.675344 | **0.673663** | +0.00042 (0.888) | +0.00000 (0.000) | **0.674342** | +0.00028 (0.850) | +0.00000 (0.000) | 0.01244 | 0.65075 |
| `P2C_LUPI_MultiFamily_HospShuffle_V3_L2_Consensus[best_canary|logit_rmse]` | 0.679914 | **0.673609** | +0.00036 (0.678) | -0.00005 (0.461) | **0.674450** | +0.00039 (0.782) | +0.00011 (0.579) | 0.01938 | 0.64830 |
| `P2B2_MultiFamily_HospShuffle_V3_L2_PSO[best_canary|logit_rmse|d7_l2_16]` | 0.680250 | **0.673597** | +0.00035 (0.667) | -0.00007 (0.449) | **0.674548** | +0.00049 (0.831) | +0.00021 (0.649) | 0.01732 | 0.64871 |
| `P2C_LUPI_MultiFamily_HospShuffle_V3_L2_Consensus[ewma_final|cross_entropy]` | 0.679162 | **0.673583** | +0.00034 (0.677) | -0.00008 (0.453) | **0.674375** | +0.00032 (0.723) | +0.00003 (0.528) | 0.01855 | 0.64862 |
| `P2B1_MultiFamily_HospShuffle_V3_L2_Consensus[ewma_final|cross_entropy]` | 0.679251 | **0.673522** | +0.00028 (0.653) | -0.00014 (0.393) | **0.674281** | +0.00022 (0.673) | -0.00006 (0.466) | 0.01954 | 0.64855 |
| `P2C_LUPI_MultiFamily_HospShuffle_V3_L2_Consensus[ewma_final|rmse]` | 0.678586 | **0.673439** | +0.00019 (0.622) | -0.00022 (0.380) | **0.674111** | +0.00005 (0.559) | -0.00023 (0.372) | 0.01378 | 0.64866 |
| `P2C_LUPI_MultiFamily_HospShuffle_V3_L2_Consensus[ewma_final|logit_rmse]` | 0.679224 | **0.673385** | +0.00014 (0.571) | -0.00028 (0.353) | **0.674193** | +0.00014 (0.611) | -0.00015 (0.417) | 0.01969 | 0.64850 |
| `P2B1_MultiFamily_HospShuffle_V3_L2_Consensus[ewma_final|logit_rmse]` | 0.679318 | **0.673348** | +0.00010 (0.539) | -0.00032 (0.321) | **0.674121** | +0.00006 (0.563) | -0.00022 (0.363) | 0.02066 | 0.64843 |
| `P2B1_MultiFamily_HospShuffle_V3_L2_Consensus[ewma_final|rmse]` | 0.678555 | **0.673326** | +0.00008 (0.538) | -0.00034 (0.301) | **0.674021** | -0.00003 (0.490) | -0.00032 (0.294) | 0.01713 | 0.64842 |
| `Baseline_CatBoostClassifier_Binary` | 0.675167 | **0.673246** | +0.00000 (0.000) | -0.00042 (0.112) | **0.674057** | +0.00000 (0.000) | -0.00028 (0.150) | 0.01020 | 0.65000 |
| `P2B3_MultiFamily_HospShuffle_V3_L0_PureRisk[ewma_final|logit_rmse]` | 0.676319 | **0.672242** | -0.00100 (0.159) | -0.00142 (0.077) | **0.672939** | -0.00112 (0.096) | -0.00140 (0.054) | 0.01261 | 0.64809 |
| `P2B3_MultiFamily_HospShuffle_V3_L0_PureRisk[ewma_final|cross_entropy]` | 0.676280 | **0.672240** | -0.00101 (0.162) | -0.00142 (0.076) | **0.672943** | -0.00111 (0.098) | -0.00140 (0.055) | 0.01319 | 0.64819 |
| `P2B3_MultiFamily_HospShuffle_V3_L0_PureRisk[ewma_final|rmse]` | 0.676050 | **0.672097** | -0.00115 (0.131) | -0.00157 (0.060) | **0.672777** | -0.00128 (0.070) | -0.00156 (0.032) | 0.01111 | 0.64800 |
