# 5-Way Statistical Split Audit Comparison (STS ACSD v2.81, n = 367,695)

- **Generated (UTC)**: `2026-11-25T07:18:16.266452+00:00`
- **Canonical Optimal Split**: `data/optimal_splits.parquet` (SHA256 `37f4977630288b905f01baeee37fe84952b8e71491dc9952087409d97fd46c62`)

## Head-to-Head Statistical Quality Scorecard

| Split Strategy | Domain Rows | Test Rows | Hosp Overlap | Energy Dist (32D) | Multi-RBF MMD² | Mean \|SMD\| (206) | Max \|SMD\| (206) | Max KS (18) | Max W1 (18) | Max Miss Drift | Region TV | Year TV | Rare Tail Δ | POAF Δ | Hosp-Grouped Adv AUC (207) | Core Adv AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_naive_2017_temporal` | 367,695 | 61,552 | **99.5%** | 0.00682 | 0.000441 | 0.03305 | **0.2793** | 0.0321 | 0.0421 | 0.0605 | 0.0102 | 1.0000 | 0.00127 | 0.00638 | **0.5908** | 0.6052 |
| `B_naive_patient_stratified` | 367,695 | 73,539 | **100.0%** | 0.00011 | 0.000008 | 0.00655 | **0.1102** | 0.0092 | 0.0189 | 0.0029 | 0.0056 | 0.0050 | 0.00017 | 0.00001 | **0.5003** | 0.4958 |
| `C_random_hospital_grouped` | 367,695 | 66,701 | **0.0%** | 0.00928 | 0.000639 | 0.03512 | **0.7598** | 0.0233 | 0.0243 | 0.0584 | 0.0812 | 0.0040 | 0.02118 | 0.00227 | **0.4954** | 0.6755 |
| `D_codex_matched_2014_2016` | 306,143 | 61,262 | **0.0%** | 0.00165 | 0.000132 | 0.01731 | **0.1956** | 0.0262 | 0.0489 | 0.0126 | 0.0195 | 0.0000 | 0.00429 | 0.00765 | **0.3819** | 0.6610 |
| `E_optimal_hosp_multivariate` | 367,695 | 73,240 | **0.0%** | 0.00212 | 0.000144 | 0.02278 | **0.5322** | 0.0151 | 0.0203 | 0.0126 | 0.0013 | 0.0003 | 0.00022 | 0.00000 | **0.4427** | 0.6558 |

## Inner 5-Fold Multivariate-Balanced Hospital-Disjoint CV (`train_optimal`)

| Fold | Rows | Hospitals | POAF Rate | Rare-Tail Rate | 2017 Share |
|---|---:|---:|---:|---:|---:|
| `fold_0` | 58,168 | 171 | 0.24295 | 0.09043 | 0.16729 |
| `fold_1` | 58,620 | 172 | 0.24297 | 0.08994 | 0.16718 |
| `fold_2` | 58,326 | 172 | 0.24296 | 0.09015 | 0.16773 |
| `fold_3` | 60,125 | 172 | 0.24298 | 0.08993 | 0.16727 |
| `fold_4` | 59,216 | 172 | 0.24296 | 0.09020 | 0.16735 |
