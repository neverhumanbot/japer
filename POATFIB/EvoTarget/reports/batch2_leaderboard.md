# STS CABG POAF — Batch-2 Clinical & Nested OOF Multi-Task Screening Leaderboard

- **Generated (UTC)**: `2026-11-25T08:20:02.644610+00:00`
- **Shared Contract**: Audited `foundation_raw_preop.parquet` (207 pre-incision raw cols, native categorical `Region`) + `splits.parquet`
- **Track 2A (Pure Row-Deterministic Clinical)**: 24 tested → **1 kept**
- **Track 2B (Nested Hospital-Grouped OOF Aux Priors)**: 10 tested → **0 kept**

## 1. Milestone Performance (5-Fold Hospital-Grouped CV & Sealed 2017 Holdout)

| Milestone | Features | Model | 5-Fold OOF AUC | 2017 Holdout AUC | 2017 *De Novo* AUC | 2017 PR-AUC | 2017 Logloss | ΔAUC vs B1 (95% CI) |
|---|---:|---|---:|---:|---:|---:|---:|---|
| B1 (Audited Raw Preop + Region) | 207 | CatBoost | 0.67812 | **0.67776** | 0.67801 | 0.38815 | 0.52271 | baseline |
| B1 (Audited Raw Preop + Region) | 207 | LightGBM | 0.67754 | **0.67810** | 0.67840 | 0.38799 | 0.52263 | baseline |
| B1 (Audited Raw Preop + Region) | 207 | CB+LGB 50/50 | 0.67844 | **0.67828** | 0.67856 | 0.38861 | 0.52251 | baseline |
| B1 + Track 2A (1 clinical) | 208 | CatBoost | 0.67807 | **0.67757** | 0.67783 | 0.38862 | 0.52274 | -0.00020 [-0.00046, +0.00006] |
| B1 + Track 2A (1 clinical) | 208 | LightGBM | 0.67787 | **0.67804** | 0.67837 | 0.38814 | 0.52262 | -0.00006 [-0.00042, +0.00030] |
| B1 + Track 2A (1 clinical) | 208 | CB+LGB 50/50 | 0.67857 | **0.67814** | 0.67844 | 0.38897 | 0.52253 | -0.00015 [-0.00039, +0.00010] |
| B1 + Track 2A + 2B (1+0) | 208 | CatBoost | 0.67807 | **0.67757** | 0.67783 | 0.38862 | 0.52274 | -0.00020 [-0.00046, +0.00006] |
| B1 + Track 2A + 2B (1+0) | 208 | LightGBM | 0.67787 | **0.67804** | 0.67837 | 0.38814 | 0.52262 | -0.00006 [-0.00042, +0.00030] |
| B1 + Track 2A + 2B (1+0) | 208 | CB+LGB 50/50 | 0.67857 | **0.67814** | 0.67844 | 0.38897 | 0.52253 | -0.00015 [-0.00039, +0.00010] |

## 2. Auxiliary Target Pre-Incision 5-Fold Hospital-Grouped OOF AUCs (Track 2B Sub-Models)

| Nested OOF Prior | 5-Fold Hospital-Grouped OOF AUC |
|---|---:|
| `oof_logit_mortalty` | 0.80476 |
| `oof_logit_cpvntlng` | 0.76533 |
| `oof_logit_crenfail` | 0.83654 |
| `oof_logit_cpleff` | 0.68445 |
| `oof_logit_cppneum` | 0.72391 |
| `oof_logit_csepsis` | 0.73133 |
| `oof_logit_cotarrst` | 0.74593 |
| `oof_logit_cvte` | 0.64589 |

## 3. All 34 Tested Batch-2 Features (Kept vs Rejected)

| Status | Track | Feature | Family | Uni AUC | CB LOI ΔAUC | CB Wins | LGB LOI ΔAUC | Joint Rank |
|---|---|---|---|---:|---:|---:|---:|---:|
| **KEPT** | `batch2a_clinical` | `diabetes_age_decoupled` | `metabolic_hematologic_coag` | 0.6440 (+) | +0.000144 | 3/5 | +0.000242 | 3 |
| rejected | `batch2a_clinical` | `hfpef_diastolic_substrate_score` | `structural_diastolic_valve` | 0.5869 (+) | +0.000141 | 4/5 | +0.000247 | 17 |
| rejected | `batch2a_clinical` | `prior_revasc_scar_burden` | `vascular_coronary_reclaimed` | 0.5070 (+) | +0.000070 | 3/5 | +0.000214 | 121 |
| rejected | `batch2b_oof_priors` | `oof_logit_cpvntlng` | `nested_oof_multitask_priors` | 0.5453 (+) | -0.000001 | 3/5 | +0.000029 | 24 |
| rejected | `batch2a_clinical` | `height_sex_standardized_z` | `anthropometric_geometry` | 0.5305 (+) | -0.000026 | 2/5 | +0.000005 | 8 |
| rejected | `batch2a_clinical` | `lv_remodeling_eccentricity` | `structural_diastolic_valve` | 0.5127 (+) | -0.000029 | 3/5 | -0.000037 | 57 |
| rejected | `batch2b_oof_priors` | `oof_logit_cpleff` | `nested_oof_multitask_priors` | 0.5624 (+) | -0.000081 | 1/5 | -0.000055 | 39 |
| rejected | `batch2b_oof_priors` | `oof_intrathoracic_inflam_cascade` | `nested_oof_multitask_priors` | 0.5575 (+) | -0.000101 | 3/5 | -0.000062 | 38 |
| rejected | `batch2b_oof_priors` | `oof_logit_crenfail` | `nested_oof_multitask_priors` | 0.5694 (+) | -0.000102 | 2/5 | +0.000050 | 34 |
| rejected | `batch2a_clinical` | `preop_antithrombotic_bleed_load` | `pharmacology_acuity_interactions` | 0.5010 (-) | -0.000118 | 3/5 | -0.000098 | 59 |
| rejected | `batch2a_clinical` | `bb_contraindicated_x_age` | `pharmacology_acuity_interactions` | 0.5038 (+) | -0.000120 | 1/5 | -0.000056 | 111 |
| rejected | `batch2a_clinical` | `atrial_nodal_stenosis_share` | `vascular_coronary_reclaimed` | 0.5029 (+) | -0.000127 | 1/5 | +0.000141 | 56 |
| rejected | `batch2a_clinical` | `age_centered_quadratic` | `anthropometric_geometry` | 0.5002 (-) | -0.000131 | 0/5 | +0.000181 | 5 |
| rejected | `batch2a_clinical` | `systolic_mr_combined_load` | `structural_diastolic_valve` | 0.5113 (+) | -0.000162 | 1/5 | +0.000164 | 45 |
| rejected | `batch2a_clinical` | `bsa_height_excess_residual` | `anthropometric_geometry` | 0.5126 (+) | -0.000169 | 2/5 | +0.000026 | 22 |
| rejected | `batch2a_clinical` | `glycemic_insulin_stress_tier` | `metabolic_hematologic_coag` | 0.5194 (-) | -0.000189 | 2/5 | -0.000054 | 27 |
| rejected | `batch2a_clinical` | `intrinsic_hepatic_inr_excess` | `metabolic_hematologic_coag` | 0.5205 (+) | -0.000191 | 1/5 | +0.000069 | 74 |
| rejected | `batch2a_clinical` | `oxygen_delivery_do2_proxy` | `metabolic_hematologic_coag` | 0.5103 (-) | -0.000233 | 0/5 | -0.000075 | 88 |
| rejected | `batch2b_oof_priors` | `oof_reconstructed_sts_morb_mort` | `nested_oof_multitask_priors` | 0.5703 (+) | -0.000234 | 1/5 | -0.000042 | 23 |
| rejected | `batch2a_clinical` | `cardiorenal_anemia_syndrome` | `metabolic_hematologic_coag` | 0.5157 (+) | -0.000244 | 1/5 | +0.000088 | 85 |
| rejected | `batch2b_oof_priors` | `oof_logit_cppneum` | `nested_oof_multitask_priors` | 0.5420 (+) | -0.000247 | 2/5 | -0.000116 | 33 |
| rejected | `batch2a_clinical` | `aortic_valve_calcific_load` | `structural_diastolic_valve` | 0.5300 (+) | -0.000251 | 1/5 | +0.000088 | 51 |
| rejected | `batch2a_clinical` | `carotid_polyvascular_burden` | `vascular_coronary_reclaimed` | 0.5275 (+) | -0.000267 | 0/5 | -0.000099 | 48 |
| rejected | `batch2a_clinical` | `refractory_antianginal_burden` | `pharmacology_acuity_interactions` | 0.5131 (+) | -0.000295 | 0/5 | +0.000158 | 83 |
| rejected | `batch2a_clinical` | `diffuse_cad_x_age_burden` | `vascular_coronary_reclaimed` | 0.5738 (+) | -0.000307 | 1/5 | -0.000128 | 42 |
| rejected | `batch2b_oof_priors` | `oof_logit_cotarrst` | `nested_oof_multitask_priors` | 0.5608 (+) | -0.000316 | 1/5 | +0.000228 | 58 |
| rejected | `batch2a_clinical` | `amiodarone_x_age_protection` | `pharmacology_acuity_interactions` | 0.5016 (+) | -0.000323 | 1/5 | -0.000095 | 38 |
| rejected | `batch2a_clinical` | `mitral_ischemic_etiology_risk` | `structural_diastolic_valve` | 0.5338 (+) | -0.000376 | 0/5 | +0.000238 | 37 |
| rejected | `batch2a_clinical` | `systemic_marrow_inflam_triad` | `metabolic_hematologic_coag` | 0.5048 (+) | -0.000410 | 1/5 | +0.000009 | 141 |
| rejected | `batch2a_clinical` | `age_gompertz_atrial_hazard` | `anthropometric_geometry` | 0.6449 (+) | -0.000428 | 0/5 | +0.000093 | 2 |
| rejected | `batch2b_oof_priors` | `oof_logit_mortalty` | `nested_oof_multitask_priors` | 0.5758 (+) | -0.000437 | 2/5 | -0.000072 | 13 |
| rejected | `batch2a_clinical` | `epidemiologic_age_race_sex_bsa` | `anthropometric_geometry` | 0.6598 (+) | -0.000592 | 1/5 | +0.000143 | 1 |
| rejected | `batch2b_oof_priors` | `oof_logit_cvte` | `nested_oof_multitask_priors` | 0.5626 (+) | -0.000611 | 1/5 | +0.000079 | 22 |
| rejected | `batch2b_oof_priors` | `oof_logit_csepsis` | `nested_oof_multitask_priors` | 0.5637 (+) | -0.000664 | 1/5 | -0.000089 | 37 |
