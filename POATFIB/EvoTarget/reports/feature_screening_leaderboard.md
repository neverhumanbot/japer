# STS AFib pre-incision feature screen

Audited raw baseline with 207 predictors. Screened 58 candidates on fixed subsamples of folds 0 and 1. Confirmed 10 candidates on all five hospital folds.

## Frozen 2017 comparison

Paired evaluation on 61,552 labeled cases. Changes are frozen portfolio minus raw baseline.

| Metric | Raw baseline | Frozen portfolio | Change |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.674429 | 0.674429 | +0.000000 |
| Average precision | 0.383369 | 0.383369 | +0.000000 |
| Log loss | 0.524388 | 0.524388 | +0.000000 |
| Top-decile lift | 1.807120 | 1.807120 | +0.000000 |

Frozen additions: none.
Published additions: none.

## Development results

| Candidate | Screen ΔAUC | Confirmation ΔAUC | Portfolio ΔAUC | Portfolio wins | Disposition |
| --- | ---: | ---: | ---: | ---: | --- |
| race_adjusted_age_risk | +0.001099 | -0.000463 | -0.000463 | 0 | Did not meet the pooled AUC gain and fold-wins rule |
| score_hatch | +0.000392 | -0.000179 | -0.000179 | 2 | Did not meet the pooled AUC gain and fold-wins rule |
| cardioprotective_prophylaxis_bundle | +0.000339 | -0.000248 | -0.000248 | 1 | Did not meet the pooled AUC gain and fold-wins rule |
| score_poaf_mariscalco | +0.000313 | -0.000090 | -0.000090 | 2 | Did not meet the pooled AUC gain and fold-wins rule |
| pulmonary_copd_osa_burden | +0.000166 | -0.000063 | -0.000063 | 3 | Did not meet the pooled AUC gain and fold-wins rule |
| age_x_height_sex_proxy | +0.000101 | -0.000498 | -0.000498 | 0 | Did not meet the pooled AUC gain and fold-wins rule |
| acuity_status_admit_mi_score | +0.000093 | -0.000011 | -0.000011 | 3 | Did not meet the pooled AUC gain and fold-wins rule |
| frailty_gait_speed_mps | +0.000084 | -0.000038 | -0.000038 | 0 | Did not meet the pooled AUC gain and fold-wins rule |
| natriuretic_peptide_harmonized_log | +0.000033 | -0.000181 | -0.000181 | 2 | Did not meet the pooled AUC gain and fold-wins rule |
| n_territories_severe_stenosis | +0.000023 | -0.000014 | -0.000014 | 0 | Did not meet the pooled AUC gain and fold-wins rule |
| pulmonary_htn_right_heart_load | +0.000003 | — | — | — | Not shortlisted |
| num_diseased_vessels_true_count | +0.000000 | — | — | — | Not shortlisted |
| log_wbc_inflammatory | +0.000000 | — | — | — | Not shortlisted |
| log_platelet_to_wbc_ratio | +0.000000 | — | — | — | Not shortlisted |
| cath_to_surg_log_days | +0.000000 | — | — | — | Not shortlisted |
| disability_payor_vulnerability | +0.000000 | — | — | — | Not shortlisted |
| missing_core_preop_labs_count | +0.000000 | — | — | — | Not shortlisted |
| score_cha2ds2_vasc | -0.000014 | — | — | — | Not shortlisted |
| beta_blocker_continuity_tier | -0.000088 | — | — | — | Not shortlisted |
| cardiorenal_stretch_index | -0.000106 | — | — | — | Not shortlisted |
| occult_preop_arrhythmia_proxy | -0.000115 | — | — | — | Not shortlisted |
| lvedd_indexed_bsa | -0.000169 | — | — | — | Not shortlisted |
| combined_pulmonary_deficit | -0.000178 | — | — | — | Not shortlisted |
| smoking_paradox_ordinal | -0.000186 | — | — | — | Not shortlisted |
| score_sts_comorbidity_burden | -0.000194 | — | — | — | Not shortlisted |
| lmain_prox_lad_severity | -0.000198 | — | — | — | Not shortlisted |
| egfr_age_normalized_ratio | -0.000286 | — | — | — | Not shortlisted |
| lvedv_teichholz_bsa_idx | -0.000290 | — | — | — | Not shortlisted |
| albi_nutritional_hepatic_score | -0.000319 | — | — | — | Not shortlisted |
| lvsd_indexed_bsa | -0.000349 | — | — | — | Not shortlisted |
| age_excess_55_nonlinear | -0.000358 | — | — | — | Not shortlisted |
| egfr_ckdepi_2021 | -0.000377 | — | — | — | Not shortlisted |
| meld_xi_score | -0.000413 | — | — | — | Not shortlisted |
| race_x_age_x_bsa_substrate | -0.000445 | — | — | — | Not shortlisted |
| age_exact_coalesced | -0.000448 | — | — | — | Not shortlisted |
| valvular_atrial_stretch_burden | -0.000487 | — | — | — | Not shortlisted |
| coronary_leaman_jeopardy_score | -0.000526 | — | — | — | Not shortlisted |
| ckd_kdigo_stage | -0.000570 | — | — | — | Not shortlisted |
| bmi_kg_m2 | -0.000617 | — | — | — | Not shortlisted |
| weight_to_ibw_devine_ratio | -0.000620 | — | — | — | Not shortlisted |
| score_mathew_preop | -0.000632 | — | — | — | Not shortlisted |
| lv_fractional_shortening | -0.000658 | — | — | — | Not shortlisted |
| biatrial_remodeling_composite | -0.000777 | — | — | — | Not shortlisted |
| preop_workup_intensity | -0.000791 | — | — | — | Not shortlisted |
| hct_to_hgb_mchc_proxy | -0.000795 | — | — | — | Not shortlisted |
| pulmonary_pressure_lv_dysfunction | -0.000805 | — | — | — | Not shortlisted |
| score_com_af_burgos | -0.000863 | — | — | — | Not shortlisted |
| bmi_u_shape_dev | -0.000923 | — | — | — | Not shortlisted |
| degenerative_vs_familial_age | -0.000924 | — | — | — | Not shortlisted |
| cockcroft_gault_crcl | -0.000966 | — | — | — | Not shortlisted |
| beta_blocker_withdrawal_flag | -0.000984 | — | — | — | Not shortlisted |
| wbc_to_hgb_stress_ratio | -0.000997 | — | — | — | Not shortlisted |
| preop_catecholamine_instability | -0.001103 | — | — | — | Not shortlisted |
| age_to_albumin_frailty_ratio | -0.001106 | — | — | — | Not shortlisted |
| age_x_bsa_la_size_proxy | -0.001167 | — | — | — | Not shortlisted |
| sa_av_nodal_artery_stenosis | -0.001189 | — | — | — | Not shortlisted |
| abg_pco2_to_po2_ratio | -0.001296 | — | — | — | Not shortlisted |
| anemia_who_deficit_hgb | -0.001545 | — | — | — | Not shortlisted |
