# Clinical Journal Analyses Receipt (`test_full_21k`, N=21,241, 74 Unseen Hospitals)

## 1. Standard Bedside Clinical Scores vs. Dual-Paradigm Fusion
| Model / Clinical Score | 21k AUC (95% 74-Hosp CI) | AP | ECE | Slope | O:E | Fusion Δ bps (95% CI) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **Dual-Paradigm Fusion** | 0.6767 (0.6686–0.6858) | 0.3777 | 0.00724 | 0.983 | 0.994 | — |
| EvoTarget-FT (Gen-20) | 0.6765 (0.6685–0.6856) | 0.3775 | 0.00850 | 0.976 | 0.996 | — |
| JAIN Ultra GP v2 | 0.6756 (0.6673–0.6850) | 0.3758 | 0.00805 | 1.002 | 0.991 | — |
| CatBoost 50k (Binary) | 0.6742 (0.6659–0.6835) | 0.3744 | 0.00778 | 1.056 | 1.001 | — |
| TabPFN-3.5 50k (Binary) | 0.6738 (0.6659–0.6827) | 0.3728 | 0.00683 | 1.002 | 1.002 | — |
| KumoTabular-Large (Binary) | 0.6715 (0.6632–0.6807) | 0.3693 | 0.00702 | 1.040 | 1.004 | — |
| TabFM v1.0.0 (Binary) | 0.6712 (0.6628–0.6805) | 0.3697 | 0.00901 | 0.963 | 0.992 | — |
| Mathew Preop Proxy | 0.6451 (0.6344–0.6558) | 0.3435 | 0.00974 | 0.965 | 0.997 | +316.1 [+245.1, +386.1] |
| Mariscalco POAF Score | 0.6190 (0.6102–0.6281) | 0.3097 | 0.01898 | 0.962 | 1.004 | +577.0 [+495.6, +656.9] |
| Age Alone | 0.6397 (0.6292–0.6503) | 0.3367 | 0.01268 | 0.955 | 0.996 | +369.6 [+290.4, +449.0] |
| Burgos COM-AF Score | 0.5760 (0.5659–0.5863) | 0.2843 | 0.00682 | 1.067 | 0.993 | +1006.3 [+927.3, +1095.3] |
| HATCH Score | 0.5737 (0.5640–0.5839) | 0.2872 | 0.00599 | 1.006 | 0.994 | +1029.3 [+953.5, +1109.9] |
| CHA2DS2-VASc Score | 0.5753 (0.5658–0.5845) | 0.2835 | 0.00583 | 1.084 | 0.996 | +1013.5 [+927.2, +1105.0] |

## 2. Decile Risk Stratification (Dual-Paradigm Fusion)
| Decile | N | Events | Predicted Risk | Observed Risk (95% CI) | Enrichment |
| :---: | ---: | ---: | ---: | :--- | ---: |
| D1 | 2125 | 129 | 6.47% | 6.07% (5.13–7.17%) | 0.25× |
| D2 | 2124 | 255 | 11.80% | 12.01% (10.69–13.46%) | 0.49× |
| D3 | 2124 | 319 | 16.07% | 15.02% (13.56–16.60%) | 0.62× |
| D4 | 2124 | 396 | 19.42% | 18.64% (17.04–20.36%) | 0.77× |
| D5 | 2124 | 471 | 22.30% | 22.18% (20.46–23.99%) | 0.91× |
| D6 | 2124 | 567 | 25.04% | 26.69% (24.86–28.62%) | 1.10× |
| D7 | 2124 | 621 | 28.08% | 29.24% (27.34–31.21%) | 1.20× |
| D8 | 2124 | 676 | 31.70% | 31.83% (29.88–33.84%) | 1.31× |
| D9 | 2124 | 774 | 36.58% | 36.44% (34.42–38.51%) | 1.50× |
| D10 | 2124 | 952 | 46.85% | 44.82% (42.72–46.94%) | 1.84× |

## 3. Clinical Utility — Decision Curve Analysis Operating Points
| Threshold | Flagged | Sens | Spec | PPV | NPV | Net Benefit (Fusion vs All) | Avoided / 100 | ΔNB vs Mariscalco (P>0) |
| :---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| 15% | 77.8% | 91.5% | 26.6% | 28.5% | 90.6% | 0.1240 vs 0.1093 | 8.3 | +0.0079 (1.000) |
| 20% | 63.1% | 81.3% | 42.7% | 31.3% | 87.7% | 0.0890 vs 0.0537 | 14.2 | +0.0154 (1.000) |
| 25% | 45.1% | 64.5% | 61.1% | 34.8% | 84.3% | 0.0587 vs -0.0094 | 20.4 | +0.0180 (1.000) |
| 30% | 29.3% | 45.6% | 75.9% | 37.8% | 81.3% | 0.0325 vs -0.0815 | 26.6 | +0.0185 (1.000) |
| 35% | 17.5% | 29.8% | 86.5% | 41.5% | 79.3% | 0.0174 vs -0.1647 | 33.8 | +0.0153 (1.000) |
| 40% | 9.8% | 18.2% | 92.8% | 44.8% | 78.0% | 0.0078 vs -0.2618 | 40.4 | +0.0075 (0.999) |

## 4. Subgroup Fairness & Clinical Transport (74-Hospital Cluster CIs)
| Group | Subgroup | N | POAF % | Fusion AUC (95% CI) | ECE | O:E | Δ vs CB50k (bps) | Δ vs Mariscalco (bps) |
| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| Overall | All Unseen-Hospital Patients | 21,241 | 24.3% | 0.6767 (0.6686–0.6858) | 0.0072 | 0.994 | +24.5 | +577.0 |
| Sex | Male | 16,062 | 25.1% | 0.6719 (0.6610–0.6836) | 0.0085 | 0.985 | +21.1 | +509.0 |
| Sex | Female | 5,172 | 22.0% | 0.6891 (0.6754–0.7018) | 0.0109 | 1.031 | +37.2 | +658.1 |
| Age | Age < 60 yr | 6,283 | 13.2% | 0.6667 (0.6527–0.6803) | 0.0074 | 0.993 | -1.2 | +1332.5 |
| Age | Age 60–69 yr | 7,881 | 24.4% | 0.6149 (0.5997–0.6282) | 0.0113 | 1.016 | +45.4 | +999.9 |
| Age | Age 70–79 yr | 5,627 | 33.0% | 0.6173 (0.5974–0.6377) | 0.0134 | 0.996 | +43.2 | +1018.9 |
| Age | Age >= 80 yr | 1,450 | 38.3% | 0.5843 (0.5449–0.6237) | 0.0321 | 0.924 | +84.7 | +449.3 |
| Race / Ethnicity | White | 17,206 | 25.7% | 0.6688 (0.6591–0.6791) | 0.0083 | 0.995 | +24.2 | +528.4 |
| Race / Ethnicity | Black | 1,822 | 17.1% | 0.6849 (0.6544–0.7148) | 0.0136 | 1.034 | +13.8 | +701.8 |
| Race / Ethnicity | Asian | 842 | 18.8% | 0.6534 (0.6071–0.7027) | 0.0317 | 0.905 | +62.4 | +374.0 |
| Race / Ethnicity | Hispanic / Latino | 1,433 | 16.5% | 0.7146 (0.6743–0.7532) | 0.0244 | 0.940 | +37.3 | +505.4 |
| Comorbidity | Diabetes Mellitus | 10,456 | 24.1% | 0.6812 (0.6711–0.6918) | 0.0106 | 1.023 | +24.7 | +655.6 |
| Comorbidity | No Diabetes | 10,738 | 24.5% | 0.6736 (0.6615–0.6861) | 0.0116 | 0.968 | +27.4 | +508.1 |
| Comorbidity | LVEF < 40% | 2,893 | 25.8% | 0.6787 (0.6546–0.7050) | 0.0231 | 0.979 | +26.4 | +606.2 |
| Comorbidity | LVEF >= 40% | 17,764 | 24.1% | 0.6764 (0.6668–0.6865) | 0.0073 | 0.997 | +22.9 | +551.9 |
| Comorbidity | Chronic Lung Disease | 4,643 | 27.0% | 0.6710 (0.6515–0.6891) | 0.0175 | 1.022 | +19.3 | +455.2 |
| Comorbidity | CKD (eGFR < 60) | 4,561 | 28.9% | 0.6439 (0.6274–0.6610) | 0.0118 | 0.989 | +22.5 | +619.1 |
| Acuity | Elective CABG | 7,887 | 25.0% | 0.6652 (0.6491–0.6801) | 0.0118 | 0.997 | +3.3 | +562.2 |
| Acuity | Urgent / Emergent CABG | 13,349 | 23.9% | 0.6830 (0.6742–0.6930) | 0.0081 | 0.993 | +35.8 | +570.3 |
| Stress Slice | No Recorded Preop AF | 21,103 | 24.3% | 0.6771 (0.6691–0.6862) | 0.0068 | 0.994 | +24.6 | +577.4 |
| Stress Slice | Rare Clinical Tails | 1,917 | 26.2% | 0.6935 (0.6586–0.7201) | 0.0273 | 0.918 | +44.7 | +589.1 |
| Stress Slice | 2017 Unseen Hospitals | 3,555 | 22.5% | 0.6579 (0.6349–0.6827) | 0.0264 | 0.904 | +14.4 | +570.4 |
| Hospital Volume | Low-Vol Centers (n<=141) | 1,864 | 26.0% | 0.6818 (0.6492–0.7122) | 0.0310 | 1.058 | +34.3 | +640.4 |
| Hospital Volume | Mid-Vol Centers | 5,282 | 23.0% | 0.6851 (0.6713–0.6981) | 0.0122 | 0.960 | +21.4 | +564.2 |
| Hospital Volume | High-Vol Centers (n>341) | 14,095 | 24.6% | 0.6727 (0.6625–0.6838) | 0.0080 | 0.998 | +24.7 | +568.0 |

## 5. Hospital-Level Transportability & Local Recalibration
- Evaluable Unseen Hospitals: **70 / 74**
- Median Center AUC: **0.6749** (IQR 0.6434–0.7021); beats 50k CatBoost in **52.9%** of centers
- Uncalibrated Center O:E (n>=50): median **0.958** (IQR 0.856–1.135)
- Prospective Chronological Local Intercept Recalibration (2014–2016 onboarding → 2017 deployment, n=40 centers): Mean |Obs−Pred| rate error drops from **5.63 pp** → **5.01 pp** (unshrunk) → **4.84 pp** (empirical-Bayes $m_0=100$); Mean Brier improves **0.16914 → 0.16777**

## 6. Top 12 Clinical Drivers of Dual-Paradigm Fusion Log-Odds
- Standardized surrogate linear $R^2$ on Fusion log-odds: **0.8896** | Logit corr(GP v2, EvoTarget-FT) = **0.9591**
| Rank | Clinical Feature | Domain | Std. β | Spearman r |
| :---: | :--- | :--- | ---: | ---: |
| 1 | Age (years) | Demographics | +0.8425 | +0.8114 |
| 2 | Body Surface Area (m²) | Anthropometry | +0.3697 | +0.2073 |
| 3 | White Race (indicator) | Demographics | +0.1364 | +0.2574 |
| 4 | Heart Failure (CHF / Prior HF) | Cardiac Function | +0.0894 | +0.1449 |
| 5 | Mitral Regurgitation Grade | Valvular | +0.0870 | +0.1871 |
| 6 | Preop Amiodarone Therapy | Pharmacology | -0.0862 | -0.0727 |
| 7 | Diabetes Mellitus | Metabolic | -0.0700 | -0.0700 |
| 8 | White Blood Cell Count | Inflammatory | +0.0649 | -0.0181 |
| 9 | Female Sex | Demographics | -0.0594 | -0.1484 |
| 10 | Chronic Lung Disease Severity | Pulmonary | +0.0538 | +0.0907 |
| 11 | Tricuspid Regurgitation Grade | Valvular | +0.0532 | +0.1427 |
| 12 | CKD-EPI eGFR (mL/min/1.73m²) | Renal Reserve | -0.0484 | -0.3573 |
