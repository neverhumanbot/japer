# Clean Pre-Incision POAF Benchmark Receipt (`NEVERHUMAN` vs. 5 Binary Alternatives)

National STS ACSD v2.81 isolated-CABG cohort (`368,132` procedures; `367,695` labeled; `1,091` U.S. centers; strictly `207` pre-incision predictors; `5,000`-draw hospital-cluster 95% CIs, `seed = 42`).

## 1. Unseen-Hospital Cold Start (74 Unseen Centers, 0% Overlap, `N = 21,241`)
| Model (50k Train Unless Noted) | ROC-AUC (95% CI) | NEVERHUMAN Margin [95% CI] | AP | ECE | Slope | O:E |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| **NEVERHUMAN (Proprietary)** | 0.6767 (0.6686--0.6858) | Reference (#1 @ 50k) | 0.3777 | 0.0072 | 0.983 | 0.994 |
| AutoGluon 1.5 (`best_quality`, binary) | 0.6745 (0.6663--0.6837) | +21.8 [+7.0, +36.5]* (P>0=0.998) | 0.3743 | 0.0046 | 1.000 | 0.991 |
| CatBoost (5-fold binary) | 0.6742 (0.6659--0.6835) | +24.5 [+7.2, +42.2]* (P>0=0.997) | 0.3744 | 0.0078 | 1.056 | 1.001 |
| PriorLabs TabPFN-3.5 (binary) | 0.6738 (0.6659--0.6827) | +29.2 [+8.2, +53.5]* (P>0=0.998) | 0.3728 | 0.0068 | 1.002 | 1.002 |
| NVIDIA KumoTabular-Large v2 (binary) | 0.6715 (0.6632--0.6807) | +52.0 [+21.6, +83.3]* (P>0=1.000) | 0.3693 | 0.0070 | 1.040 | 1.004 |
| Google TabFM v1.0.0 (binary) | 0.6712 (0.6628--0.6805) | +55.3 [+28.6, +81.5]* (P>0=1.000) | 0.3697 | 0.0090 | 0.963 | 0.992 |
| Mariscalco POAF Bedside Proxy | 0.6190 (0.6102--0.6281) | +577.0 [+495.6, +656.9]* (P>0=1.000) | 0.3097 | 0.0190 | 0.962 | 1.004 |
| CatBoost Full Registry (346k, 7x Ref) | 0.6773 (0.6695--0.6862) | -6.4 [-23.0, +10.5] (P>0=0.221) | 0.3776 | 0.0076 | 1.023 | 1.000 |

## 2. Seen-Hospital In-Network Study (All 1,091 Centers in Train, `1,086` Test Centers, `N = 21,241`)
| Model | Unseen (0% ovlp) | Seen 207 Raw | Seen +EB50k | Seen +EB346k | Native site208 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Mariscalco POAF Proxy | 0.6190 | 0.6276 | --- | --- | --- |
| Google TabFM v1.0.0 | 0.6712 | 0.6775 | 0.6918 | 0.6986 | --- |
| NVIDIA KumoTabular-Large v2 | 0.6715 | 0.6764 | 0.6911 | 0.6977 | --- |
| PriorLabs TabPFN-3.5 | 0.6738 | 0.6795 | 0.6919 | 0.6982 | --- |
| CatBoost (50k binary) | 0.6742 | 0.6777 | 0.6902 | 0.6966 | 0.6886 |
| AutoGluon 1.5 (`best_quality`) | 0.6745 | 0.6790 | 0.6912 | 0.6974 | 0.6891 |
| **NEVERHUMAN** | 0.6767 | 0.6794 | 0.6909 | 0.6974 | 0.6916 |
| CatBoost Full Registry (346k, 7x) | 0.6773 | 0.6831 | --- | --- | **0.7014** |

## 3. Train & Inference Time Showcase (`50,000` Train -> `21,241` Holdout Patients)
| Model | Train / Prep Time | 21,241-Pt Batch Infer | Latency (ms/pt) | 1-Pt Bedside (ms) | Hardware / Peak VRAM |
| :--- | ---: | ---: | ---: | ---: | :--- |
| NEVERHUMAN | 1290.7 s (21.5 min) | 4.11 s | 0.193 | 45.2 | 1x NVIDIA GeForce RTX 3090 + CPU (0.5 GB) |
| CatBoost (5-fold binary) | 49.2 s (0.8 min) | 0.15 s | 0.007 | 3.0 | CPU (20 threads) (CPU / mixed) |
| AutoGluon 1.5 (best_quality) | 1797.1 s (30.0 min) | 80.99 s | 3.81 | 28510.7 | 1x NVIDIA GeForce RTX 4090 + 48 CPUs (CPU / mixed) |
| NVIDIA KumoTabular-Large v2 | 197.1 s (3.3 min) | 481.69 s | 22.68 | N/A (in-context) | NVIDIA A100-PCIE-40GB (6.8 GB) |
| PriorLabs TabPFN-3.5 | 0.9 s (0.0 min) | 515.29 s | 24.26 | N/A (in-context) | NVIDIA A100-PCIE-40GB (7.7 GB) |
| Google TabFM v1.0.0 | 394.9 s (6.6 min) | 1326.73 s | 62.46 | N/A (in-context) | NVIDIA A100-PCIE-40GB (22.1 GB) |

## 4. False Positive Rate (FPR) & U.S. (200k/yr) / Global (1M/yr) Clinical Impact Scaling
| Model (Unseen Hospitals) | FPR @ 70% Sens | Spared / yr (200k U.S.) | Spared / yr (1M Global) | FPR @ 80% Sens | Top-25% PPV | Modeled POAF Prevented (U.S. / Global) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **NEVERHUMAN (Proprietary)** | 44.10% | Ref (Lowest) | Ref (Lowest) | 55.54% | 38.93% | 8,758 / 43,790 |
| AutoGluon 1.5 (`best_quality`, binary) | 44.42% | +490 | +2,448 | 55.97% | 38.68% | 8,703 / 43,515 |
| CatBoost (5-fold binary) | 44.58% | +734 | +3,672 | 55.86% | 38.74% | 8,716 / 43,578 |
| PriorLabs TabPFN-3.5 (binary) | 44.76% | +1,007 | +5,037 | 55.82% | 38.81% | 8,733 / 43,663 |
| NVIDIA KumoTabular-Large v2 (binary) | 45.28% | +1,798 | +8,992 | 56.24% | 38.57% | 8,678 / 43,388 |
| Google TabFM v1.0.0 (binary) | 45.14% | +1,582 | +7,909 | 56.98% | 38.81% | 8,733 / 43,663 |
| Mariscalco POAF Bedside Proxy | 76.30% | +48,755 | +243,774 | 76.30% | 33.82% | 7,610 / 38,049 |
