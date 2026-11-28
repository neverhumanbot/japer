# Executive Brief: Pre-Incision POAF AI Across Unseen & Seen U.S. Hospitals

**For Health-System CEOs, CMOs & Chief AI Officers**  
*NEVERHUMAN Research & HonorHealth Heart Care / HonorHealth Research Institute · STS ACSD v2.81 · November 2026*  
**[Download 2-Page Formatted PDF](evotarget-poaf-ceo-brief.pdf)** · **[NEJM AI Manuscript (PDF)](../nejm_ai/main.pdf)** · **[npj Digital Medicine (PDF)](../npj_digital_medicine/main.pdf)** · **[CTO Brief (PDF)](evotarget-poaf-cto-brief.pdf)**

---

## Executive Snapshot

| **368,132 / 1,091** | **0.6767 Cold-Start / 0.7014 In-Net** | **20.4–33.8 Avoided / 100** | **+6.7 pts vs. Ext. Median** |
| :---: | :---: | :---: | :---: |
| **National STS CABG Cohort** | **Dual-Regime Registry Frontier** | **Bedside Clinical Utility (DCA)** | **Literature Reconciliation** |
| 89,337 POAF events (`24.30%`); audited `638 → 207` pre-incision inputs with skip-logic resolution | **Dual-Paradigm Fusion** beats Mariscalco POAF (`+577 bps`) & all 6 binary AI baselines on 74 unseen hospitals (`79.2%` of 7× gap); EB site priors lift seen hospitals to `0.6974–0.7014` | **7.4× decile risk gradient** (`6.1%–44.8%`); avoids `20.4–33.8` unnecessary prophylactic interventions per 100 patients (`P>0 = 1.000`) | Published AUCs `>0.72` use post-op/ICU data, valve cohorts, or waveforms; PROBAST external median is **`0.61`** |

---

## 1. Why Published Clinical AI Claims (0.85–0.97 AUC) Collapse at New Hospitals

Postoperative atrial fibrillation (POAF) complicates **nearly 1 in 4 isolated CABG operations** (`24.30%`), driving stroke, ICU prolongation, readmissions, and high surgical cost—and the highest-leverage preventive interventions (such as posterior pericardiotomy or targeted amiodarone) must be chosen **before skin incision**. Yet two PROBAST critical appraisals show published POAF models drop from `0.71` apparent AUC to a **`0.61` external median**.

Our forensic audit of **368,132 isolated-CABG surgeries across 1,091 U.S. centers** (STS ACSD v2.81) exposes why:
1. **Skip-logic collider bias & early-column leakage:** Dropping structurally blank uncomplicated recoveries (`Complics == 2`, `63.3%`) induces collider bias, and positional column slicing retains seven hidden postoperative duration leaks (`LOS`, `PLOS`, `ORDuration`, etc.). Purging all leaks leaves strictly **207 pre-incision predictors**.
2. **Naive calendar splits reuse 99.5% of hospitals:** Default 2017 holdouts reuse 998/1,003 hospitals, hiding a **`−1.99` AUC-point drop** on unseen 2017 hospitals (`0.6778 → 0.6579`). Our controlled **100% Seen-Hospital** experiment isolates `+28–64 bps` of pure hospital-overlap inflation even with zero `HospID`.
3. **Binary `{0,1}` labels trap every AI vendor:** Across 74 strictly unseen U.S. hospitals (`N = 21,241`), all six world-leading binary tabular families (CatBoost, LightGBM, XGBoost, TabPFN-3.5, KumoTabular-Large v2, TabFM v1.0.0) compress within `0.6712–0.6746`.

---

## 2. Two Breakthroughs & Two Deployment Regimes

1. **Cold-Start Unseen Hospitals (0% Overlap, 74 Centers):** Pairing 1,000-generation continuous-target clinical self-attention (**EvoTarget-FT**, `0.6765`, verified zero-collapse across 256,000 fits) with 6-fix parallel genetic programming (**JAIN Ultra GP v2**, `0.6756`, `+39.8* bps` vs. legacy) produces **Dual-Paradigm Fusion (`0.6767` AUC, `0.0072` ECE, slope `0.983`, O:E `0.994`)**—beating Mariscalco POAF by **`+577.0 bps`**, every 50k binary AI baseline by **`+20.3–55.3 bps`**, and a 3-way binary blend by **`+9.9* bps`**, recovering **`79.2%`** of a 7× national data expansion (`50k → 346k`), winning **24 of 25 demographic/volume subgroups**, and avoiding **20.4–33.8 unnecessary interventions per 100 patients** at 25%–35% decision thresholds.
2. **In-Network Seen Hospitals (All 1,091 Centers in Train, 100% Overlap):** Once a surgical center has local registry history, incorporating strictly train-only Empirical-Bayes (EB) institutional priors lifts every 50k model by **`+174–213 bps`** (nearly +2 full AUC points) into **`0.6958–0.6986`** (Dual-Paradigm Fusion `+EB346k` **`0.6974`**; native 50k `site208` champion **`0.6916`**, `+29.0* bps` over CatBoost `site208`), pushes full-registry 346k CatBoost past `0.70` to **`0.7014`** (`slope 1.014`, `O:E 0.999`, `1.96×` lift), and cuts prospective 2017 center rate error by **14%** (`5.63 → 4.84 pp`).

---

## 3. Dual-Regime Scoreboard (`N = 21,241` Natural-Prior Holdout)

| Model (207 Strictly Pre-Incision Predictors) | Unseen Hosp. (0% ovlp) | Seen Hosp. (207 raw) | Seen + 346k EB Prior | Unseen Fusion Δ [74-Hosp 95% CI] |
| :--- | ---: | ---: | ---: | :--- |
| **Dual-Paradigm Fusion (`GP v2 + EvoTarget-FT`)** | **0.6767** | **0.6794** | **0.6974** *(50k site208: **0.6916**)* | **#1 Cold-Start (79.2% of 7× gap; 0.0072 ECE)** |
| **EvoTarget-FT (50k, Gen-20 $z^*$)** | **0.6765** | 0.6793 | 0.6972 | +1.4 `[-4.0, +6.9]` *(vs. Bin. FT: **+18.9***)* |
| **JAIN Ultra GP v2 (50k, 6 evolved fams)** | **0.6756** | 0.6788 | 0.6969 | +10.3 `[-5.6, +25.8]` *(vs. Legacy: **+39.8***)* |
| CatBoost 50k (5-fold binary ref) | 0.6742 | 0.6777 | 0.6966 | **+24.5** `[+7.2, +42.2]`* |
| PriorLabs TabPFN-3.5 (50k native) | 0.6738 | 0.6795 | 0.6982 | **+29.2** `[+8.2, +53.5]`* |
| LightGBM / XGBoost (5-fold binary) | 0.6736 | 0.6780 / 0.6774 | 0.6960 / 0.6958 | **+30.5 / +31.1*** (all CIs > 0) |
| NVIDIA Kumo-L v2 / Google TabFM v1.0 | 0.6715 / 0.6712 | 0.6764 / 0.6775 | 0.6977 / 0.6986 | **+52.0 / +55.3*** (all CIs > 0) |
| *Mariscalco POAF Score (STS proxy)* | *0.6190* | *0.6276* | *—* | **+577.0** `[+495.6, +656.9]`* |
| *7× Full-Registry CatBoost (`346,454`)* | *0.6773* | *0.6831* | ***0.7014*** | *−6.4 (Fusion beats on AP, ECE & 2017 unseen)* |

---

## 4. Safe Clinical Pilot & Honest Governance Boundaries

1. **Map 207 Routine Pre-Incision Fields (Week 1):** Connect existing STS/EHR intake data with zero post-incision variables.
2. **Dual-Tier Shadow Scoring (Weeks 2–3):** Deploy cold-start Dual-Paradigm Fusion at new sites and add single-intercept EB recalibration once historical site volume accumulates (`−14%` center rate error).
3. **Prospective Confirmation Gate (STS v4.20+):** Freeze model weights and site-intercept calibration prior to prospective multi-center confirmation and randomized prophylaxis trial testing (e.g., PALACS) before changing bedside care.

**Next Reads:** [NEJM AI Original Article (PDF)](../nejm_ai/main.pdf) · [npj Digital Medicine Article (PDF)](../npj_digital_medicine/main.pdf) · [CTO Brief (PDF)](evotarget-poaf-cto-brief.pdf) · [Seen-Hospital Receipt](../../reports/seen_hospital_study/seen_hospital_master_receipt.md)
