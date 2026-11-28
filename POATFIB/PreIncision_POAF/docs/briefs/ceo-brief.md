# Executive Brief: Outperforming Frontier Tabular AI on Unseen & Seen U.S. Hospitals — At 117×–323× Faster Speed

**For Health-System CEOs, CMOs, Chief AI Officers & Clinical Quality Leaders**  
*NEVERHUMAN Research & HonorHealth Heart Care · STS ACSD v2.81 · November 2026*  
**[Download 2-Page Formatted PDF](neverhuman-poaf-ceo-brief.pdf)** · **[NEJM AI Manuscript (PDF)](../nejm_ai/main.pdf)** · **[npj Digital Medicine (PDF)](../npj_digital_medicine/main.pdf)** · **[CTO Brief (PDF)](neverhuman-poaf-cto-brief.pdf)**

---

## Executive Snapshot

| **368,132 CABG Cases** | **0.6767 / 0.7014 AUC** | **4.11 s (0.193 ms/pt)** | **20.4–33.8 Avoided / 100** |
| :---: | :---: | :---: | :---: |
| **1,091 U.S. Centers** | **Unseen / In-Net Frontier** | **117×–323× Faster Infer** | **Clinical Net Benefit** |
| Audited STS v2.81 cohort (`24.30%` POAF); strictly 207 pre-incision EHR variables | **#1 at 50k** on 74 unseen hospitals (beats all 5 binary alternatives, all CIs `> 0`); seen + EB reaches `0.697–0.701` | `19.7×` faster than AutoGluon (`81.0 s`), `117×–323×` faster than Kumo/TabPFN/TabFM; `0.53 GB` VRAM | Across `25%–35%` thresholds (`P>0 = 1.000` vs. POAF score); **7.4×** decile spread; **24/25** subgroup wins |

---

## 1. The Clinical & Validation Problem

Postoperative atrial fibrillation (POAF) complicates nearly **1 in 4** isolated CABG surgeries, driving stroke, ICU stay, readmission, and excess mortality. High-leverage prophylaxis (such as posterior left pericardiotomy) must be chosen **before skin incision**.

Yet PROBAST appraisals show published POAF models drop to a **`0.61` external median** (higher published AUCs `> 0.72` rely on post-operative/ICU fields, mixed valve-plus-CABG cohorts, or raw ECG/echo streams). Our national audit revealed two real-world barriers:
1. **Calendar-2017 splits leak 99.5% of hospitals**, hiding a **`−1.99` AUC-point drop** on truly unseen centers (`0.6778 → 0.6579`).
2. **In-context tabular foundation models impose heavy serving latency**: scoring 21,241 patients on an A100-40GB GPU requires **8.0–22.1 minutes** (`22.7–62.5 ms/patient`) and **12–41 GB** reserved VRAM, while 30-minute AutoGluon `best_quality` writes **10.1 GB** of models and takes **81.0 s** batch / **28.5 s** single-patient bedside.

---

## 2. Head-to-Head Results Across Both Hospital Regimes (`N = 21,241`)

All five public comparators were trained strictly on binary `{0, 1}` targets on identical 50,000-patient five-fold hospital-grouped cohorts (strictly 207 pre-incision variables):

| Model (50k Parity Unless Noted) | Unseen AUC (74 hosp) | NEVERHUMAN Margin (95% CI) | Seen Native 208 | Seen +EB346k | Train Time | 21k Batch Infer |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| **NEVERHUMAN (Proprietary)** | **0.6767** | **#1 @ 50k (79.2% of 7× gap)** | **0.6916** | **0.6974** | **21.5 min** | **4.11 s** |
| AutoGluon 1.5 (`best_quality`) | 0.6745 | **+21.8 [+7.0, +36.5]\*** | 0.6891 | 0.6974 | 30.0 min | 80.99 s |
| CatBoost (5-fold binary) | 0.6742 | **+24.5 [+7.2, +42.2]\*** | 0.6886 | 0.6966 | 0.8 min | 0.15 s |
| PriorLabs TabPFN-3.5 (50k native) | 0.6738 | **+29.2 [+8.2, +53.5]\*** | --- | 0.6982 | 0.9 s | 515.3 s |
| NVIDIA KumoTabular-Large v2 | 0.6715 | **+52.0 [+21.6, +83.3]\*** | --- | 0.6977 | 3.3 min | 481.7 s |
| Google TabFM v1.0.0 | 0.6712 | **+55.3 [+28.6, +81.5]\*** | --- | 0.6986 | 6.6 min | 1,326.7 s |
| *Mariscalco POAF Bedside Proxy* | *0.6190* | ***+577.0 [+495.6, +656.9]\**** | *---* | *---* | *---* | *< 0.01 s* |
| *7× Full-Registry CatBoost (346k)* | *0.6773* | *−6.4 (NH beats on AP & ECE)* | ***0.7014*** | *---* | *6.2 min* | *0.30 s* |

---

## 3. False-Positive Reduction & U.S. (200k/yr) / Global (1M/yr) Scaled Clinical Impact

- **Empirical STS Morbidity & Mortality Excess (`N = 367,695`):** POAF carries nearly double the operative mortality (`2.78%` vs. `1.42%`, `+1.36 pp`), double the prolonged ventilation rate (`11.03%` vs. `5.52%`), and triple the renal failure rate (`3.48%` vs. `1.14%`).
- **Lowest False Positive Rate at 70% & 80% Capture:** On 74 unseen hospitals, NEVERHUMAN achieves the lowest FPR at 70% (`44.10%`, dropping to `41.22%` on seen `site208`) and 80% (`55.54%`) capture---sparing **490–1,798 U.S. patients/yr** (**2,448–8,992 globally**) from needless prophylaxis vs. the 5 binary AI models, and **48,755 U.S. patients/yr** (**243,774 globally**) vs. Mariscalco POAF.
- **Modeled Events Prevented & Lives Saved (Top-25% PALACS Targeting, 45% RRR):** Captures **19,462 true U.S. POAF patients/yr** (**97,312 globally**), modeling **8,758 U.S. POAF episodes prevented/yr** (**43,790 globally**), **$87.6M–$131.4M/yr** in U.S. hospital cost avoidance, and (under conservative 25%–50% causal attribution of the `+1.36 pp` STS mortality excess; 100% upper bound) **30–60 (upper 119) U.S. operative deaths averted/yr** (**149–298, upper 596 globally**).
- **Governance:** Mortality projections reflect transparent epidemiological modeling (PALACS was powered for POAF prevention); prospective confirmation of frozen interfaces on STS ACSD v4.20+ is required prior to clinical care changes.

**Next Reads:** [NEJM AI Original Article (PDF)](../nejm_ai/main.pdf) · [npj Digital Medicine Article (PDF)](../npj_digital_medicine/main.pdf) · [CTO Technical Brief (PDF)](neverhuman-poaf-cto-brief.pdf) · [Verified Receipt (MD)](../../reports/clean_publication_receipt.md)
