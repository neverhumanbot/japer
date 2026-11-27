# Executive Brief: Outperform Frontier AI Models on Unseen Hospitals — With 7× Less Data

**For Health-System CEOs, CMOs & Chief AI Officers**  
*NEVERHUMAN Research & HonorHealth Heart Care / HonorHealth Research Institute · STS ACSD v2.81 · November 2026*  
**[Download 2-Page Formatted PDF](evotarget-poaf-ceo-brief.pdf)** · **[Read CTO Technical Brief](cto-brief.md)** ([PDF](evotarget-poaf-cto-brief.pdf)) · **[Read Full Manuscript](../paper/main.pdf)** · **[Distributed Compute Playbook](../distributed_compute/README.md)**

---

## Executive Snapshot

| **368,132 / 1,091** | **0.67668 AUC (79.2%)** | **1.85× (~44.9%)** | **−1.99 AUC pts** |
| :---: | :---: | :---: | :---: |
| **National STS CABG Cohort** | **7× Scaling Gap Recovered** | **Top-Decile Risk Enrichment** | **Validation Illusion Exposed** |
| 89,337 POAF cases (`24.30%`); evaluated on **74 strictly unseen** U.S. holdout hospitals (`0.0%` overlap) | **Dual-Paradigm Fusion** (`JAIN Ultra GP v2 + EvoTarget-FT`) recovers **79.2%** of a 50k → 346k expansion & beats 346k CatBoost in AP & ECE (`0.00724`) | Nearly **1 in 2** top-decile patients develop POAF before the first incision (`1.85×` baseline) | Naive calendar holdouts (`99.5%` hospital overlap) inflate apparent AUC by nearly 2 full points |

---

## 1. The Clinical AI Mirage Costing Health Systems Millions

Postoperative atrial fibrillation (POAF) complicates **nearly 1 in 4 isolated coronary artery bypass grafting (CABG) procedures** (`24.30%` nationwide), driving stroke, prolonged ICU length of stay, readmissions, elevated mortality, and heavy episodic cost. Yet dozens of commercial vendor pitches and published papers advertise **0.85–0.97 AUC** — scores that routinely collapse when deployed inside a new hospital.

Our forensic audit of **368,132 isolated-CABG surgeries across 1,091 U.S. centers** in the Society of Thoracic Surgeons Adult Cardiac Surgery Database (STS ACSD v2.81) explains why:

1. **Hidden postoperative leakage:** Fields such as `LOS`, `PLOS`, and `ORDuration` quietly contaminate "pre-operative" datasets alongside STS complication skip-logic collider errors (`Complics → COtAFib`). Purging all leaks reduced 638 raw fields to **207 strictly pre-incision predictors**.
2. **Naive calendar splits reuse 99.5% of training hospitals:** Holding out calendar year 2017 puts 998 of 1,003 hospitals right back into the test set. On the exact same 207-feature CatBoost model, apparent AUC drops from **0.6778** on naive 2017 to **0.6579** on genuinely unseen 2017 hospitals (**−1.99 AUC points**) — dwarfing incremental model tuning.
3. **Binary `{0,1}` labels trap every AI vendor inside a tight glass ceiling:** Across 74 never-seen U.S. hospitals (`N = 21,241`), all six world-leading tabular AI families trained on raw binary outcomes cluster within a narrow `0.67115–0.67423` band.

---

## 2. Two Breakthroughs: Evolve Targets AND Evolve Programs

Healthcare leaders are often told that breaking clinical AI plateaus requires years of multi-hospital data aggregation or complex "AI feature factories." Our audit proves otherwise: adding 82 clinical risk composites left 50k tree and foundation-model discrimination flat. Two orthogonal advances broke the ceiling on **strictly 207 routine pre-incision EHR variables**:

1. **1,000-Generation Target Evolution (`EvoTarget-FT`, `0.67653` AUC):** Strictly inside the 50,000-patient training cohort (zero test labels ever touched), multi-family evolutionary search distills a smooth continuous risk target ($z^*$) verified across **1,000 generations (256,000 fits)** with **zero late-stage collapse** (`0.67592` at Gen 1,000). At Generation 20, **EvoTarget-FT** reaches **`0.67653` ROC-AUC** (`+18.9 bps [+2.6, +35.8]` over identical binary FT; **74.6%** of the 7× scaling gap) and lifts **five distinct model families** with 74-hospital cluster intervals strictly above zero. A strictly nested 207-raw 2×2 control confirmed **65.2%** of this lift comes purely from zero-leakage target smoothing (`<1 bp` outer-fold effect).
2. **Re-Engineered Parallel Genetic Programming (`JAIN Ultra GP v2`, `0.67565` AUC):** Fixing six code-level bottlenecks in GPU-guided genetic programming lifted unseen-hospital AUC by **`+39.8 bps [+19.6, +60.1]`** over legacy JAIN (`0.67167 → 0.67565`, `0.00805 ECE`), surpassing all 6 binary tabular ELO baselines standalone with fresh 7-fold CV verification (`r = 0.950`).
3. **Dual-Paradigm Fusion (`GP v2 + EvoTarget-FT`, `0.67668` AUC / `0.37767` AP / `0.00724` ECE):** Combining symbolic/tree GP pipelines with continuous-target clinical self-attention recovers **79.2% of a 7× national data expansion** (`50,000 → 346,454` patients), clears every individual ELO baseline **and** the 3-way binary paradigm blend (`+9.9 bps [+0.6, +19.6]`), and beats 7× 346k CatBoost on Average Precision, ECE, 2017 unseen hospitals (`0.65794`), and rare clinical tails (`0.69347`; EvoTarget-FT `0.69438`).

---

## 3. Unseen-Hospital Scoreboard at a Glance (`N = 21,241`, 74 Unseen U.S. Centers)

| Model (207 Strictly Pre-Incision Predictors) | Unseen-Hosp AUC ↑ | Fusion / Flagship Gain [74-Hosp 95% CI] | Avg. Prec. | ECE (21k) | Executive Verdict |
| :--- | ---: | :--- | ---: | ---: | :--- |
| **Dual-Paradigm Fusion (GP v2 + EvoTarget-FT)** | **0.67668** | **#1 Overall (79.2% of 7× gap)** | **0.37767** | **0.00724** | **Beats 346k CB on AP, ECE, 2017 unseen & tails** |
| **EvoTarget-FT (50k, Gen-20 $z^*$)** | **0.67653** | **+18.9 bps** `[+2.6, +35.8]` vs. Bin. FT | 0.37754 | 0.00850 | **74.6% of 7× gap; 0.69438 rare tails; zero collapse** |
| **JAIN Ultra GP v2 (50k, 6 evolved GP fams)** | **0.67565** | **+39.8 bps** `[+19.6, +60.1]` vs. Legacy | 0.37582 | 0.00805 | **Standalone GP beats all 6 binary ELOs; 0.65809 2017** |
| Tuned CatBoost (50k Binary Ref) | 0.67423 | Fusion: **+24.5 bps** `[+7.2, +42.2]` | 0.37438 | 0.00778 | Best binary GBDT; surpassed under clustering |
| PriorLabs TabPFN-3.5 (50k Native) | 0.67376 | Fusion: **+29.2 bps** `[+8.2, +53.5]` | 0.37284 | 0.00683 | Strongest zero-shot TFM; beaten on unseen sites |
| LightGBM / XGBoost (50k Binary) | 0.67363 | Fusion: **+30.5 / +31.0 bps** (CI > 0) | 0.37251 | 0.00603 | Tight binary plateau (XGB: `0.67357`) |
| NVIDIA KumoTabular-Large v2 (50k) | 0.67147 | Fusion: **+52.0 bps** `[+21.6, +83.3]` | 0.36925 | 0.00702 | Bagged 5×10k TFM; lifted `+35.2 bps` on $z^*$ |
| Google TabFM v1.0.0 (50k) | 0.67115 | Fusion: **+55.3 bps** `[+28.6, +81.5]` | 0.36967 | 0.00901 | Bagged 5×10k TFM; lifted `+26.7 bps` on $z^*$ |
| *3-Way Binary Paradigm Blend* | *0.67568* | Fusion: **+9.9 bps** `[+0.6, +19.6]` | *0.37563* | — | *Fusion strictly clears multi-paradigm binary blend* |
| *7× Full-Registry CatBoost (346k)* | *0.67732* | *−6.4 bps (7× more data)* | *0.37761* | *0.00760* | *National 346k reference; Fusion closes 79.2%* |

---

## 4. 30-Day Low-Risk Health-System Shadow Pilot & Governance Boundaries

1. **Map 207 Routine Pre-Op STS/EHR Fields (Week 1):** Connect existing pre-incision surgical intake data with zero post-op variables and benchmark local POAF incidence against the national `11.7%–34.8%` hospital range.
2. **Silent Prospective Shadow Scoring (Weeks 2–3):** Run frozen Dual-Paradigm Fusion inference (20 compact transformer + 30 lightweight GP passes) silently prior to incision with lightweight local intercept recalibration.
3. **Gate Clinical Action on Formal Protocol (Week 4+):** Deploy top-decile stratification (`~44.9%` POAF) inside prospective clinical trials or IRB-approved quality-improvement protocols before making bedside prophylaxis changes.

**Honest scientific scope:** Although holdout test labels never reached training, evolution, or GP workers, 140+ model–target arms were benchmarked retrospectively; hospital-cluster CIs condition on selected runs. Strictly nested 207-raw EvoTarget-FT (`+12.3 bps [-3.3, +28.5]`) resolves outer-fold concern (`+0.8 bps` nesting gap) while spanning zero at 74 centers. Prospective confirmation of the frozen pipeline on modern STS ACSD v4.20+ cohorts and randomized prophylaxis testing remain required prior to routine bedside use.

---

**Next Reads:** [2-Page CEO Brief (PDF)](evotarget-poaf-ceo-brief.pdf) · [CTO Technical & Architecture Brief (MD)](cto-brief.md) ([PDF](evotarget-poaf-cto-brief.pdf)) · [Full Manuscript (PDF)](../paper/main.pdf) · [Paper v4 Master Receipt](../../reports/paper_v4_master_receipt.md)
