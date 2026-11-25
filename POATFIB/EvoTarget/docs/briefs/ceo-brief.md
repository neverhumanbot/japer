# Executive Brief: Better Pre-Incision Risk Prediction on Unseen Hospitals

**For Health-System CEOs, CMOs & Chief AI Officers**<br>
*NEVERHUMAN Research & HonorHealth Heart Care / HonorHealth Research Institute · STS ACSD v2.81 · November 2026*<br>
**[Download 2-Page Formatted PDF](evotarget-poaf-ceo-brief.pdf)** · **[Read CTO Technical Brief](cto-brief.md)** ([PDF](evotarget-poaf-cto-brief.pdf)) · **[Read Full 17-Page Manuscript](../paper/main.pdf)** · **[Distributed Compute Playbook](../distributed_compute/README.md)**

---

## Executive Snapshot

| **368,132 / 1,091** | **66.7%** | **1.85× (~44.9%)** | **−1.99 AUC pts** |
| :---: | :---: | :---: | :---: |
| **National STS CABG Cohort** | **7× Scaling Gap Recovered** | **Top-Decile Risk Enrichment** | **Validation Illusion Exposed** |
| 89,337 POAF events among 367,695 eligible cases (`24.30%`); evaluated on **74 strictly unseen** U.S. holdout hospitals (`0.0%` overlap) | **EvoTarget-FT** (`0.67629` AUC) closes two-thirds of the AUC gap between CatBoost trained on 50k and 346k cases | Nearly **1 in 2** patients ranked in the highest risk decile before incision later developed POAF (`1.85×` baseline) | Naive calendar holdouts (`99.5%` hospital overlap) inflate apparent AUC by nearly 2 full points |

---

## 1. The Risk of Evaluating on Familiar Hospitals

Postoperative atrial fibrillation (POAF) complicates **nearly 1 in 4 isolated coronary artery bypass grafting (CABG) procedures** (`24.30%` nationwide), and is associated with stroke, prolonged ICU stays, and readmissions. Published AUCs can differ substantially with the prediction window, available predictors, and hospital mix. [Published POAF evidence](../paper/POAF_LITERATURE_REVIEW.md)

Our forensic audit of **368,132 isolated-CABG surgeries across 1,091 U.S. centers** in the Society of Thoracic Surgeons Adult Cardiac Surgery Database (STS ACSD v2.81) explains why:

1. **Hidden postoperative leakage:** Fields such as `LOS`, `PLOS`, and `ORDuration` quietly contaminate "pre-operative" datasets alongside STS complication skip-logic collider errors (`Complics → COtAFib`). Purging all leaks reduced 638 raw fields to **207 strictly pre-incision predictors**.
2. **Naive calendar splits reuse 99.5% of training hospitals:** Holding out calendar year 2017 puts 998 of 1,003 hospitals right back into the test set. On the exact same 207-feature CatBoost model, apparent AUC drops from **0.6778** on naive 2017 to **0.6579** on genuinely unseen 2017 hospitals (**−1.99 AUC points**) — dwarfing incremental model tuning.
3. **Binary `{0,1}` labels trap every AI vendor inside a tight glass ceiling:** Across 74 never-seen U.S. hospitals (`N = 21,241`), all six world-leading tabular AI families trained on raw binary outcomes cluster within a narrow `0.67115–0.67423` band.

---

## 2. Better Supervision Closes Two-Thirds of the Data-Scaling Gap

Healthcare leaders are often told that breaking clinical AI plateaus requires years of multi-hospital data aggregation or complex, brittle "AI feature factories." Adding 82 clinical risk composites left discrimination essentially unchanged for the 50k tree and tabular foundation models. Continuous supervision offered a separate way to improve the benchmark.

**EvoTarget-FT** introduces a two-stage paradigm:
- **Stage 1 (Offline Target Discovery):** Multi-family evolutionary search strips intra-class label noise inside the training cohort to distill a smooth, continuous clinical severity target ($z^*$). Even among original non-POAF (`y=0`) training patients, elevated $z^*$ was associated with **3.33× higher postoperative sepsis, 2.51× higher renal failure, and 2.33× higher operative mortality** on complications never shown during target solving.
- **Stage 2 (Lean Pre-Incision Deployment):** A compact clinical self-attention transformer trains on $z^*$ using **strictly 207 routine pre-incision EHR variables** — averaging 20 lightweight passes at inference with zero heavy feature pipelines.

On 74 strictly unseen U.S. hospitals (`N = 21,241`), **EvoTarget-FT reaches `0.67629` ROC-AUC**, outperforming every individual binary GBDT and frontier Tabular Foundation Model (PriorLabs TabPFN-3.5, NVIDIA KumoTabular-Large v2, Google TabFM v1.0.0) with 74-hospital cluster intervals strictly above zero, and **recovering 66.7% of a 7× national dataset expansion** (`50,000 → 346,454` patients).

---

## 3. Three Strategic Plays Where EvoTarget-FT Creates Immediate Value

| Strategic Play | Executive Action | Measurable Impact |
| :--- | :--- | :--- |
| **Play 01: Precision Clinical Trial Enrichment** | Enrich preventive POAF prophylaxis trials (colchicine, amiodarone, posterior pericardiotomy) using pre-incision top-decile scores. | Concentrating POAF incidence to **~44.9% (`1.85×` lift)** could help target enrollment for a trial that tests clinical benefit. |
| **Play 02: Pre-Incision Care-Pathway Triage** | Risk-stratify CABG patients *before anesthesia induction* using existing STS/EHR intake fields. | Routes high-risk surgical patients proactively to pharmacy review, electrolyte optimization, and ICU telemetry readiness. |
| **Play 03: Vendor-Proof AI Procurement** | Adopt our zero-hospital-overlap audit contract as enterprise purchasing policy. | Separates real cross-hospital transportability from 0.90-AUC benchmark theater before committing multi-year IT budgets. |

---

## 4. Unseen-Hospital Scoreboard at a Glance (`N = 21,241`, 74 Unseen U.S. Centers)

| Model (207 Strictly Pre-Incision Predictors) | Unseen-Hosp AUC ↑ | EvoTarget Gain [74-Hosp 95% CI] | Avg. Prec. | ECE (21k) | Executive Verdict |
| :--- | ---: | :--- | ---: | ---: | :--- |
| **EvoTarget-FT (50k, Gen-20 $z^*$)** | **0.67629** | **Featured Model** | **0.37691** | 0.00955 | **66.7% of 7× scaling gap recovered; 1.85× lift** |
| Tuned CatBoost (50k Binary Ref) | 0.67423 | **+20.6 bps** `[+3.1, +39.1]` | 0.37438 | 0.00778 | Best binary GBDT; surpassed under clustering |
| PriorLabs TabPFN-3.5 (50k Native) | 0.67376 | **+25.3 bps** `[+2.2, +50.1]` | 0.37284 | 0.00683 | Strongest zero-shot TFM; beaten on unseen sites |
| LightGBM / XGBoost (50k Binary) | 0.67363 | **+26.6 bps** `[+5.3, +46.1]` | 0.37251 | **0.00603** | Tight binary plateau (XGB: `0.67357`, `+27.2 bps`) |
| NVIDIA KumoTabular-Large v2 (50k) | 0.67147 | **+48.2 bps** `[+15.8, +80.4]` | 0.36925 | 0.00702 | Bagged 5×10k TFM; lifted `+35.2 bps` on $z^*$ |
| Google TabFM v1.0.0 (50k) | 0.67115 | **+51.4 bps** `[+22.8, +79.4]` | 0.36967 | 0.00901 | Bagged 5×10k TFM; lifted `+26.7 bps` on $z^*$ |
| *Same-Arch Binary FT-Transformer* | *0.67468* | *+16.0 bps `[-2.1, +35.3]`* | *0.37551* | *0.00643* | *CI spans zero (`95.9%` pos.): arch + $z^*$ act jointly* |
| *7× Full-Registry CatBoost (346k)* | *0.67732* | *−10.3 bps (7× more data)* | *0.37761* | *0.00760* | *National 346k reference; EvoTarget closes 2/3* |

- **Prespecified clinical subgroups:** Gains persist on patients with **no recorded preoperative AFib** (`n = 21,103`: **0.67676** vs. `0.67465`, `+21.2 bps [+3.5, +40.0]`) and widen on **rare high-complexity clinical tails** (`n = 1,917`: **0.69350** vs. `0.68899`, `+45.1 bps [+2.1, +83.6]`).

---

## 5. Illustrative Timeline for a Silent Feasibility Pilot

1. **Map 207 Routine Pre-Op STS/EHR Fields (Week 1):** Connect existing pre-incision surgical intake data with zero post-op variables and benchmark local POAF incidence against the national `11.7%–34.8%` hospital range.
2. **Silent Prospective Shadow Scoring (Weeks 2–3):** Run frozen 20-pass EvoTarget-FT inference silently prior to incision and assess local calibration. If recalibration is needed, use a separate local development sample.
3. **Gate Clinical Action on Formal Protocol (Week 4+):** Deploy top-decile stratification (`~44.9%` POAF) inside prospective clinical trials or IRB-approved quality-improvement protocols before making bedside prophylaxis changes.

---

## 6. Governance, Risk & Honest Scientific Scope

Responsible clinical AI makes its boundaries explicit before deployment:

- **Retrospective selection exposure (103 arms):** Although holdout test labels never reached training workers, 103 model–target combinations were benchmarked retrospectively before featuring EvoTarget-FT; hospital-cluster intervals condition on selected runs.
- **Risk stratification ≠ proven prophylaxis benefit:** Identifying high-risk patients (`~44.9%` top decile) is a necessary prerequisite, but proving that model-guided medication or surgical prophylaxis improves outcomes requires a prospective randomized trial.
- **Treatment-requiring ascertainment:** STS `COtAFib` captures clinically treated AF/flutter rather than all silent telemetry episodes (continuous telemetry detects 6–10 pp more AFib).
- **Mandatory local intercept recalibration:** Because institutional POAF rates span `11.7%–34.8%` threefold across U.S. centers, new sites must pair frozen model weights with lightweight local intercept recalibration before setting clinical thresholds.
- **Next clinical milestone (STS v4.20+ confirmation):** Prospective confirmation of this frozen pipeline on modern post-2017 STS ACSD v4.20+ cohorts is the required validation gate prior to routine clinical deployment.

---

**Next Reads:** [2-Page CEO Brief (PDF)](evotarget-poaf-ceo-brief.pdf) · [CTO Technical & Architecture Brief (MD)](cto-brief.md) ([PDF](evotarget-poaf-cto-brief.pdf)) · [Full 17-Page Manuscript (PDF)](../paper/main.pdf) · [Red-Team Statistical Audit](../paper/RED_TEAM_REVIEW.md) · [Machine Receipts](../../reports/paper_red_team_receipt.json)
