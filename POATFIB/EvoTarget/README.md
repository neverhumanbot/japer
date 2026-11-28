# EvoTarget-FT & JAIN Ultra GP v2: Pre-Incision Postoperative Atrial Fibrillation Prediction (STS ACSD v2.81)

Hospital-disjoint and in-network pre-incision prediction of treatment-requiring postoperative atrial fibrillation (POAF) across **368,132 isolated-CABG procedures at 1,091 U.S. surgical centers** — conducted by **NEVERHUMAN Research** and **HonorHealth Heart Care & HonorHealth Research Institute**.

[![NEJM AI PDF](https://img.shields.io/badge/NEJM_AI-Original_Article_PDF-9E1B32.svg)](docs/nejm_ai/main.pdf)
[![npj Digital Medicine PDF](https://img.shields.io/badge/npj_Digital_Medicine-Article_PDF-007A78.svg)](docs/npj_digital_medicine/main.pdf)
[![CEO Brief PDF](https://img.shields.io/badge/CEO_Brief-PDF-D9480F.svg)](docs/briefs/evotarget-poaf-ceo-brief.pdf)
[![CTO Brief PDF](https://img.shields.io/badge/CTO_Brief-PDF-1E3A5F.svg)](docs/briefs/evotarget-poaf-cto-brief.pdf)
[![Technical Paper v4 PDF](https://img.shields.io/badge/Technical_Paper_v4-PDF-006D77.svg)](docs/paper/main.pdf)
[![Distributed Compute](https://img.shields.io/badge/Distributed_Compute-Playbook-256ABF.svg)](docs/distributed_compute/README.md)

| Clinical Journal Editions (Submission-Ready) | Executive & Technical Briefs | Full Technical Paper & Audits |
| :---: | :---: | :---: |
| **[NEJM AI Original Article (PDF)](docs/nejm_ai/main.pdf)** ([TeX](docs/nejm_ai/main.tex) · [Figs](docs/nejm_ai/figures/))<br>**[npj Digital Medicine Article (PDF)](docs/npj_digital_medicine/main.pdf)** ([TeX](docs/npj_digital_medicine/main.tex) · [Figs](docs/npj_digital_medicine/figures/)) | **[CEO Brief (PDF)](docs/briefs/evotarget-poaf-ceo-brief.pdf)** ([MD](docs/briefs/ceo-brief.md))<br>**[CTO Brief (PDF)](docs/briefs/evotarget-poaf-cto-brief.pdf)** ([MD](docs/briefs/cto-brief.md)) | **[Paper v4 Manuscript (PDF)](docs/paper/main.pdf)** ([TeX](docs/paper/main.tex))<br>*[Seen-Hospital Receipt](reports/seen_hospital_study/seen_hospital_master_receipt.md) · [Clinical & DCA Receipt](reports/clinical_journal_analyses.md)* |

[Unseen-Hospital Results](#headline-results-74-unseen-us-hospitals) · [Seen-Hospital Study](#seen-hospital-in-network-study-all-1091-hospitals-in-train) · [Cohort & Data Contract](#cohort--zero-leakage-data-contract) · [Reproduce & Verify](#reproduce-and-verify) · [Documentation](#documentation)

---

## Headline Results (74 Unseen U.S. Hospitals, `N = 21,241`)

All 50k models train on identical 1,017-hospital cohorts using strictly **207 raw pre-incision predictors** and are scored on **74 strictly unseen holdout hospitals** (`0.0%` hospital overlap, natural clinical prevalence `24.29%` via uniform fold-OOF Platt scaling + dev-only Bayes prior shift `−1.136479`).

| Model (207 Strictly Pre-Incision Predictors) | Train `N` | 21k ROC-AUC ↑ | Fusion / Flagship Margin [74-Hosp 95% CI] | Avg. Prec. | ECE (21k) |
| :--- | ---: | ---: | :--- | ---: | ---: |
| **Dual-Paradigm Fusion (`JAIN GP v2 + EvoTarget-FT`)** | **50,000** | **0.67668** | **#1 Overall (79.2% of 7× scaling gap; beats 346k CB on AP & ECE)** | **0.37767** | **0.00724** |
| **EvoTarget-FT (Gen-20 $z^*$ + Dual-View PLR Attn.)** | **50,000** | **0.67653** | **+18.9 bps** `[+2.6, +35.8]` vs. Binary FT (zero collapse to Gen 1,000) | 0.37754 | 0.00850 |
| **JAIN Ultra GP v2 (6 evolved symbolic/tree GP fams)** | **50,000** | **0.67565** | **+39.8 bps** `[+19.6, +60.1]` vs. Legacy JAIN (beats all 6 binary ELOs) | 0.37582 | 0.00805 |
| *EvoTarget-FT (Strictly Nested 207-Raw Gen-20)* | *50,000* | *0.67588* | *65.2% pure zero-leakage smoothing (`+0.8 bps` outer-fold gap)* | *0.37637* | *0.00742* |
| Tuned CatBoost (5-fold binary reference) | 50,000 | 0.67423 | Fusion: **+24.5 bps** `[+7.2, +42.2]` | 0.37438 | 0.00778 |
| PriorLabs TabPFN-3.5 (50k native, e8) | 50,000 | 0.67376 | Fusion: **+29.2 bps** `[+8.2, +53.5]` | 0.37284 | 0.00683 |
| LightGBM / XGBoost (5-fold binary) | 50,000 | 0.67363 | Fusion: **+30.5 / +31.0 bps** (all CIs > 0) | 0.37251 | 0.00603 |
| Legacy JAIN Ultra GPU (1-worker, 20k, RMSE) | 50,000 | 0.67167 | Fusion: **+50.1 bps** `[+27.1, +73.2]` | 0.36993 | 0.00905 |
| NVIDIA KumoTabular-Large v2 (5×10k bag, e40) | 50,000 | 0.67147 | Fusion: **+52.0 bps** `[+21.6, +83.3]` | 0.36925 | 0.00702 |
| Google TabFM v1.0.0 (5×10k bag, e80) | 50,000 | 0.67115 | Fusion: **+55.3 bps** `[+28.6, +81.5]` | 0.36967 | 0.00901 |
| *Mariscalco POAF Bedside Score (calibrated)* | *50,000* | *0.6190* | Fusion: **+577.0 bps** `[+495.6, +656.9]` (`+316–1029 bps` across 6 scores) | *0.3097* | *0.01898* |
| *3-Way Binary Paradigm Blend* | *50,000* | *0.67568* | Fusion: **+9.9 bps** `[+0.6, +19.6]` *(strictly > 0)* | *0.37563* | — |
| *7× Full-Registry CatBoost Reference (207 raw)* | *346,454* | *0.67732* | *−6.4 bps (Fusion beats on AP, ECE, 2017 unseen & tails)* | *0.37761* | *0.00760* |

- **Clinical Utility & Equity ([clinical_journal_analyses.md](reports/clinical_journal_analyses.md)):** Monotonic **7.4× decile risk stratification** (`6.07%` D1 → `44.82%` D10); avoids **20.4–33.8 unnecessary prophylactic interventions per 100 patients** across 25%–35% thresholds (`P>0 = 1.000` vs. Mariscalco POAF); outperforms CatBoost 50k in **24 of 25 prespecified demographic, comorbidity, acuity, and hospital-volume subgroups**; chronological `2014–16 → 2017` empirical-Bayes site intercept onboarding cuts center-level MAE by **14%** (`5.63 → 4.84 pp`).

---

## Seen-Hospital In-Network Study (All 1,091 Hospitals in Train, `N = 21,241`)

In a matched **100.0% Seen-Hospital** split guaranteeing that **all 1,091 U.S. surgical centers** appear in both the 346,454-case development set and the nested 50,000-case balanced training benchmark (`0` unseen test centers; `1,086` seen validation hospitals; exact matched `N = 21,241` natural-prior holdout; full receipt in [seen_hospital_master_receipt.md](reports/seen_hospital_study/seen_hospital_master_receipt.md)):

| Model Family | Unseen Hosp. (0% ovlp, 207) | Seen Hosp. (100%, 207 raw) | Seen + 50k EB Site Prior | Seen + 346k EB Site Prior | Site-Prior Gain (bps) | In-Net ECE |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mariscalco POAF Score | 0.6190 | 0.6276 | — | — | — | 0.0166 |
| Google TabFM v1.0.0 (`5×10k`) | 0.6712 | 0.6775 | 0.6918 | 0.6986 | +210 | 0.0093 |
| NVIDIA KumoTabular-L v2 (`5×10k`) | 0.6715 | 0.6764 | 0.6911 | 0.6977 | +213 | 0.0080 |
| XGBoost (5-fold binary) | 0.6736 | 0.6774 | 0.6895 | 0.6958 | +184 | 0.0100 |
| LightGBM (5-fold binary) | 0.6736 | 0.6780 | 0.6897 | 0.6960 | +180 | 0.0084 |
| PriorLabs TabPFN-3.5 (50k native) | 0.6738 | 0.6795 | 0.6919 | 0.6982 | +187 | 0.0086 |
| CatBoost 50k (5-fold binary) | 0.6742 | 0.6777 | 0.6902 | 0.6966 | +188 | 0.0089 |
| Binary FT-Transformer (Gen-0) | 0.6746 | 0.6784 | 0.6890 | 0.6958 | +174 | 0.0067 |
| JAIN Ultra GP v2 (6-family) | 0.6756 | 0.6788 | 0.6904 | 0.6969 | +181 | 0.0098 |
| EvoTarget-FT (Gen-20 $z^*$) | 0.6765 | 0.6793 | 0.6906 | 0.6972 | +179 | 0.0117 |
| **Dual-Paradigm Fusion (`GP v2 + FT`)** | **0.6767** | **0.6794** | **0.6909** | **0.6974** | **+180** | **0.0110** |
| CatBoost Full Registry (`346k`, 7×) | 0.6773 | 0.6831 | — | **0.7014** | +183 | 0.0083 |

- **Native 50k `site208` champion:** End-to-end `Dual-Paradigm Fusion (site208)` reaches **`0.6916`** (`0.6844–0.6987`), beating `CatBoost site208` (`0.6887`) by **`+29.0 bps [+16.6, +41.5], P>0 = 1.000`**.
- **Implicit hospital fingerprinting:** Even with zero `HospID`, hospital overlap inflates patient-only AUCs by `+28–64 bps` (twice as large for tabular foundation models `+49–64 bps` as for Dual-Paradigm Fusion `+28 bps`). High-capacity 40k–346k models absorb site-coding signatures (`r = +0.122–0.143` with true site POAF deviation), whereas 10k-context bagged TFMs show ~30% lower entanglement (`r = +0.088–0.092`), avoiding double-counting under additive 346k EB site updating.

> **Governance note:** All intervals condition on retrospective model selection across 140+ evaluated arms (`B=5,000`, `seed=42`, [paper_v4_master_receipt.json](reports/paper_v4_master_receipt.json) & [seen_hospital_master_receipt.json](reports/seen_hospital_study/seen_hospital_master_receipt.json)). See the [NEJM AI Manuscript](docs/nejm_ai/main.pdf), [npj Digital Medicine Manuscript](docs/npj_digital_medicine/main.pdf), and [Red-Team Statistical Audit](docs/paper/RED_TEAM_REVIEW.md) for full scope boundaries and the path to frozen prospective validation on STS ACSD v4.20+.

---

## Cohort & Zero-Leakage Data Contract

This study uses the full 368,132-row, 638-column STS ACSD v2.81 isolated-CABG export from 2014–2017 (verified SHA-256: `89cab3de7977b463009baa71455683075b5dfbca0151df7ba07c06b882b5ed38`). No patient-level STS records or raw hospital identifiers are included in this public repository; only de-identified aggregate receipts, figures, manuscripts, and code are archived.

The primary outcome resolves the v2.81 `Complics` skip logic (`Complics=1, COtAFib=1` positive; `Complics=2` or `Complics=1, COtAFib=2` negative; [STS ACSD v2.81 specifications, page 838](https://www.sts.org/sites/default/files/documents/ACSD_DataSpecificationsV2_81.pdf#page=838)). Exhaustive auditing purged seven early-column postoperative duration leaks (`LOS`, `PLOS`, `ORDuration`, `SIDuration`, `SurgtoMort`, `SurgtoRead`, `SurgtoSternInf`), leaving strictly 207 pre-incision predictors.

| Year | Source rows | Labeled | Positive | Unknown |
| --- | ---: | ---: | ---: | ---: |
| 2014 | 58,790 | 58,716 | 13,911 | 74 |
| 2015 | 123,248 | 123,106 | 29,435 | 142 |
| 2016 | 124,479 | 124,321 | 30,709 | 158 |
| 2017 | 61,615 | 61,552 | 15,282 | 63 |
| **Total** | **368,132** | **367,695** | **89,337** | **437** |

---

## Reproduce and Verify

Run from `POATFIB/EvoTarget/` with Python 3.12 and `pdflatex`/`bibtex`:

```sh
# 1. Deterministically rebuild BOTH submission-ready clinical journal manuscripts
make -C docs/nejm_ai -B
make -C docs/npj_digital_medicine -B

# 2. Run automated journal compliance, word-count, overfull, and privacy/DUA audit
python3 scripts/36_audit_journal_manuscripts.py

# 3. Rebuild technical v4 manuscript, executive briefs, and numeric audit
make -C docs/paper -B
bash docs/briefs/scripts/build_briefs.sh --check
python3 scripts/20_audit_paper_numbers.py
```

---

## Documentation

| Area | Quick Links |
| :--- | :--- |
| **Submission-Ready Journal Editions** | **[NEJM AI Original Article (PDF)](docs/nejm_ai/main.pdf)** ([TeX](docs/nejm_ai/main.tex) · [Figures](docs/nejm_ai/figures/)) · **[npj Digital Medicine Article (PDF)](docs/npj_digital_medicine/main.pdf)** ([TeX](docs/npj_digital_medicine/main.tex) · [Figures](docs/npj_digital_medicine/figures/)) |
| **Aggregate Clinical & Seen-Hospital Receipts** | **[Clinical/DCA/Equity Receipt (MD)](reports/clinical_journal_analyses.md)** ([JSON](reports/clinical_journal_analyses.json)) · **[Seen-Hospital Master Receipt (MD)](reports/seen_hospital_study/seen_hospital_master_receipt.md)** ([JSON](reports/seen_hospital_study/seen_hospital_master_receipt.json)) · **[v4 Master Receipt (MD)](reports/paper_v4_master_receipt.md)** ([JSON](reports/paper_v4_master_receipt.json)) |
| **Executive & Technical Briefs** | **[CEO Brief (PDF)](docs/briefs/evotarget-poaf-ceo-brief.pdf)** ([MD](docs/briefs/ceo-brief.md)) · **[CTO Brief (PDF)](docs/briefs/evotarget-poaf-cto-brief.pdf)** ([MD](docs/briefs/cto-brief.md)) · [Build Script](docs/briefs/scripts/build_briefs.sh) |
| **Technical Manuscript & Audits** | **[Paper v4 Manuscript (PDF)](docs/paper/main.pdf)** · [LaTeX Source](docs/paper/main.tex) · [POAF Literature Review](docs/paper/POAF_LITERATURE_REVIEW.md) · [Red-Team Review](docs/paper/RED_TEAM_REVIEW.md) |
| **Distributed Multi-Node Compute** | **[Playbook Index](docs/distributed_compute/README.md)** · [SSH & Bootstrap](docs/distributed_compute/01_ssh_fabric_and_bootstrap.md) · [CPU Pull Queue](docs/distributed_compute/02_cpu_ssh_pull_queue_cluster.md) · [GPU Fan-Out / DDP / DiLoCo](docs/distributed_compute/03_gpu_fanout_ddp_and_diloco.md) · [Co-Tenancy & Ops](docs/distributed_compute/04_cotenancy_guardrails_and_ops.md) · [`xbabe5`–`xbabe9` Audit](docs/distributed_compute/05_xbabe5_9_spindown_audit_and_harvest.md) |
| **Research Notes & Supplements** | **[Full Docs Index](docs/README.md)** · [Dataset Guide](dataset/DATASET_README.md) · [Feature-Matched Split](FEATURE_MATCHED_EVALUATION.md) · [DOPE-ECHO Report](dope_echo/DOPE_ECHO_FINAL_REPORT.md) |
