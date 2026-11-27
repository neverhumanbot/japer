# EvoTarget-FT: Pre-Incision Postoperative Atrial Fibrillation Prediction (STS ACSD v2.81)

Hospital-disjoint pre-incision prediction of treatment-requiring postoperative atrial fibrillation (POAF) across **368,132 isolated-CABG procedures at 1,091 U.S. surgical centers** — conducted by **NEVERHUMAN Research** and **HonorHealth Heart Care & HonorHealth Research Institute**.

[![CEO Brief PDF](https://img.shields.io/badge/CEO_Brief-PDF-D9480F.svg)](docs/briefs/evotarget-poaf-ceo-brief.pdf)
[![CTO Brief PDF](https://img.shields.io/badge/CTO_Brief-PDF-1E3A5F.svg)](docs/briefs/evotarget-poaf-cto-brief.pdf)
[![Manuscript PDF](https://img.shields.io/badge/Manuscript_v4-PDF-006D77.svg)](docs/paper/main.pdf)
[![Distributed Compute](https://img.shields.io/badge/Distributed_Compute-Playbook-256ABF.svg)](docs/distributed_compute/README.md)
[![Leakage & Contract Tests](https://img.shields.io/badge/pytest-48%2F48_passing-16A34A.svg)](scripts/tests/test_leakage_and_features.py)

| Executives & Clinical Leaders | CTOs & ML Architects | Full Technical Paper & Compute |
| :---: | :---: | :---: |
| **[CEO Brief (PDF)](docs/briefs/evotarget-poaf-ceo-brief.pdf)** ([MD](docs/briefs/ceo-brief.md))<br>*Clinical ROI, 79.2% scaling recovery & governance* | **[CTO Brief (PDF)](docs/briefs/evotarget-poaf-cto-brief.pdf)** ([MD](docs/briefs/cto-brief.md))<br>*1,000-gen $z^*$, JAIN Ultra GP v2, 2×2 nested proof & CLI* | **[Paper v4 Manuscript (PDF)](docs/paper/main.pdf)** ([TeX](docs/paper/main.tex))<br>*[Distributed Compute Playbook](docs/distributed_compute/README.md) · [v4 Master Receipt](reports/paper_v4_master_receipt.md)* |

[Headline Results](#headline-results-74-unseen-us-hospitals) · [Cohort & Data Contract](#cohort--zero-leakage-data-contract) · [Artifacts](#artifacts) · [Reproduce & Verify](#reproduce-and-verify) · [Documentation](#documentation)

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
| *3-Way Binary Paradigm Blend* | *50,000* | *0.67568* | Fusion: **+9.9 bps** `[+0.6, +19.6]` *(strictly > 0)* | *0.37563* | — |
| *7× Full-Registry CatBoost Reference (207 raw)* | *346,454* | *0.67732* | *−6.4 bps (Fusion beats on AP, ECE, 2017 unseen & tails)* | *0.37761* | *0.00760* |

> **Governance note:** All intervals condition on retrospective model selection across 140+ evaluated arms (`B=5,000`, `seed=42`, [reports/paper_v4_master_receipt.json](reports/paper_v4_master_receipt.json)). See the [CEO Brief](docs/briefs/evotarget-poaf-ceo-brief.pdf), [CTO Brief](docs/briefs/evotarget-poaf-cto-brief.pdf), and [Red-Team Statistical Audit](docs/paper/RED_TEAM_REVIEW.md) for full scope boundaries and the path to frozen prospective validation on STS ACSD v4.20+.

---

## Cohort & Zero-Leakage Data Contract

This workspace uses the full 368,132-row, 638-column STS ACSD v2.81 isolated CABG export from 2014–2017. The builder streams and hashes the original archive member. The verified SHA-256 is `89cab3de7977b463009baa71455683075b5dfbca0151df7ba07c06b882b5ed38`.

The primary outcome uses the v2.81 `Complics` skip logic. A positive is `Complics=1, COtAFib=1`. A negative is `Complics=2` or `Complics=1, COtAFib=2`. The STS outcome represents postoperative AF/flutter requiring treatment ([STS ACSD v2.81 data specifications, page 838](https://www.sts.org/sites/default/files/documents/ACSD_DataSpecificationsV2_81.pdf#page=838)).

| Year | Source rows | Labeled | Positive | Unknown |
| --- | ---: | ---: | ---: | ---: |
| 2014 | 58,790 | 58,716 | 13,911 | 74 |
| 2015 | 123,248 | 123,106 | 29,435 | 142 |
| 2016 | 124,479 | 124,321 | 30,709 | 158 |
| 2017 | 61,615 | 61,552 | 15,282 | 63 |
| **Total** | **368,132** | **367,695** | **89,337** | **437** |

---

## Artifacts

- [Raw pre-incision predictors](data/foundation_raw_preop.parquet) contain 207 predictors and `row_id`. `Region` keeps all nine categorical codes.
- [Unlabeled predictors](data/foundation_unlabeled_preop.parquet) contain the same schema for the 437 unknown-outcome rows. Their row IDs follow the labeled IDs.
- [Labels](data/foundation_targets.parquet) contain the binary `PRIMARY_TARGET_COtAFib` and eight existing auxiliary outcomes. An auxiliary value of `-1` represents an unknown outcome.
- [Splits](data/splits.parquet) preserve staged assignments across 5 hospital-grouped folds and the 2017 calendar slice. Use `row_id` for alignment and exclude it from model inputs.
- [Schema audit](data/schema_audit.json) assigns each raw column one disposition, documents the timing source for each raw predictor, and records hashes for the four Parquets above.
- [Augmented predictors](data/foundation_augmented.parquet), [feature registry](features/registry.json), and the `features/tested` and `features/kept` directories record the audited 92-feature clinical portfolio.
- [Screening report](reports/feature_screening_leaderboard.md), [score receipt](reports/baseline_and_screening_summary.json), [red-team hospital-cluster receipt](reports/paper_red_team_receipt.json), and [master benchmark receipt](reports/paper_v3_master_receipt.json) store complete paired evaluation receipts.
- [Feature-matched evaluation](FEATURE_MATCHED_EVALUATION.md) documents the label-blind matched hospital split and modular dataset interface ([dataset/DATASET_README.md](dataset/DATASET_README.md)).

---

## Reproduce and Verify

Run from this directory with Python 3.12 and dependencies in [requirements.txt](requirements.txt):

```sh
# 1. Verify all 12 manuscript .tex files, 95 citations & 15 cross-refs against JSON receipts
python3 scripts/20_audit_paper_numbers.py

# 2. Run all 48 leakage, skip-logic & feature-contract unit tests
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q

# 3. Deterministically rebuild the 17-page manuscript and both 2-page executive briefs
make -C docs/paper -B
bash docs/briefs/scripts/build_briefs.sh --check

# 4. Rebuild foundation & screening pipeline (uses read-only archive stream)
OPENBLAS_NUM_THREADS=1 python3 scripts/01_build_clean_foundation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/03_fast_catboost_screen.py
```

---

## Documentation

| Area | Quick Links |
| :--- | :--- |
| **Executive & Technical Briefs** | **[CEO Brief (PDF)](docs/briefs/evotarget-poaf-ceo-brief.pdf)** ([MD](docs/briefs/ceo-brief.md)) · **[CTO Brief (PDF)](docs/briefs/evotarget-poaf-cto-brief.pdf)** ([MD](docs/briefs/cto-brief.md)) · [Build Script](docs/briefs/scripts/build_briefs.sh) |
| **Publication Manuscript** | **[17-Page Manuscript (PDF)](docs/paper/main.pdf)** · [LaTeX Source](docs/paper/main.tex) · [POAF Literature Review](docs/paper/POAF_LITERATURE_REVIEW.md) · [Red-Team Review](docs/paper/RED_TEAM_REVIEW.md) |
| **Distributed Multi-Node Compute** | **[Playbook Index](docs/distributed_compute/README.md)** · [SSH & Bootstrap](docs/distributed_compute/01_ssh_fabric_and_bootstrap.md) · [CPU Pull Queue](docs/distributed_compute/02_cpu_ssh_pull_queue_cluster.md) · [GPU Fan-Out / DDP / DiLoCo](docs/distributed_compute/03_gpu_fanout_ddp_and_diloco.md) · [Co-Tenancy & Ops](docs/distributed_compute/04_cotenancy_guardrails_and_ops.md) · [`xbabe5`–`xbabe9` Audit](docs/distributed_compute/05_xbabe5_9_spindown_audit_and_harvest.md) |
| **Research Notes & Supplements** | **[Full Docs Index](docs/README.md)** · [Dataset Guide](dataset/DATASET_README.md) · [Feature-Matched Split](FEATURE_MATCHED_EVALUATION.md) · [DOPE-ECHO Report](dope_echo/DOPE_ECHO_FINAL_REPORT.md) |
