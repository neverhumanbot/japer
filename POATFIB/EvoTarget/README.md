# EvoTarget-FT: Pre-Incision Postoperative Atrial Fibrillation Prediction (STS ACSD v2.81)

Hospital-disjoint pre-incision prediction of treatment-requiring postoperative atrial fibrillation (POAF) across **368,132 isolated-CABG procedures at 1,091 U.S. surgical centers** — conducted by **NEVERHUMAN Research** and **HonorHealth Heart Care & HonorHealth Research Institute**.

[![CEO Brief PDF](https://img.shields.io/badge/CEO_Brief-2--Page_PDF-D9480F.svg)](docs/briefs/evotarget-poaf-ceo-brief.pdf)
[![CTO Brief PDF](https://img.shields.io/badge/CTO_Brief-2--Page_PDF-1E3A5F.svg)](docs/briefs/evotarget-poaf-cto-brief.pdf)
[![Manuscript PDF](https://img.shields.io/badge/Manuscript-17--Page_PDF-006D77.svg)](docs/paper/main.pdf)
[![Distributed Compute](https://img.shields.io/badge/Distributed_Compute-Playbook-256ABF.svg)](docs/distributed_compute/README.md)

| Executives & Clinical Leaders | CTOs & ML Architects | Full Technical Paper & Compute |
| :---: | :---: | :---: |
| **[CEO Brief (PDF)](docs/briefs/evotarget-poaf-ceo-brief.pdf)** ([MD](docs/briefs/ceo-brief.md))<br>*Clinical opportunity, external validation & governance* | **[CTO Brief (PDF)](docs/briefs/evotarget-poaf-cto-brief.pdf)** ([MD](docs/briefs/cto-brief.md))<br>*EvoTarget-FT architecture, paired comparisons & reproducibility* | **[17-Page Manuscript (PDF)](docs/paper/main.pdf)** ([TeX](docs/paper/main.tex))<br>*[Distributed Compute Playbook](docs/distributed_compute/README.md) · [Receipts](reports/paper_v3_master_receipt.json)* |

---

## Headline Unseen-Hospital Results (`74` Strictly Unseen U.S. Hospitals, `N = 21,241`)

All 50k models consume identical **207 strictly raw pre-incision predictors** (`0.0%` hospital overlap, natural `24.29%` prevalence via development-only Bayes prior shift `-1.136479`; 5,000-draw whole-hospital cluster bootstrap CIs condition on retrospective selection across 103 evaluated arms):

| Model (207 Strictly Pre-Incision Predictors) | Train `N` | 21k ROC-AUC ↑ | EvoTarget-FT Advantage [74-Hosp 95% CI] | Avg. Prec. | ECE (21k) |
| :--- | ---: | ---: | :--- | ---: | ---: |
| **EvoTarget-FT (Gen-20 $z^*$ + Dual-View PLR Attn.)** | **50,000** | **0.67629** | **#1 Model (66.7% of 7× scaling gap; 1.85× lift)** | **0.37691** | 0.00955 |
| Tuned CatBoost (5-fold binary reference) | 50,000 | 0.67423 | **+20.6 bps** `[+3.1, +39.1]` | 0.37438 | 0.00778 |
| PriorLabs TabPFN-3.5 (50k native, e8) | 50,000 | 0.67376 | **+25.3 bps** `[+2.2, +50.1]` | 0.37284 | 0.00683 |
| LightGBM (5-fold binary) | 50,000 | 0.67363 | **+26.6 bps** `[+5.3, +46.1]` | 0.37251 | **0.00603** |
| XGBoost (5-fold binary) | 50,000 | 0.67357 | **+27.2 bps** `[+5.6, +47.0]` | 0.37219 | 0.00685 |
| NVIDIA KumoTabular-Large v2 (5×10k bag, e40) | 50,000 | 0.67147 | **+48.2 bps** `[+15.8, +80.4]` | 0.36925 | 0.00702 |
| Google TabFM v1.0.0 (5×10k bag, e80) | 50,000 | 0.67115 | **+51.4 bps** `[+22.8, +79.4]` | 0.36967 | 0.00901 |
| *Same-Architecture Binary FT-Transformer (ablation)* | *50,000* | *0.67468* | *+16.0 bps `[-2.1, +35.3]` (95.9% pos. draws)* | *0.37551* | *0.00643* |
| *7× Full-Registry CatBoost Reference (207 raw)* | *346,454* | *0.67732* | *−10.3 bps (7× national data reference)* | *0.37761* | *0.00760* |

## Repository Contents

- **[CEO Brief (2-Page PDF)](docs/briefs/evotarget-poaf-ceo-brief.pdf)** ([Markdown](docs/briefs/ceo-brief.md)) and **[CTO Brief (2-Page PDF)](docs/briefs/evotarget-poaf-cto-brief.pdf)** ([Markdown](docs/briefs/cto-brief.md))
- **[Compiled 17-Page Manuscript (PDF)](docs/paper/main.pdf)** and [LaTeX source](docs/paper/main.tex)
- **[Distributed Multi-Node Compute Playbook](docs/distributed_compute/README.md)** and [reusable templates](docs/distributed_compute/templates/)
- **[Research and Supplement Index](docs/README.md)**, including the complete 103-arm leaderboard, feature screening, full-registry ensembles, and counterfactual embedding experiments
- [POAF and STS literature review](docs/paper/POAF_LITERATURE_REVIEW.md) and [statistical red-team review](docs/paper/RED_TEAM_REVIEW.md)
- [Label-blind matched evaluation](FEATURE_MATCHED_EVALUATION.md) and [dataset interface](dataset/DATASET_README.md)
- [Primary and subgroup confidence intervals](reports/paper_red_team_receipt.json), [benchmark and calibration receipts](reports/paper_v3_master_receipt.json), and [feature-augmentation intervals](reports/agentic_fe_cluster_receipt.json)

## Rebuild & Audit

Rebuild the manuscript and both 2-page executive briefs with pdfLaTeX and BibTeX, and verify every reported number against shipped aggregate receipts:

```sh
python3 scripts/20_audit_paper_numbers.py
make -C docs/paper -B
bash docs/briefs/scripts/build_briefs.sh --check
```

The bundle contains aggregate results and publication artifacts only. Record-level STS data, split assignments, saved predictions, and model weights remain in governed research storage. See the archival notes for original experiment details and the manuscript for the current interpretation.
