# POATFIB — Self-Evolving Agent Networks (SEAN) for Postoperative Atrial Fibrillation

The [EvoTarget-FT study bundle](EvoTarget/README.md) contains the current STS v2.81 pre-incision POAF manuscript, bibliography, figures, and aggregate results.

Two coordinated manuscripts on **Self-Evolving Agent Networks (SEAN)** — a governed system of
language-model agents that autonomously *discover* clinical features and *invent* algorithm
components for predicting STS-defined treated postoperative atrial fibrillation/flutter
(`TARGET_COtAFib`), evaluated honestly under a single-use holdout firewall.

- `NEJM/nejm_poaf.tex` — clinical framing. First author: **Zain Khalpey, MD**.
- `IEEE/ieee_poaf.tex` — technical framing (IEEEtran). First author: **Jepson Taylor**.

## Headline (honest) result

On a leakage-controlled **50,000/10,000** STS-derived split (2016–17 train / held-out 2017 test;
147 preoperative predictors; gates passed: 0 feature-row hash overlap, adversarial-shift AUROC
0.509, target drift 0.0014), the field sits in a **moderate, overlapping band** and SEAN reaches
the top of it:

| Model | Held-out AUROC |
| --- | ---: |
| **SEAN** (best fair holdout) | **0.69196** |
| AutoGluon (default) | 0.6905 |
| CatBoost (default) | 0.6874 |
| TabPFN (default, baseline only) | 0.6869 |
| LightGBM (default) | 0.6863 |
| Random Forest (default) | 0.6747 |
| XGBoost (default) | 0.6687 |

SEAN **leads**, but its margin over the best default is within the AUROC sampling interval
(~0.011) — a moderate field, **not a decisive separation**. The contribution is an auditable,
trustworthy discovery instrument and an honest measurement of the ceiling. SEAN also discovers
theory-backed features (the top of 2,182 is `metabolicriskcalcul = Age×BSA+INR`) and novel
algorithm components that generalize where signal exists (HIGGS +0.0023; a frozen benchmark
+0.037). The "young-and-sick" age signal is a cohort-selection confound (POAF 50.8% in <60 vs
24.1% in >75), not rediscovered physiology.

## Grounding receipts

- `/tmp/apex/preds/` — per-patient predictions (`preds_*.{npy,txt}`, `y_test.npy`), `ranking.json`.
- `…/sts_mimo_afib_50k10kprime_manifest.json` — split/gate manifest.
- `data/clinical_analyses.json` — real DeLong / calibration (raw→Platt) / subgroup, from
  `compute_clinical_analyses.py` (cross-checked to `ranking.json`).
- `data/engineered_features.json` — discovered-feature formulae + citations.

## Build

```bash
cd POATFIB
python3 compute_clinical_analyses.py     # -> */data/clinical_analyses.json
python3 make_figures.py                  # vector PDF + PNG into NEJM/IEEE figures
cd NEJM  && latexmk -pdf nejm_poaf.tex
cd ../IEEE && latexmk -pdf ieee_poaf.tex
```

## Before submission (`TO CONFIRM` in both papers)

STS database version + `COtAFib` dictionary; IRB/DUA; the 2,182-feature ranking receipt + hash;
funding, conflicts, data/code availability, CRediT, and AI-writing disclosure.
