# POATFIB — Postoperative Atrial Fibrillation Prediction Archive & Active Study Suite

> **Superseding Publication Notice (Important Reconciliation):**  
> The archived early drafts below (`NEJM/nejm_poaf.tex` and `IEEE/ieee_poaf.tex`, reporting `0.6920` AUC for SEAN on a 2016–2017 / held-out 2017 50k/10k split) evaluated a **hospital-overlapping temporal split** before our full national 1,091-hospital audit. In our superseding **[Pre-Incision POAF National Benchmark (`PreIncision_POAF/`)](PreIncision_POAF/README.md)**, we show that hospital overlap alone inflates apparent pre-incision AUC by `+28–64 bps` (and up to `+2.0` AUC points with calendar/site effects): on **74 strictly unseen hospitals (0% overlap)**, **NEVERHUMAN** establishes the leakage-free 50k cold-start benchmark at **`0.6767` AUC** (`+577 bps` over Mariscalco POAF; `+6.7` AUC points above the PROBAST published external median of `0.61`, beating AutoGluon 1.5, CatBoost, TabPFN-3.5, KumoTabular-Large v2, and TabFM v1.0.0 with all 95% cluster CIs `> 0`), while on our matched **100% Seen-Hospital In-Network study (all 1,091 centers in training)** we achieve **`0.6916`** (50k native `site208`), **`0.6974`** (50k NEVERHUMAN `+EB346k`), and **`0.7014`** (346k CatBoost `site208`)—surpassing and fully reconciling the earlier `0.6920` overlapping-split estimate.

### Active Submission-Ready Manuscripts & Briefs (`PreIncision_POAF/`)

- ***NEJM AI* Original Article (PDF):** [POATFIB/PreIncision_POAF/docs/nejm_ai/main.pdf](PreIncision_POAF/docs/nejm_ai/main.pdf) · [LaTeX](PreIncision_POAF/docs/nejm_ai/main.tex)
- ***npj Digital Medicine* Research Article (PDF):** [POATFIB/PreIncision_POAF/docs/npj_digital_medicine/main.pdf](PreIncision_POAF/docs/npj_digital_medicine/main.pdf) · [LaTeX](PreIncision_POAF/docs/npj_digital_medicine/main.tex)
- **Executive & Technical Briefs (2 pp each):** **[CEO Brief (PDF)](PreIncision_POAF/docs/briefs/neverhuman-poaf-ceo-brief.pdf)** · **[CTO Brief (PDF)](PreIncision_POAF/docs/briefs/neverhuman-poaf-cto-brief.pdf)**
- **Complete Study Index & Verified Receipts:** [PreIncision_POAF/README.md](PreIncision_POAF/README.md) · [Clean Master Receipt (MD)](PreIncision_POAF/reports/clean_publication_receipt.md)

---

## Archived Earlier Drafts (SEAN 2016–2017 / 2017 Hospital-Overlapping Split)

Two earlier coordinated manuscripts on **Self-Evolving Agent Networks (SEAN)** — evaluated on an earlier 50,000/10,000 2016–2017 train / 2017 test STS split (`147` preoperative predictors, hospital-overlapping temporal holdout) — are preserved here for historical provenance:

- `NEJM/nejm_poaf.tex` — archived clinical draft (`0.6920` SEAN vs. `0.6905` AutoGluon / `0.6874` CatBoost on overlapping 2017 split).
- `IEEE/ieee_poaf.tex` — archived technical draft (IEEEtran).

Please use the active **NEJM AI** and **npj Digital Medicine** manuscripts in [PreIncision_POAF/](PreIncision_POAF/README.md) for all current clinical, methodological, unseen-hospital, seen-hospital, speed, and population-impact results.
