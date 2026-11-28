# POATFIB — Postoperative Atrial Fibrillation Prediction Archive & Active Study Suite

> **Superseding Publication Notice (Important Reconciliation):**  
> The archived early drafts below (`NEJM/nejm_poaf.tex` and `IEEE/ieee_poaf.tex`, reporting `0.6920` AUC for SEAN on a 2016–2017 / held-out 2017 50k/10k split) evaluated a **hospital-overlapping temporal split** before our full national 1,091-hospital audit. In our superseding **[EvoTarget-FT & JAIN Ultra GP v2 study suite](EvoTarget/README.md)**, we prove that hospital overlap alone inflates apparent pre-incision AUC by `+28–64 bps` (and up to `+2.0` AUC points with calendar/site effects): on **74 strictly unseen hospitals (0% overlap)** our Dual-Paradigm Fusion establishes the leakage-free cold-start benchmark at **`0.6767` AUC** (`+577 bps` over Mariscalco POAF; `+6.7` AUC points above the published external median of `0.61`), while in our matched **100% Seen-Hospital In-Network study (all 1,091 centers in training)** we achieve **`0.6916`** (50k native `site208`), **`0.6974`** (50k Fusion `+EB346k`), and **`0.7014`** (346k CatBoost `site208`)—surpassing and fully reconciling the earlier `0.6920` overlapping-split estimate.

### Active Submission-Ready Manuscripts & Briefs (`EvoTarget/`)

- ***NEJM AI* Original Article (PDF):** [POATFIB/EvoTarget/docs/nejm_ai/main.pdf](EvoTarget/docs/nejm_ai/main.pdf) · [LaTeX](EvoTarget/docs/nejm_ai/main.tex)
- ***npj Digital Medicine* Research Article (PDF):** [POATFIB/EvoTarget/docs/npj_digital_medicine/main.pdf](EvoTarget/docs/npj_digital_medicine/main.pdf) · [LaTeX](EvoTarget/docs/npj_digital_medicine/main.tex)
- **Technical Paper v4 (PDF):** [POATFIB/EvoTarget/docs/paper/main.pdf](EvoTarget/docs/paper/main.pdf) · **[CEO Brief (PDF)](EvoTarget/docs/briefs/evotarget-poaf-ceo-brief.pdf)** · **[CTO Brief (PDF)](EvoTarget/docs/briefs/evotarget-poaf-cto-brief.pdf)**
- **Complete Study Index & Receipts:** [EvoTarget/README.md](EvoTarget/README.md) · [Seen-Hospital Master Receipt](EvoTarget/reports/seen_hospital_study/seen_hospital_master_receipt.md)

---

## Archived Earlier Drafts (SEAN 2016–2017 / 2017 Hospital-Overlapping Split)

Two earlier coordinated manuscripts on **Self-Evolving Agent Networks (SEAN)** — evaluated on an earlier 50,000/10,000 2016–2017 train / 2017 test STS split (`147` preoperative predictors, hospital-overlapping temporal holdout) — are preserved here for historical provenance:

- `NEJM/nejm_poaf.tex` — archived clinical draft (`0.6920` SEAN vs. `0.6905` AutoGluon / `0.6874` CatBoost on overlapping 2017 split).
- `IEEE/ieee_poaf.tex` — archived technical draft (IEEEtran).

Please use the active **NEJM AI** and **npj Digital Medicine** manuscripts in [EvoTarget/](EvoTarget/README.md) for all current clinical, methodological, unseen-hospital, and seen-hospital results.
