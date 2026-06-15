# POATFIB — Engineered Intelligence for Post-Operative Atrial Fibrillation

Two manuscripts on an **engineered-intelligence** (EI) system — algorithm decomposition into a stage
grammar + an agent scripting language (ZYAL) + an agent fleet + genetic evolution + adversarial judges +
a holdout firewall — that searches the space of *complete* tabular ML pipelines for STS post-operative
atrial fibrillation (POAF), with the cohort used as a deliberately hard test bed.

- **NEJM/** — clinical framing (NEJM AI style). First author: **Zain Khalpey, MD**. (`nejm_poaf.tex`)
- **IEEE/** — technical framing (IEEEtran). First author: **Jepson Taylor**. (`ieee_poaf.tex`)

Each folder is self-contained: `*.tex`, `references.bib`, `figures/*.{png,eps}`, `data/*.{csv,json}`,
`tables/table_temporal.tex`, plus the verified `FACTS.md` / `paper_stats.json` used to ground every number.

## Headline (honest) result

On a fixed 10,000-train / 3,000-holdout STS-derived split (147 registry predictors, 9 outcomes; POAF
prevalence ~33%), the engineered #1 pipeline **E3** (LDA-112 → BrownianBridgeRankExpander →
MonotoneRunAtlasExpander → TopologicalSubspace, license-clean) reaches **holdout AUROC 0.6893
(95% CI 0.670–0.709)**. A raw LDA baseline reaches 0.6852 and the best human model 0.6806; **the 95%
confidence intervals overlap heavily** (Hanley–McNeil SE ≈ 0.011). DeLong tests: E3 vs baseline
Δ=0.0042 (p=0.048, borderline); E3 vs best human Δ=0.0087 (**p=0.099, not significant**).

**Interpretation:** engineered intelligence *reaches* the ~0.689 ceiling that bounds registry-only POAF
prediction; it does **not** significantly surpass strong human baselines. Two further findings stand on
their own: (1) **cross-validation does not predict held-out performance** in this regime
(Spearman(CV, holdout) = −0.06 over 500 generations), which is why a held-out firewall is essential; and
(2) the apparent "young-and-sick" inversion is a **cohort selection confound**, not rediscovered
physiology. The contribution is an **auditable search** and an **honest characterization of the ceiling**,
not a superior bedside predictor.

## Fairness protocol

Human models (all human baselines, recreated published methods, TabPFN v2.5) receive **only the 146
default registry features** — no engineered features, no LDA downsample, and never any `TARGET_*` column.
Engineered-intelligence models receive the engineered features + the LDA-112 downsample; MIMO models use
the 8 auxiliary outcomes as training labels only, never at inference. The 50k/10k temporal external
validation re-creates the engineered features and applies the identical downsample. **TabPFN v2.5 is
reported only as a human baseline**, never inside the EI results.

## Build

```bash
cd NEJM && latexmk -pdf nejm_poaf.tex      # -> nejm_poaf.pdf
cd ../IEEE && latexmk -pdf ieee_poaf.tex   # -> ieee_poaf.pdf
```

`TO CONFIRM` placeholders (IRB/DUA, funding, disclosures, AI-assisted-writing disclosure, CRediT
contributions) must be completed before submission.
