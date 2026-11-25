# 04. Why a Naive 2017 Temporal Holdout Is NOT a Statistically Valid Primary Test Set

- **Paper Section**: *Methods — Critique of Naive Temporal Splitting* & *Discussion — External Validation Design*
- **Core Claim**: Using 2017 as the sole primary holdout test set in STS ACSD v2.81 is **statistically invalid as a fair generalization benchmark** due to **five compounding quantitative pathologies**: (1) **99.5% hospital leakage**, (2) **3× inter-hospital POAF dispersion**, (3) **half-year seasonality confounding (H2-2014 vs H1-2017)**, (4) **6-percentage-point documentation/missingness drift**, and (5) **strong multivariate covariate shift (Adversarial AUC 0.62–0.77)**. Consequently, 2017 belongs as an embedded **temporal stress slice on unseen hospitals (`stress_2017_unseen_hosp`)**, not as the sole primary test benchmark.
- **Evidence & Source Artifacts**: `scripts/05_optimal_test_split.py`, `reports/split_audit_comparison.json`, `reports/split_audit_comparison_80_20.json`.
- **Status**: `Verified`

---

## 1. The Five Quantitative Pathologies of a Naive 2017 Holdout

### Pathology 1 — 99.5% Hospital Leakage (Fails External Site Generalization)
- Across 2017 (`n = 61,552` labeled patients), **`998` of `1,003` hospitals (99.5%)** already appear in the 2014–2016 training set!
- Only 5 hospitals are genuinely unseen in 2017. Any GBDT or neural model can implicitly fingerprint hospital-level lab ordering, missingness patterns, and regional practice styles from the 207 preoperative columns — **failing to test true transportability to new surgical centers**.

### Pathology 2 — Massive Inter-Hospital POAF Rate Dispersion (3× Across Centers)
- Even among isolated-CABG programs with substantial volume, center-level POAF rates vary **3-fold** across hospitals:
  - 5th percentile hospital: **`11.7%`**
  - 25th percentile: **`19.2%`**
  - Median hospital: **`24.1%`**
  - 75th percentile: **`28.3%`**
  - 95th percentile hospital: **`34.8%`**
  - Cross-hospital standard deviation: **`6.97 percentage points`**
- Testing on 99.5% identical hospitals lets models exploit center-specific base-rate priors rather than patient-level clinical physiology.

### Pathology 3 — Half-Year Seasonality Confounding (H2-2014 vs H1-2017)
- STS v2.81 began July 1, 2014 and ended June 30, 2017:
  - `2014` contains **H2 only** (Jul–Dec, lower winter respiratory peak → POAF **`23.69%`**).
  - `2017` contains **H1 only** (Jan–Jun, peak winter/early-spring respiratory season → POAF **`24.83%`**, **`+1.14 pp`** higher than 2014 and **`+0.64 pp`** higher than 2014–2016 pooled!).
- A naive 2017 split conflates true secular practice drift with truncated half-year seasonal bias, giving **`Year TV Distance = 1.0000`** (zero temporal overlap).

### Pathology 4 — Systematic Documentation & Missingness Drift (+5–6 pp)
- Echocardiographic and angiographic reporting density shifted sharply across the v2.81 lifecycle (`2014–2016` vs `2017`):
  - `LVSD` missingness shifted by **`6.05 pp`** (`42.4%` → `48.5%`)
  - `LVEDD` missingness shifted by **`6.04 pp`** (`41.7%` → `47.7%`)
  - `PctStenAM` missingness shifted by **`5.6 pp`**
  - `PctStenDiag3` missingness shifted by **`5.4 pp`**
- In our audited 5-way benchmark (`reports/split_audit_comparison_80_20.json`), naive 2017 exhibited **`Max Missingness Drift = 0.0605`** (nearly **5× worse** than our optimal hospital-disjoint split's `0.0126`) and **`Max |SMD| = 0.2793`**.

### Pathology 5 — Strong Multivariate Covariate Shift (Adversarial Validation AUC 0.62–0.77)
Training a 3-fold LightGBM domain classifier (`train` vs `2017`) strictly on the 207 pre-incision features reveals severe covariate separability:

| Pairwise Domain Contrast | Adversarial Validation ROC-AUC (207 Pre-Incision Features) | Interpretation |
|---|---:|---|
| `2017 (H1)` vs `2014 (H2)` | **`0.7661`** | Severe multi-feature & seasonal drift |
| `2017 (H1)` vs `2015` | **`0.6527`** | Moderate-to-high drift |
| `2017 (H1)` vs `2016` | `0.5582` | Mild adjacent-year drift |
| `2017 (H1)` vs `2014–2016 (all)` | **`0.6217`** (`0.5908` hospital-grouped; `0.6052` core `<20%` missing) | Clearly non-exchangeable holdout |
| 32-D Clinical Manifold Energy Distance | **`0.00682`** (`Multi-RBF MMD² = 0.000441`) | **3×–5× worse** distributional mismatch than multivariate hospital-matched splits |

---

## 2. Methodological Consequence for Clinical AI Benchmarking (Paper Recommendation)

A single naive 2017 split simultaneously **over-optimistically leaks 99.5% of hospitals** while **pessimistically confounding H1 seasonality and documentation missingness drift** — mixing two completely distinct clinical deployment questions into one biased metric!

**Our Principled Two-Tier Solution** (aligned with Collins et al., *BMJ* 2024 TRIPOD+AI & Steyerberg 2019 *Clinical Prediction Models*):
1. **Primary Benchmark (`test_full`, 5% Hospital-Disjoint Multivariate Optimal Split)**: **0.0% hospital overlap**, spanning all 4 years (full seasonality), all 9 regions, all 207 pre-incision features, missingness structure, rare clinical tails, and all 9 clinical targets — isolating true **unseen-hospital transportability** with minimal covariate mismatch (`Energy Distance`, `MMD²`, `SMD`, `Wasserstein W1`).
2. **Embedded Temporal + Unseen-Hospital Stress Slice (`stress_2017_unseen_hosp`)**: Evaluates 2017 patients **specifically at unseen hospitals** inside `test_full`, cleanly measuring temporal + spatial generalization simultaneously without hospital memorization!
3. *(Historical 2014–2016 → 2017 temporal score is still archived transparently in `reports/batch2_summary.json` and Codex's `FEATURE_MATCHED_EVALUATION.md` for direct comparison with prior literature.)*

## References
1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378.
2. Steyerberg EW, Harrell FE Jr. Prediction models need appropriate internal, internal–external, and external validation. *J Clin Epidemiol*. 2016;69:245–247.
3. Riley RD, Ensor J, Snell KIE, et al. External validation of clinical prediction models using big datasets from e-health records or IPD meta-analysis. *BMJ*. 2019;364:l4379.
