# 01. Cohort Source Provenance & STS ACSD v2.81 Schema Verification

- **Paper Section**: *Methods — Study Population & Data Source*
- **Core Claim**: The authoritative cohort comprises **368,132 isolated-CABG records** (638 raw columns, 1,091 hospitals, 9 U.S. census regions) exported under **100% STS Adult Cardiac Surgery Database (ACSD) Version 2.81** (`OnDemandVrsn == '2.81.1'`), spanning the exact 3-year v2.81 era (H2 2014 – H1 2017). Multiple STS integer codings are non-monotonic and must be explicitly decoded before clinical feature engineering.
- **Evidence & Source Artifacts**: `data/schema_audit.json`, `data/foundation_raw_preop.parquet`, `references/sts-v2.81.pdf`.
- **Status**: `Verified`

---

## 1. Authoritative Raw Source Verification

| Property | Verified Value |
|---|---|
| Source CSV | `STS_Combined_2014_2017_Full.csv` (verified via archival manifest) |
| Byte Size | `585,144,357` bytes |
| SHA-256 | `89cab3de7977b463009baa71455683075b5dfbca0151df7ba07c06b882b5ed38` |
| Total CSV Records (excl. header) | `368,132` (`0` malformed-width records) |
| Labeled Eligible Cohort (`PRIMARY_TARGET_COtAFib` ∈ {0,1}) | `367,695` (`437` unknown-outcome rows isolated in `foundation_unlabeled_preop.parquet`) |
| Unique Hospitals (`HospID`) | `1,091` across `9` U.S. census regions (`Region`) |
| STS Database Version (`DataVrsn`) | `2.81` for **100.00%** of rows (`OnDemandVrsn == '2.81.1'`) |

> **Manuscript Note (Refuting Legacy Documentation Errors)**: Earlier internal PDF summaries (`STS_All_Years_Key_Variables_Combined.pdf`) erroneously claimed schema "v2.83" and misstated several integer codebooks. Direct verification against `DataVrsn` and the official STS ACSD v2.81 Data Collection Specifications (`references/sts-v2.81.pdf`) confirms uniform v2.81 compliance across all four calendar years.

## 2. Exact Temporal Coverage (Half-Year Boundary Effect)

Because STS ACSD v2.81 went live on **July 1, 2014** and transitioned to v2.9 on **July 1, 2017**, the boundary years are **exact half-years**:

| `DataYear` | Calendar Window | Total Raw Records | Share | POAF Rate (Labeled) |
|---|---|---:|---:|---:|
| `2014` | H2 only (Jul–Dec 2014) | `58,790` | 15.97% | `23.69%` |
| `2015` | Full year (Jan–Dec 2015) | `123,248` | 33.48% | `24.21%` |
| `2016` | Full year (Jan–Dec 2016) | `124,479` | 33.81% | `24.41%` |
| `2017` | H1 only (Jan–Jun 2017) | `61,615` | 16.74% | `24.83%` |

This half-year asymmetry (H2-only 2014 vs H1-only 2017) is a key driver of why a naive 2017 temporal split confounds seasonal respiratory/winter POAF incidence with temporal drift (see [04_why_2017_holdout_is_not_statistically_valid.md](04_why_2017_holdout_is_not_statistically_valid.md)).

## 3. Verified Non-Monotonic STS v2.81 Coding Pitfalls

Direct empirical cross-tabulation against POAF incidence and the official v2.81 codebook identified critical non-monotonic encodings that cause silent clinical errors if treated as raw linear integers:

| Variable(s) | Exact STS v2.81 Coding | Empirical POAF / Modeling Insight |
|---|---|---|
| `VDInsufA`, `VDInsufM`, `VDInsufT` | `0=None, 1=Trace, 2=Mild, 3=Moderate, 4=Severe, 5=Not Documented` | **Never treat `5` as severe**; must mask `5 → NaN` before ordinal valve severity scoring (`VDInsufM in {3,4}` strongly elevates POAF). |
| `ChrLungD` | `1=No, 2=Mild, 3=Moderate, 4=Severe, 5=Severity Unknown, 6=Unknown` | `5` and `6` are unknown codes, not extreme COPD; must remap before pulmonary composites. |
| `CarShock`, `Resusc` | `2=No, 3=Yes at procedure, 4=Yes <24h` | There is **no code `1`** in v2.81; binary acute instability flags must test `isin([3, 4])`. |
| `HmO2` | `2=No, 3=PRN, 4=Continuous, 5=Unknown` | `5` must be masked to NaN. |
| `TobaccoUse` | `1=Never, 2/3/4=Current, 5=Former` | Classic **smoker's paradox**: Current smokers (`19.3%–21.5%` POAF, younger) vs Never (`23.9%`) vs Former (`27.6%`, older/remodeled atria). |
| `MedAmiodarone` | `2=Yes, 3=No, 4=Unknown` | Preoperative amiodarone (`2`) is strongly **protective** (`19.3%` POAF vs `24.5%` no amiodarone). |
| `MedBeta` / `MedBetaTher` | `1=Yes, 2=No, 3=Contraindicated` | Contraindication (`3`) marks high-risk POAF (`26.6%–27.1%`); `MedBetaTher==1 & MedBeta!=1` captures acute **beta-blocker withdrawal**. |
| `NumDisV` | `1=0 vessels, 2=1 vessel, 3=2 vessels, 4=3 vessels` | Offset by `+1` relative to vessel count. |
| `RaceCaucasian` / `RaceBlack` | `1=Yes, 2=No` | Pronounced epidemiologic gradient (`25.5%` POAF Caucasian vs `15.9%` Black, consistent with Mathew 2004 & Andrei 2024). |
| `Gender` | `1=Male, 2=Female` | Male `25.1%` POAF vs Female `21.5%` (larger absolute atrial size / height). |

## References
1. Society of Thoracic Surgeons. *STS Adult Cardiac Surgery Database Data Specifications, Version 2.81*. Chicago, IL: STS; 2014.
2. Shahian DM, Jacobs JP, Badhwar V, et al. The Society of Thoracic Surgeons 2018 Adult Cardiac Surgery Risk Models: Part 1—Background, Design Considerations, and Model Development. *Ann Thorac Surg*. 2018;105(5):1411–1418.
3. Mathew JP, Fontes ML, Tudor IC, et al. A Multicenter Risk Index for Atrial Fibrillation After Cardiac Surgery. *JAMA*. 2004;291(14):1720–1729.
