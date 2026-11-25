# 03. Strict Pre-Incision Feature Space & Zero-Leakage Schema Audit (638 → 207)

- **Paper Section**: *Methods — Candidate Predictors & Leakage Prevention*
- **Core Claim**: Every single one of the **638 raw columns** in `STS_Combined_2014_2017_Full.csv` was audited against `INPUT_VARIABLE_STS.md`, `OUTPUT_VARIABLE_STS.md`, and the official STS ACSD v2.81 specifications (`sts-v2.81.pdf`). Exactly **207 strictly pre-incision predictors** survive (206 numeric/ordinal + 9-level categorical `Region`), purging 100% of post-incision, outcome, STS-calculator, administrative, and hidden early-column leaks.
- **Evidence & Source Artifacts**: `data/schema_audit.json`, `scripts/foundation_contract.py`, `scripts/01_build_clean_foundation.py`.
- **Status**: `Verified`

---

## 1. Exhaustive Disjoint Partition of All 638 Raw Columns

No column is implicitly kept; every column is assigned to an explicit, disjoint category verified programmatically by `test_batch2_contract.py` and `test_optimal_split_and_dataset.py`:

1. **Kept Strictly Pre-Incision Predictors (`207` columns)**:
   - Comprises the legacy 147 clean preoperative predictors **plus ~60 reclaimed STS v2.81 preoperative clinical predictors** (e.g., detailed mitral/aortic valve etiology & lesion codes, coronary vessel stenosis percentages, carotid stenosis, home oxygen, preop lab panels, and antiarrhythmic/antithrombotic medication granularity).
   - `Region` (9 U.S. census divisions: `ENCENT`, `ESCENT`, `MIDATL`, `MOUNTAIN`, `NEWENG`, `PACIFIC`, `SOATL`, `WNCENT`, `WSCENT`) is preserved natively as a pandas `category` rather than coerced to NaN.
   - `CathtoSurg` intervals outside `[0, 365]` days are masked as invalid data-entry outliers.
2. **Purged Hidden Early-Column Leaks Before `OPApp` (col 335) — `7` Critical Catches**:
   - Even though STS columns are largely ordered chronologically up to operative approach (`OPApp`, col 335), **7 post-operative / intra-operative duration and outcome-timing fields appear in early column positions** and would cause catastrophic data leakage if sliced naively by column index:
     - `LOS` (Hospital Length of Stay)
     - `PLOS` (Postoperative Length of Stay)
     - `SurgtoMort` (Days from Surgery to Mortality)
     - `SurgtoRead` (Days from Surgery to Readmission)
     - `SurgtoSternInf` (Days from Surgery to Deep Sternal Infection)
     - `ORDuration` (Operating Room Duration)
     - `SIDuration` (Skin-Incision-to-Closure Duration)
3. **Purged Post-Incision, Intra-Op, Discharge & Outcome Columns (`OUTPUT_VARIABLE_STS.md` + all post-`OPApp` columns)**:
   - Purges cardiopulmonary bypass / cross-clamp times (`PerfusTm`, `XClampTm`), intraoperative blood products, graft counts, `OpCAB`, `OPApp`, `Complics`, `COtAFib`, all postoperative complications, discharge medications (`DCAmiodarone`, `DCCoum`, etc.), and all pre-computed STS risk-calculator probability outputs (`PredMort`, `PredMM`, `PredLong`, etc.).
4. **Purged Site, Provider, Temporal & Administrative Identifiers**:
   - `HospID`, `SiteID`, `SurgID`, `Surgorder`, `DataYear`, `DataVrsn`, `OnDemandVrsn`, `CountryUS` (undocumented export-level timing mapping).
   - `HospID` and `DataYear` are stored **strictly in `splits.parquet`** for hospital-disjoint grouping and audit stratification — never passed to any model matrix `X`.

## 2. Resolution of Historically Conflicted Variable Timings

Earlier project notes conflicted on the clinical timing of four important predictors. Verification against the official STS ACSD v2.81 sequence numbers resolved each conclusively:

| Variable | STS v2.81 Section / Seq | Timing Verdict | Justification |
|---|---|---|---|
| `Status` | Operative Risk / Seq 1975 (`1=Elective, 2=Urgent, 3=Emergent, 4=Salvage`) | **SAFE Pre-Incision** | Determined preoperatively by surgical acuity prior to anesthesia/incision. |
| `Dialysis` | Preoperative Risk Factors | **SAFE Pre-Incision** | Preoperative chronic dialysis status (postop equivalent is purged `CRenDial`). |
| `CVA`, `CVAWhen` | Preoperative Risk Factors | **SAFE Pre-Incision** | Prior cerebrovascular history (postop equivalent is purged `CNStrokP`). |
| `ArrhythAFib`, `ArrhythAFibDur` | Preoperative Arrhythmia History | **SAFE Pre-Incision** | Preoperative arrhythmia history (0.52% non-null; postop equivalent is `COtAFib`). Also exposed as `has_preop_afib_hx` for *de novo* reporting. |

## 3. Automated Zero-Leakage Contract

Every downstream dataset (`dataset/full/`, `dataset/small_50k_10k/`), candidate feature registry (`features/tested/`, `features/batch2_tested/`), and loader call enforces `is_forbidden(col, audit["forbidden_model_keys"]) == False` via automated pytest suites (`scripts/tests/test_batch2_contract.py`, `scripts/tests/test_optimal_split_and_dataset.py`).
