# 02. Primary Target Definition, STS Skip-Logic Proof & Legacy Collider-Bias Resolution

- **Paper Section**: *Methods — Outcome Definitions* & *Discussion — Registry Skip-Logic & Collider Bias*
- **Core Claim**: Under STS ACSD v2.81 (Seq 4930), postoperative atrial fibrillation (`COtAFib`) is gated by the parent complication flag `Complics`. Reconstructing the true clinical label yields **89,337 POAF cases out of 367,695 labeled patients (24.30%)**, matching published isolated-CABG epidemiology (20%–30%) and resolving a severe double bug (collider bias + label inversion) present in legacy project downsamples.
- **Evidence & Source Artifacts**: `data/foundation_targets.parquet`, `data/foundation_unlabeled_preop.parquet`, `scripts/01_build_clean_foundation.py`.
- **Status**: `Verified`

---

## 1. Empirical Proof of STS v2.81 `Complics → COtAFib` Skip-Logic

In STS ACSD v2.81, data abstractors record `COtAFib` **only when `Complics == 1`** ("Did the patient experience any postoperative complication?"). When `Complics == 2` (No complication), `COtAFib` is intentionally left blank (`NaN`). Cross-validating all 368,132 records against independent discharge antiarrhythmic/anticoagulation therapies (`DCAmiodarone`, `DCCoum`) and `Age` proves exact label semantics beyond doubt:

| Raw STS Pattern | Row Count | Share of Labeled | Mapped `PRIMARY_TARGET_COtAFib` | Mean Age (yr) | Discharge Amiodarone (`DCAmiodarone=1`) | Discharge Warfarin (`DCCoum=1`) | Clinical Meaning |
|---|---:|---:|:---:|---:|---:|---:|---|
| `Complics == 2` (`COtAFib` NaN) | `232,783` | 63.31% | **`0`** | `63.18` | `14.65%` | `1.83%` | Uncomplicated postoperative course |
| `Complics == 1 & COtAFib == 1` | `89,337` | **24.30%** | **`1`** | **`68.49`** | **`74.33%`** | **`16.49%`** | **True Postoperative Atrial Fibrillation** |
| `Complics == 1 & COtAFib == 2` | `45,575` | 12.40% | **`0`** | `64.70` | `16.58%` | `4.92%` | Non-AFib postoperative complication |
| `Complics` NaN (389) or `Complics==1 & COtAFib` NaN (48) | `437` | Excluded | Isolated | — | — | — | Indeterminate outcome → `foundation_unlabeled_preop.parquet` |

**Key Clinical Confirmation**:
- Patients with `Complics == 1 & COtAFib == 1` are **+5.3 years older** than uncomplicated patients (`68.49` vs `63.18`), receive discharge amiodarone at **5.1×** the baseline rate (`74.33%` vs `14.65%`), and receive discharge warfarin at **9.0×** the baseline rate (`16.49%` vs `1.83%`).
- The resulting overall incidence of **24.30%** aligns directly with multi-center STS isolated-CABG POAF registries (22%–28%; Magee 2007, Mariscalco 2014, Andrei 2024).

## 2. Root-Cause Audit of Legacy Downsample Failure Mode

Prior internal downsamples (`50k10kprime`, `train.csv` 16k) exhibited two compounding data-pipeline defects that invalidated clinical interpretation:
1. **Severe Conditioning-on-a-Collider Bias**: Requiring `COtAFib.notna()` silently discarded all **232,783 uncomplicated patients** (`63.3%` of the cohort!), restricting modeling strictly to the subset with *at least one postoperative complication* (`Complics == 1`). Conditioning on a post-treatment outcome (`Complics == 1`) induces spurious inverse correlations between POAF and severe non-AFib complications (e.g., renal failure, sepsis, mortality).
2. **Label Polarity Inversion**: Copying an erroneous PDF legend mapped `COtAFib == 2 → 1` and `COtAFib == 1 → 0`, accidentally predicting *non-AFib complications among complicated patients* rather than POAF!

Both defects are eliminated at the root in `data/foundation_targets.parquet`.

## 3. Complete 9-Outcome Table (Primary + 8 Auxiliary STS Complications)

All 8 auxiliary complications follow the exact same `Complics` skip-logic (`Complics == 2 → 0`, unknown coded `-1`) and are kept strictly isolated in `targets_all.parquet` (never leaked as inputs):

| Target Column | Positives | Valid Rows | Natural Prevalence | Clinical Endpoint |
|---|---:|---:|---:|---|
| `PRIMARY_TARGET_COtAFib` | `89,337` | `367,695` | `24.296%` | Postoperative Atrial Fibrillation (Primary) |
| `TARGET_CPVntLng` | `25,261` | `367,648` | `6.871%` | Prolonged Ventilation (>24 h) |
| `TARGET_CPlEff` | `11,645` | `367,627` | `3.168%` | Pleural Effusion Requiring Drainage |
| `TARGET_CPPneum` | `8,034` | `367,635` | `2.185%` | Postoperative Pneumonia |
| `TARGET_Mortalty` | `6,645` | `367,476` | `1.808%` | Operative / 30-Day Mortality |
| `TARGET_CRenFail` | `6,337` | `367,642` | `1.724%` | Postoperative Renal Failure |
| `TARGET_COtArrst` | `5,363` | `367,659` | `1.459%` | Postoperative Cardiac Arrest |
| `TARGET_CSepsis` | `2,638` | `367,361` | `0.718%` | Postoperative Sepsis |
| `TARGET_CVTE` | `2,435` | `367,637` | `0.662%` | Venous Thromboembolism (DVT/PE) |

- **Preoperative AFib Sensitivity Slice (`has_preop_afib_hx`)**: `0.52%` of patients have documented preoperative AFib (`ArrhythAFib == 1`). Rather than silently dropping them, we preserve `has_preop_afib_hx` in `splits.parquet` so every model reports both full-cohort performance and strictly ***de novo* POAF** performance (`has_preop_afib_hx == 0`).
