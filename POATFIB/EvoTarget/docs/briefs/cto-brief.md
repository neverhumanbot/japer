# Technical & Architecture Brief: Dual-Regime Pre-Incision POAF Prediction Across 1,091 U.S. Hospitals

**For CTOs, VPs of Engineering, Clinical ML Architects & Tabular Foundation Model Teams**  
*NEVERHUMAN Research & HonorHealth Heart Care · STS ACSD v2.81 · November 2026*  
**[Download 2-Page Formatted PDF](evotarget-poaf-cto-brief.pdf)** · **[NEJM AI Manuscript (PDF)](../nejm_ai/main.pdf)** · **[npj Digital Medicine (PDF)](../npj_digital_medicine/main.pdf)** · **[CEO Brief (PDF)](evotarget-poaf-ceo-brief.pdf)**

---

## Engineering Snapshot

| **0.6767 Cold / 0.7014 In-Net** | **1,000 Gens (256k Fits)** | **+39.8* bps (`JAIN GP v2`)** | **Implicit Fingerprinting** |
| :---: | :---: | :---: | :---: |
| **Dual-Regime Frontier (207 Raw)** | **Zero-Collapse $z^*$ + 2×2 Proof** | **Parallel GPU-Guided GP** | **100% Seen-Hospital Study** |
| Beats all 6 binary ELOs (`+20.3–55.3*`), 3-way binary blend (`+9.9*`), & bedside proxies (`+316–1,029*`); **79.2%** of 7× gap; `20.4–33.8` avoided/100 | **EvoTarget-FT** peaks Gen 20 (`0.6765`, `+18.9*`), plateaus `0.6759` at Gen 1,000; nested 207-raw confirms `65.2%` pure smoothing (`+0.8` nesting) | 6 fixes lift legacy `0.6717 → 0.6756` (CV-7 `r=0.959`); `+9.1*` binary-$y$ stop fix; `site208` Fusion leads native 50k (`0.6916`, `+29.0*` vs CB) | Overlap lifts 207-raw AUCs `+28–64 bps` (TFMs 2× Fusion); 40k–346k models absorb site signatures (`r=0.122–0.143`) vs 10k-shard TFMs (`0.088–0.092`) |

---

## 1. Zero-Leakage Data Contract & Literature Reconciliation (`0.61–0.98` AUCs)

Auditing all **368,132 isolated-CABG records across 1,091 U.S. centers** (STS ACSD v2.81, SHA-256 `89cab3de7977b463009baa71455683075b5dfbca0151df7ba07c06b882b5ed38`) established:
1. **Skip-logic reconstruction (`Complics → COtAFib`):** Recovers **367,695 labeled cases** (`89,337` POAF, `24.30%`), eliminating complete-case collider bias across 232,783 uncomplicated recoveries (`63.31%`).
2. **Strict `638 → 207` pre-incision audit:** Purges 7 early-column postoperative leaks (`LOS`, `PLOS`, `ORDuration`, `SIDuration`, `SurgtoMort`, `SurgtoRead`, `SurgtoSternInf`), bypass/cross-clamp times, discharge drugs, STS calculators, and site IDs.
3. **Why higher published AUCs (`0.72–0.98`) do not contradict `0.677–0.701`:** Every higher-AUC report uses post-incision/ICU variables (Magee `0.72`, Cho `0.73–0.89`, Cao MIMIC-IV `0.81`, Ma `0.93`), mixed valve+CABG cohorts (Mathew `0.77`, Burgos COM-AF `0.78 → 0.576` unseen), or continuous waveforms/multimodal streams (Han AI-ECG `0.71`, Khalpey Digital Twin `0.89`). PROBAST appraisals place the published external median at **`0.61`**, while Pollock et al.'s 5-center STS pooled model (`0.68`) matches our frontier tightly.

---

## 2. Dual-Paradigm Architecture + Controlled Cold-Start vs. In-Network Regimes

- **Regime 1 — Unseen-Hospital Cold Start (0.0% Overlap, 74 Centers, `N = 21,241`):** Multivariate anticlustering cuts 32-D Energy Distance 4.1× (`0.00427`) and `Max|SMD|` 3.5× (`0.1028`). **Dual-Paradigm Fusion** (`0.4626 GP v2 + 0.5374 EvoTarget-FT`, fold-OOF Platt + dev-only Bayes shift `−1.136479`) reaches **`0.6767` AUC**, **`0.3777` AP**, **`0.0072` ECE**, **slope `0.983`**, **O:E `0.994`**, **7.4× decile gradient** (`6.07% → 44.82%`), wins **24/25 subgroups**, and avoids **20.4–33.8 interventions / 100 patients** (`P>0 = 1.000` vs. Mariscalco POAF).
- **Regime 2 — Seen-Hospital In-Network (100.0% All-1,091-Hospitals-In-Train, `1,086` Test Centers, `N = 21,241`):**
  - **Patient-only (`207 raw`, zero `HospID`):** Every model jumps **`+28–64 bps`** purely from hospital familiarity—twice as large for tabular FMs (`TabFM +64`, `TabPFN +57`, `Kumo +49`) as for Fusion (`+28` → `0.6794`, strictly beating Kumo `+30.4*`, XGB `+20.8*`, TabFM `+18.9*`, CB `+16.9*`, LGBM `+14.5*`, tying TabPFN-50k `-0.4`).
  - **Native 50k `site208` champion:** End-to-end `Dual-Paradigm Fusion (site208)` leads at **`0.6916`**, beating `CatBoost site208` (`0.6887`) by **`+29.0 bps [+16.6, +41.5], P>0 = 1.000`**.
  - **Closed-form Bayesian site-intercept updating (`Platt_OOF + dev_shift + EB346k`):** Lifts every 50k model **`+174–213 bps`** into **`0.6958–0.6986`** (Fusion **`0.6974`**; 10k-shard TabPFN/TabFM **`0.6986`** due to ~30% lower implicit hospital entanglement `r=0.088–0.092` vs `0.122–0.143`) and lifts 346k CatBoost (`site208`) to **`0.7014`** (`slope 1.014`, `O:E 0.999`, `1.96×` lift).

---

## 3. Complete Verification & Deterministic Build CLI

```bash
# 1. Deterministically rebuild BOTH submission-ready clinical journal PDFs
make -C docs/nejm_ai -B && make -C docs/npj_digital_medicine -B

# 2. Run automated journal compliance, prose word-count, overfull & DUA audit
python3 scripts/36_audit_journal_manuscripts.py

# 3. Verify Paper v4 numbers + rebuild technical manuscript & 2-page PDF briefs
python3 scripts/20_audit_paper_numbers.py
make -C docs/paper -B && bash docs/briefs/scripts/build_briefs.sh --check
```

**Next Reads:** [NEJM AI Original Article (PDF)](../nejm_ai/main.pdf) · [npj Digital Medicine Article (PDF)](../npj_digital_medicine/main.pdf) · [Seen-Hospital Receipt](../../reports/seen_hospital_study/seen_hospital_master_receipt.md) · [Clinical & DCA Receipt](../../reports/clinical_journal_analyses.md)
