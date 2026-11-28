# Technical & Latency Brief: NEVERHUMAN vs. Tabular Foundation Models, CatBoost & AutoGluon Across 1,091 U.S. Hospitals

**For CTOs, VPs of Engineering, Clinical ML Architects & Platform Teams**  
*NEVERHUMAN Research & HonorHealth Heart Care · STS ACSD v2.81 · November 2026*  
**[Download 2-Page Formatted PDF](neverhuman-poaf-cto-brief.pdf)** · **[NEJM AI Manuscript (PDF)](../nejm_ai/main.pdf)** · **[npj Digital Medicine (PDF)](../npj_digital_medicine/main.pdf)** · **[CEO Brief (PDF)](neverhuman-poaf-ceo-brief.pdf)**

---

## Engineering Snapshot

| **0.6767 / 0.6916 AUC** | **+21.8 to +55.3\* bps** | **4.11 s / 45.2 ms** | **21.5 min Offline Train** |
| :---: | :---: | :---: | :---: |
| **Unseen / Native Seen 208** | **Strict 74-Hosp Cluster Wins** | **0.193 ms/pt · 0.53 GB VRAM** | **28% Faster than AutoGluon** |
| #1 at 50k parity on both unseen (74 hosp) and native `site208` seen (1,086 hosp) splits | Beats AutoGluon (`+21.8*`), CatBoost (`+24.5*`), TabPFN (`+29.2*`), Kumo (`+52.0*`), TabFM (`+55.3*`) | `19.7×` faster than AutoGluon batch (`630×` faster 1-pt) & `117×–323×` faster than Kumo/TabPFN/TabFM | Compact deployment footprint vs. `10.1 GB` AutoGluon stack and `12–41 GB` TFM VRAM |

---

## 1. Zero-Leakage Data Contract & Dual Hospital Split Architecture

Auditing all **368,132 isolated-CABG records across 1,091 U.S. centers** (STS ACSD v2.81) established:
1. **Skip-logic outcome reconstruction (`Complics → COtAFib`):** Recovers **367,695 labeled cases** (`89,337` POAF events, `24.30%`), removing complete-case collider bias across 232,783 uncomplicated recoveries (`63.31%`).
2. **Strict `638 → 207` pre-incision variable contract:** Purges 7 early-column postoperative duration leaks (`LOS`, `PLOS`, `ORDuration`, `SIDuration`, `SurgtoMort`, `SurgtoRead`, `SurgtoSternInf`), bypass/cross-clamp times, discharge drugs, STS calculators, and site IDs.
3. **Matched Two-Regime Design (`50,000` 5-fold hospital-grouped train $\to$ `21,241` natural-prior holdout):**
   - **Regime 1 — Unseen Hospitals (0.0% overlap, 74 centers):** Multivariate anticlustering (`Max|SMD| = 0.103`).
   - **Regime 2 — Seen Hospitals (100.0% overlap, all 1,091 centers in train / 1,086 test centers):** Evaluated both on 207 patient-only features (exposing `+28–64 bps` familiarity inflation; 2× larger for TFMs) and with train-only Empirical-Bayes hospital priors (`site208` and closed-form `+EB50k` / `+EB346k`).

---

## 2. Full 6-Model Accuracy, Training Time & Inference Speed Matrix

| Model (50k Parity Unless Noted) | Unseen AUC | NEVERHUMAN $\Delta$ Unseen | Seen 207 Raw | Seen Native 208 | Seen +EB346k | Train / Prep | 21k Batch Infer | Batch ms/pt | 1-Pt Bedside | Hardware / Peak Memory |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **NEVERHUMAN** | **0.6767** | **#1 @ 50k** | **0.6794** | **0.6916** | **0.6974** | **21.5 m** | **4.11 s** | **0.193** | **45.2 ms** | 1× RTX 3090 + CPU (0.53 GB VRAM) |
| AutoGluon 1.5 (`best_quality`) | 0.6745 | **+21.8 [+7.0, +36.5]\*** | 0.6790 | 0.6891 | 0.6974 | 30.0 m | 80.99 s | 3.81 | 28.5 s | 1× RTX 4090 + 48 CPUs (10.1 GB disk) |
| CatBoost (5-fold binary) | 0.6742 | **+24.5 [+7.2, +42.2]\*** | 0.6777 | 0.6886 | 0.6966 | 0.8 m | 0.15 s | 0.007 | 3.0 ms | CPU 20 threads (0 GB VRAM) |
| PriorLabs TabPFN-3.5 | 0.6738 | **+29.2 [+8.2, +53.5]\*** | 0.6795 | --- | 0.6982 | 0.9 s | 515.3 s | 24.26 | In-context | A100-40GB (7.7 / 14.7 GB VRAM) |
| NVIDIA KumoTabular-Large v2 | 0.6715 | **+52.0 [+21.6, +83.3]\*** | 0.6764 | --- | 0.6977 | 3.3 m | 481.7 s | 22.68 | In-context | A100-40GB (6.8 / 12.0 GB VRAM) |
| Google TabFM v1.0.0 | 0.6712 | **+55.3 [+28.6, +81.5]\*** | 0.6775 | --- | 0.6986 | 6.6 m | 1,326.7 s | 62.46 | In-context | A100-40GB (22.1 / 40.8 GB VRAM) |
| *7× CatBoost Full Registry (346k)* | *0.6773* | *−6.4 [−23.0, +10.5]* | *0.6831* | ***0.7014*** | *---* | *6.2 m* | *0.30 s* | *0.014* | *---* | CPU 20 threads |

---

## 3. False Positive Rate (FPR) & U.S. (200k/yr) / Global (1M/yr) Clinical Impact

- **Lowest FPR at Fixed Capture (74 Unseen Hospitals):** At 70% sensitivity, NEVERHUMAN achieves the lowest FPR (**`44.10%`** vs. `44.42%–45.28%` across the 5 binary AI models and `76.30%` Mariscalco POAF; dropping to **`41.22%`** on seen `site208`), sparing **490–1,798 U.S. patients/yr** (**2,448–8,992 globally**) vs. AI comparators and **48,755 U.S. patients/yr** (**243,774 globally**) vs. Mariscalco POAF. At 80% sensitivity, NEVERHUMAN again leads at **`55.54%` FPR** (`424–2,184` U.S. / `2,118–10,921` global spared/yr).
- **Modeled Top-25% Prophylaxis Impact (PALACS 45% RRR + STS Mortality Excess `2.78%` vs. `1.42%`):** Captures **19,462 U.S. POAF cases/yr** (**97,312 globally**), modeling **8,758 U.S. POAF episodes prevented/yr** (**43,790 globally**), **$87.6M–$131.4M/yr** in U.S. hospital cost avoidance, and (under conservative 25%–50% causal attribution; 100% upper bound) **30–60 (upper 119) U.S. operative deaths averted/yr** (**149–298, upper 596 globally**).

---

## 4. Reproducible Build & Compliance CLI

```bash
# 1. Deterministically rebuild BOTH submission-ready clinical journal editions
make -C docs/nejm_ai -B && make -C docs/npj_digital_medicine -B

# 2. Rebuild & verify 2-page CEO & CTO executive briefs
bash docs/briefs/scripts/build_briefs.sh && bash docs/briefs/scripts/build_briefs.sh --check

# 3. Run automated compliance, privacy & link auditor
python3 scripts/41_audit_ip_clean_publication.py \
  --nejm-dir docs/nejm_ai \
  --npj-dir docs/npj_digital_medicine \
  --scan-dirs docs reports README.md
```

**Next Reads:** [NEJM AI Original Article (PDF)](../nejm_ai/main.pdf) · [npj Digital Medicine Article (PDF)](../npj_digital_medicine/main.pdf) · [CEO Executive Brief (PDF)](neverhuman-poaf-ceo-brief.pdf) · [Clean Benchmark Receipt (MD)](../../reports/clean_publication_receipt.md)
