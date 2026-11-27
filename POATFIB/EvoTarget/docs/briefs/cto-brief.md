# Technical & Architecture Brief: Breaking Tabular ML's Binary Ceiling via 1,000-Gen Target Evolution & Parallel GP

**For CTOs, VPs of Engineering, Clinical ML Architects & Tabular Foundation Model Teams**  
*NEVERHUMAN Research & HonorHealth Heart Care · STS ACSD v2.81 · November 2026*  
**[Download 2-Page Formatted PDF](evotarget-poaf-cto-brief.pdf)** · **[Read CEO Executive Brief](ceo-brief.md)** ([PDF](evotarget-poaf-ceo-brief.pdf)) · **[Read Full Manuscript](../paper/main.pdf)** · **[Distributed Compute Playbook](../distributed_compute/README.md)**

---

## Engineering Snapshot

| **0.67668 AUC / 0.00724 ECE** | **1,000 Gens (256k Fits)** | **+39.8 bps (`JAIN GP v2`)** | **65.2% Strictly Nested** |
| :---: | :---: | :---: | :---: |
| **Dual-Paradigm Fusion (207 Raw)** | **Zero-Collapse Target Evolution** | **Parallel GPU-Guided GP** | **2×2 Causal Proof (Lim. #2)** |
| `+24.5` to `+55.3 bps` over all 6 binary ELOs & `+9.9 bps` over 3-way binary blend (all CIs > 0); **79.2%** of 7× gap | **EvoTarget-FT** peaks Gen 20 (`0.67653`, `+18.9* bps` vs. Bin. FT), lifts **5 families** CI > 0, plateaus `0.67592` at Gen 1,000 | 6 fixes lift legacy `0.67167 → 0.67565` (beats all 6 binary ELOs); `+9.1* bps` binary-$y$ stop fix; CV-7 `r = 0.950` | Nested 207-raw $z^*$ captures `+12.3 bps` pure smoothing, **`+0.8 bps`** outer nesting (null), `+5.7* bps` 289→207 distillation |

---

## 1. Zero-Leakage Data Contract & Why Binary `{0,1}` Labels Stall Every Paradigm

Auditing all **368,132 isolated-CABG records** across **1,091 U.S. centers** (STS ACSD v2.81, SHA-256 `89cab3de7977b463009baa71455683075b5dfbca0151df7ba07c06b882b5ed38`) established three non-negotiable contracts:
1. **Skip-logic outcome reconstruction:** Reconstructing `Complics → COtAFib` skip logic yields **367,695 labeled cases** (`89,337` POAF, `24.30%`).
2. **Strict `638 → 207` pre-incision cutoff:** Purged 7 hidden early-position leaks (`LOS`, `PLOS`, `ORDuration`, `SIDuration`, `SurgtoMort`, `SurgtoRead`, `SurgtoSternInf`), perfusion times, discharge drugs, STS calculators, and center IDs.
3. **Hospital-disjoint anticluster split (`0.0%` overlap):** Naive 2017 splitting reuses `99.5%` of training hospitals (`+1.99` AUC inflation). Multivariate anticlustering partitioned 1,091 hospitals into **1,017 train (`346,454`) / 74 strictly unseen holdout (`21,241`, `5,160` POAF)**, with a nested `50,000 / 10,000` context-budget split recalibrated via uniform fold-OOF Platt + strictly development-only Bayes shift (`−1.136479`).

---

## 2. Dual-Paradigm Architecture: 1,000-Gen EvoTarget-FT + Re-Engineered JAIN Ultra GP v2

```text
[1,017 Dev Hospitals | 50k Nested Cohort | ZERO y_test]
   │
   ├─► Track 1: 1,000-Gen Target Evolution (256k fits, 5-node SSH pull queue) → EvoTarget-FT (207 Raw)
   │      • EWMA hardening (β_g: 0.286 → 0.159 at Gen 20) + 20% confident core gene-lock (zero collapse to Gen 1,000)
   │      • Strictly nested 5-fold 207-raw 2×2 causal proof: +12.3 bps pure smoothing (65.2%), +0.8 bps nesting, +5.7* bps distill
   │      • Dual-View PLR Transformer (512-Q + signed-log, 6-freq PLR, 24+1 tokens, 3L/6H 192d, Soft-BCE + 0.15× rank + SWA)
   │      • Gen-20 Peak: 0.67653 AUC (+18.9* bps vs. Bin. FT; 74.6% of 7× gap; 0.69438 rare clinical tails)
   │
   ├─► Track 2: JAIN Ultra GP v2 (GPU CUDA r882 Guided + 24-Worker Parallel GP, 207 Raw)
   │      • Fixes 6 bottlenecks: 24 parallel workers, 8 GiB RSS, full 50k fit, 64-bin Newton histgbm2 (+ with_stop_target),
   │        5-fold hospital-grouped ROC-AUC fitness, fresh CV-7 audit (r = 0.950), 64-thread Rayon fold-bagging
   │      • 6 evolved symbolic/linear/tree families: ensemble, winsor->pair_ratio->sigmoid->ridge_gcv, pair_ratio->elasticnet,
   │        histgbm2(logloss), oblique_hinge+cheb+pair_prod->ridge, histgbm → 0.67565 AUC (+39.8* bps vs. Legacy JAIN)
   │
   └─► Dual-Paradigm Fusion (Nested-OOF Logit Simplex: 46.3% GP v2 + 53.7% EvoTarget-FT)
          • Fold-OOF Platt scaling + dev-only shift (-1.136479) → 0.67668 AUC / 0.37767 AP / 0.00724 ECE / 0.51770 LogLoss
          • 79.2% of 7× 50k→346k gap recovered; beats 3-way binary blend (+9.9* bps) & beats 346k CB on AP, ECE, 2017 unseen
```

---

## 3. Complete Unseen-Hospital Scoreboard & Causal Ablations (`test_full_21k`, 74 Hospitals, `B = 5,000`)

| Model / Ablation (207 Raw Unless Noted) | Train `N` | 21k AUC | Fusion / Flagship Gain [74-Hosp 95% CI] | AP | LogLoss | Brier | ECE | 2017 Un. |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **Dual-Paradigm Fusion (GP v2 + EvoTarget-FT)** | **50,000** | **0.67668** | **#1 Overall (79.2% of 7× gap)** | **0.37767** | **0.51770** | **0.17123** | **0.00724** | **0.65794** |
| **EvoTarget-FT (Gen-20 $z^*$ + Clinical Attn)** | **50,000** | **0.67653** | vs. Bin. FT: **+18.9 `[+2.6, +35.8]`** | 0.37754 | 0.51796 | 0.17129 | 0.00850 | 0.65730 |
| *EvoTarget-FT (Strictly Nested 207-Raw G20)* | *50,000* | *0.67588* | *vs. Bin. FT: +12.3 `[-3.3, +28.5]` (65.2%)* | *0.37637* | *0.51797* | *0.17129* | *0.00742* | *0.65535* |
| *EvoTarget-FT (Gen-1,000 $z^*$ Stability)* | *50,000* | *0.67592* | *Zero collapse across 256,000 fits* | *0.37656* | *0.51857* | *0.17154* | *0.01370* | *0.65710* |
| **JAIN Ultra GP v2 Ensemble (6 evolved fams)** | **50,000** | **0.67565** | vs. Legacy JAIN: **+39.8 `[+19.6, +60.1]`** | 0.37582 | 0.51793 | 0.17135 | 0.00805 | **0.65809** |
| *Gen-0 Fair Fusion (GP v2 + Binary FT, no $z^*$)* | *50,000* | *0.67623* | vs. Bin. FT: **+15.8 `[+1.0, +30.3]`** | *0.37670* | **0.51761** | **0.17119** | **0.00460** | **0.65873** |
| CatBoost (5-fold tuned binary ref) | 50,000 | 0.67423 | Fusion: **+24.5 `[+7.2, +42.2]`** | 0.37438 | 0.51846 | 0.17146 | 0.00778 | 0.65650 |
| PriorLabs TabPFN-3.5 (50k native, e8) | 50,000 | 0.67376 | Fusion: **+29.2 `[+8.2, +53.5]`** | 0.37284 | 0.51844 | 0.17152 | 0.00683 | 0.65315 |
| LightGBM / XGBoost (5-fold binary) | 50,000 | 0.67363 | Fusion: **+30.5 / +31.0** (all CIs > 0) | 0.37251 | 0.51906 | 0.17162 | 0.00603 | 0.65707 |
| Legacy JAIN Ultra GPU (1w, 20k, RMSE) | 50,000 | 0.67167 | Fusion: **+50.1 `[+27.1, +73.2]`** | 0.36993 | 0.52171 | 0.17234 | 0.00905 | 0.65003 |
| NVIDIA KumoTabular-Large v2 (5×10k) | 50,000 | 0.67147 | Fusion: **+52.0 `[+21.6, +83.3]`** | 0.36925 | 0.51951 | 0.17191 | 0.00702 | 0.65338 |
| Google TabFM v1.0.0 (5×10k, e80) | 50,000 | 0.67115 | Fusion: **+55.3 `[+28.6, +81.5]`** | 0.36967 | 0.51973 | 0.17200 | 0.00901 | 0.65240 |
| *Ablation: 3-Way Binary Paradigm Blend* | *50,000* | *0.67568* | Fusion: **+9.9 `[+0.6, +19.6]`** | *0.37563* | — | — | — | *0.65722* |
| *7× Full-Registry CatBoost (207 raw)* | *346,454* | *0.67732* | *−6.4 (Fusion beats on AP, ECE, 2017)* | *0.37761* | *0.51713* | *0.17100* | *0.00760* | *0.65785* |

### Cross-Family Gen-20 Target $z^*$ Causal Lift (5 of 7 Paradigms Strictly > 0) & GP Binary-Stop Fix

| Comparison (`test_full_21k`, 74 Hospitals, `B = 5,000`) | Baseline AUC | Candidate AUC | Δ (bps) | 74-Hospital 95% CI | Draws > 0 |
| :--- | ---: | ---: | ---: | :--- | ---: |
| **Google TabFM v1.0** (5×10k Reg, $z^*$ vs. `{0,1}`) | 0.66410 | **0.67382** | **+97.2** *(+26.7 vs cls)* | **`[+63.7, +132.7]`** | **100.0%** |
| **NVIDIA KumoTabular-L v2** (5×10k Reg, $z^*$ vs. `{0,1}`) | 0.67220 | **0.67499** | **+27.9** *(+35.2 vs cls)* | **`[+6.0, +51.5]`** | **99.4%** |
| **PriorLabs TabPFN-3.5** (5×10k Reg, $z^*$ vs. `{0,1}`) | 0.67134 | **0.67395** | **+26.1** | **`[+6.0, +48.8]`** | **99.7%** |
| **EvoTarget-FT** (Gen-20 $z^*$ vs. Binary FT) | 0.67465 | **0.67653** | **+18.9** | **`[+2.6, +35.8]`** | **98.8%** |
| **CatBoost CE** (Gen-20 $z^*$ vs. Gen-0 `{0,1}`) | 0.67408 | **0.67538** | **+12.9** | **`[+0.1, +26.0]`** | **97.5%** |
| **GP v2 on Global $z^*$** (`with_stop_target` vs. no binary stop) | 0.67441 | **0.67532** | **+9.1** | **`[+0.5, +18.0]`** | **98.0%** |
| **EvoTarget-FT 289→207 Distillation** (Global-289 vs. Global-207) | 0.67596 | **0.67653** | **+5.7** | **`[+0.3, +10.6]`** | **98.2%** |
| *EvoTarget-FT Outer-Fold Nesting* (Global-207 vs. Nested-207) | *0.67588* | *0.67596* | *+0.8* | *`[-2.4, +4.4]` (null!)* | *68.2%* |

---

## 4. 5-Minute Zero-Trust Audit & Verification CLI

```bash
# 1. Verify all 12 manuscript .tex files, 95 citations & 21 Paper v4 JSON receipts (0 errors)
python3 scripts/20_audit_paper_numbers.py

# 2. Rebuild Paper v4 master receipt (78 models, 61 contrasts, B=5,000, seed=42)
python3 scripts/27_build_paper_v4_master_receipt.py

# 3. Run all 48 leakage, skip-logic & feature-contract unit tests
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q

# 4. Deterministically rebuild full manuscript + both 2-page PDF briefs
make -C docs/paper -B
bash docs/briefs/scripts/build_briefs.sh --check
```

---

**Next Reads:** [2-Page CTO Brief (PDF)](evotarget-poaf-cto-brief.pdf) · [CEO Executive Brief (MD)](ceo-brief.md) ([PDF](evotarget-poaf-ceo-brief.pdf)) · [Full Manuscript (PDF)](../paper/main.pdf) · [Paper v4 Master Receipt](../../reports/paper_v4_master_receipt.md)
