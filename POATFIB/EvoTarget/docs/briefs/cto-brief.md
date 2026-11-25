# Technical & Architecture Brief: Breaking Tabular ML's Binary-Label Ceiling via Evolutionary Target Distillation

**For CTOs, VPs of Engineering, Clinical ML Architects & Tabular Foundation Model Teams**<br>
*NEVERHUMAN Research & HonorHealth Heart Care · STS ACSD v2.81 · November 2026*<br>
**[Download 2-Page Formatted PDF](evotarget-poaf-cto-brief.pdf)** · **[Read CEO Executive Brief](ceo-brief.md)** ([PDF](evotarget-poaf-ceo-brief.pdf)) · **[Read Full 17-Page Manuscript](../paper/main.pdf)** · **[Distributed Compute Playbook](../distributed_compute/README.md)**

---

## Engineering Snapshot

| **0.67629 AUC** | **+26 to +97 bps** | **638 → 207** | **601 Cores / 142 fits/s** |
| :---: | :---: | :---: | :---: |
| **EvoTarget-FT (207 Raw)** | **Cross-Paradigm $z^*$ Lift** | **Zero-Leakage Contract** | **Zero-NFS Cluster Engine** |
| `+20.6` to `+51.4 bps` over all 6 binary 50k ELO baselines (every 74-hosp 95% CI > 0) | Single frozen Gen-20 target lifts all 6 families; strictly > 0 CIs on all 3 bagged TFMs | `Complics` collider fixed; 7 early leaks purged; `0.0%` hospital overlap; 48/48 pytest | 10-node SSH-piped pull queue; `uint16` genomes; `mmap`; zero `y_test` on workers |

---

## 1. Zero-Leakage Data Contract & Why Binary `{0,1}` Labels Stall All 6 Tabular Paradigms

Auditing all **368,132 isolated-CABG records** across **1,091 U.S. centers** (STS ACSD v2.81, SHA-256 `89cab3de7977b463009baa71455683075b5dfbca0151df7ba07c06b882b5ed38`) established three non-negotiable data engineering contracts before modeling:

1. **Skip-logic outcome reconstruction:** Under STS Sequence 4930, `COtAFib` is populated only when `Complics == 1`; uncomplicated courses (`Complics == 2`, `232,783` rows, `63.31%`) leave `COtAFib` structurally null. Naive `notna()` filtering induces severe collider bias. Reconstructing skip logic yields **367,695 labeled cases** (`89,337` POAF, `24.30%`).
2. **Strict `638 → 207` pre-incision cutoff:** Purged 7 hidden early-position post-operative/duration fields (`LOS`, `PLOS`, `ORDuration`, `SIDuration`, `SurgtoMort`, `SurgtoRead`, `SurgtoSternInf`), perfusion times, discharge drugs, STS risk calculators, and center IDs.
3. **Hospital-disjoint anticluster split (`0.0%` overlap):** Naive 2017 splitting reuses `99.5%` of training hospitals (`998/1,003`) and inflates matched CatBoost AUC by **+1.99 points** (`0.6778` vs. `0.6579` on within-2017 unseen centers). Multivariate anticlustering partitioned the 1,091 hospitals into **1,017 train (`346,454`) / 74 strictly unseen holdout (`21,241`, `5,160` POAF)**, cutting 32-D Energy Distance `4.1×` (`0.01732 → 0.00427`) and `Max |SMD|` `3.5×` (`0.3572 → 0.1028`), with a nested `50,000 / 10,000` balanced context-budget split mapped back to natural prevalence via the strictly development-only Bayes shift (`−1.136479`).

**Why every binary-label model plateaus:** Unmeasured intraoperative triggers and threefold center-level incidence dispersion (`11.7%–34.8%`) inject heavy quantization noise into `{0,1}` labels. Consequently, all six frontier tabular ELO families trained on raw binary labels at `N = 50,000` collapse into a razor-thin **30.8 bps** band (`0.67115–0.67423`), and adding 82 deterministic clinical features leaves 50k AUC flat (`−7.7` to `+1.8 bps`).

---

## 2. Two-Stage EvoTarget-FT Architecture

```text
[1,017 Dev Hospitals | ZERO y_test]
   │
   ├─► Stage 1: EWMA Gene-Locked Multi-Family PSO (5,120 offline fits, 10-node SSH pull queue)
   │      • Whole-hospital shuffled teachers: CatBoost + LightGBM + XGBoost + L2 Ridge (289 feat.)
   │      • EWMA step hardening (β_g: 0.286 → 0.159 at Gen 20) + 20% confident core gene-lock
   │      • Emits single frozen continuous risk vector z* ∈ [0,1]^50k
   │      • Independent biological check (original y=0): top-15% z* has 3.33× sepsis, 2.51× renal failure, 2.33× mortality
   │
   └─► Stage 2: Dual-View PLR Clinical Self-Attention Transformer (STRICTLY 207 RAW FEATURES)
          • Fold-internal Dual Views: 512-quantile Gaussianized + [0.5%, 99.5%]-winsorized signed-log(1+|x|)
          • 6 learnable PLR sinusoidal frequencies/view + missingness embeddings + 4 row-missingness summaries
          • T=24 clinical subsystem tokens + [CLS] (d_model=192) → 3 Pre-Norm GELU attention layers (6 heads)
          • Loss: Soft BCE vs. z* + 0.15× gap-weighted pairwise ranking + top-3 SWA (4 seeds × 5 folds)
          • Deployment: 20 lightweight passes + fold-OOF Platt scaling + dev-only shift (-1.136479) → 0.00955 ECE
```

---

## 3. Complete Unseen-Hospital Scoreboard & Controlled Comparisons (`test_full_21k`, 74 Hospitals)

| Model / Ablation (207 Raw Unless Noted) | Train `N` | 21k AUC | Δ vs. EvoTarget [74-Hosp 95% CI] | AP | LogLoss | Brier | ECE | 2017 Unseen |
| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **EvoTarget-FT (Gen-20 $z^*$ + Clinical Attn)** | **50,000** | **0.67629** | **Featured Reference** | **0.37691** | **0.51818** | **0.17128** | 0.00955 | **0.65715** |
| CatBoost (5-fold tuned binary ref) | 50,000 | 0.67423 | **+20.6 bps `[+3.1, +39.1]`** | 0.37438 | 0.51846 | 0.17146 | 0.00778 | 0.65650 |
| PriorLabs TabPFN-3.5 (50k native, e8) | 50,000 | 0.67376 | **+25.3 bps `[+2.2, +50.1]`** | 0.37284 | 0.51844 | 0.17152 | 0.00683 | 0.65315 |
| LightGBM (5-fold binary) | 50,000 | 0.67363 | **+26.6 bps `[+5.3, +46.1]`** | 0.37251 | 0.51906 | 0.17162 | **0.00603** | 0.65707 |
| XGBoost (5-fold binary) | 50,000 | 0.67357 | **+27.2 bps `[+5.6, +47.0]`** | 0.37219 | 0.51903 | 0.17162 | 0.00685 | 0.65572 |
| NVIDIA KumoTabular-Large v2 (5×10k) | 50,000 | 0.67147 | **+48.2 bps `[+15.8, +80.4]`** | 0.36925 | 0.51951 | 0.17191 | 0.00702 | 0.65338 |
| Google TabFM v1.0.0 (5×10k, e80) | 50,000 | 0.67115 | **+51.4 bps `[+22.8, +79.4]`** | 0.36967 | 0.51973 | 0.17200 | 0.00901 | 0.65240 |
| *Ablation: Same-Arch Binary FT-Transf.* | *50,000* | *0.67468* | *+16.0 bps `[-2.1, +35.3]`* | *0.37551* | *0.51848* | *0.17138* | *0.00643* | *0.65716* |
| *Ablation: 3-Way Binary Paradigm Blend* | *50,000* | *0.67568* | *+6.0 bps `[-5.8, +17.3]`* | — | — | — | — | *0.65722* |
| *7× Full-Registry CatBoost (207 raw)* | *346,454* | *0.67732* | *−10.3 bps (7× data ceiling)* | *0.37761* | *0.51713* | *0.17100* | *0.00760* | *0.65785* |
| *346k TabM-K16 Multi-Task (289 feat.)* | *346,454* | *0.67802* | **+10.2 bps `[+2.9, +17.6]`** *(289 vs 207)* | — | — | — | — | — |

### Cross-Paradigm Target Comparison: Identical Frozen Gen-20 Target $z^*$ vs. Binary `{0,1}`

| Family (Exact Same Frozen Gen-20 $z^*$) | Binary Control AUC | Gen-20 $z^*$ AUC | Δ ROC-AUC (bps) | 74-Hospital 95% CI | Draws > 0 |
| :--- | ---: | ---: | ---: | :--- | ---: |
| **Google TabFM v1.0.0** (5×10k Reg) | 0.66410 | **0.67382** | **+97.2** *(+26.7 vs cls)* | **`[+62.7, +132.5]`** | **100.0%** |
| **NVIDIA KumoTabular-L v2** (5×10k Reg) | 0.67220 | **0.67499** | **+27.9** *(+35.2 vs cls)* | **`[+6.2, +51.2]`** | **99.4%** |
| **PriorLabs TabPFN-3.5** (5×10k Reg) | 0.67134 | **0.67395** | **+26.1** | **`[+6.8, +48.9]`** | **99.8%** |
| PriorLabs TabPFN-3.5 (50k Native Reg) | 0.67282 | 0.67413 | +13.1 | `[-3.9, +32.1]` | 92.4% |
| XGBoost CE (5-fold) | 0.67357 | 0.67487 | +12.9 | `[-3.5, +27.7]` | 93.8% |
| LightGBM Multi-Seed CE (3-seed) | 0.67320 | 0.67447 | +12.7 | `[-5.2, +29.0]` | 91.7% |
| CatBoost CE (5-fold) | 0.67408 | 0.67497 | +8.9 | `[-1.2, +19.8]` | 95.8% |

**Transparent nuanced / negative findings:**
- Same-architecture Binary FT-Transformer (`+16.0 bps [-2.1, +35.3]`, `95.9%` positive draws) is compatible with a small advantage or no gain from the evolved target in this architecture.
- Adding 82 deterministic clinical features at 50k leaves AUC flat and lowers observed 15-bin ECE point estimates `11%–42%` (74-hospital CIs span zero), while delivering hospital-cluster-confirmed representation gains on 346k multi-task neural networks (`TabM-K16` `+10.2 bps [+2.9, +17.6]`, `PLR-ResNet` `+10.1 bps [+2.2, +18.3]`).
- On the small within-2017 unseen-hospital slice (`n = 3,555`, 63 centers), EvoTarget-FT (`0.65715`) ties binary FT (`0.65716`).

---

## 4. Zero-NFS Distributed Compute & 5-Minute Verification CLI

Offline Stage-1 evolution and heterogeneous model fan-out ran across a **10-node LAN + Cloud fleet** (`601` cores, `10+` GPUs, zero shared NFS/Slurm) documented in [docs/distributed_compute/](../distributed_compute/README.md): `/dev/shm` `mmap_mode='r'`, `uint16` genome streaming (`~75 KB`), in-node sufficient-stats reduction (`>50×` WAN cut), OS idle priority wrappers (`ionice -c 3 nice -n 19 chrt -i 0`), live external-load backpressure, and strict `y_test` quarantine on the controller.

```bash
# Run from POATFIB/EvoTarget in Japer.
# Check citations, cross-references, and manuscript artifacts.
python3 scripts/20_audit_paper_numbers.py

# Rebuild the manuscript and verify both PDF briefs.
make -C docs/paper -B
bash docs/briefs/scripts/build_briefs.sh --check
```

The full test suite passed 48 tests in the research workspace, which also contains the original distributed launchers.

---

## 5. Hard Technical Limits & Next Engineering Frontier

1. **Selection exposure (103 evaluated arms):** Holdout intervals are conditional on selected runs and do not adjust for adaptive search.
2. **Teacher vs. student feature asymmetry:** Stage-1 teachers consumed 289 clinical features while deployed EvoTarget-FT consumes 207 raw inputs. Matched-budget raw-only teachers are prioritized for v2 to cleanly separate feature distillation from evolutionary label smoothing.
3. **Non-nested global search & split sensitivity:** Stage-1 evolution operated globally on the 50k development cohort (internal canary trace `0.67831` is search telemetry, not nested CV). Outer-nested evolution paired with label-blind matched hospital splits ([FEATURE_MATCHED_EVALUATION.md](../../FEATURE_MATCHED_EVALUATION.md)) closes this loop.
4. **Prospective frozen STS v4.20+ validation:** Freeze target recipe, teacher inputs, and transformer weights before scoring an unused post-2017 STS cohort with lightweight local intercept recalibration.

---

**Next Reads:** [2-Page CTO Brief (PDF)](evotarget-poaf-cto-brief.pdf) · [CEO Executive Brief (MD)](ceo-brief.md) ([PDF](evotarget-poaf-ceo-brief.pdf)) · [Full 17-Page Manuscript (PDF)](../paper/main.pdf) · [Distributed Compute Playbook](../distributed_compute/README.md) · [Audit Receipts](../../reports/paper_red_team_receipt.json)
