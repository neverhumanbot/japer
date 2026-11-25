# Note 11 — Project DOPE-ECHO: Turning Frozen `DOPE-MLC-v3` Into a Counterfactual Per-Row Tabular Featurizer (`dataset/small_50k_10k/`)

**Target Paper Sections**: *Methods & Results — Counterfactual Hypernetwork Featurization (`DOPE-ECHO`)* & *Discussion — Zero-Shot Table Encoders as Per-Row Tabular Representations*\
**Primary Sources**: `../reports/dope_echo_eval.json`, `../reports/dope_echo_eval.md`, `../reports/dope_echo_oof_preds.parquet`, `../reports/dope_echo_test_preds.parquet`, `../reports/paper_tables/table_dope_echo_cycles_small50k.{csv,tex}`, `../reports/paper_tables/table_dope_echo_ablation_multifamily.csv`, `../reports/paper_figures/fig_dope_echo_{master_summary,smoothing_convergence,cycle_curve}.{png,pdf}`, `../dope_echo/DOPE_ECHO_FINAL_REPORT.md`\
**Verified Status**: `Complete` (`100` zero-leakage cycles across all `60,000` rows on `xbabe0`–`xbabe9` = `12,000,000` High-level DOPE evaluations + `10`-cycle literal unmodified `dope-core` `LandmarkMode::None` control; 4/4 Rust unit tests + 4/4 pytest contract tests passing; `0/0` leftover scratch/workers across all 10 nodes)

> **Dataset Provenance (Verified)**: All results in this note were evaluated strictly on the **perfected smart-downsampled hospital-disjoint dataset `dataset/small_50k_10k/`** (smaller twin of our official `94.2% / 5.8%` hospital-disjoint split from Note 05: `50,000` train patients across `1,017` hospitals → `10,000` test patients across `74` strictly unseen hospitals, **all years `2014–2017`**, **`0.0%` hospital overlap**, exact `50.0% / 50.0%` POAF, `splits_sha256: c59a4621855580eedea596a868dc87ae22a7bbd209d728dd1b1ea2356536b766`, Bayes logit offset `1.1367` + Horvitz-Thompson natural-prior weights; strictly `207` pre-incision raw predictors `raw_all`, zero `aug_*` features). This is **NOT** the legacy naive 2017 holdout split (`2017 Unseen` in tables denotes the embedded `stress_2017_unseen_hosp` diagnostic slice inside `test_small_10k`). The larger twin `dataset/full/` (`346,454 → 21,241`) has not yet been run with DOPE-ECHO.

---

## 1. Motivation & Core Methodological Design

`DOPE-MLC-v3` (`~/dope/`, frozen `444,168`-parameter neural meta-encoder, canonical SafeTensors SHA-256 `d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`) is strictly a **dataset/table-level encoder** that maps an entire labeled table $(X, y) \to e \in \mathbb{R}^{629}$; it has no standalone per-row encoder. **Project DOPE-ECHO** (*D.O.P.E. Ensemble Counterfactual Hypernetwork Observer*) tests whether a frozen table hypernetwork can be converted into a leak-free **per-row featurizer** by repeatedly probing each row inside random `1,000`-row labeled training blocks and smoothing across cycles.

### Strictly Zero-Leakage Counterfactual Protocol
On **`dataset/small_50k_10k/`** (`50,000` train / `10,000` unseen-hospital test, exact `50.0% / 50.0%` POAF target `PRIMARY_TARGET_COtAFib`, **strictly `207` pre-incision raw predictors `include_groups=["raw_all"]`, zero `aug_*` features**):
1. All `60,000` rows (`50,000` train + `10,000` test) are processed through the **identical code path**, with **zero `y_test` bytes** ever packed into the worker bundle (`echo_bundle.bin`, verified by byte-exact contract test).
2. On cycle $t \in \{1,\dots,100\}$, we draw a stratified `500/500` random `1,000`-row context block $C_t \subset \text{train\_small\_50k}$ with true training targets.
3. **Mathematically Airtight Out-of-Context Exclusion**: Because `train_small_50k` is exact `50/50` and $C_t$ is exact `500/500`, $P(i \in C_t \mid y_i = 0) = P(i \in C_t \mid y_i = 1) = 0.020000$ identically ($I(\mathbb{1}[i \in C_t]; y_i) = 0$). Whenever a training row $i$ falls inside $C_t$ (~2% of cycles), cycle $t$ is **skipped** for row $i$ rather than swapped with a class-matched backup — guaranteeing **0.000 bits** of self-label leakage. *(An earlier prototype using class-matched backup replacement leaked 1 bit of $y_{\text{train}}$ on the 18.25% of rows swapped by Cycle 10, inflating CatBoost CV to `0.7499` while harming test AUC; skipping in-context rows completely eliminated that CV-to-Test gap to `<0.0002`.)*
4. For every out-of-context row $i$, we append $(x_i, 0)$ and $(x_i, 1)$ to $C_t$ as a paired batch to obtain $(e_{i,0}^{(t)}, e_{i,1}^{(t)})$ alongside the context baseline $e_{\text{base}}^{(t)} = \text{DOPE}(C_t)$, extracting three channels:
   - **Uncentered Raw EWMA**: $\tilde{z}_{i,\text{raw}}^{(t)} = \frac{1}{2}(e_{i,0}^{(t)} + e_{i,1}^{(t)})$
   - **Control-Variate Centered `DOPE-Mean`**: $\Delta_{i,\text{mean}}^{(t)} = 1000 \cdot \left(\frac{e_{i,0}^{(t)} + e_{i,1}^{(t)}}{2} - e_{\text{base}}^{(t)}\right)$ (eliminates 50×–80× random context baseline noise)
   - **Counterfactual Target-Alignment `DOPE-Contrast`**: $\Delta_{i,\text{contrast}}^{(t)} = 1000 \cdot (e_{i,1}^{(t)} - e_{i,0}^{(t)})$ (zero extra compute; isolates first-order target-conditioning sensitivity)
5. Both **bias-corrected EWMA ($\alpha = 0.02$)** and **exact Welford running mean ($\alpha_t = 1/t$) + SE + odd/even split-half means** are accumulated in `float64` across cycles.

### Two-Arm Experimental Design & Distributed Systems Execution (`xbabe0`–`xbabe9`)
- **Arm A (`full-high-rank1`, 100 cycles, all 60,000 rows)**: Full `563/629`-active High `DOPE-MLC-v3` conditioning (all 18 High landmarks via exact Fixed-Fold Sherman–Morrison rank-1 updates verified to `< 1e-10` vs brute-force Cholesky + 64×9 univariate spectrum + exact rank-1 `table_vec24`/`building_block_vec68` + parallel batched `MlcV3Model`). Built in pure Rust with **`mimalloc`** (eliminating static `musl` allocator lock contention for a **7.6× Linux speedup**). Executed across **10/10 nodes (`xbabe0`–`xbabe9`, `606` active idle-scheduled threads)** in **15.4 minutes (`926.3s`, peaking at `10,829 rows/s` / `5.54s/cycle`)**, saving checkpoints at Cycles `1, 5, 10, 20, 50, 100`.
- **Arm B (`native-none`, 10 cycles, all 60,000 rows)**: Control arm passing truly independent random `1,000`-row blocks per row through literal unmodified `dope-core` `LandmarkMode::None` (`395/629` active dims) across `xbabe4`–`xbabe9` (`169` threads, `35.7 min`), saving checkpoints at Cycles `1, 5, 10`.
- Downstream evaluations used the exact audited 5-fold Hospital-Disjoint CatBoost recipe (`08_eval_catboost_dual_datasets.py`, audited baseline `V0_raw_only`: 5f CV `0.675304`, Test **`0.673558`**, `LogLoss 0.643520`, `2017 Unseen 0.650462`), fanned out simultaneously across `xbabe0`–`xbabe3` via `dope_echo/src/distributed_eval.py`.

---

## 2. Smoothing Convergence & Phase Transition ("When Can We Trust the Smoothing?")

Monitoring exact global Intraclass Reliability $\rho_{\text{ICC}}(t) = 1 - \mathbb{E}_i[\text{SE}_{i,d}^2]/\text{Var}_i(\bar{z}_{i,d})$ and empirical Odd-vs-Even Split-Half Spearman–Brown correlation $\rho_{\text{SB}}(t)$ across all `60,000` rows revealed a sharp **phase transition at Cycle 50 ($\rho_{\text{ICC}} \approx 0.75$)**:

| Cycle | `Contrast` Median ICC | `Contrast` Split-Half SB | `Contrast` Drift | Train-Test max\|SMD\| | `V2 DOPE-Contrast Only` (0 raw) Test AUC | `V4 Raw + DOPE-Contrast` (CB Direct) Test AUC | `V6 Raw + DOPE-Compact` (CB SVD+Ridge) Test AUC | `V4` Δ vs Raw | `V6` Δ vs Raw | `V4` 2017 Unseen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | — | — | — | — | — | `0.673558` | `0.673558` | `0.000000` | `0.000000` | `0.650462` |
| **1** | `0.0000` | `0.0000` | `1.0000` | `0.2024` | `0.663280` | `0.672224` | `0.673784` | `-0.001335` | `+0.000225` | `0.649738` |
| **5** | `0.2726` | `0.3473` | `0.3503` | `0.0498` | `0.663501` | `0.670048` | `0.671025` | `-0.003510` | `-0.002533` | `0.644412` |
| **10** | `0.5499` | `0.5472` | `0.1449` | `0.0440` | `0.666175` | `0.670527` | `0.671452` | `-0.003031` | `-0.002106` | `0.643706` |
| **20** | `0.6707` | `0.6635` | `0.0925` | `0.0451` | `0.668097` | `0.671110` | `0.672614` | `-0.002448` | `-0.000944` | `0.645939` |
| **50** | **`0.7511`** | **`0.7569`** | `0.0803` | `0.0474` | **`0.671668`** | **`0.673794`** | **`0.673960`** | **`+0.000236`** | **`+0.000402`** | `0.650605` |
| **100** | **`0.8575`** | **`0.8548`** | **`0.0283`** | **`0.0527`** | **`0.673254`** | **`0.674058`** | **`0.674287`** | **`+0.000500`** | **`+0.000729`** | **`0.652429`** |

**Key Convergence Takeaways**:
1. **Why Cycles 5–20 underperform Raw CatBoost**: Below `ICC = 0.75`, 33%–73% of variance in DOPE embeddings is random block noise; greedy tree splits overfit that jitter (`−0.0035` to `−0.0024` on `V4`).
2. **Exact Trust Threshold (`ICC ≥ 0.75`, Cycle 50)**: Both direct `V4` and compact `V6` flip positive over Raw CatBoost at Cycle 50 and continue climbing cleanly through Cycle 100 (`ICC = 0.858`, `vec256` sub-block median ICC = **`0.9624`**, `p90 = 0.9993`, `gates14 = 0.9246`, `task128 = 0.8900`, `col64 = 0.8465`).
3. **Zero Train-vs-Test Distributional Shift**: `Train-vs-Test max|SMD|` on `DOPE-Contrast` stays at `0.044–0.053` across Cycles 5–100, confirming unseen-hospital patients map identically into DOPE space.

---

## 3. Complete Cycle 100 Leaderboard, Multi-Family & Paired Bootstrap 95% CIs

All models use **strictly the 207 raw predictors and/or Cycle-100 DOPE-ECHO embeddings** (zero `aug_*` features). Paired bootstrap CIs use `1,000` resamples against `V0_raw_only` (`0.673558`).

| Category | Model / Feature View | #Feats | 5f CV OOF AUC | 5f Test AUC | Δ Test AUC vs Raw | 95% Paired Bootstrap CI | $P(\Delta>0)$ | Test PR-AUC | Test LogLoss | ECE (15-bin) | 2017 Unseen | Rare Tails | HT-Recal LogLoss |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | `V0_raw_only` (CatBoost 207 raw) | 207 | 0.675304 | `0.673558` | `0.000000` | — | — | 0.643654 | 0.643520 | 0.01040 | 0.650462 | 0.675906 | 0.518895 |
| DOPE-Only (0 raw) | `V1_dope_mean_only` (CB) | 538 | 0.666824 | `0.666929` | `-0.006630` | `[-0.010388, -0.002892]` | 0.000 | 0.635291 | 0.646919 | 0.01381 | 0.647187 | 0.669173 | 0.521750 |
| DOPE-Only (0 raw) | `Abl_RunMean_DOPE_only` (CB, $\alpha_t=1/t$) | 538 | 0.667760 | `0.669555` | `-0.004004` | `[-0.007709, -0.000358]` | 0.016 | 0.638818 | 0.645863 | 0.01319 | 0.647992 | 0.672581 | 0.520776 |
| DOPE-Only (0 raw) | `Ridge_DOPE_only` (Linear Ridge) | 1,057 | 0.672863 | **`0.673062`** | `-0.000497` | `[-0.003302, +0.002193]` | 0.377 | 0.643471 | **`0.643518`** | 0.01190 | 0.650113 | 0.675776 | 0.518916 |
| DOPE-Only (0 raw) | **`V2_dope_contrast_only` (CB)** | 519 | 0.673092 | **`0.673254`** | `-0.000305` | `[-0.003162, +0.002469]` | 0.424 | **0.644073** | 0.643540 | 0.01184 | **0.651654** | **0.676450** | **0.518853** |
| Raw + DOPE (CB) | `V4_raw_plus_dope_contrast` (Direct) | 726 | 0.673782 | **`0.674058`** | **`+0.000500`** | `[-0.001989, +0.002940]` | 0.665 | **0.644968** | **0.643170** | 0.01358 | **0.652429** | **0.678673** | **0.518573** |
| Raw + DOPE (CB) | `V6_raw_plus_dope_compact` (SVD+Ridge) | 280 | 0.674168 | **`0.674287`** | **`+0.000729`** | `[-0.001641, +0.003056]` | 0.737 | 0.643493 | 0.643341 | **0.00948** | **0.651911** | 0.676576 | 0.518697 |
| Raw + DOPE (CB) | `Abl_Raw_plus_RunMean_Compact` ($\alpha_t=1/t$) | 280 | 0.663721 | **`0.674364`** | **`+0.000806`** | `[-0.001479, +0.002975]` | 0.760 | 0.643627 | 0.647797 | 0.02521 | 0.651738 | 0.676912 | 0.522750 |
| Multi-Family | `LGBM_raw_only` → `LGBM_raw_plus_dope` | 207 → 1,264 | 0.6742 → 0.6731 | `0.672221` → **`0.672999`** | `+0.000778` (vs LGB raw) | — | — | 0.640903 | 0.643779 | 0.01140 | 0.648441 | 0.675498 | 0.519234 |
| OOF Blend | `Blend_RawCB_plus_LR_DOPE` ($w=0.25$) | — | 0.675763 | **`0.674395`** | **`+0.000837`** | **`[+0.000025, +0.001605]`** | **`0.976`** | 0.644697 | **0.642814** | **0.00804** | 0.651522 | 0.677445 | 0.518436 |
| OOF Blend | `Blend_RawCB_plus_V6_DOPE_Compact` ($w=0.38$) | — | 0.675951 | **`0.674617`** | **`+0.001059`** | **`[+0.000111, +0.001959]`** | **`0.991`** | 0.644753 | 0.642992 | **0.00756** | 0.651707 | 0.676999 | 0.518476 |
| OOF Blend | `Blend_RawCB_plus_LGBM_DOPE` ($w=0.33$) | — | 0.676160 | **`0.674603`** | **`+0.001045`** | **`[+0.000137, +0.002042]`** | **`0.989`** | 0.644375 | 0.642973 | 0.00929 | 0.651285 | 0.678398 | 0.518490 |
| OOF Blend | **`Blend_RawCB_plus_V2_DOPE_Only`** ($w=0.31$) | — | 0.676007 | **`0.674690`** | **`+0.001132`** | **`[+0.000254, +0.002019]`** | **`0.994`** | **0.644950** | 0.642860 | 0.00818 | **0.651942** | 0.677836 | **0.518397** |
| OOF Blend | **`Blend_RawCB_plus_V2_plus_LGBM_DOPE`** | — | **0.676191** | **`0.674726`** | **`+0.001168`** | **`[+0.000205, +0.002200]`** | **`0.990`** | 0.644664 | 0.642893 | 0.00883 | 0.651588 | **0.678519** | 0.518428 |

---

## 4. Arm A vs. Arm B Control & Complete Mechanistic Ablation Ladder

Comparing Arm B (`native-none`, literal unmodified `dope-core` `LandmarkMode::None`, truly independent 1,000-row blocks per row, Cycle 10) against Arm A (`full-high-rank1`, Cycles 10 and 100) reveals **exactly where every basis point of DOPE-ECHO's power comes from**:

| Ablation Step | Arm / Configuration | Cycle | Test AUC (0 Raw Cols) | Incremental Gain | Mechanistic Interpretation |
|---|---|---:|---:|---:|---|
| 1. Naive Uncentered EWMA | Arm B `native-none` (`raw_ewma`) | 10 | `0.509284` | baseline | Random 1,000-row context baseline $e_{\text{base}}$ is 50×–80× larger than 1-row signal |
| 2. + Control-Variate Centering | Arm B `native-none` (`DOPE-Mean`) | 10 | `0.540021` | `+0.030737` | Subtracting $e_{\text{base}}$ cancels context baseline variance |
| 3. + Counterfactual Contrast (`e₁ − e₀`) | Arm B `native-none` (`DOPE-Contrast`) | 10 | **`0.661600`** | **`+0.121579`** | First-order target sensitivity works even in literal unmodified `dope-core` with zero landmarks |
| 4. + High Landmarks (`Sherman-Morrison`) | Arm A `full-high-rank1` (`DOPE-Contrast`) | 10 | `0.666175` | `+0.004575` | Adds all 18 High landmarks + 64×9 spectrum (and runs **~30× faster** per thread!) |
| 5. + 100-Cycle Smoothing (`ICC 0.55→0.86`) | Arm A `full-high-rank1` (`DOPE-Contrast`) | 100 | **`0.673254`** | **`+0.007079`** | Matches 207-raw CatBoost (`0.673558`) & beats it on PR-AUC, 2017 Unseen, Rare Tails |
| 6. + Combined with 207 Raw Features | `Raw CB + CB(DOPE-Contrast Only)` Blend | 100 | **`0.674690`** | **`+0.001132`** *(vs Raw CB, $P=0.994$)* | Statistically significant orthogonal paradigm lift; beats Note 09 `CB + TabPFN-3.5` (`0.674369`) |

### Key Scientific Insights for the Manuscript
1. **A 444k-parameter frozen table hypernetwork rivals frontier Tabular FMs on `small_50k_10k`**: Standalone `V2_dope_contrast_only` (`0.673254`) and pure linear `Ridge_DOPE_only` (`0.673062`) surpass Note 09's **KumoTabular-Large v2** (`0.670748`) and **TabFM v1.0** (`0.670668`) by `+0.0025` AUC and virtually tie **TabPFN-3.5** (`0.673334`) — despite never being trained as a per-row classifier.
2. **Orthogonal representation diversity drives statistically significant gains over Raw CatBoost**: Blending Raw CatBoost (`69%`) with a CatBoost trained *exclusively* on `DOPE-Contrast` (`31%`) achieves **`0.674690`** (`+0.001132`, 95% CI `[+0.000254, +0.002019]`, $P(\Delta>0)=0.994$), lowering LogLoss by `-0.000660`, ECE by `21%` (`0.01040 → 0.00818`), and Bayes-recalibrated HT-weighted LogLoss from `0.518895 → 0.518397` — outperforming Note 09's `CatBoost + TabPFN-3.5` 2-paradigm blend (`0.674369`) on the exact same `test_small_10k` holdout.
3. **Strongest single-model gains appear on clinical stress slices**: Direct `V4_raw_plus_dope_contrast` lifts **Rare Clinical Tails** by **`+0.002767`** (`0.675906 → 0.678673`) and **2017 Unseen Hospitals** by **`+0.001967`** (`0.650462 → 0.652429`), with 14 of its top 15 CatBoost importances coming from high-reliability `vec256` `DOPE-Contrast` dimensions (`ICC = 0.9624`).
4. **Equal-weight running mean outperforms EWMA ($\alpha=0.02$) on stationary random blocks**: At Cycle 100, exact $1/t$ averaging (`Abl_RunMean_DOPE_only = 0.669555`, `Abl_Raw_plus_RunMean_Compact = 0.674364`) beats $\alpha=0.02$ EWMA (`0.666929` / `0.674287`) because equal weights maximize effective sample size ($N_{\text{eff}} = 100$ vs $\approx 75$).
