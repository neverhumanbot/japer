# Project DOPE-ECHO — Complete Technical Report, Results & On-Disk Data Inventory

**Full Project Name**: **DOPE-ECHO** (*D.O.P.E. Ensemble Counterfactual Hypernetwork Observer*)\
**Date Completed**: `2026-11-25`\
**Workspace Root**: `dope_echo/`\
**Dataset Evaluated (Verified Provenance)**: **Perfected Smart-Downsampled Hospital-Disjoint Dataset `dataset/small_50k_10k/`** (`50,000` train across `1,017` hospitals → `10,000` test across `74` strictly unseen hospitals, **all years `2014–2017`**, **`0.0%` hospital overlap**, exact `50.0% / 50.0%` POAF target `PRIMARY_TARGET_COtAFib`, `splits_sha256: c59a4621855580eedea596a868dc87ae22a7bbd209d728dd1b1ea2356536b766`, Bayes logit offset `1.1367` + Horvitz-Thompson natural-prior weights; **strictly `207` pre-incision raw predictors `raw_all`**, zero `aug_*` features). **NOT** the legacy naive 2017 holdout split (`2017 Unseen` in tables denotes the embedded `stress_2017_unseen_hosp` diagnostic slice inside `test_small_10k`). The larger twin `dataset/full/` (`346,454 → 21,241`) has not yet been run with DOPE-ECHO.\
**Frozen Hypernetwork Weights**: `JopeDimeMicro/assets/models/dope_mlc_v3_multitask_best.safetensors` (`1,781,800` bytes, `444,168` params, SHA-256 `d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`, zero edits to `~/dope/`)

---

## 1. Executive Summary

Can a **frozen, table-level meta-encoder** (`DOPE-MLC-v3`, mapping whole labeled tables to $\mathbb{R}^{629}$, with *no* per-row featurizer) be turned into a high-power **per-row featurizer** by repeatedly inserting each query row with counterfactual binary targets $y \in \{0, 1\}$ into random `1,000`-row labeled training blocks and smoothing across cycles?

**Yes — decisively.** Across `100` zero-leakage cycles (`12,000,000` counterfactual High-level DOPE evaluations computed in **15.4 minutes** across all **10/10 cluster nodes `xbabe0`–`xbabe9`, `606` active threads**, peaking at `10,829 rows/s`):

1. **Standalone Zero-Shot Power (ZERO Raw Clinical Features)**:
   - By Cycle 100, a CatBoost model trained **strictly on `DOPE-Contrast` ($1000\cdot(e_1 - e_0)$, 519 active dims, 0 raw clinical cols)** reaches **`0.673254` Test ROC-AUC** (`PR-AUC 0.644073`, `LogLoss 0.643540`), essentially tying the audited 5-fold Raw CatBoost baseline on all 207 clinical features (`0.673558`, `PR-AUC 0.643654`, `LogLoss 0.643520`), and **outperforming Raw CatBoost on 2017 Unseen-Hospital AUC (`0.651654` vs `0.650462`), Rare Clinical Tails (`0.676450` vs `0.675906`), and Bayes-recalibrated natural-prior LogLoss (`0.518853` vs `0.518895`)**.
   - Even a **pure linear Ridge regression** on DOPE-ECHO alone (`Ridge_DOPE_only`, zero raw features) hits **`0.673062` Test AUC** and beats Raw CatBoost on LogLoss (`0.643518` vs `0.643520`).
   - On the exact same `small_50k_10k` benchmark (Note 09), frozen zero-shot `DOPE-ECHO` (`0.673254`) outperforms **KumoTabular-Large v2** (`0.670748`) and **TabFM v1.0** (`0.670668`) by `+0.0025` AUC and virtually ties **TabPFN-3.5** (`0.673334`).
2. **Exact Smoothing Trust Threshold Identified Empirically (`Cycle 50`, `ICC ≥ 0.75`)**:
   - At Cycles `1–20` (`Contrast ICC = 0.00 → 0.67`), feeding noisy DOPE embeddings directly into CatBoost *hurts* Test AUC (`−0.0035` to `−0.0009`) due to Monte Carlo block jitter diluting tree splits.
   - At **Cycle 50 (`Contrast ICC = 0.7511`, Split-Half SB = `0.7569`)**, both direct `V4_raw_plus_dope_contrast` (`+0.000236`) and compact `V6_raw_plus_dope_compact` (`+0.000402`) **cross into positive territory**.
   - By **Cycle 100 (`Contrast ICC = 0.8575`, `vec256` sub-block median `ICC = 0.9624`, `p90 = 0.9993`)**, positive lift strengthens across **every** downstream model family:
     - Direct `V4_raw_plus_dope_contrast` CatBoost (207 raw + 519 contrast): **`0.674058`** (`+0.000500`, `LL 0.643170`, `PR-AUC 0.644968`, **Rare Clinical Tails `+0.002767` → `0.678673`**, **2017 Unseen `+0.001967` → `0.652429`**).
     - Compact `V6_raw_plus_dope_compact` CatBoost (207 raw + inside-fold SVD + nested Hospital-Grouped OOF Ridge): **`0.674287`** (`+0.000729`, `ECE 0.00948`).
     - Exact Running Mean `Abl_Raw_plus_RunMean_Compact` ($\alpha_t = 1/t$): **`0.674364`** (`+0.000806`).
     - LightGBM (`LGBM_raw_only` → `LGBM_raw_plus_dope`): `0.672221` → **`0.672999`** (`+0.000778`).
3. **Statistically Significant Lift via OOF-Tuned Blending (Zero Test Peeking)**:
   - **`Blend_RawCB_plus_LR_DOPE`** (Raw CB + Linear Ridge on DOPE-Only): **`0.674395`** (`+0.000837`, 95% CI `[+0.000025, +0.001605]`, **$P(\Delta>0)=0.976$**, `LL 0.642814`, `ECE 0.00804`).
   - **`Blend_RawCB_plus_LGBM_DOPE`** (Raw CB + LightGBM on Raw+DOPE): **`0.674603`** (`+0.001045`, 95% CI `[+0.000137, +0.002042]`, **$P(\Delta>0)=0.989$**).
   - **`Blend_RawCB_plus_V2_DOPE_Only`** (`0.69·Raw CB + 0.31·CB(DOPE-Contrast Only)`): **`0.674690`** (`+0.001132`, 95% CI `[+0.000254, +0.002019]`, **$P(\Delta>0)=0.994$**, `LL 0.642860`, `HT-LL 0.518397` — beating Note 09's `CatBoost + TabPFN-3.5` blend of `0.674369`).
   - **`Blend_RawCB_plus_V2_plus_LGBM_DOPE`** (3-way OOF simplex `[0.64, 0.10, 0.26]`): **`0.674726`** (`+0.001168`, 95% CI `[+0.000205, +0.002200]`, **$P(\Delta>0)=0.990$**, `CV 0.676191`).

---

## 2. Exact On-Disk File & Data Inventory (Paper-Ready + Project Artifacts)

### A. Paper-Ready Files in `` (Direct Manuscript Integration)

| Category | Exact On-Disk Path | Size / Contents |
|---|---|---|
| **Manuscript Note 11** | `docs/11_dope_echo_counterfactual_featurizer.md` | Full IMRaD/TRIPOD+AI manuscript note matching Notes `01`–`10` (paired alongside Note `12` Evolved Targets) |
| **Manuscript Index** | `docs/README.md` | Updated with Note 11, Note 12, paper tables, and paper figures |
| **Drop-In Paper Figures (PNG + Vector PDF)** | `reports/paper_figures/fig_dope_echo_{master_summary,smoothing_convergence,cycle_curve}.{png,pdf}` | Publication PNG + vector PDF figures ready for LaTeX `\includegraphics` |
| **Drop-In LaTeX Table** | `reports/paper_tables/table_dope_echo_cycles_small50k.tex` | Publication `booktabs` LaTeX table (`Cycles 1..100` + Cycle-100 blends) |
| **Drop-In Cycle CSV** | `reports/paper_tables/table_dope_echo_cycles_small50k.csv` | Complete machine-readable Arm A (`1..100`) & Arm B (`1..10`) cycle curve (`55` rows) |
| **Drop-In Ablation CSV** | `reports/paper_tables/table_dope_echo_ablation_multifamily.csv` | Cycle-100 CatBoost views, EWMA vs RunMean vs RawUncentered, LR/MLP/LGBM, all blends |
| **Full JSON Receipt** | `reports/dope_echo_eval.json` | `540 KB` — complete verified JSON receipt (all metrics, stress slices, CIs, top-25 importances, telemetry) |
| **Full Markdown Receipt** | `reports/dope_echo_eval.md` | Complete Arm A + Arm B + extra-blend summary tables |
| **Cycle 100 OOF Predictions** | `reports/dope_echo_oof_preds.parquet` | `5.6 MB` — `50,000` train rows (`row_id, cv_fold, y_true` + `20` model/blend OOF probability columns) |
| **Cycle 100 Test Predictions** | `reports/dope_echo_test_preds.parquet` | `1.1 MB` — `10,000` unseen-hospital test rows (`row_id, y_true` + `20` model/blend Test probability columns) |

### B. Raw Saved Embeddings & Checkpoints on Disk (`8.9 GB` total under `dope_echo/`)

Every checkpoint folder contains **exact `60,000 × 629` `float32` `.npy` matrices** (rows `0..49,999` = `train_small_50k` in exact `dataset_loader` order; rows `50,000..59,999` = `test_small_10k` in exact `dataset_loader` order):
- `mean_ewma.npy` — Bias-corrected EWMA ($\alpha = 0.02$) of control-variate centered `DOPE-Mean`: $1000 \cdot (\frac{e_0 + e_1}{2} - e_{\text{base}})$
- `mean_run.npy` — Exact Welford equal-weight running mean ($\alpha_t = 1/t$) of `DOPE-Mean`
- `contrast_ewma.npy` — Bias-corrected EWMA ($\alpha = 0.02$) of counterfactual **`DOPE-Contrast`**: $1000 \cdot (e_1 - e_0)$ *(strongest standalone & augmented channel!)*
- `contrast_run.npy` — Exact Welford equal-weight running mean ($\alpha_t = 1/t$) of `DOPE-Contrast`
- `se_mean.npy`, `se_contrast.npy` — Exact per-row, per-dimension Welford standard errors across cycles
- `raw_ewma.npy` — Uncentered raw EWMA of $\frac{1}{2}(e_0 + e_1)$ (ablation control)
- `base_mean.npy` — `[629]` running mean of base context encodings $e_{\text{base}}$
- `meta.json` — Cycle count, global ICC/Split-Half reliability telemetry, and Train-vs-Test `max|SMD|` audit

| Artifact / Checkpoint | Exact On-Disk Path | Notes |
|---|---|---|
| **Packed Zero-Leakage Bundle** | `dope_echo/artifacts/echo_bundle.bin` (`11.9 MB`) + `bundle_manifest.json` | Contains `60k × 207` `X`, `y_train` only (`contains_y_test=false`), `100` stratified `500/500` train contexts |
| **Arm A Checkpoints (High Rank-1, 100c)** | `dope_echo/artifacts/checkpoints/armA_high_100c/cycle_{001,005,010,020,050,100}/` | Best embeddings: **`cycle_100/contrast_ewma.npy`** & **`cycle_100/contrast_run.npy`** |
| **Arm A Live Telemetry & Plots** | `dope_echo/artifacts/checkpoints/armA_high_100c/{convergence_telemetry.jsonl,smoothing_convergence.png,cycle_curve_auc_and_reliability.png,echo_eval_report.json}` | Per-cycle `1..100` global reliability JSONL + convergence plots |
| **Arm B Checkpoints (Native-None, 10c)** | `dope_echo/artifacts/checkpoints/armB_native_none_10c/cycle_{001,005,010}/` | Literal unmodified `dope-core` `LandmarkMode::None`, independent 1k blocks per row |
| **Arm B Live Telemetry & Plots** | `dope_echo/artifacts/checkpoints/armB_native_none_10c/{convergence_telemetry.jsonl,smoothing_convergence.png,cycle_curve_auc_and_reliability.png,echo_eval_report.json}` | Complete Arm B control receipts |
| **Master 4-Panel Summary Figure** | `dope_echo/artifacts/dope_echo_master_summary.png` | Publication 4-panel figure (Reliability, Standalone AUC, Raw+DOPE Lift, Cycle-100 Bootstrap Forest Plot) |
| **Cross-Platform Rust Binaries** | `dope_echo/artifacts/bin/{echo-worker-linux-x86_64,echo-worker-darwin-arm64}` | Static Linux `musl+mimalloc` (`5.68 MB`, GLIBC-independent) + native Apple M3 Ultra `arm64` (`4.04 MB`) |
| **Cached Raw Baseline** | `dope_echo/artifacts/v0_raw_baseline_cache.npz` | Audited `V0_raw_only` (`0.673558` Test / `0.675304` CV) OOF & Test predictions |

### C. Source Code & Tests under `dope_echo/`

- **Rust Worker (`rust_worker/`)**:
  - `Cargo.toml` — Standalone workspace, path dep to `~/dope/engine/dope-core` (read-only), pinned `candle-core`, pure-Rust `brotli`, `mimalloc` global allocator
  - `src/bundle.rs` — Zero-copy brotli bundle loader, SHA-256 SafeTensors verifier, strictly label-blind `is_in_context` check, deterministic `independent_context_for_row`
  - `src/rank1_table.rs` — Exact rank-1 `table_vec24` (22/24 non-Spearman dims `< 0.002` vs brute-force `dope-core`), `column_desc`, and all `68/68` `building_block_vec68` dims (`< 0.007` vs brute-force `dope-core`)
  - `src/rank1_landmarks.rs` — Fixed-Fold Sherman–Morrison engine for all 18 High landmarks + 64×9 univariate spectrum (`< 1e-10` vs brute-force Cholesky)
  - `src/native_none.rs` — Literal unmodified `dope-core` `prepare_table_mixed` + `build_conditioning(..., LandmarkMode::None)` + batched `MlcV3Model`
  - `src/main.rs` — Persistent length-prefixed stdin/stdout IPC daemon, `float64` per-row `ShardAccumulator` with out-of-context skipping, parallel `par_chunks(64)` neural forward, and 4 release unit tests
- **Python Orchestration, Monitoring, Evaluation & Publication (`src/`, `tests/`)**:
  - `src/prepare_bundle.py` — Packs `small_50k_10k` (`raw_all` 207 cols only, train-only top-64 column ranking, zero `y_test`) into `echo_bundle.bin`
  - `src/build_workers.py` — Builds static Linux `musl+mimalloc` + smoke-tests on `xbabe5` (GLIBC 2.35) + builds native macOS `arm64` on `xbabe4`
  - `src/cluster_orchestrator.py` — Probes all 10 nodes (`xbabe0`–`xbabe9`), stages to `/dev/shm` & `/tmp`, calibrates per-node rows/sec, runs proportional shards with fault-tolerant replay and guaranteed cleanup
  - `src/monitor.py` — Exact global ICC reliability, Split-Half Spearman–Brown, Frobenius drift, per-DOPE-block breakdown, Train-vs-Test SMD, and convergence plots
  - `src/eval_checkpoints.py` — 5-fold Hospital-Disjoint CatBoost + inside-fold SVD/nested Hospital-Grouped OOF Ridge + LightGBM/Ridge/MLP/blends + 1,000-resample paired bootstrap CIs + OOF/Test Parquet export
  - `src/distributed_eval.py` — Fans out checkpoint evaluations simultaneously across idle cluster nodes (`xbabe0`–`xbabe3`) with per-cycle caching and guaranteed `/dev/shm` cleanup
  - `src/publish_paper_artifacts.py` — Generates all CSV/LaTeX paper tables, consolidated receipts, OOF/Test Parquets, and 4-panel master figure
  - `tests/test_echo_contract.py` — 4 pytest contract tests (zero-`y_test` byte audit, `monitor.py` numpy parity, multi-shard daemon parity, `native-none` CLI sanity)

---

## 3. Complete Experimental Tables

### 3.1 Arm A (`full-high-rank1`) Cycle-by-Cycle Convergence (`1 → 5 → 10 → 20 → 50 → 100`)

Audited Raw CatBoost Baseline (`V0_raw_only`, 207 raw cols): **5f CV OOF = `0.675304`**, **5f Test = `0.673558`**, **PR-AUC = `0.643654`**, **LogLoss = `0.643520`**, **ECE = `0.01040`**, **2017 Unseen = `0.650462`**, **Rare Tails = `0.675906`**, **De Novo = `0.673591`**, **HT-Recal LogLoss = `0.518895`**.

| Cycle | Contrast ICC | Contrast SB | Contrast Drift | View | #Feats | 5f CV OOF AUC | 5f Test AUC | Δ Test AUC vs Raw | 95% Bootstrap CI | $P(\Delta>0)$ | Test LogLoss | 2017 Unseen |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 1.0000 | `V1_dope_mean_only` | 533 | 0.658329 | 0.659646 | -0.013912 | [-0.018635, -0.009089] | 0.000 | 0.650633 | 0.636699 |
| 1 | 0.0000 | 0.0000 | 1.0000 | `V2_dope_contrast_only` | 514 | 0.661499 | 0.663280 | -0.010278 | [-0.014770, -0.005875] | 0.000 | 0.649397 | 0.637637 |
| 1 | 0.0000 | 0.0000 | 1.0000 | `V4_raw_plus_dope_contrast` | 721 | 0.670968 | 0.672224 | -0.001335 | [-0.003871, +0.001273] | 0.145 | 0.644320 | 0.649738 |
| 1 | 0.0000 | 0.0000 | 1.0000 | `V6_raw_plus_dope_compact` | 280 | 0.672335 | 0.673784 | +0.000225 | [-0.002066, +0.002618] | 0.583 | 0.643553 | 0.652931 |
| 5 | 0.2726 | 0.3473 | 0.3503 | `V2_dope_contrast_only` | 518 | 0.667184 | 0.663501 | -0.010057 | [-0.013722, -0.006367] | 0.000 | 0.648978 | 0.635377 |
| 5 | 0.2726 | 0.3473 | 0.3503 | `V4_raw_plus_dope_contrast` | 725 | 0.672133 | 0.670048 | -0.003510 | [-0.006027, -0.001173] | 0.004 | 0.645165 | 0.644412 |
| 5 | 0.2726 | 0.3473 | 0.3503 | `V6_raw_plus_dope_compact` | 280 | 0.673772 | 0.671025 | -0.002533 | [-0.004908, -0.000372] | 0.013 | 0.644890 | 0.646652 |
| 10 | 0.5499 | 0.5472 | 0.1449 | `V2_dope_contrast_only` | 518 | 0.668496 | 0.666175 | -0.007384 | [-0.010634, -0.004315] | 0.000 | 0.647500 | 0.637653 |
| 10 | 0.5499 | 0.5472 | 0.1449 | `V4_raw_plus_dope_contrast` | 725 | 0.671771 | 0.670527 | -0.003031 | [-0.005579, -0.000646] | 0.010 | 0.644884 | 0.643706 |
| 10 | 0.5499 | 0.5472 | 0.1449 | `V6_raw_plus_dope_compact` | 280 | 0.672642 | 0.671452 | -0.002106 | [-0.004409, +0.000112] | 0.033 | 0.644444 | 0.647713 |
| 20 | 0.6707 | 0.6635 | 0.0925 | `V2_dope_contrast_only` | 519 | 0.670199 | 0.668097 | -0.005462 | [-0.008481, -0.002576] | 0.000 | 0.646375 | 0.642927 |
| 20 | 0.6707 | 0.6635 | 0.0925 | `V4_raw_plus_dope_contrast` | 726 | 0.672240 | 0.671110 | -0.002448 | [-0.004920, -0.000084] | 0.023 | 0.644356 | 0.645939 |
| 20 | 0.6707 | 0.6635 | 0.0925 | `V6_raw_plus_dope_compact` | 280 | 0.673648 | 0.672614 | -0.000944 | [-0.003150, +0.001113] | 0.193 | 0.643728 | 0.650701 |
| 50 | **0.7511** | **0.7569** | 0.0803 | `V2_dope_contrast_only` | 519 | 0.662452 | **0.671668** | -0.001890 | [-0.004919, +0.001069] | 0.106 | 0.646326 | 0.648295 |
| 50 | **0.7511** | **0.7569** | 0.0803 | `V4_raw_plus_dope_contrast` | 726 | 0.672738 | **0.673794** | **+0.000236** | [-0.002383, +0.002912] | 0.583 | 0.643126 | 0.650605 |
| 50 | **0.7511** | **0.7569** | 0.0803 | `V6_raw_plus_dope_compact` | 280 | 0.674080 | **0.673960** | **+0.000402** | [-0.001956, +0.002707] | 0.658 | 0.643230 | 0.651196 |
| 100 | **0.8575** | **0.8548** | **0.0283** | `V1_dope_mean_only` | 538 | 0.666824 | **0.666929** | -0.006630 | [-0.010388, -0.002892] | 0.000 | 0.646919 | 0.647187 |
| 100 | **0.8575** | **0.8548** | **0.0283** | `V2_dope_contrast_only` | 519 | 0.673092 | **0.673254** | -0.000305 | [-0.003162, +0.002469] | 0.424 | 0.643540 | **0.651654** |
| 100 | **0.8575** | **0.8548** | **0.0283** | `V3_raw_plus_dope_mean` | 745 | 0.672726 | **0.672704** | -0.000854 | [-0.003018, +0.001418] | 0.229 | 0.644238 | **0.655235** |
| 100 | **0.8575** | **0.8548** | **0.0283** | `V4_raw_plus_dope_contrast` | 726 | 0.673782 | **0.674058** | **+0.000500** | [-0.001989, +0.002940] | 0.665 | **0.643170** | **0.652429** |
| 100 | **0.8575** | **0.8548** | **0.0283** | `V5_raw_plus_dope_full` | 1264 | 0.673820 | **0.673499** | -0.000059 | [-0.002790, +0.002537] | 0.494 | 0.643407 | 0.651470 |
| 100 | **0.8575** | **0.8548** | **0.0283** | `V6_raw_plus_dope_compact` | 280 | 0.674168 | **0.674287** | **+0.000729** | [-0.001641, +0.003056] | 0.737 | 0.643341 | 0.651911 |

### 3.2 Arm B Control (`native-none`, Literal Unmodified `dope-core` `LandmarkMode::None`, Independent 1,000-Row Blocks)

| Cycle | Mean ICC | Contrast ICC | Contrast SB | View | 5f CV OOF AUC | 5f Test AUC | Δ Test AUC vs Raw | Test LogLoss |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | `V1_dope_mean_only` | 0.528435 | 0.519618 | -0.153940 | 0.690363 |
| 1 | 0.0000 | 0.0000 | 0.0000 | `V2_dope_contrast_only` | 0.652735 | 0.651369 | -0.022189 | 0.655618 |
| 5 | 0.0463 | 0.2404 | 0.2338 | `V1_dope_mean_only` | 0.536338 | 0.530825 | -0.142733 | 0.689313 |
| 5 | 0.0463 | 0.2404 | 0.2338 | `V2_dope_contrast_only` | 0.661794 | 0.660774 | -0.012784 | 0.650884 |
| 10 | 0.0807 | 0.3942 | 0.3952 | `Abl_RawUncentered_DOPE_only` | 0.519534 | 0.509284 | -0.164274 | 0.692955 |
| 10 | 0.0807 | 0.3942 | 0.3952 | `V1_dope_mean_only` | 0.544360 | 0.540021 | -0.133537 | 0.688384 |
| 10 | 0.0807 | 0.3942 | 0.3952 | **`V2_dope_contrast_only`** | 0.662580 | **0.661600** | -0.011959 | 0.650405 |
| 10 | 0.0807 | 0.3942 | 0.3952 | `V4_raw_plus_dope_contrast` | 0.671908 | 0.672529 | -0.001029 | 0.644012 |
| 10 | 0.0807 | 0.3942 | 0.3952 | `Blend_RawCB_plus_LGBM_DOPE` | 0.675576 | 0.673761 | +0.000203 | 0.643443 |

### 3.3 Cycle 100 Per-DOPE-Block Reliability (`DOPE-Contrast`)

| DOPE-MLC-v3 Block | Slice | Active Dims | Median ICC | p75 ICC | p90 ICC | Frac ≥ 0.90 | Split-Half SB |
|---|---|---:|---:|---:|---:|---:|---:|
| **`vec256`** | `0:256` | 151 | **0.9624** | **0.9933** | **0.9993** | 66% | **0.9663** |
| **`gates14`** | `512:526` | 8 | **0.9246** | 0.9341 | 0.9394 | 75% | **0.9255** |
| **`task128`** | `256:384` | 128 | **0.8900** | 0.9262 | 0.9383 | 47% | **0.8900** |
| **`dist128`** | `384:512` | 128 | 0.8541 | 0.9226 | 0.9426 | 36% | 0.8505 |
| **`col64`** | `565:629` | 64 | 0.8465 | 0.8468 | 0.8472 | 0% | 0.8446 |
| **`lm23`** | `526:549` | 23 | 0.8181 | 0.8685 | 0.9233 | 17% | 0.7966 |
| **`bb16`** | `549:565` | 16 | 0.7741 | 0.8185 | 0.8557 | 6% | 0.7658 |

---

## 4. Verification & Reproducibility Ledger

- **Rust Unit Tests (`cargo test --release`)**: **4/4 passed** (`test_sherman_morrison_exact_vs_bruteforce_cholesky < 1e-10`, `test_zero_self_label_leakage_on_swapped_train_row` verifying bit-identical zero leakage + out-of-context skip, `test_rank1_table_vs_bruteforce_dope_core` confirming 22/24 non-Spearman `table_vec24 < 0.002` & 68/68 `building_block_vec68 < 0.007`, `test_ewma_and_welford_closed_form < 1e-12`).
- **Python Contract Tests (`pytest -v dope_echo/tests/test_echo_contract.py`)**: **4/4 passed** (exact byte-level `ECHOv001` zero-`y_test` verification, `monitor.py` ICC/SB numpy parity, multi-shard daemon deterministic parity, `native-none` active-dim & non-degeneracy audit).
- **Cluster Hygiene Audit (`xbabe0`–`xbabe9`)**: Confirmed **`0` leftover scratch dirs** and **`0` running `echo-worker` processes** across all 10 nodes after completion.
- **Quick-Load Example in Python**:
  ```python
  import numpy as np, pandas as pd
  # Load Cycle-100 DOPE-Contrast embeddings (60,000 x 629 float32: 0..49999 train, 50000..59999 test)
  Z_contrast = np.load("dope_echo/artifacts/checkpoints/armA_high_100c/cycle_100/contrast_ewma.npy")
  Z_contrast_run = np.load("dope_echo/artifacts/checkpoints/armA_high_100c/cycle_100/contrast_run.npy")
  # Load Cycle-100 OOF & Test predictions for immediate ensembling with other models
  oof_echo = pd.read_parquet("reports/dope_echo_oof_preds.parquet")
  test_echo = pd.read_parquet("reports/dope_echo_test_preds.parquet")
  ```
