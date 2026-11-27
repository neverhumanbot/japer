# Research Notes & Publication Manuscript Index (`docs/`) — STS ACSD v2.81 POAF Study

## Executive & Technical Briefs (2-Page Summaries)

| Audience | Formatted 2-Page PDF | Markdown Companion | Focus |
| :--- | :--- | :--- | :--- |
| **Health System CEOs, CMOs & Clinical AI Leaders** | **[CEO Brief (PDF)](briefs/evotarget-poaf-ceo-brief.pdf)** | [ceo-brief.md](briefs/ceo-brief.md) | Clinical AI validation illusion, 79.2% data-scaling recovery, Dual-Paradigm Fusion (`0.67668`), trial enrichment & governance |
| **CTOs, VP Engineering & ML Architects** | **[CTO Brief (PDF)](briefs/evotarget-poaf-cto-brief.pdf)** | [cto-brief.md](briefs/cto-brief.md) | 1,000-gen zero-collapse $z^*$, strictly nested 207-raw 2×2 causal proof, `JAIN Ultra GP v2` (`+39.8* bps`) & 5-min audit CLI |

Rebuild & verify both 2-page PDFs deterministically via `bash docs/briefs/scripts/build_briefs.sh --check` ([build_briefs.sh](briefs/scripts/build_briefs.sh)).

## Multi-Node Distributed Compute Playbook (`distributed_compute/`)

- **[Playbook Overview & Decision Matrix](distributed_compute/README.md)** — 10-node LAN + Cloud topology (`601` cores, `10+` GPUs), 5-pattern selector, & 10-minute fresh-cluster spin-up checklist.
- [01 — SSH Fabric, Isolated Venvs & Ephemeral Staging](distributed_compute/01_ssh_fabric_and_bootstrap.md) · [02 — Zero-NFS SSH Pull-Queue CPU Engine](distributed_compute/02_cpu_ssh_pull_queue_cluster.md) · [03 — GPU Fan-Out, VPC NCCL DDP & Async DiLoCo](distributed_compute/03_gpu_fanout_ddp_and_diloco.md) · [04 — Co-Tenancy Guardrails & Operational Post-Mortems](distributed_compute/04_cotenancy_guardrails_and_ops.md) · [05 — `xbabe5`–`xbabe9` Pre-Spindown Audit Ledger](distributed_compute/05_xbabe5_9_spindown_audit_and_harvest.md).

## Manuscript and supporting evidence

- [Manuscript PDF](paper/main.pdf) and [LaTeX source](paper/main.tex). Build with [the Makefile](paper/Makefile).
- [POAF literature review](paper/POAF_LITERATURE_REVIEW.md), with primary results, design differences, exact searches, and [the STS screening index](paper/literature/sts-query-screen.tsv).
- [Red-team review](paper/RED_TEAM_REVIEW.md) and [hospital-resampling receipt](../reports/paper_red_team_receipt.json). Use these for the current interpretation of validation, calibration, and model selection.
- [Label-blind matched hospital split](../FEATURE_MATCHED_EVALUATION.md) and [modular dataset guide](../dataset/DATASET_README.md).
- [Paper coordination log](../paper_chat.md) and [numeric audit script](../scripts/20_audit_paper_numbers.py).

The notes below retain findings from earlier experiments. The manuscript and red-team review document the interpretation after checking outcome-aware partition selection, the nested 10k sample, test-set model ranking, and hospital-level uncertainty.

The full-scale ensembles, counterfactual DOPE-ECHO embeddings, and JAIN and elastic-net arms remain available with their methods and results. The DOPE-ECHO experiments use the nested 10k sample. [Full-scale ensemble results](10_championship_multiparadigm_ensemble.md) · [DOPE-ECHO experiments](11_dope_echo_counterfactual_featurizer.md) · [Additional continuous-target arms](12_evolved_continuous_targets_and_multiparadigm_lift.md) · [Complete model leaderboard](../reports/evolved_targets_eval.json)

## Manuscript Section Map (IMRaD + TRIPOD+AI)

| Note File | Target Paper Section | Core Contribution / Finding | Status |
|---|---|---|---|
| [01_cohort_source_and_v281_schema.md](01_cohort_source_and_v281_schema.md) | *Methods: Cohort & Data Source* | 368,132-record STS ACSD v2.81 provenance (SHA-256), half-year H2-2014/H1-2017 boundaries, verified non-monotonic integer codings | `Verified` |
| [02_target_definition_and_legacy_bug.md](02_target_definition_and_legacy_bug.md) | *Methods: Outcomes* & *Discussion* | Empirical `Complics → COtAFib` skip-logic proof (`24.30%` POAF), resolution of legacy `zain_gain` collider-bias + label-inversion bug, 9-outcome table | `Verified` |
| [03_strict_preincision_leakage_audit.md](03_strict_preincision_leakage_audit.md) | *Methods: Predictors & Leakage* | Exhaustive 638 → 207 pre-incision audit, 7 hidden early-column leaks (`LOS`, `PLOS`, `ORDuration`, etc.), timing resolution of `Status`/`Dialysis`/`CVA`/`ArrhythAFib` | `Verified` |
| [04_why_2017_holdout_is_not_statistically_valid.md](04_why_2017_holdout_is_not_statistically_valid.md) | *Methods: Critique of Temporal Split* | Calendar coverage, 99.5% overlap among 2017 hospitals, center-level incidence dispersion, 6 percentage points of missingness drift, and covariate-shift diagnostics | `Verified` |
| [05_optimal_94_6_split_and_50k10k_downsampling.md](05_optimal_94_6_split_and_50k10k_downsampling.md) | *Methods & Results: Optimal Splits* | Outcome-aware **94.2% / 5.8%** hospital split (5,160 holdout events, max \|SMD\| 0.1028). Nested 50k/10k class-balanced samples, clipped distance diagnostics, and class-ratio weights for sensitivity analysis | `Verified` |
| [06_clinical_domain_features_and_screening.md](06_clinical_domain_features_and_screening.md) | *Methods & Results: Feature Engineering* | Batch-1 (58 scores) vs Track-2A (24 orthogonal physiological features; `diabetes_age_decoupled` kept) vs Track-2B (10 nested Hospital-Grouped OOF multi-task risk priors, aux AUCs up to `0.83654`); 2017 blend AUC `0.67828` | `Verified` |
| [07_modular_dataset_and_toggle_benchmarks.md](07_modular_dataset_and_toggle_benchmarks.md) | *Results: Dual CatBoost Benchmarks* | Feature toggles for both data scales, raw and augmented CatBoost comparisons, cross-scale transfer, and separate within-2017 unseen-hospital results | `Verified` |
| [08_engineering_and_reproducibility_notes.md](08_engineering_and_reproducibility_notes.md) | *Supplementary Methods / TRIPOD+AI* | CatBoost `one_hot_max_size=16` (+0.00013 AUC, 2.1× faster), LightGBM `libgomp` fork-livelock fix & 358-CPU cluster fan-out, full SHA-256 ledger, 4/4 pytest verification, TRIPOD+AI checklist | `Verified` |
| [09_tabular_foundation_models_50k_benchmark.md](09_tabular_foundation_models_50k_benchmark.md) | *Results & Discussion: Tabular FMs vs. GBDT* | Head-to-head `SMALL 50k/10k` + Cross-Scale `test_full` (`21,241`) benchmark of **TabPFN-3.5** (`full50k_native` `0.673334` ≈ CatBoost `0.673558`, beats LL/PR/ECE!), **KumoTabular-Large v2** (`0.670748`, CUDA fix + Shard-3 class-flip caught), **TabFM v1.0** (`0.670434`/`0.670668`), `V3` vs `V1` clinical-feature lift & calibration gain, and **2-Paradigm Blend** (`CB + TabPFN-Nat`: **`0.674369`** `10k` / **`0.674908`** `21k`) | `Verified` |
| [10_championship_multiparadigm_ensemble.md](10_championship_multiparadigm_ensemble.md) | *Results & Discussion: Championship 6-Paradigm Ensemble* | Full-scale comparison of 18 models across six paradigms. The GBDT and multitask-neural blend reached 0.678418 AUC. Augmentation AUC and whole-hospital intervals for TabM and PLR-ResNet appear in Figure 3 | `Verified` |
| [11_dope_echo_counterfactual_featurizer.md](11_dope_echo_counterfactual_featurizer.md) | *Methods & Results: Counterfactual Hypernetwork Featurization (`DOPE-ECHO`)* | Counterfactual row embeddings from the frozen 444k-parameter DOPE-MLC-v3 table encoder. The Cycle-100 DOPE-only model reached 0.673254 and its blend with raw CatBoost reached 0.674690 on the nested balanced 10k sample | `Verified` |
| [12_evolved_continuous_targets_and_multiparadigm_lift.md](12_evolved_continuous_targets_and_multiparadigm_lift.md) | *Methods, Results & Discussion: Evolutionary Continuous Targets (`z*`) & JAIN Ultra* | EWMA target search, 103-arm retrospective leaderboard, and 50k model ablations. Includes PLR-TabM, JAIN Ultra, FitAuto ElasticNet, and other exploratory variants. The current manuscript reports hospital-level intervals for the headline comparisons | `Verified` |

## Related Top-Level Project References
- `../README.md` — Foundation directory overview and contract
- `../FEATURE_MATCHED_EVALUATION.md` — Collaborating Codex agent's 2014–2016 feature-matched split documentation (benchmarked head-to-head in Note 05)
- `../dataset/DATASET_README.md` — Copy-paste usage guide for `dataset_loader.py` (`scale="full"` vs `scale="small_50k_10k"`)
- `../dope_echo/DOPE_ECHO_FINAL_REPORT.md` — Full standalone technical report, checkpoint schema, and on-disk inventory for Project DOPE-ECHO
- `../reports/paper_tables/` — Drop-in CSV & LaTeX (`booktabs`) manuscript tables (`table_main_small_50k_benchmark.{csv,tex}`, `table_championship_full_346k.{csv,tex}`, `table_championship_paradigm_ablation.csv`, `table_evolved_targets_*.{csv,tex}`, `table_dope_echo_cycles_small50k.{csv,tex}`, `table_dope_echo_ablation_multifamily.csv`, `table_augmentation_lift_v3_vs_v1.csv`, `table_bayes_calibration_and_cross_scale.csv`, `table_paired_bootstrap_cis.csv`)
- `../reports/paper_figures/` — Publication-ready PNG & vector PDF figures (`fig_evolved_targets_*.{png,pdf}`, `fig_dope_echo_master_summary.{png,pdf}`, `fig_dope_echo_smoothing_convergence.{png,pdf}`, `fig_dope_echo_cycle_curve.{png,pdf}`)
- `../reports/evolved_targets_eval.{json,md}` & `../reports/evolved_targets_{oof,test_10k,test_full_21k}_preds.parquet` & `../reports/evolved_targets_evolution_telemetry.parquet` — Full evolved-target receipts, OOF/test Parquets, and generation-by-generation telemetry

