# 08. Engineering, Compute, Reproducibility & TRIPOD+AI Notes

- **Paper Section**: *Supplementary Methods — Implementation Details, Reproducibility & TRIPOD+AI Alignment*
- **Core Claim**: Every modeling, feature-engineering, and splitting decision is deterministic, leakage-tested by automated pytest suites, cryptographically fingerprinted by SHA-256, and engineered for fast CPU reproducibility without relying on contended GPUs.
- **Evidence & Source Artifacts**: `scripts/tests/test_batch2_contract.py`, `scripts/tests/test_optimal_split_and_dataset.py`, `data/schema_audit.json`, `agent_chat.md`.
- **Status**: `Verified`

---

## 1. Key Practical ML Engineering Findings

### CatBoost Native Categorical Encoding (`one_hot_max_size=16` vs Default `2`)
`Region` has exactly 9 U.S. census divisions (`ENCENT` .. `WSCENT`). Under CatBoost's default `one_hot_max_size=2`, a 9-level category triggers online permutation-based Target Statistics (CTR) and high-order categorical combinations, incurring both extra variance and a heavy CPU penalty. Controlled single-fold benchmark on `train_2014_2016` (24 CPU threads, identical seed/features):

| Setting | Fold-0 Validation ROC-AUC | Best Iteration | Wall Time | Per-Iteration Speed |
|---|---:|---:|---:|---:|
| `one_hot_max_size=2` (default CTR) | `0.681811` | `249` | `15.3 s` | `61.4 ms/iter` |
| **`one_hot_max_size=16` (exact one-hot)** | **`0.681942` (`+0.000131`)** | `408` | **`11.9 s`** | **`29.2 ms/iter` (2.1× faster)** |

Full 5-fold Hospital-Grouped CV on 207 raw predictors confirmed the gain: `0.677686` (`one_hot_max_size=2`) → **`0.677947` (`+0.000261`)** with 3.5× faster total 5-fold screening wall time. All final CatBoost evaluations lock `one_hot_max_size=16`.

### Multi-Node Cluster Fan-Out & LightGBM `libgomp` Fork/Thread Contention Fix
- Active cluster topology leveraged: **`xbabe0`** (128 CPUs, load 1.84, 106 GB free RAM), **`xbabe3`** (128 CPUs, load 9.70, 50 GB free RAM), **`xbabe1`** (102 CPUs, load 2.97, 34 GB free RAM) = **358 idle CatBoost/LightGBM-ready CPUs**, offloading heavily loaded `xbabe2` (`load avg 248.8`).
- **Critical Systems Catch**: Running LightGBM 4.6.0 on 346k rows inside either `ThreadPoolExecutor(5)` or Linux `fork()`-based `ProcessPoolExecutor(8)` triggered a `libgomp` OpenMP spin-wait livelock (burning ~780% CPU per forked child with zero progress). Switching to **independent fresh OS `python3` worker processes fanned across `xbabe0`/`xbabe1`/`xbabe3`** eliminated the livelock completely (17–25 s/fold, completing all 120 nested LightGBM fits in 5.1 min wall time and all 30 CatBoost 2500-iter models in 6.2 min wall time).
- Storage discipline: ZSTD-compressed Parquet + clean ML-ready CSVs (`PRIMARY_TARGET_COtAFib` as final rightmost column; no `row_id` or aux targets) cleanly fit in ~500 MB (out of 40 GB free on `/`).

## 2. Exact Zero-Leakage Nested OOF Protocol (Zero Cross-Fold & Zero Cross-Scale Leakage)

For Track-2B multi-task auxiliary risk priors (`aug_b2b__*`), two strict statistical invariants are enforced:
1. **Exact Inner-Outer Nesting (`C(5,3)=10` + `C(5,4)=5`)**:
   - When evaluating outer fold $k \in \{0,1,2,3,4\}$, training rows in fold $j \ne k$ receive auxiliary-risk logits predicted by the 3-fold LightGBM model trained strictly on $\{0,1,2,3,4\} \setminus \{k, j\}$ (never seeing either fold $k$'s hospitals or fold $j$'s own labels!).
   - Outer validation rows in fold $k$ receive logits predicted by the 4-fold model trained on $\{0,1,2,3,4\} \setminus \{k\}$.
   - Held-out test rows receive the 5-fold ensemble average of the 5 four-fold models (zero test labels or test hospitals ever touched).
2. **Zero Cross-Scale Leakage (`full` vs `small_50k_10k`)**:
   - `dataset/full/` fits its `aug_b2b__*` models strictly on `train_full` (`~349.3k` rows).
   - `dataset/small_50k_10k/` **re-fits its `aug_b2b__*` models from scratch strictly on `train_small_50k` (`50,000` rows only)** so the small benchmark never borrows statistical power from the excluded ~299k training patients.

## 3. Multi-Agent Coordination & Provenance Ledger

- Independent workstreams between Antigravity and Codex are coordinated append-only via `agent_chat.md` with strict file-ownership isolation (all Codex Batch-1 and `feature_matched_splits.parquet` artifacts preserved untouched and audited head-to-head).
- Core Immutable Source, Split & Dual-Dataset SHA-256 Receipts:
  - `STS_Combined_2014_2017_Full.csv`: `89cab3de7977b463009baa71455683075b5dfbca0151df7ba07c06b882b5ed38`
  - `data/splits.parquet` (2014–2016 CV + 2017 holdout): `34709b5d5ba6aec0d51b11636b29991b5cfe9e2718f1436132b5615c10578599`
  - `data/feature_matched_splits.parquet` (Codex 2014–2016 matched): `3ea4a6e5cc8c330bc887d169a00939cd6020067e324f6499cee26305e759528e`
  - `data/optimal_splits.parquet` (Official 94.2/5.8, ≥5,000 POAF=1): `f7f3c3f8fb4ab691836628b8da25a933cf8a73de4ab1318fb677faade7eb2051`
  - `data/small_50k_10k_splits.parquet` (Exact 50/50 50k/10k): `edd74118b28a73dac7824f1f0d62100da2d0d1b604f0256f9a94e655a90c4ac6`
  - `dataset/full/train.csv`: `fe4bb0d1ed064d2a3c5854d429b4814035d478ffc01624aeccb7c790022399d4`
  - `dataset/full/test.csv`: `3a54d1e577301d003c3c76e7bd4783a2202f5630f2497230b90fde41d156084a`
  - `dataset/small_50k_10k/train.csv`: `56aac7aa969aa553f36950a215d8f7d86cdba67c808056379c77098bf9ad7f79`
  - `dataset/small_50k_10k/test.csv`: `01fe961e504f8807783546998799757b938ff372951c7643129523f9f9851acd`
- Automated Verification Gate: `pytest -v scripts/tests/test_optimal_split_and_dataset.py scripts/tests/test_batch2_contract.py` → **4 / 4 PASSED (100%)**.

## 4. TRIPOD+AI Compliance Mapping

| TRIPOD+AI Domain | How This Pipeline Complies | Primary Documentation |
|---|---|---|
| Source & Eligibility | Complete 368,132-record audit; 437 unknown outcomes explicitly isolated | `docs/01`, `docs/02` |
| Outcome Definition | Empirical proof of STS `Complics` skip-logic; *de novo* sensitivity slice | `docs/02` |
| Predictors & Leakage | Exhaustive 638 → 207 pre-incision audit; automated pytest forbidden-key gate | `docs/03` |
| Validation Splitting | 0% hospital overlap; full-feature Energy/MMD² balance; embedded 2017 unseen-hosp stress slice | `docs/04`, `docs/05` |
| Class Imbalance & Calibration | Exact Bayes prior-shift offset + Horvitz-Thompson recovery + ECE/slope/Brier reporting | `docs/05`, `docs/07` |
| Uncertainty & Ablation | 1,000-resample paired bootstrap 95% CIs across toggleable raw vs augmented views | `docs/06`, `docs/07` |
