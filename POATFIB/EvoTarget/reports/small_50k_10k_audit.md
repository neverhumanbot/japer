# Smart 50k/10k Target-Balanced & Class-Conditionally Feature-Matched Downsample Audit

- **Generated (UTC)**: `2026-11-25T07:53:07.675995+00:00`
- **Canonical Artifact**: `data/small_50k_10k_splits.parquet` (SHA256 `edd74118b28a73dac7824f1f0d62100da2d0d1b604f0256f9a94e655a90c4ac6`)
- **Exact Balance**: `train_small_50k` = **25,000 pos / 25,000 neg (50.00%)** across **1017** hospitals; `test_small_10k` = **5,000 pos / 5,000 neg (50.00%)** across **74** unseen hospitals (**0.0% hospital overlap**, **0 duplicate rows**)
- **Bayes Prior-Shift Logit Offset**: `+1.136479` (`logit_natural = logit_balanced - 1.136479`)

## Class-Conditional Multivariate Covariate Fidelity

| Contrast | Parent N | Subset N | Energy Dist (32D) | Multi-RBF MMD² | Mean \|SMD\| (206) | Max \|SMD\| (206) | Max KS (18) | Max W1 (18) | Max Miss Drift | Region TV | Year TV | Rare Tail Δ | Max Aux Cond Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train_50k vs train_full (Y=1)` | 84,177 | 25,000 | 0.00072 | 0.000047 | 0.00544 | **0.1475** | 0.0057 | 0.0128 | 0.0064 | 0.0005 | 0.0003 | 0.00033 | 0.00213 |
| `train_50k vs train_full (Y=0)` | 262,277 | 25,000 | 0.00062 | 0.000034 | 0.00727 | **0.1950** | 0.0081 | 0.0167 | 0.0099 | 0.0003 | 0.0002 | 0.00017 | 0.00089 |
| `test_10k vs test_full (Y=1)` | 5,160 | 5,000 | 0.00000 | 0.000000 | 0.00193 | **0.0154** | 0.0026 | 0.0060 | 0.0017 | 0.0005 | 0.0005 | 0.00072 | 0.00094 |
| `test_10k vs test_full (Y=0)` | 16,081 | 5,000 | 0.00000 | 0.000000 | 0.00671 | **0.0735** | 0.0154 | 0.0299 | 0.0083 | 0.0011 | 0.0002 | 0.00033 | 0.00249 |
| `test_10k vs train_50k (overall)` | 50,000 | 10,000 | 0.00444 | 0.000300 | 0.03003 | **0.2685** | 0.0238 | 0.0357 | 0.0362 | 0.0162 | 0.0058 | 0.00208 | 0.00407 |

## All 9 Targets — Natural vs Balanced Enrichment (Zero Collider Distortion)

| Target | `train_full` Rate | `train_small_50k` Rate | Train Enrichment | `test_full` Rate | `test_small_10k` Rate | Test Enrichment |
|---|---:|---:|---:|---:|---:|---:|
| `PRIMARY_TARGET_COtAFib` | 0.2430 | **0.5000** | 2.06× | 0.2429 | **0.5000** | 2.06× |
| `TARGET_Mortalty` | 0.0180 | **0.0225** | 1.25× | 0.0188 | **0.0228** | 1.21× |
| `TARGET_CSepsis` | 0.0070 | **0.0105** | 1.49× | 0.0095 | **0.0146** | 1.53× |
| `TARGET_CPVntLng` | 0.0688 | **0.0838** | 1.22× | 0.0672 | **0.0812** | 1.21× |
| `TARGET_CPPneum` | 0.0218 | **0.0282** | 1.29× | 0.0230 | **0.0292** | 1.27× |
| `TARGET_CVTE` | 0.0065 | **0.0087** | 1.33× | 0.0083 | **0.0107** | 1.28× |
| `TARGET_CRenFail` | 0.0172 | **0.0235** | 1.37× | 0.0184 | **0.0259** | 1.41× |
| `TARGET_CPlEff` | 0.0317 | **0.0387** | 1.22× | 0.0319 | **0.0393** | 1.23× |
| `TARGET_COtArrst` | 0.0145 | **0.0177** | 1.22× | 0.0160 | **0.0202** | 1.27× |
