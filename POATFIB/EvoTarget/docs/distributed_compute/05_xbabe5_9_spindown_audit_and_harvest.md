# 05 — `xbabe5`–`xbabe9` Pre-Spindown Audit Ledger & Preservation Record

**Last successful workload checks (UTC)** 22:33–22:37Z.
**Last SSH check (UTC)** 23:31Z. All five cloud hosts were unreachable. Provider instance and volume status remain unconfirmed.

---

## 1. Pre-Spindown Live Workload Verification

Every cloud GPU node was inspected over read-only SSH prior to spindown (`nvidia-smi`, `ps -ef`, `tmux ls`, `screen -ls`, `crontab -l`, `docker ps`, `df -h`, and full git/receipt diffs). **All 7 GPUs were at `0%` utilization and `0 MiB` VRAM with zero active compute jobs:**

| Node | Public IP / VPC Private IP | Hardware Profile | Final Disk | Final GPU / Active Compute | Non-Critical Daemons / Orphans Observed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`xbabe5`** | `<REDACTED_PUBLIC_IP>` / `192.0.2.10` | 60c EPYC 7H12 · 442 GiB · **2× A100-40GB** | `967G / 993G` (**98%**) | **0% / 0 MiB / 0 jobs** | Default `jupyter-lab`/`glances`, external 1s telemetry SSH probe, stale `keep_ckpts.sh` (`H53_*`) |
| **`xbabe6`** | `<REDACTED_PUBLIC_IP>` / `198.51.100.10` *(sep. VPC)* | 30c Xeon 8358 · 222 GiB · **1× A10-24GB** | `1.4T / 1.4T` (**99%**) | **0% / 0 MiB / 0 jobs** | Default `jupyter-lab`/`glances`, 1s probe, stale `bin/x6_disk_watch.sh` |
| **`xbabe7`** | `<REDACTED_PUBLIC_IP>` / `192.0.2.11` | 60c EPYC 7H12 · 442 GiB · **2× A100-40GB** | `964G / 993G` (**98%**) | **0% / 0 MiB / 0 jobs** | Default `jupyter-lab`/`glances`, 1s probe, stale `keep_ckpts.sh`, 54 orphaned `multiprocessing.spawn` + 3 `resource_tracker` PIDs (`PPID=1`, 0% GPU) |
| **`xbabe8`** | `<REDACTED_PUBLIC_IP>` / `192.0.2.12` | 30c EPYC 7H12 · 221 GiB · **1× A100-40GB** | `493G / 497G` (**100%**) | **0% / 0 MiB / 0 jobs** | Default `jupyter-lab`/`glances`, 1s probe |
| **`xbabe9`** | `<REDACTED_PUBLIC_IP>` / `192.0.2.13` | 30c EPYC 7H12 · 221 GiB · **1× A100-40GB** | `487G / 497G` (**99%**) | **0% / 0 MiB / 0 jobs** | Default `jupyter-lab`/`glances`, 1s probe |

- **Zero `tmux` sessions, zero `screen` sessions, zero user `crontab` entries, and zero Docker containers** across all five nodes at the last successful check.
- **Reachability and forensic receipts:** Saved at `<on-prem-archive>/xbabe5-9-retirement-20261125/inventory/xbabe{5..9}-final-idle.txt` (`22:46Z`) and `<on-prem-archive>/xbabe5_9_spindown_archive/forensic_audit_logs/`.

---

## 2. Permanent Preservation Inventory (Local on `xbabe2` + GitHub)

All STS POAF research code, receipts, tables, figures, and the 17-page manuscript were already 100% local on `xbabe2` and committed to `neverhumanbot/japer` (`POATFIB/EvoTarget/`). All core `jsr` training modules (`anchored_rnnt.py`, `mfa_anchors.py`, `recipe_opts.py`, `mwer.py`, `h41_mwer.sh`, `test_train_stream_ctx.py`) were verified present on `github.com/neverhuman/jsr` `origin/main`.

Salvaged cloud scripts, patches, STS worker receipts, and forensic logs were copied to `xbabe2` across two indexed directories (strictly excluding multi-GB bulk datasets, audio streams, and model weights):

| Local Permanent Archive Path | Files | Total Size | Preserved Contents |
| :--- | ---: | ---: | :--- |
| **`<on-prem-archive-volume>/spindown-x5-x9-20261125/`** *(SHA-256 verified)* | **35+ archives** | **163 MB** | **Verified Git bundles and STS artifacts:** All 5 `~/jsr` full `git bundle --all` snapshots (`xbabe5`–`9`, 27 MB each, 145–146 refs — **verified `has_562dd31=1`**, capturing the sole remote-only commit absent from GitHub & local `~/jsr`!), `xbabe5/6/7-code.tgz` & `dirty/xbabe5..7.txt`, `sts-codex/` (`10/10` SHA-256 `OK`: 116 coordinator files, 67 `xbabe8` STS code/result files, both worker SHA-256 revisions, pinned `dope-core 820ac79`, `0001a-encoder-blocks-a1l-tree-d5f231e.patch`), and `inventory/inv-xbabe5..9.txt`. *(Disconnect note: at `~22:40:25Z` all 5 hosts went unreachable simultaneously, truncating `{xbabe8,xbabe9}-code.tgz`, `dirty/{xbabe8,xbabe9}.txt`, and `xbabe5-ops.tgz` at `0 B` — committed counterparts exist for several named JSR files, while exact untracked versions and several worker training logs are unverified. Three small non-STS result folders not inside `grok-preserve/results/` were noted: `xbabe9:~/jope-bin103-cov4vs16-x9-out` [17.5 MB, summary JSON captured], `xbabe6:~/exact-isa/runs{,-rc2}/x6` [11.9 MB], and `xbabe5:~/v5_m1v2_harvest` [17 MB].)* |
| **`<on-prem-archive>/xbabe5_9_spindown_archive/`** *(SHA-256 `MANIFEST.json`)* | **103** | **0.477 MB** | Salvaged multi-node DDP/co-tenancy scripts (`d2f_ddp5.sh`, `d2d_ddp4.sh`, `h50_topo.sh`, `tc_deploy.sh`, `cotenant_launch.sh`, `a1l_launch_cotenant.sh`, `keep_ckpts.sh`, `xhost.sh`, `ema_patch.py`, `xbabe5-launchers.tgz`) plus 25 pre-spindown audit/forensic logs (`max file: 31.5 KB`). |
| **`<on-prem-archive>/xbabe5-9-retirement-20261125/`** | **607** | **18.972 MB** | Full STS coordinator source/receipts snapshot, `xbabe8-sts-code-results`, `dope-rust-dependency-source-820ac79`, and per-node `22:34–22:46Z` inventory & idle confirmation receipts. |
| **`<on-prem-workspace>/jope_plan/sts_*`** | — | Local | All `xbabe8` STS/selfplay manifests and 11 SHA-256-matched self-play milestone checkpoints. |

---

## 3. Post-Spindown Reconfiguration Checklist (Active 5-Node LAN Fleet)

For LAN-only training while the cloud nodes are unreachable, keep the remaining 5-node LAN fleet (`xbabe0`–`xbabe4`: 4× Threadripper PRO 3995WX + 1× Apple M3 Ultra + RTX 4090 + RTX 3090) running smoothly with four housekeeping steps:

1. **Pass explicit space-separated `--hosts` for LAN-only CPU sweeps** (or update `DEFAULT_CLUSTER_SPECS` in `scripts/target_solver/bootstrap_cluster.py` when you next run target evolution):
   ```bash
   python3 scripts/target_solver/bootstrap_cluster.py \
     --hosts xbabe0 xbabe1 xbabe2 xbabe3 xbabe4
   ```
2. **Prune stale SSH aliases when convenient:** Comment out or remove `Host xbabe5` through `Host xbabe9` from `~/.ssh/config` so accidental fleet loops don't wait for TCP timeouts.
3. **Detach external fleet dashboard probes:** Remove `xbabe5`–`xbabe9` from the external 1-second telemetry monitor (`<REDACTED_PUBLIC_IP>`) to eliminate dead SSH reconnect noise.
4. **Provisioning Replacement Cloud GPUs Later:** When new cloud GPUs are rented, assign fresh aliases in `~/.ssh/config`, verify same-VPC private subnet membership if running NCCL DDP, and follow the 10-minute spin-up checklist in [README.md](README.md#10-minute-fresh-cluster-spin-up-checklist).
