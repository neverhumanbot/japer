# Multi-Node Distributed Compute Playbook (`docs/distributed_compute/`)

This playbook captures the complete multi-node distributed compute architecture, orchestration patterns, co-tenancy guardrails, and operational lessons learned across our 10-node heterogeneous LAN + Cloud GPU fleet (`xbabe0`–`xbabe9`). Every pattern documented here ran in production across **601 CPU cores**, **10+ GPUs**, **88,320+ evolutionary tree fits**, **multi-node PyTorch NCCL DDP**, and **WAN+LAN asynchronous DiLoCo** — with **zero shared NFS/Lustre storage** and **zero test-label leakage**.

> **Fleet Status Note (2026-11-25):** Cloud nodes `xbabe5`–`xbabe9` were idle at the last successful check and later became unreachable over SSH. Provider instance and volume status remain unconfirmed. See [05_xbabe5_9_spindown_audit_and_harvest.md](05_xbabe5_9_spindown_audit_and_harvest.md) for preservation evidence and recovery steps. IP addresses in this playbook are documentation examples.

The reusable templates are included here. References to original experiment launchers and model training scripts refer to the archived research workspace.

---

## Playbook Index

| Guide | Focus | When to Read |
| :--- | :--- | :--- |
| **[01_ssh_fabric_and_bootstrap.md](01_ssh_fabric_and_bootstrap.md)** | SSH fabric, isolated `~/.venvs/`, `git bundle` & `pydeps.tar.gz` side-loading, `/dev/shm` staging | First 10 minutes on any fresh cluster |
| **[02_cpu_ssh_pull_queue_cluster.md](02_cpu_ssh_pull_queue_cluster.md)** | Zero-NFS SSH-piped pull-queue CPU engine (`scripts/target_solver/`), 135–142 fits/s, damped backpressure | Embarrassingly parallel / evolutionary / CV sweeps across LAN + WAN CPUs |
| **[03_gpu_fanout_ddp_and_diloco.md](03_gpu_fanout_ddp_and_diloco.md)** | Heterogeneous GPU model fan-out, VPC private-LAN PyTorch NCCL DDP, and WAN+LAN Async HTTP DiLoCo | Multi-GPU model training, Tabular FMs, or distributed neural training |
| **[04_cotenancy_guardrails_and_ops.md](04_cotenancy_guardrails_and_ops.md)** | Multi-project GPU co-tenancy (`sitecustomize.py`, `fleet_guard`, `idle_watch.sh`), disk guards, & post-mortems | Sharing nodes safely across multiple teams/agents without OOM or disk crashes |
| **[05_xbabe5_9_spindown_audit_and_harvest.md](05_xbabe5_9_spindown_audit_and_harvest.md)** | Pre-spindown audit ledger for `xbabe5`–`xbabe9`, archive manifest summary, & fleet retirement checklist | Auditing preserved assets or cleaning retired hosts out of configs |

### Drop-In Executable Templates (`templates/`)

| Script | Purpose |
| :--- | :--- |
| [templates/node_setup_uv_torch.sh](templates/node_setup_uv_torch.sh) | Fresh cloud GPU node Python/`uv`/PyTorch provisioning template |
| [templates/gpu_preflight_guard.sh](templates/gpu_preflight_guard.sh) | Fail-closed GPU idle & disk-space preflight guard before launching jobs |
| [templates/launch_ddp_template.sh](templates/launch_ddp_template.sh) | VPC private-LAN `torchrun` NCCL DDP launcher with `PIN_TREE_SHA256` & step-boundary throttle |
| [templates/diloco_reverse_tunnels.sh](templates/diloco_reverse_tunnels.sh) | SSH reverse-tunnel helper for WAN+LAN Async HTTP DiLoCo workers |
| [templates/sitecustomize_vram_cap.py](templates/sitecustomize_vram_cap.py) | Zero-code-change per-process CUDA VRAM ceiling (`ULTRA_CUDA_MEM_FRACTION`) |

---

## Fleet Reference Topology (10-Node Peak Architecture)

| Tier | Host Alias | Hardware Profile | Network Topology | Primary Roles Proven | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LAN** | `xbabe0` | 64c/128T Threadripper PRO 3995WX · 126 GiB RAM | Local 10GbE LAN | CPU pull-queue (52w), multi-seed GBDT fan-out | **Active** |
| **LAN** | `xbabe1` | Threadripper PRO 3995WX · 63 GiB RAM · **RTX 4090** | Local 10GbE LAN | CPU pull-queue (52w), KumoTabular-Large v2 GPU | **Active** |
| **LAN** | `xbabe2` | 64c/128T Threadripper PRO 3995WX · 126 GiB RAM | Local 10GbE LAN | **Primary coordinator / controller** (holds `y_test`) | **Active** |
| **LAN** | `xbabe3` | 64c/128T Threadripper PRO 3995WX · 63 GiB RAM · **RTX 3090** | Local 10GbE LAN | CPU pull-queue (52w), TabFM v1.0.0 GPU | **Active** |
| **LAN** | `xbabe4` | Apple M3 Ultra (32c) · **512 GiB Unified RAM** | Local 10GbE LAN (macOS) | CPU pull-queue (26w via `spawn`), CatBoost sweeps | **Active** |
| **Cloud** | `xbabe5` | 60c EPYC 7H12 · 443 GiB RAM · **2× A100-40GB** | VPC-A (`192.0.2.10`, `eno1`) | DDP `MASTER_ADDR` (rank 0/1), CPU pull-queue (54w) | *SSH unreachable 2026-11-25* |
| **Cloud** | `xbabe6` | 30c Xeon 8358 · 222 GiB RAM · **1× A10-24GB** | **VPC-B (`198.51.100.10`, isolated)** | Standalone GPU fan-out (neural multi-task), DiLoCo | *SSH unreachable 2026-11-25* |
| **Cloud** | `xbabe7` | 60c EPYC 7H12 · 443 GiB RAM · **2× A100-40GB** | VPC-A (`192.0.2.11`, `eno1`) | DDP peer (rank 3+), 2× DiLoCo/A100, CPU pull-queue (54w) | *SSH unreachable 2026-11-25* |
| **Cloud** | `xbabe8` | 30c EPYC 7H12 · 221 GiB RAM · **1× A100-40GB** | VPC-A (`192.0.2.12`, `eno1`) | TabPFN-3.5 native 50k GPU, DDP peer, CPU pull-queue (27w) | *SSH unreachable 2026-11-25* |
| **Cloud** | `xbabe9` | 30c EPYC 7H12 · 221 GiB RAM · **1× A100-40GB** | VPC-A (`192.0.2.13`, `eno1`) | EvoTarget-FT / PLR-TabM GPU, DDP peer, CPU pull-queue (27w) | *SSH unreachable 2026-11-25* |

---

## Decision Matrix: Choosing the Right Multi-Node Pattern

| Workload Shape | Recommended Pattern | Transport & Sync | Shared Storage Needed? | Reference Implementation |
| :--- | :--- | :--- | :--- | :--- |
| Thousands of fast CPU fits (GBDT, Ridge, evolutionary target search, CV sweeps) | **Pattern 1: SSH-Piped Pull Queue** | Length-prefixed `zlib` frames over `ssh` stdin/stdout; in-node sufficient stats | **No** (`/dev/shm` RAM disk) | `scripts/target_solver/` |
| Independent GPU/CPU model training (TabPFN, Kumo, TabFM, PyTorch neural arms) | **Pattern 2: Zero-Leakage Model Fan-Out** | `scp` + `pydeps.tar.gz` + `rsync -az` partial pred sync (`y_test` stays local) | **No** (`/dev/shm` per job) | `scripts/16_cluster_top_models_fanout.py` |
| Tightly coupled single model across multiple cloud GPUs in the **same VPC** | **Pattern 3: VPC Private-LAN NCCL DDP** | `torch.distributed.run` over `NCCL_SOCKET_IFNAME=eno1` + `PIN_TREE_SHA256` | **No** (`git bundle` deploy) | [03_gpu_fanout_ddp_and_diloco.md](03_gpu_fanout_ddp_and_diloco.md#pattern-b) |
| Large neural training spanning **separate VPCs or WAN + LAN GPUs** | **Pattern 4: Async HTTP DiLoCo** | Periodic outer-optimizer HTTP sync (`--sync-every 10`) to LAN hub | **No** | [03_gpu_fanout_ddp_and_diloco.md](03_gpu_fanout_ddp_and_diloco.md#pattern-c) |
| Sharing expensive GPUs/CPUs safely with concurrent teams or research agents | **Pattern 5: Co-Tenancy Guardrails** | `ULTRA_CUDA_MEM_FRACTION`, `fleet_guard` / `ddp_util_target`, OS idle wrappers | **No** | [04_cotenancy_guardrails_and_ops.md](04_cotenancy_guardrails_and_ops.md) |

---

## 10-Minute Fresh-Cluster Spin-Up Checklist

1. **Provision & Verify Networking:** Confirm SSH key auth works non-interactively (`BatchMode=yes`) to all nodes; verify which cloud GPUs share a private L2/L3 VPC subnet (`ip -4 addr show`) before attempting NCCL DDP.
2. **Harden `~/.ssh/config`:** Add `BatchMode yes`, `ServerAliveInterval 30`, `ServerAliveCountMax 6`, `IdentitiesOnly yes`, and stable host aliases (`node0`, `node1`, ...).
3. **Bootstrap Isolated Worker Venvs:** Run `python3 scripts/target_solver/bootstrap_cluster.py --hosts <host1> <host2> ...` (creates `~/.venvs/sts_cb_worker` in ~15 seconds in parallel without touching system Python; add `--check-only` for a read-only probe) and verify `reports/target_solver/cluster_manifest.json`.
4. **Dry-Run Before Launching Full Jobs:**
   - For CPU pull-queue sweeps: run `--mode dry-run-cluster` to confirm every node receives the `/dev/shm` bundle, obeys idle scheduling (`chrt -i 0` / `nice`), and cleans up cleanly on `SIGTERM`.
   - For GPU workloads: verify `nvidia-smi` idle preflight (`<5%` util, `<500 MiB` VRAM) and check root disk free space (`df -h /` — **keep >100 GB free**; never let cloud disks cross 90%).
   - For NCCL DDP: run a 10-step numeric parity test (`nccl_parity.py`) and pin `PIN_TREE_SHA256` across all ranks before launching multi-hour training.
5. **Pin Co-Tenancy & Disk Guards:** Enable `ddp_util_target` (for DDP) or `fleet_guard` / `idle_watch.sh` (for standalone jobs) and attach a checkpoint pruner (`keep_ckpts.sh`) so logs and checkpoints never fill the root volume.
