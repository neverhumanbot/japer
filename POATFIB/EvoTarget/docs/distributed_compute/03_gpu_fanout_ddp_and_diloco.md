# 03 — GPU Fan-Out, Multi-Node VPC NCCL DDP & Async WAN DiLoCo

Beyond CPU pull-queues, the `xbabe0`–`xbabe9` fleet executed three distinct GPU orchestration patterns. Selecting the right pattern depends primarily on **network topology** (same private VPC vs. cross-VPC/WAN) and **model coupling** (independent models vs. single synchronized neural network).

---

## Pattern A: Zero-Leakage Heterogeneous GPU/CPU Model Fan-Out {#pattern-a}

**Reference code:** `scripts/16_cluster_top_models_fanout.py` + `scripts/cluster_gpu_worker.py`<br>
**Production receipt:** the original run artifact `reports/top_models_evolved/cluster_fanout_receipt.json`

### Role-to-Hardware Assignment Architecture
Instead of forcing every model onto one machine, the controller maps each model paradigm to the GPU VRAM/compute profile it fits best — executing all 7 nodes concurrently:

| Node | Accelerator | Assigned Role | Runtime (Verified) |
| :--- | :--- | :--- | ---: |
| `xbabe8` | A100-40GB | `tabpfn_reg` (TabPFN-3.5 50k Native + 5×10k Bag) | `4,074 s` |
| `xbabe9` | A100-40GB | `neural_champion` (`PLR_TabM_Large_v2_400H` + `RealMLP_ResNet_PLR`) | `3,472 s` |
| `xbabe6` | A10-24GB | `neural_multitask` (`PLR_TabM_MultiTask_AuxReg_V1` + `V3_Clinical82`) | `1,635 s` |
| `xbabe1` | RTX 4090 (24GB) | `kumo_reg` (NVIDIA KumoTabular-Large v2, 5×10k Bag) | `3,941 s` |
| `xbabe3` | RTX 3090 (24GB) | `tabfm_reg` (Google TabFM v1.0.0, 5×10k Bag) | `2,886 s`* |
| `xbabe0` | 128T Threadripper | `gbdt_multiseed` (3-seed CatBoost + LightGBM + XGBoost) | `4,155 s`* |
| `xbabe4` | M3 Ultra (512 GiB) | `catboost_m3` (3-seed multi-depth CatBoost LogitRMSE sweep) | `588 s` |

*\*Note on long-run SSH drops (`rc=255`):* Because `cluster_gpu_worker.py` writes each completed arm's predictions atomically to `/dev/shm/zain_top_models_cluster/preds/<prefix>.npz`, partial results survive an SSH disconnect and are harvested cleanly via `rsync -az`, while subsequent re-runs skip already-completed prefixes automatically.

### Four Mandatory Guardrails Built Into Pattern A
1. **Strict `y_test` isolation:** `build_zero_leak_bundle()` constructs `zero_leak_bundle.npz` containing `X_tr`, `X_t10`, `X_tf`, `folds`, train-only `y_tr_aux8`, and train-only evolved targets (`target_gen20_p2b2`, etc.). **Zero holdout labels (`y_test`) ever leave `xbabe2`.**
2. **Fail-closed `nvidia-smi` idle preflight:** Before launching any GPU role, the controller probes `nvidia-smi --query-gpu=utilization.gpu,memory.used` and refuses to land on a GPU occupied by another project.
3. **Portable dependency side-loading:** Packs required wheel trees (`pydeps.tar.gz`) once and streams them into `/dev/shm`, avoiding environment drift on fresh cloud images.
4. **Automatic prefix resume + `rsync -az` recovery:** Every finished arm persists immediately; interrupted roles resume from the next missing arm without recomputing completed seeds or folds.

---

## Pattern B: Multi-Node PyTorch NCCL DDP Over Private VPC LAN {#pattern-b}

**Salvaged reference scripts:** `d2f_ddp5.sh`, `d2d_ddp4.sh`, `h50_topo.sh`, `nccl_parity.py` (archived in `<on-prem-archive>/xbabe5_9_spindown_archive/salvaged_from_tmp/xbabe-fleet/scripts/`).

### Network Topology Rule (Critical Lesson)
Cloud GPU nodes can only join a synchronous PyTorch NCCL DDP ring if they share a **high-bandwidth private VPC L2/L3 subnet**:
- **`xbabe5` (`192.0.2.10`), `xbabe7` (`192.0.2.11`), `xbabe8` (`192.0.2.12`), `xbabe9` (`192.0.2.13`)** all shared VPC subnet `192.0.2.0/24` on interface `eno1` → formed fast 4-rank to 6-rank NCCL rings with `MASTER_ADDR=192.0.2.10`.
- **`xbabe6` (`198.51.100.10`)** lived in a **separate VPC subnet** → routing NCCL collectives over public WAN collapses throughput and triggers timeouts. Always keep cross-VPC nodes on Pattern A (standalone fan-out) or Pattern C (Async DiLoCo).

### Measured H50 Fabric & NCCL Transport Receipts (`xbabe5`–`xbabe9`, PyTorch `2.7.0+cu128`, NCCL `2.26.2`)
- **Cloud A100 VPC (MTU 9000, ConnectX-6 `mlx5_0`):** Ping RTT `0.20–0.42 ms`, 8-stream TCP `79–109 Gbit/s` (30-vCPU hosts `xbabe8`/`9` retransmit at 4–8 streams), RoCE v2 `ib_write_bw` **`111 Gbit/s`**, `ib_send_lat` **`1.37 µs`**.
- **Site ↔ Cloud WAN:** Port `:22` inbound only, RTT `28–29 ms`, `110–125 Mbit/s` cloud→site, `235–250 Mbit/s` site→cloud.
- **Cross-host GDR (`NCCL_NET_GDR_LEVEL=SYS`):** Raises cross-host all-reduce from **`3.4 → 6.44 GB/s` (`1.8×`, `NET/IBext_v10/1/GDRDMA`)** vs. pure sockets `1.8–2.0 GB/s`. *(Quirk: Torch NCCL `2.26.2` still loads `NET/IBext_v10` RDMA even when `NCCL_IB_DISABLE=1` is exported!)*
- **Intra-node PCIe A100 SHM (`xbabe5` g0 ↔ g1, P2P disabled by VM):** Safe default SHM is `3.0 GB/s` (`64.3 ms` exposed FP32 comm, proven over 49k DDP steps). Adding `NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3` raises synthetic throughput to `16.9 GB/s` (`6.4 ms`), **but BF16 comm adds zero win once transport is tuned (`7.0 ms` vs. `6.4 ms`)**.

> [!CAUTION]
> **Critical Safety Warning on `NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3`:**
> On `2026-11-23T08:37–08:39Z`, enabling `SHM/CE/CE` **deadlocked `DDP.__init__` / `state_checksum` inside lazy `cuLibraryGetModule` on P2P-disabled PCIe A100s**. Keep `SHM/CE/CE` strictly **opt-in** (never enable by default!) and always pair it with `CUDA_MODULE_LOADING=EAGER` + a live `nccl_parity.py` DDP-init broadcast preflight before trusting it on a fresh VM image.

### Production-Safe NCCL + `torchrun` Launch Profile
Extracted from `h50_topo.sh`, `d2f_ddp5.sh`, and `research/H50_fleet_training/README.md` (see runnable template in [templates/launch_ddp_template.sh](templates/launch_ddp_template.sh)):

```bash
# 1. Fail-closed code-tree verification across every rank (zero silent drift)
GOT=$(cd "$WORKTREE" && git ls-files -co --exclude-standard python/ultra \
  | grep -E '\.py$' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)
[ "$GOT" = "$PIN_TREE_SHA256" ] || { echo "REFUSED: tree hash mismatch"; exit 1; }

# 2. Recommended production-safe cross-host GDR profile (tuned-gdr)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"          # e.g., "0,1" on 2x A100 nodes
export OMP_NUM_THREADS=4
export NCCL_SOCKET_IFNAME=eno1               # bind strictly to private VPC NIC
export NCCL_NET_GDR_LEVEL=SYS                # 3.4 -> 6.44 GB/s cross-host GDRDMA
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET            # log negotiated transport in every receipt
# Experimental opt-in ONLY (requires preflight!):
# export CUDA_MODULE_LOADING=EAGER NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3

# 3. Launch detached with CPU pinning & synchronized step-boundary throttling
setsid nohup taskset -c "$CPUS" nice -n 10 \
  "$PY" -m torch.distributed.run \
    --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC" \
    --master_addr 192.0.2.10 --master_port 29750 \
    -m train.trainer --run-dir "$RUN_DIR" \
      ddp_native=true ddp_global_batches=true ddp_bf16_comm=false \
      ddp_util_target=0.87 >> "$RUN_DIR/stdout.log" 2>&1 < /dev/null &
```

*Verified 6-rank production receipt (`H53_RC1_s7919` across `xbabe5/7/8/9`):* `13,017 audio-h/h`, `gpu_peak_gb=25.29 GB`, `rank_fwd_s=3.7–4.1 s`, `rank_bwd_comm_s=6.84 s`, synchronized `rank_throttle_sleep_s=0.26 s` across all 6 ranks holding 5-min rolling GPU util cleanly at `74–85%`.

### Also Supported: Static `musl` Rust CPU Sharding (`DOPE-ECHO`)
For deterministic high-throughput CPU feature synthesis (444k-param `DOPE-MLC-v3` counterfactual hypernetwork), the original run artifact `dope_echo/src/build_workers.py` compiles a static `x86_64-unknown-linux-musl` + `mimalloc` binary (plus native macOS `arm64`) and the original run artifact `dope_echo/src/cluster_orchestrator.py` shards rows proportional to calibrated node throughput — completing 100 counterfactual cycles across **606 scheduled CPU threads in `926.3 s`**.

### Three Non-Negotiable DDP Rules Learned on `xbabe5`–`xbabe9`
1. **NEVER `SIGSTOP` a DDP rank:** External process pausers (`idle_watch.sh`, `fleet_guard` SIGSTOP mode) will freeze one rank mid-AllReduce and trigger a fatal NCCL collective timeout across the entire cluster. Always throttle DDP exclusively via **synchronized step-boundary sleeping** (`ddp_util_target=0.83`–`0.87`, where `sleep = max` over all ranks).
2. **Verify numeric parity first (`nccl_parity.py`):** Before committing multi-hour runs, verify 1-node vs. multi-node DDP matches within tolerance at fixed global batch exposure (`ddp_global_batches=true`).
3. **Log `NCCL_DEBUG_SUBSYS=INIT,NET`:** Capturing the negotiated socket/SHM transport in `launch.log` immediately exposes accidental fallback to slow interfaces or mismatched MTU.

---

## Pattern C: Async HTTP DiLoCo Over WAN + LAN GPUs {#pattern-c}

**Use case:** Scaling neural pretraining across heterogeneous GPUs that do **not** share a private VPC (e.g., combining local RTX 4090s with cross-VPC cloud A100s/A10s, or packing 2 independent learners per 40GB A100).

### Architecture & Key Flags (from `dope-v73-gpu/launch_x7.sh` & `launch_x8.sh`)
- **Outer-optimizer HTTP hub:** A lightweight coordinator server on LAN (port `:877x`, exposed over SSH reverse tunnel or trusted tunnel) holds global weights and serves outer pseudo-gradient updates.
- **Inner-loop autonomy:** Each worker runs `--sync-every 10` (or `20`–`50`) local AdamW steps independently on its GPU, eliminating per-step WAN latency stalls and surviving individual worker restarts gracefully.
- **Dual-worker A100 packing:** On 40GB A100 nodes (`xbabe5`, `xbabe7`), two workers share one physical GPU cleanly using `--gpu-mem-fraction 0.45` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

```bash
# Launch pattern for 2 DiLoCo workers sharing one 40GB A100 (or 1 worker on 24GB GPU)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -m diloco_worker \
  --hub-url "http://${HUB_HOST}:8770" \
  --gpu-mem-fraction 0.45 \
  --sync-every 10 \
  --seed "$WORKER_SEED"
```
