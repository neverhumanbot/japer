# 02 — Zero-NFS SSH-Piped Pull-Queue CPU Cluster (`scripts/target_solver/`)

Our evolutionary continuous-target solver (`EvoTarget-FT` Stage 1) needed to execute **tens of thousands of hospital-shuffled CatBoost, LightGBM, XGBoost, and Ridge fits** across 10 heterogeneous machines spanning local 10GbE Threadrippers, a macOS Apple M3 Ultra, and WAN cloud EPYC/Xeon nodes — without installing Ray, Dask, Celery, or NFS.

The resulting 3-file engine (`bootstrap_cluster.py`, `cluster_pool.py`, `cluster_worker.py`) achieved **135–142 fits/sec** in Phase 1 (`76,800` fits across 10 nodes / 601 cores / 326–545 workers) and **7.6 fits/sec** in Phase 2 (`11,520` high-capacity multi-family fits per track across 4 tracks = `46,080` fits), while transparently yielding CPU whenever neighbor jobs ran on any node.

---

## 1. End-to-End Architecture

```mermaid
sequenceDiagram
    participant Ctrl as Coordinator (xbabe2)
    participant SSH as Persistent SSH Pipe (stdin/stdout)
    participant W as Remote Node Daemon (cluster_worker.py)
    participant Pool as Local Worker Pool (fork/spawn, 1 thread/fit)

    Ctrl->>SSH: 1. One-time init: zlib(X_all, y_anchor, hosp_id) (~14 MB)
    SSH->>W: Unpack to /dev/shm/<run_id>/ & mmap_mode='r' (1 page-cache copy)
    W->>Pool: Spawn N idle-priority workers (ionice -c 3 nice -n 19 chrt -i 0)
    loop Every Generation g = 1..20
        Ctrl->>SSH: 2. Push quantized uint16 genomes (~75 KB/genome, cached per node)
        loop Dynamic Pull Queue (micro-batches)
            Ctrl->>SSH: 3. Dispatch micro-batch (genome_id, seeds, family)
            W->>Pool: Execute single-threaded fits in parallel
            Pool-->>W: Out-of-sample predictions per fit
            W-->>SSH: 4. Reduce locally -> (pred_sum, pred_sq_sum, pred_cnt) (~250 KB) + live health
            SSH-->>Ctrl: Return sufficient stats + (load_1m, ram_avail_gb)
            Ctrl->>Ctrl: Update damped active_cap backpressure; re-queue if node drops
        end
    end
    Ctrl->>SSH: 5. Shutdown frame + rm -rf /dev/shm/<run_id>
```

---

## 2. Seven Engineering Techniques That Made WAN + LAN Scaling Work

### 1. Binary Framed RPC Over Plain SSH Pipes (`write_frame` / `read_frame`)
- Each remote daemon runs over a single persistent `subprocess.Popen(["ssh", host, ...], stdin=PIPE, stdout=PIPE, stderr=PIPE)`.
- **Critical stdout firewall:** On line 1 of `cluster_worker.py`, `sys.stdout` is redirected to `sys.stderr` while keeping a dedicated binary handle to the original fd 1. That guarantees stray C++ warnings from CatBoost, LightGBM, or OpenMP can **never** corrupt length-prefixed binary frames (`struct.pack("!I", len(comp)) + zlib.compress(pickle.dumps(payload, protocol=4), level=1)`).

### 2. Zero-Copy Read-Only Memory Mapping (`mmap_mode='r'`)
- `X_all.npy` (`50,000 × 289` float32 ≈ `57.8 MB`) is written once to `/dev/shm` and loaded with `np.load(..., mmap_mode="r")`.
- On Linux (`fork`), all 52–54 worker processes share **one physical OS page-cache copy** instead of duplicating `54 × 58 MB ≈ 3.1 GB` per node.
- On macOS (`xbabe4`, M3 Ultra), workers use `spawn` and re-attach to the `/tmp` mmap cleanly without pickling `X_all` across processes.

### 3. Strict Single-Thread Pinning Inside Every Worker
Before importing `numpy` or any GBDT library, `cluster_worker.py` pins every BLAS/OpenMP runtime to 1 thread:

```python
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
```

Every `CatBoostRegressor`, `LightGBM`, and `XGBoost` call additionally passes `thread_count=1` / `n_jobs=1` and `used_ram_limit="512mb"`. This prevents the classic **fork-vs-`libgomp` livelock** and eliminates thread oversubscription across 50+ concurrent fits.

### 4. `uint16` Genome Quantization + Per-Node Caching
- Sending a raw `float32[50_000]` target vector takes `200 KB`; quantizing `[0, 1]` targets into little-endian `uint16` (`pack_genome_u16`, `1.5e-5` resolution) cuts wire size in half (`100 KB` raw, **~75 KB compressed**).
- Each `_NodeConn` tracks `cached_genomes: set[int]` per generation so every genome is pushed to a WAN cloud node **at most once**, regardless of how many micro-batches that node pulls.

### 5. In-Node Sufficient-Statistics Reduction (>50× Bandwidth Reduction)
- Returning raw float predictions for 64 fits across 50,000 rows would transfer `~12.8 MB` per micro-batch over WAN.
- Instead, `cluster_worker.py` reduces every micro-batch locally into three dense arrays: `(pred_sum, pred_sq_sum, pred_cnt)` plus scalar OOS metrics (`fast_pearson_r`, exact tie-aware `fast_roc_auc_numpy`), slashing return traffic to **~250 KB** while preserving exact pooled mean and epistemic standard deviation across all nodes.

### 6. Neighbor-Friendly OS Idle Wrappers & Damped Backpressure
- **OS scheduler wrapping:** Linux workers launch under `ionice -c 3 nice -n 19 chrt -i 0` (`SCHED_IDLE`, automatic graceful fallback if restricted); macOS workers launch under `nice -n 20`. Interactive shells, SSH, and foreground GPU training always preempt CPU workers immediately.
- **Live external-load backpressure:** Every batch response returns `read_node_health()` (`load_1m`, `ram_avail_gb`). When external jobs start on a shared node, the coordinator dynamically shrinks `node.active_cap` (with hysteresis/damping) while idle nodes seamlessly pull the remaining queue items.

### 7. Automatic Dead-Node Eviction & Work Re-Queueing
- Because WAN SSH pipes occasionally drop (`rc=255`) or stall (`batch_timeout_sec=180.0`), the dynamic pull-queue wraps every dispatch in timeout/exception handling: if a node disconnects, `node.alive = False`, its unfinished micro-batch is **immediately returned to the shared `queue.Queue`**, and surviving LAN/cloud nodes finish the generation with zero lost fits.

---

## 3. Verified Production Throughput Receipts

| Phase | Workload | Active Fleet | Total Fits | Sustained Throughput | Reference Receipt |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **Phase 1 (Fast Sweep)** | 8 evolutionary runs (`01`–`08`), depth-4/5 CatBoost, 50k–60k rows | 10 nodes (`xbabe0`–`xbabe9`, 601 cores) | `337,920` across 8 runs (`76,800` deep runs) | **133–142 fits/sec** | the original run artifact `reports/target_solver/summary_report.md` |
| **Phase 2 (Multi-Family)** | 4 tracks (`P2B1`, `P2B2`, `P2B3`, `P2C`), whole-hospital shuffled CatBoost + LightGBM + XGBoost + Ridge, 289 features | 10 nodes (`xbabe0`–`xbabe9`, 326–545 workers) | `46,080` (`11,520`/track) | **7.4–7.7 fits/sec** | [reports/target_solver_phase2/phase2_summary_report.md](../../reports/target_solver_phase2/phase2_summary_report.md) |

---

## 4. Running on a New Cluster

```bash
# 1. Edit DEFAULT_CLUSTER_SPECS in scripts/target_solver/bootstrap_cluster.py (or pass --hosts)
python3 scripts/target_solver/bootstrap_cluster.py \
  --hosts xbabe0 xbabe1 xbabe2 xbabe3 xbabe4

# 2. Smoke-test cluster framing & throughput
python3 scripts/11_run_target_solver.py --mode dry-run-cluster

# 3. Launch Phase-2 multi-family whole-hospital evolution on active fleet
python3 scripts/12_phase2_highcap_multifamily_solver.py \
  --hosts xbabe0 xbabe1 xbabe2 xbabe3 xbabe4
```
