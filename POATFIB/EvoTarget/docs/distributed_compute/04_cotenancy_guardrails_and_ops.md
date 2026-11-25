# 04 — Multi-Project Co-Tenancy Guardrails & Operational Post-Mortems

Running multiple research programs (`STS POAF`, `DOPE`, `JSR`) and multiple autonomous agents simultaneously across shared GPU/CPU boxes requires explicit runtime isolation. This guide captures the exact co-tenancy toolkit developed on `xbabe5`–`xbabe9` and the four concrete failure modes we hit so future clusters never repeat them.

---

## 1. GPU Co-Tenancy Toolkit (Zero Code Modification Required)

### A. Per-Process VRAM Hard Ceiling via `sitecustomize.py`
When packing two PyTorch jobs onto a single 40GB A100 (e.g., two `0.45` fractions = 18 GB each, leaving 4 GB headroom for CUDA context), enforce memory caps **outside the model repository** via a lightweight `PYTHONPATH` hook (`$HOME/jsr-data/cotenant/sitecustomize.py`):

```python
# $HOME/jsr-data/cotenant/sitecustomize.py (see templates/sitecustomize_vram_cap.py)
# CRITICAL: os.environ.pop() removes the env var immediately in the main process so
# spawned DataLoader worker subprocesses NEVER inherit it or initialize CUDA contexts!
import os
_frac = os.environ.pop("ULTRA_CUDA_MEM_FRACTION", "")
if _frac:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(_frac), 0)
        print(f"cotenant: cuda memory fraction {_frac}", flush=True)
```

Invoke any co-tenant job with:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ULTRA_CUDA_MEM_FRACTION=0.45
export PYTHONPATH="$HOME/jsr-data/cotenant:${PYTHONPATH:-}"
```

### B. Whole-Card Thermal & Utilization Guard (`cotenant_launch.sh` + `fleet_guard`)
When two independent (non-DDP) training processes share one GPU, launch each inside its own process group (`setsid`) and attach a lightweight monitor (`fleet_guard`) that measures **whole-card utilization** and throttles duty cycle when combined GPU utilization exceeds `GPU_TARGET=0.87` or CPU exceeds `0.85`:

```bash
# Extracted from salvaged cotenant_launch.sh
setsid nohup taskset -c "$CPUS" nice -n 10 \
  "$PY" -m train.trainer --run-dir "$RUN_DIR" "$@" >> "$RUN_DIR/stdout.log" 2>&1 < /dev/null &
pid=$!; sleep 2
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
echo "$pid" > "$RUN_DIR/PID"; echo "$pgid" > "$RUN_DIR/PGID"

setsid nohup nice -n 5 "$PY" -m ultra.fleet_guard \
  --pgid "$pgid" --gpu "$GPU" --gpu-target 0.87 --cpu-target 0.85 \
  --min-duty 0.0 --log "$RUN_DIR/fleet_guard.jsonl" >> "$RUN_DIR/guard.log" 2>&1 < /dev/null &
```

### C. Opportunistic Idle-Fill Yielding (`gpu_vram_guard.sh` & `idle_watch.sh`)
For **low-priority opportunistic / sweep jobs** (NEVER synchronous DDP!), attach a 1-second watchdog that inspects `nvidia-smi --query-compute-apps=pid,used_memory` for any foreign PID:
- **`idle_watch.sh` (CPU/single-GPU sweeps):** Sends `SIGSTOP` to the background process group within 1 second whenever a higher-priority job appears, and `SIGCONT` after the GPU stays idle for a cooldown window (e.g., 15–30 seconds).
- **`gpu_vram_guard.sh` (VRAM yielder):** For jobs holding significant VRAM where `SIGSTOP` does not release device memory, gracefully checkpoints and exits (`SIGTERM`) when a foreign job requests VRAM, allowing priority jobs to run without `CUDA out of memory`.

### D. External Fleet Telemetry Probe
Our fleet dashboard polled all active nodes over lightweight 1-second SSH probes reading `nvidia-smi` + `/proc/stat`. Keep telemetry strictly read-only, non-blocking, and bound to a single dedicated SSH connection per host so monitoring never inflates load averages.

### E. Empirical CPU Interference Guard, `deadman.sh` & `relaunch_with.sh`
- **Measured CPU Co-Tenant Interference (`a1-h53-interference@20261122-23`):** Even zero-GPU "spare-CPU" training on synchronous DDP nodes (`xbabe5/7/8/9`) slowed multi-node JSR H53 by **`−3.6%` (`×2`) to `−4.5%` (`×1`)**, recovering **`+3.9%`** immediately on pause. Always isolate secondary CPU work with `chrt -i 0 taskset -c <core> nice -n 19` + `OMP/MKL/OPENBLAS_NUM_THREADS=1` and keep heavy CPU jobs off active synchronous DDP ranks.
- **Heartbeat Dead-Man Switch (`deadman.sh`):** Polls `guard.heartbeat` every 60 s; if the central guard goes stale (`> 300 s`), writes a `STOPPED` sentinel and runs `pkill -STOP` on opportunistic CPU workers automatically.
- **Graceful Live Override (`relaunch_with.sh`):** Sends `SIGTERM` so the trainer saves an atomic checkpoint, merges new `--set k=v` overrides into `<run_dir>/config.json` `argv`, and relaunches cleanly under `launch_guarded.sh`.
- **Orphan Prevention in Python (`PR_SET_PDEATHSIG`):** Register Linux `ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)` inside every forked/spawned worker (as in `scripts/target_solver/cluster_worker.py`) so killed parents never leave sleeping `PPID=1` zombies.

---

## 2. Checkpoint Retention & Disk-Fill Protection (`keep_ckpts.sh`)

Long-running multi-GPU sweeps can easily exhaust 500 GB–1.4 TB cloud root disks overnight if every intermediate checkpoint is retained. Use the hard-link retention pattern extracted from salvaged `keep_ckpts.sh`:

1. Trainer writes checkpoints to `$RUN_DIR/ckpt-*.pt` alongside `$RUN_DIR/ckpt-*.sha256` and prunes old working checkpoints aggressively.
2. Sidecar `keep_ckpts.sh` hard-links every completed `.pt` + `.sha256` into `$HOME/protected/$RUN_NAME/` (`ln` takes zero extra disk blocks), retains only:
   - Milestone branch points (e.g., every $k \times 10{,}000$ training units),
   - Explicitly pinned checkpoints listed in `KEEP`,
   - The latest 2 checkpoints + latest EMA weight snapshot,
   and unlinks everything else.
3. Monitors `df -BG --output=avail` every 10 seconds and raises an immediate alert whenever free space falls below **60 GB**.

---

## 3. Four Real Production Post-Mortems From `xbabe5`–`xbabe9`

| # | Incident Observed on `xbabe5`–`xbabe9` | Root Cause | Permanent Fix / Playbook Rule |
| :-: | :--- | :--- | :--- |
| **1** | **Root disk exhaustion (`98%–100%` full on ALL 5 cloud nodes!)** (`xbabe8` hit `493G/497G = 100%`, `xbabe6` `1.4T/1.4T = 99%`) | Overlapping datasets, teacher caches, and unpruned checkpoints across concurrent projects (`jsr`, `jope`, `dope`) on a single root volume. | Mount a separate scratch volume for datasets/checkpoints; enforce a hard **85% disk-usage preflight gate** before starting any new run; always run `keep_ckpts.sh` and delete stale caches after harvesting receipts. |
| **2** | **54 orphaned `multiprocessing.spawn` + 3 `resource_tracker` PIDs on `xbabe7`** (`PPID=1`, sleeping 25–38h in `futex_wait_queue`) | Parent trainer killed with `kill -9 $PID` instead of terminating the entire process group (`kill -TERM -$PGID`), leaving worker processes pinned to RAM and file descriptors. | Always launch background jobs in their own session (`setsid`), record `PGID` alongside `PID`, and terminate with `kill -TERM -- -$(cat PGID)` followed by `kill -KILL -- -$(cat PGID)` after 10s grace. |
| **3** | **Long-running SSH fan-out drops (`rc=255`) on `xbabe0` & `xbabe3`** (observed in `cluster_fanout_receipt.json`) | Multi-hour synchronous SSH commands eventually hit transient NAT/idle resets if not detached or checkpointed. | Write per-arm predictions atomically on completion, add `ServerAliveInterval 30` / `ServerAliveCountMax 6`, enable prefix-based skip on re-run, and recover partial artifacts with `rsync -az`. |
| **4** | **Volatile `/dev/shm` receipts at risk on node shutdown** (33 JSON receipts existed only in `/dev/shm` during our pre-spindown audit) | Fast training loops wrote final evaluation JSONs directly into `/dev/shm` run folders for speed. | Treat `/dev/shm` strictly as ephemeral scratch for input bundles and intermediate temp files; **immediately copy every JSON receipt, log, and metric summary to persistent coordinator storage** upon job completion. |
