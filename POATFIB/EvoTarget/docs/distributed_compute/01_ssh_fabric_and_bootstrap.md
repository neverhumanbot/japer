# 01 — SSH Fabric, Zero-Touch Bootstrapping & Ephemeral Staging

All five distributed patterns in this repository operate **without NFS, Lustre, Slurm, Kubernetes, or Docker Swarm**. Every node joins the fleet over plain OpenSSH using isolated user-space virtual environments and `/dev/shm` RAM-disk staging.

---

## 1. Hardened `~/.ssh/config` Template

Always use non-interactive SSH with explicit keepalives so long-running jobs detect half-open WAN drops promptly rather than hanging indefinitely:

```sshconfig
Host worker-*
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    BatchMode yes
    StrictHostKeyChecking accept-new
    ConnectTimeout 8
    ServerAliveInterval 30
    ServerAliveCountMax 6
    Compression no

# Example: LAN coordinator & peers (10GbE)
Host worker-lan0
    HostName 203.0.113.10

# Example: macOS Apple Silicon peer
Host worker-mac
    HostName 203.0.113.14
    User agent

# Example: Cloud GPU node (public IP for SSH; record VPC private IP in comments for NCCL)
Host worker-cloud0
    HostName <PUBLIC_IP>   # VPC private LAN: 192.0.2.10 (eno1)
```

> **Why `Compression no`?** Our binary RPC frames (`cluster_worker.py`) and `pydeps.tar.gz` archives are already compressed (`zlib` level 1 / `gzip`). Enabling SSH-level compression burns CPU cycles on already-compressed bytes.

Verify fleet connectivity in one command before bootstrapping:

```bash
for h in worker-lan0 worker-mac worker-cloud0; do
  ssh -o ConnectTimeout=6 "$h" 'echo "$(hostname) | cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu) | disk=$(df -h / | awk "NR==2{print \$5}")"'
done
```

---

## 2. Idempotent Isolated-Venv Bootstrapping (`bootstrap_cluster.py`)

`scripts/target_solver/bootstrap_cluster.py` provisions an entire heterogeneous Linux + macOS cluster in **10–15 seconds in parallel** with three strict guarantees:

1. **Zero-touch system Python:** Never runs `sudo`, never edits `/usr/bin/python3`, and never modifies other users' or projects' virtual environments.
2. **Version-pinned isolation:** Probes preferred interpreters first; if missing or mismatched (e.g., `catboost != 1.2.10`), creates an isolated `~/.venvs/sts_cb_worker` (`uv venv` / `python3 -m venv`) and installs pinned wheel versions.
3. **Capability & safety discovery:** Probes every node for OS architecture, logical cores, 1-minute load average, `MemAvailable`, `/dev/shm` free capacity, and idle scheduler support (`chrt -i 0`, `ionice -c 3`, `nice -n 19` on Linux; `nice -n 20` / `taskpolicy` on macOS), emitting a machine-readable `reports/target_solver/cluster_manifest.json`.

### Adding a New Node to `DEFAULT_CLUSTER_SPECS`

Each node is configured via a frozen `NodeSpec` tuple in `bootstrap_cluster.py`:

```python
NodeSpec(
    host="worker-cloud0",
    tier="cloud",                        # "lan" or "cloud"
    os_type="linux",                     # "linux" or "darwin"
    preferred_python="~/.venvs/sts_cb_worker/bin/python3",
    fallback_pythons=("/usr/bin/python3",),
    scratch_root="/dev/shm",             # "/tmp" on macOS
    reserve_cores=4,                     # always left free for OS / SSH / neighbors
    min_free_ram_gb=16.0,                # safety floor before throttling workers
    max_workers_cap=54,                  # memory-bandwidth / L3-cache sweet spot
)
```

**Throughput rule of thumb:** On 64-core/128-thread AMD Threadripper PRO 3995WX nodes, capping `max_workers_cap=52` outperformed using all 128 hyperthreads because single-threaded CatBoost fits saturate DRAM/L3 bandwidth well before 128 threads, while leaving 76+ hardware threads free for concurrent interactive work.

### Running Bootstrap & Dry-Run Verification

```bash
# 1. Bootstrap isolated venvs and generate reports/target_solver/cluster_manifest.json
python3 scripts/target_solver/bootstrap_cluster.py \
  --hosts xbabe0 xbabe1 xbabe2 xbabe3 xbabe4

# 2. Validate end-to-end SSH framing, /dev/shm sync, and cleanup before launching real sweeps
python3 scripts/11_run_target_solver.py --mode dry-run-cluster
```

---

## 3. Code & Dependency Sync Without Shared Storage

Across our projects we used three lightweight distribution mechanisms depending on workload complexity:

### A. In-Memory Bundle over SSH Stdin (`cluster_pool.py`)
- Best for **Python worker daemons with small data matrices** (`<50 MB`).
- The controller serializes `X_all.npy`, `y_anchor.npy`, `is_anchored.npy`, `hosp_id.npy`, and `cluster_worker.py` into one zlib-compressed payload (~14 MB for 50,000 × 289 `float32`), pipes it directly to `ssh <host> ...` stdin, and writes it into `/dev/shm/sts_target_solver_<run_id>/` (Linux) or `/tmp/...` (macOS).
- **Zero persistent files left behind:** An `atexit` + `SIGTERM`/`SIGINT` cleanup hook removes every remote staging directory automatically.

### B. Portable `pydeps.tar.gz` Side-Loading (`16_cluster_top_models_fanout.py`)
- Best for **niche ML packages** (`tabpfn`, `kumoai`/KumoTabular, `tabfm`, custom `torch` extensions) that aren't installed on every remote cloud image.
- The controller packs the exact site-packages subdirectories from a verified reference venv into a single tarball, `scp`s it once to `/dev/shm/<job>/pydeps.tar.gz`, extracts it on the worker, and prepends `PYTHONPATH=/dev/shm/<job>/pydeps`.
- Eliminates pip resolution drift, HuggingFace rate-limit failures during worker startup, and dirty remote environments.

### C. Air-Gapped / Deterministic `git bundle` + Tree-Hash Pinning (`tc_deploy.sh` & `h50_topo.sh`)
- Best for **multi-node PyTorch DDP** where every rank *must* execute byte-identical code.
- On the coordinator, package the exact commit into a single bundle (`git bundle create t.bundle lane/trainer`), `scp` to all nodes, and check out detached HEAD on each worker:

```bash
# On remote node (from salvaged tc_deploy.sh)
cd "$WORKTREE" && git fetch -q ~/t.bundle lane/trainer \
  && git checkout -q --detach FETCH_HEAD && rm -f ~/t.bundle
```

- Before starting `torch.distributed.run`, enforce a fail-closed content hash across all Python sources (`PIN_TREE_SHA256`, extracted from salvaged `h50_topo.sh`):

```bash
GOT=$(cd "$WORKTREE" && git ls-files -co --exclude-standard python/ultra \
  | grep -E '\.py$' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)
[ "$GOT" = "$PIN_TREE_SHA256" ] || { echo "REFUSED: tree hash mismatch"; exit 1; }
[ -z "$(git -C "$WORKTREE" status --porcelain python/ultra)" ] || { echo "REFUSED: dirty tree"; exit 1; }
```

This single guard prevented silent multi-node divergence where one rank accidentally ran stale local edits.

---

## 4. Ephemeral `/dev/shm` Staging & Cleanup Rules

1. **Always check `/dev/shm` free capacity first:** Cloud VMs often cap `/dev/shm` at 50% of RAM (or less in containers). Always verify `scratch_free_gb` before copying multi-GB bundles.
2. **Never store long-lived JSON receipts ONLY in `/dev/shm`:** Any crash, OOM reboot, or instance stop destroys `/dev/shm` instantaneously. Sync final receipts back to persistent disk on the coordinator immediately after each generation or job finishes.
3. **Clean up per-run directories on exit:** Namespace every remote staging path with `<run_id>` (e.g., `/dev/shm/sts_target_solver_${RUN_ID}`) so concurrent runs never collide and `rm -rf` cleanup is surgical.
