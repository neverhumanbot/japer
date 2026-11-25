#!/usr/bin/env bash
# Cross-host PyTorch DDP launcher with locale-invariant code-tree SHA-256 verification,
# safe NCCL profile selection, CPU core pinning, and PID/PGID lifecycle tracking.
#
# NEVER attach SIGSTOP-based guards (fleet_guard) to DDP ranks: use in-trainer
# synchronized step-boundary throttling (StepThrottle) instead.
#
# Environment variables:
#   REPO             Path to git checkout (default: current directory)
#   PY               Path to venv python (required)
#   PIN_TREE_SHA256  Optional 64-hex expected sha256 of tracked *.py files under CODE_SUBDIR
#   CODE_SUBDIR      Subdir to hash (default: .)
#   NCCL_PROFILE     a1env (default, safest) | tuned-gdr (cross-host RoCE/RDMA 1.8x)
#   IFNAME           Private LAN NIC for NCCL (default: eno1)
#   NNODES           Total nodes/ranks participating (default: 1)
#   NODE_RANK        This host's rank 0..NNODES-1 (default: 0)
#   MASTER_ADDR      Rank-0 private IP on $IFNAME (default: 127.0.0.1)
#   MASTER_PORT      Rendezvous port (default: 29511)
#
# Usage:
#   launch_ddp_template.sh <run_dir> <cpu_list> <gpu_list e.g. 0,1> -- <module_and_args...>
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$4" != "--" ]; then
  echo "usage: $0 <run_dir> <cpu_list> <gpu_list> -- <python -m ... args>" >&2
  exit 2
fi

RUN_DIR=$1 CPUS=$2 GPUS=$3
shift 4

REPO="${REPO:-$PWD}"
PY="${PY:?set PY to venv python}"
CODE_SUBDIR="${CODE_SUBDIR:-.}"
NCCL_PROFILE="${NCCL_PROFILE:-a1env}"
IFNAME="${IFNAME:-eno1}"

# 1. Locale-invariant code-tree hash check across all nodes (prevents silent drift)
if [ -n "${PIN_TREE_SHA256:-}" ]; then
  GOT_SHA=$(cd "$REPO" && git ls-files -co --exclude-standard "$CODE_SUBDIR" \
    | grep -E '\.py$' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)
  if [ "$GOT_SHA" != "$PIN_TREE_SHA256" ]; then
    echo "FATAL: code tree sha256 mismatch on $(hostname): got=$GOT_SHA want=$PIN_TREE_SHA256" >&2
    exit 42
  fi
fi

# 2. Safe NCCL transport profile (NEVER enable SHM_USE_CUDA_MEMCPY=1 without EAGER preflight!)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_SOCKET_IFNAME="$IFNAME"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

case "$NCCL_PROFILE" in
  a1env)
    export NCCL_IB_DISABLE=1
    ;;
  tuned-gdr)
    unset NCCL_IB_DISABLE || true
    export NCCL_NET_GDR_LEVEL=SYS
    ;;
  *)
    echo "FATAL: unknown NCCL_PROFILE=$NCCL_PROFILE (use a1env or tuned-gdr)" >&2
    exit 2
    ;;
esac

NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)

setsid nohup taskset -c "$CPUS" nice -n 10 "$PY" -m torch.distributed.run \
  --nnodes "${NNODES:-1}" \
  --node_rank "${NODE_RANK:-0}" \
  --nproc_per_node "$NPROC" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --master_port "${MASTER_PORT:-29511}" \
  "$@" >> "$RUN_DIR/stdout.log" 2>&1 < /dev/null &

PID=$!
sleep 2
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
echo "$PID" > "$RUN_DIR/PID"
echo "$PGID" > "$RUN_DIR/PGID"
echo "host=$(hostname) profile=$NCCL_PROFILE torchrun_pid=$PID pgid=$PGID nproc=$NPROC gpus=$GPUS run_dir=$RUN_DIR"
