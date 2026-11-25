#!/usr/bin/env bash
# Fail-closed GPU preflight guard & bounded P-slot parallel campaign runner.
# Distilled from JOPE v2 run_race2_gpu.sh & cluster_gpu_worker.py on xbabe5-xbabe9.
#
# Enforces:
#   1. Exact SHA-256 verification of worker script before launch
#   2. Physical GPU UUID verification (catches PCIe renumbering across reboots)
#   3. Fail-closed idle-GPU gate (< MAX_IDLE_MIB used VRAM & 0 compute apps)
#   4. Never-clobber output directory check
#   5. Bounded P-slot concurrency with launch stagger & hard per-job timeout
#   6. Background 30s GPU VRAM/utilization telemetry log
#
# Usage:
#   GPU_INDEX=0 EXPECTED_UUID="GPU-..." WORKER=worker.py WORKER_SHA=<sha256> \
#   OUT_DIR=/path/to/out P=2 STAGGER=30 RUN_TIMEOUT=3600 \
#   bash gpu_preflight_guard.sh jobs.txt
#
# Each non-comment line in jobs.txt is passed as arguments to: python3 -B "$WORKER" <line>
set -euo pipefail

JOBS_FILE="${1:?usage: $0 <jobs.txt>}"
GPU_INDEX="${GPU_INDEX:?set GPU_INDEX}"
EXPECTED_UUID="${EXPECTED_UUID:?set EXPECTED_UUID from nvidia-smi -L}"
WORKER="${WORKER:?set WORKER script path}"
WORKER_SHA="${WORKER_SHA:?set expected WORKER_SHA}"
OUT_DIR="${OUT_DIR:?set non-existing OUT_DIR}"
P="${P:-2}"
STAGGER="${STAGGER:-30}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
MAX_IDLE_MIB="${MAX_IDLE_MIB:-500}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1

# 1. Worker SHA-256 check
GOT_SHA=$(sha256sum "$WORKER" | cut -d' ' -f1)
[ "$GOT_SHA" = "$WORKER_SHA" ] || { echo "FATAL: worker sha256 mismatch ($GOT_SHA != $WORKER_SHA)" >&2; exit 3; }

# 2. Physical GPU UUID check
GOT_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
[ "$GOT_UUID" = "$EXPECTED_UUID" ] || { echo "FATAL: GPU $GPU_INDEX is $GOT_UUID, expected $EXPECTED_UUID" >&2; exit 3; }

# 3. Fail-closed idle GPU gate
USED_MIB=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits)
APPS=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | awk 'NF' | wc -l)
if [ "$USED_MIB" -ge "$MAX_IDLE_MIB" ] || [ "$APPS" -gt 0 ]; then
  echo "FATAL: GPU $GPU_INDEX busy (${USED_MIB} MiB, ${APPS} compute apps); refusing to stomp" >&2
  exit 3
fi

# 4. Never-clobber output directory
[ ! -e "$OUT_DIR" ] || { echo "FATAL: $OUT_DIR already exists; refusing to overwrite" >&2; exit 3; }
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/campaign.log"
echo "START $(date -u +%FT%TZ) host=$(hostname) gpu=$GPU_INDEX uuid=$EXPECTED_UUID P=$P" >> "$LOG"

# 5. Background 30s GPU telemetry sampler
(
  while :; do
    echo "$(date -u +%FT%TZ) $(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" >> "$OUT_DIR/gpu_samples.txt"
    sleep 30
  done
) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

# 6. Bounded P-slot queue with stagger & SIGTERM->SIGKILL timeout
JOB_IDX=0
while read -r job_args; do
  [ -z "${job_args:-}" ] && continue
  [ "${job_args:0:1}" = "#" ] && continue
  while [ "$(jobs -rp | grep -v -x "$SAMPLER" | wc -l)" -ge "$P" ]; do wait -n; done
  JOB_IDX=$((JOB_IDX + 1))
  tag="job_$(printf '%03d' "$JOB_IDX")"
  (
    t0=$(date -u +%FT%TZ)
    # shellcheck disable=SC2086
    timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT" python3 -B "$WORKER" $job_args > "$OUT_DIR/$tag.log" 2>&1
    echo "DONE $tag rc=$? start=$t0 end=$(date -u +%FT%TZ)" >> "$LOG"
  ) &
  echo "LAUNCH $tag $(date -u +%FT%TZ) args=$job_args" >> "$LOG"
  sleep "$STAGGER"
done < "$JOBS_FILE"

while [ "$(jobs -rp | grep -v -x "$SAMPLER" | wc -l)" -gt 0 ]; do wait -n; done
echo "END $(date -u +%FT%TZ)" >> "$LOG"
