#!/usr/bin/env bash
# Supervised hub-initiated reverse SSH tunnels for cross-site DiLoCo behind NAT/firewalls.
# The DiLoCo hub listens ONLY on 127.0.0.1:$PORT on the coordinator; each remote worker
# reaches the hub at ITS OWN 127.0.0.1:$PORT without opening any public ports beyond :22.
#
# Usage (run on coordinator where hub listens on 127.0.0.1:$PORT):
#   bash diloco_reverse_tunnels.sh <PORT> <STATE_DIR> <user@ipv4> [user@ipv4 ...]
# Example:
#   bash diloco_reverse_tunnels.sh 8768 "$HOME/diloco-state/tunnels" ubuntu@<CLOUD_NODE_1_IP>
#
# Stop cleanly:
#   awk '{print $1}' "$STATE_DIR/tunnels.pids" | xargs -r kill
set -euo pipefail

PORT="${1:?usage: $0 <PORT> <STATE_DIR> <user@ipv4> ...}"
STATE="${2:?usage: $0 <PORT> <STATE_DIR> <user@ipv4> ...}"
shift 2

if [ "$#" -lt 1 ]; then
  echo "error: provide at least one user@ipv4 endpoint" >&2
  exit 2
fi

mkdir -p "$STATE"
: > "$STATE/tunnels.pids"

for dst in "$@"; do
  (
    while true; do
      ssh -N \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o BatchMode=yes \
        -R "127.0.0.1:${PORT}:127.0.0.1:${PORT}" \
        "$dst"
      echo "$(date -u +%FT%TZ) tunnel to $dst exited ($?), restarting in 5s" >> "$STATE/tunnels.log"
      sleep 5
    done
  ) &
  echo "$! $dst" >> "$STATE/tunnels.pids"
done

echo "started supervised tunnels on loopback :$PORT -> $(tr '\n' '; ' < "$STATE/tunnels.pids")"
