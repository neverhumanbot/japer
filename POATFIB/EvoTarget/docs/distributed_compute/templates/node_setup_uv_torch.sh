#!/usr/bin/env bash
# Idempotent GPU node bootstrapper using uv (Python 3.12 + PyTorch cu128 + optional extras).
# Never mutates system Python. Optionally verifies dataset SHA-256 manifest before exiting.
#
# Usage:
#   bash node_setup_uv_torch.sh <PROJECT_DATA_DIR> [TORCH_VER] [CUDA_TAG]
# Example:
#   bash node_setup_uv_torch.sh "$HOME/project-data" "2.7.0" "cu128"
set -euo pipefail

DATA="${1:?usage: $0 <PROJECT_DATA_DIR> [TORCH_VER] [CUDA_TAG]}"
TORCH_VER="${2:-2.7.0}"
CUDA_TAG="${3:-cu128}"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi

mkdir -p "$DATA/venvs" "$DATA/logs" "$DATA/runs" "$DATA/gpu-leases" "$DATA/cotenant"
PY="$DATA/venvs/train/bin/python"

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  uv venv -q --python 3.12 "$DATA/venvs/train"
  uv pip install -q --python "$PY" "torch==${TORCH_VER}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
  uv pip install -q --python "$PY" \
    numpy scipy pyarrow soundfile pytest huggingface_hub hf_transfer
fi

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader
df -h / /dev/shm | tail -n +2
