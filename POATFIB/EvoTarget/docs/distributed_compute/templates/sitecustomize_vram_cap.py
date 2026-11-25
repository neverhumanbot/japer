"""Out-of-tree per-process CUDA memory fraction cap for safe co-tenant GPU sharing.

Place this file in a dedicated directory (e.g. $DATA/cotenant/sitecustomize.py) and
inject via PYTHONPATH=$DATA/cotenant together with:
  ULTRA_CUDA_MEM_FRACTION=0.45
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CRITICAL: os.environ.pop() removes the variable immediately in the parent process so
multiprocessing.spawn DataLoader workers never import torch or create CUDA contexts.
"""
import os

_frac = os.environ.pop("ULTRA_CUDA_MEM_FRACTION", "")
if _frac:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(_frac), 0)
        print(f"cotenant: cuda memory fraction {_frac}", flush=True)
