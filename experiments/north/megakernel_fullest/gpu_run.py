"""Acquire the shared research GPU before importing any MLX code."""
import fcntl
import os
from pathlib import Path
import runpy
import sys

if not Path('/tmp/north-minimal-baseline-20260914.done').exists():
    raise SystemExit('Parent minimal baseline has not released the GPU; CPU-only work required.')
with open('/tmp/north-metal-research-gpu.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
    runpy.run_path(sys.argv[0], run_name='__main__')
