"""AIC 2026 competition toolkit."""

import os

# Fix OpenMP duplicate lib runtime conflict on Windows (Error #15 / 0xC0000005)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from .paths import default_data_root

data_root = str(default_data_root())

if data_root:
    cache_dir = os.path.join(data_root, ".cache")
    os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_dir, "torch"))
    os.environ.setdefault("UV_CACHE_DIR", os.path.join(cache_dir, "uv"))
    os.environ.setdefault("PIP_CACHE_DIR", os.path.join(cache_dir, "pip"))

__version__ = "0.1.0"

