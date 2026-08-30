"""AIC 2026 competition toolkit."""

import os

# Fix OpenMP duplicate lib runtime conflict on Windows (Error #15 / 0xC0000005)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

__version__ = "0.1.0"

