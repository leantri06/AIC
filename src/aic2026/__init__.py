"""AIC 2026 competition toolkit."""

import os

# Fix OpenMP duplicate lib runtime conflict on Windows (Error #15 / 0xC0000005)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if os.path.exists(r"D:\ai_challenge\data"):
    os.environ.setdefault("HF_HOME", r"D:\ai_challenge\data\.cache\huggingface")

__version__ = "0.1.0"

