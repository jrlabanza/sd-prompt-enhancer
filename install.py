"""
sd-forge-prompt-enhancer / install.py
=====================================

Ensures onnxruntime is available for the WD14 image-to-prompt feature.
Tries GPU build first (CUDA), falls back to CPU build.

The 33k artist list and the gradio UI itself need nothing extra - Forge
provides gradio. WD14 model files are downloaded lazily on first use of
the Image-to-Prompt section, not at install time.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _pip_install(pkg: str) -> bool:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", pkg]
        )
        return True
    except Exception as e:
        print(f"[sd-forge-prompt-enhancer] pip install {pkg} failed: {e}")
        return False


if not _have("onnxruntime"):
    print("[sd-forge-prompt-enhancer] installing onnxruntime (for image-to-prompt)")
    if not _pip_install("onnxruntime-gpu"):
        _pip_install("onnxruntime")

if not _have("PIL"):
    _pip_install("Pillow")
