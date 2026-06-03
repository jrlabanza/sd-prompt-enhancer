"""
sd-forge-prompt-enhancer / install.py
=====================================

Ensures onnxruntime is available for the WD14 image-to-prompt feature.
Tries GPU build first (CUDA), falls back to CPU build.

The 33k artist list and the gradio UI itself need nothing extra - For