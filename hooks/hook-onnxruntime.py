"""PyInstaller hook: collect onnxruntime and tokenizers."""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("onnxruntime") + ["tokenizers"]
