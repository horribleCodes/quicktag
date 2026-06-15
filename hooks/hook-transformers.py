"""PyInstaller hook: bundle all transformers.models submodules for lazy imports."""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("transformers.models") + ["regex"]
