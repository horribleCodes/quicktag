"""PyInstaller hook: collect onnxruntime and tokenizers."""

hiddenimports = [
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
    "tokenizers",
]
