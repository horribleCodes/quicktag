# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

block_cipher = None
project_root = Path(SPECPATH)
sys.path.insert(0, str(project_root / "hooks"))
from metadata_packages import RUNTIME_METADATA

package_metadata = []
for distribution in RUNTIME_METADATA:
    package_metadata += copy_metadata(distribution)

a = Analysis(
    [str(project_root / "src" / "quicktag" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=package_metadata,
    hiddenimports=["tokenizers", "onnxruntime"],
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "transformers", "pytest", "sympy", "onnx"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="quicktag",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name="quicktag",
)
