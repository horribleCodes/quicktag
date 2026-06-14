#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Building QuickTag"

if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "ERROR: Python not found. Install Python 3.11+ and try again."
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)

has_exiftool() {
    command -v exiftool &>/dev/null \
        || [[ -x assets/exiftool/exiftool ]] \
        || [[ -f assets/exiftool/exiftool && -d assets/exiftool/lib ]]
}

if ! has_exiftool; then
    echo ""
    echo "WARNING: ExifTool was not found. QuickTag needs ExifTool to write image metadata."
    echo "Install it before running the built executable:"
    echo ""
    echo "  Arch Linux:        sudo pacman -S perl-image-exiftool"
    echo "  Debian / Ubuntu:   sudo apt install libimage-exiftool-perl"
    echo "  Fedora:            sudo dnf install perl-Image-ExifTool"
    echo "  macOS:             brew install exiftool"
    echo ""
    echo "Or follow https://exiftool.org/install.html"
    echo ""
    echo "Continuing the build — metadata writing will fail at runtime without ExifTool."
    echo ""
fi

echo "==> Creating virtual environment"
$PYTHON -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing PyTorch (CPU) and project dependencies"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

echo "==> Building executable with PyInstaller"
pyinstaller quicktag.spec --noconfirm --clean

DIST_DIR="dist/linux/quicktag"
mkdir -p "$DIST_DIR/input" "$DIST_DIR/output"
cp config.example.yaml "$DIST_DIR/config.yaml"
cp tags.example.yaml "$DIST_DIR/tags.yaml"
cp docs/DIST_README_LINUX.md "$DIST_DIR/README.md"

if [[ -d assets/exiftool ]] && [[ -n "$(ls -A assets/exiftool 2>/dev/null)" ]]; then
    cp -r assets/exiftool "$DIST_DIR/exiftool"
fi

echo "==> Build complete: $DIST_DIR"
