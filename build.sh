#! /bin/bash

set -euo pipefail

echo "Building QuickTag."

echo "Checking package dependencies..."
if ! command -v python &> /dev/null; then
    echo "Python could not be found. Please install Python and try again."
    exit 1
fi

echo "Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
pip install -e ".[dev]"

echo "Building executable..."
pyinstaller quicktag.spec --noconfirm --clean

echo "Copying example files..."
DIST_DIR="dist/quicktag"
mkdir -p $DIST_DIR/input $DIST_DIR/output
cp config.example.yaml $DIST_DIR/config.yaml
cp tags.example.yaml $DIST_DIR/tags.yaml
cp -r assets/exiftool $DIST_DIR/exiftool