#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "==> Creating virtual environment"
python -m venv .venv
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

Write-Host "==> Installing PyTorch (CPU) and project dependencies"
& $Python -m pip install --upgrade pip
& $Pip install torch --index-url https://download.pytorch.org/whl/cpu
& $Pip install -e ".[dev]"

Write-Host "==> Downloading ExifTool for Windows"
$ExifToolDir = Join-Path $ProjectRoot "assets\exiftool"
New-Item -ItemType Directory -Force -Path $ExifToolDir | Out-Null

$ExifToolZip = Join-Path $env:TEMP "exiftool-13.59_64.zip"
$ExifToolUrl = "https://exiftool.org/exiftool-13.59_64.zip"
Invoke-WebRequest -Uri $ExifToolUrl -OutFile $ExifToolZip

$ExtractDir = Join-Path $env:TEMP "exiftool-extract"
if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
Expand-Archive -Path $ExifToolZip -DestinationPath $ExtractDir

$WinDir = Get-ChildItem -Path $ExtractDir -Directory | Where-Object { $_.Name -like "exiftool-*" } | Select-Object -First 1
Copy-Item -Path (Join-Path $WinDir.FullName "exiftool(-k).exe") -Destination (Join-Path $ExifToolDir "exiftool.exe") -Force
Copy-Item -Path (Join-Path $WinDir.FullName "exiftool_files") -Destination $ExifToolDir -Recurse -Force

Write-Host "==> Building executable with PyInstaller"
& $Python -m PyInstaller quicktag.spec --distpath dist/win --noconfirm --clean

$DistDir = Join-Path $ProjectRoot "dist\win\quicktag"
New-Item -ItemType Directory -Force -Path (Join-Path $DistDir "input") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DistDir "output") | Out-Null
Copy-Item -Path (Join-Path $ProjectRoot "config.example.yaml") -Destination (Join-Path $DistDir "config.yaml") -Force
Copy-Item -Path (Join-Path $ProjectRoot "tags.example.yaml") -Destination (Join-Path $DistDir "tags.yaml") -Force
Copy-Item -Path (Join-Path $ProjectRoot "docs\DIST_README_WIN.md") -Destination (Join-Path $DistDir "README.md") -Force
Copy-Item -Path $ExifToolDir -Destination (Join-Path $DistDir "exiftool") -Recurse -Force

Write-Host "==> Build complete: $DistDir"
