#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $Python)) {
    Write-Host "==> Creating virtual environment"
    python -m venv .venv
} else {
    Write-Host "==> Using existing virtual environment"
}

$depsOk = $false
& $Python -c "import torch, PyInstaller" 2>$null
if ($LASTEXITCODE -eq 0) { $depsOk = $true }

if ($depsOk) {
    Write-Host "==> Dependencies already installed"
    & $Pip install -e ".[dev]"
} else {
    Write-Host "==> Installing PyTorch (CPU) and project dependencies"
    & $Python -m pip install --upgrade pip
    & $Pip install torch --index-url https://download.pytorch.org/whl/cpu
    & $Pip install -e ".[dev]"
}

$ExifToolDir = Join-Path $ProjectRoot "assets\exiftool"
New-Item -ItemType Directory -Force -Path $ExifToolDir | Out-Null
$ExifToolExe = Join-Path $ExifToolDir "exiftool.exe"

if (-not (Test-Path $ExifToolExe)) {
    Write-Host "==> Downloading ExifTool for Windows"
    $ExifToolZip = Join-Path $env:TEMP "exiftool-13.59_64.zip"
    $ExifToolUrl = "https://exiftool.org/exiftool-13.59_64.zip"
    Invoke-WebRequest -Uri $ExifToolUrl -OutFile $ExifToolZip

    $ExtractDir = Join-Path $env:TEMP "exiftool-extract"
    if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
    Expand-Archive -Path $ExifToolZip -DestinationPath $ExtractDir

    $WinDir = Get-ChildItem -Path $ExtractDir -Directory | Where-Object { $_.Name -like "exiftool-*" } | Select-Object -First 1
    Copy-Item -Path (Join-Path $WinDir.FullName "exiftool(-k).exe") -Destination $ExifToolExe -Force
    Copy-Item -Path (Join-Path $WinDir.FullName "exiftool_files") -Destination $ExifToolDir -Recurse -Force
} else {
    Write-Host "==> Using cached ExifTool"
}

Write-Host "==> Building executable with PyInstaller"
$PyInstallerArgs = @("quicktag.spec", "--distpath", "dist/win", "--noconfirm")
if ($env:CI -ne "true") {
    $PyInstallerArgs += "--clean"
}
& $Python -m PyInstaller @PyInstallerArgs

$DistDir = Join-Path $ProjectRoot "dist\win\quicktag"
New-Item -ItemType Directory -Force -Path (Join-Path $DistDir "input") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DistDir "output") | Out-Null
Copy-Item -Path (Join-Path $ProjectRoot "config.example.yaml") -Destination (Join-Path $DistDir "config.yaml") -Force
Copy-Item -Path (Join-Path $ProjectRoot "tags.example.yaml") -Destination (Join-Path $DistDir "tags.yaml") -Force
Copy-Item -Path (Join-Path $ProjectRoot "docs\DIST_README_WIN.md") -Destination (Join-Path $DistDir "README.md") -Force
Copy-Item -Path $ExifToolDir -Destination (Join-Path $DistDir "exiftool") -Recurse -Force

Write-Host "==> Build complete: $DistDir"
